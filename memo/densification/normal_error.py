from typing import List
import torch
from memo.arguments import PipelineParams
from memo.scene.cameras import Camera
from memo.scene.gaussian_model import GaussianModel
from memo.gaussian_renderer import render_simp, render
from memo.utils.geometry_utils import depth_to_normal_with_mask
from memo.utils.general_utils import cull_stride


@torch.no_grad()
def compute_normal_error(
    gaussians: GaussianModel,
    cameras: List[Camera],
    pipe: PipelineParams,
    background: torch.Tensor,
    method: str,
    depth_ratio: float,
    max_culling_images: int,
    min_visibility: int = 2,
) -> torch.Tensor:
    """Per-Gaussian normal error summed across (sampled) views, for normal-aware densification.

    Two rasterizer calls per camera:
      1. ``render_simp`` for ``area_proj`` (per-Gaussian projected area).
      2. ``render`` for ``gidx``, the alpha-blended normal field via
         ``extra_features``, and the median and expected depths. The renderer
         rotates ``extra_features`` to the camera's frame time before alpha
         blending; no per-camera rotation is needed here.

    The normal error compares the normal field with the normal of the blended
    depth (1 - depth_ratio) expected + depth_ratio median, the depth L_N and
    L_DN align with, masked where that depth is invalid.

    ``max_culling_images`` caps the rendered (view, frame) pair count: the
    gate ``(view.t - offset) % interval == 0`` keeps a uniform subset of
    frames (with all views per kept frame), matching ``culling_with_clone``'s
    sampling pattern. Gaussians visible in fewer than ``min_visibility``
    pairs are zeroed out so they cannot drive a clone decision
    (high-variance error estimate).

    Sum-then-CDF: per-Gaussian errors are summed across pairs and returned
    unscaled. Selection is done at the caller via a quantile cutoff, which is
    scale-invariant.
    """
    assert method in ["count", "area", "none"], "Invalid method"

    base_normals = gaussians.convert_features_to_normals()
    base_normals = torch.nn.functional.normalize(base_normals, dim=-1)

    gaussian_errors = torch.zeros_like(gaussians._xyz[:, 0])
    count_vis = torch.zeros_like(gaussian_errors)

    n_cams = len(cameras)
    if max_culling_images < n_cams:
        interval = cull_stride(n_cams, max_culling_images)
        offset = interval // 2
    else:
        interval = 1
        offset = 0

    for cam in cameras:
        if interval > 1 and (cam.t - offset) % interval != 0:
            continue

        # LMDB cameras carry CPU-side transforms; move them to GPU before
        # rendering.
        cam.to_device("cuda")

        simp_render_pkg = render_simp(
            viewpoint_camera=cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            culling=None,
            apply_4d=True,
        )
        gaussians_proj_area = simp_render_pkg["area_proj"]

        gaussian_render_pkg = render(
            viewpoint_camera=cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            extra_features=base_normals,
            apply_4d=True,
            culling=None,
            require_coord=False,
            require_depth=True,
        )
        normal_field_render = gaussian_render_pkg["extra_features"]  # (3, H, W)
        gidx = gaussian_render_pkg["gidx"]
        view_to_world_transform = cam.world_view_transform[:3, :3].permute(-1, -2)

        median_depth = gaussian_render_pkg["median_depth"]
        blended_depth = torch.where(
            median_depth > 0,
            (1. - depth_ratio) * gaussian_render_pkg["expected_depth"] + depth_ratio * median_depth,
            median_depth,
        )
        blended_depth_normal, error_mask = depth_to_normal_with_mask(cam, blended_depth)
        error_mask = error_mask.squeeze(0)
        blended_depth_normal = (blended_depth_normal.permute(1, 2, 0) @ view_to_world_transform).permute(2, 0, 1)

        normal_error = 1. - (normal_field_render * blended_depth_normal).sum(dim=0)  # (H, W)
        normal_error = torch.where(error_mask, normal_error, torch.zeros_like(normal_error))

        # The rasterizer's gidx is -1 on background / non-contributing
        # pixels. Filter to foreground before index_add_ to avoid a CUDA
        # device-side assert.
        flat_idx = gidx.flatten().long()
        flat_err = normal_error.flatten()
        fg_mask = flat_idx >= 0
        flat_idx = flat_idx[fg_mask]
        flat_err = flat_err[fg_mask]

        gaussian_errors_i = torch.zeros_like(gaussians_proj_area, dtype=torch.float32)
        gaussian_errors_i.index_add_(0, flat_idx, flat_err)

        # Per-camera per-Gaussian pixel count (drives both count-method
        # normalization and the cross-camera visibility tally below).
        gaussian_count = torch.zeros_like(gaussians_proj_area, dtype=torch.float32)
        gaussian_count.index_add_(0, flat_idx, torch.ones_like(flat_err))

        if method == "count":
            valid_mask = gaussian_count > 0
            gaussian_errors_i = torch.where(valid_mask, gaussian_errors_i / gaussian_count, torch.zeros_like(gaussian_errors_i))
        elif method == "area":
            valid_area_mask = gaussians_proj_area > 0
            gaussian_errors_i = torch.where(valid_area_mask, gaussian_errors_i / gaussians_proj_area, torch.zeros_like(gaussian_errors_i))

        gaussian_errors = gaussian_errors + gaussian_errors_i

        # Cross-camera visibility: increment count_vis for each Gaussian that
        # caught >= 1 foreground pixel in this camera. Avoids unique() so the
        # deferred-assert sync-point is gone.
        count_vis = count_vis + (gaussian_count > 0).float()

    gaussian_errors = torch.where(count_vis >= min_visibility, gaussian_errors, torch.zeros_like(gaussian_errors))

    return gaussian_errors
