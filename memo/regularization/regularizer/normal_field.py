from typing import Dict, Any, List
import numpy as np
import torch
from memo.arguments import PipelineParams
from memo.scene.cameras import Camera
from memo.scene.gaussian_model import GaussianModel
from memo.regularization.normal_field import get_gaussian_std_in_direction
from memo.densification.normal_error import compute_normal_error
from memo.utils.geometry_utils import depth_to_normal_with_mask
from memo.utils.general_utils import build_rotation


def compute_normal_field_regularization(
    render_pkg: Dict[str, torch.Tensor],
    viewpoint_cam: Camera,
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Normal-field alignment loss L_N.

    Aligns the alpha-blended learned normals, rendered as extra feature
    channels in the same rasterizer call as the RGB, with the normal of the
    blended depth (1 - r) expected + r median that L_DN also reads, where r is
    `depth_ratio_for_alignment`.
    """
    oriented_normal = render_pkg["extra_features"]  # (3, H, W)
    # With bg=0 for extra channels, the alignment reference is alpha
    # instead of 1, so that semi-transparent pixels aren't penalized
    # for low normal magnitude (which is a transparency effect, not
    # a misalignment). alpha = 1 - T_final = sum(alpha_i * T_i).
    alignment_ref = render_pkg["mask"].squeeze(0)  # (H, W)

    view_to_world_transform = viewpoint_cam.world_view_transform[:3,:3].permute(-1, -2)

    ratio = config["depth_ratio_for_alignment"]
    median_depth = render_pkg["median_depth"]  # (1, H, W)
    blended_depth = torch.where(
        median_depth > 0,
        (1. - ratio) * render_pkg["expected_depth"] + ratio * median_depth,
        median_depth,
    )  # (1, H, W)
    blended_depth_normal, valid_depth_points = depth_to_normal_with_mask(
        viewpoint_cam,
        blended_depth,
    )  # (3, H, W), (H, W)
    blended_depth_normal = (
        blended_depth_normal.permute(1, 2, 0) @ view_to_world_transform
    ).permute(2, 0, 1)  # (3, H, W)
    normal_error_map = alignment_ref - (oriented_normal * blended_depth_normal).sum(dim=0)  # (H, W)
    normal_field_alignment_loss = torch.where(
        valid_depth_points.squeeze(),  # (H, W)
        normal_error_map,  # (H, W)
        torch.zeros_like(normal_error_map),  # (H, W)
    ).mean() * config["normal_field_alignment_weight"]

    return {
        "normal_field_loss": normal_field_alignment_loss,
        "normal_field_alignment_loss": normal_field_alignment_loss.detach(),
    }


@torch.no_grad()
def densify_normal_field(
    gaussians: GaussianModel,
    cameras: List[Camera],
    pipe: PipelineParams,
    background: torch.Tensor,
    config: Dict[str, Any],
    max_culling_images: int,
):
    """Normal-aware densification.

    Gaussians in the top `densification_normal_errors_quantile` of the
    normal error are split along their normal: each is shifted forward along
    its normal and a clone with the opposite normal is placed behind it.
    """
    # Get Gaussian normals
    gaussian_normals = gaussians.convert_features_to_normals()  # (N_gaussians, 3)
    gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)  # (N_gaussians, 3)

    # Compute normal errors
    normal_errors = compute_normal_error(
        gaussians=gaussians,
        cameras=cameras,
        pipe=pipe,
        background=background,
        method=config["densification_normalization_method"],  # "count" or "area" or "none"
        depth_ratio=config["depth_ratio_for_alignment"],
        max_culling_images=max_culling_images,
    )  # (N_gaussians,)

    # Compute normal errors quantile
    normal_errors_quantile = torch.quantile(normal_errors, q=1. - config["densification_normal_errors_quantile"])

    # Densification mask
    densification_mask = normal_errors > normal_errors_quantile  # (N_gaussians,)

    # Adjust scale of Gaussians to be densified. The idea is to divide the volume of the densified Gaussian by 2,
    # while taking into account the direction of the normal.
    if config["maintain_constant_volume"]:
        #   > First, we compute the local basis of the Gaussian
        local_basis = build_rotation(
            r=gaussians._rotation[densification_mask]  # (N_gaussians_to_densify, 3, n_vectors_in_basis)
        ).transpose(-1, -2)  # (N_gaussians_to_densify, n_vectors_in_basis, 3)

        #   > Then, we compute the projections of the normals on the local basis
        projections_on_local_basis = (
            gaussian_normals[densification_mask].unsqueeze(1)  # (N_gaussians_to_densify, 1, 3)
            * local_basis  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
        ).sum(dim=-1)  # (N_gaussians_to_densify, n_vectors_in_basis)

        #   > We compute the logarithm of the adjustment factors
        log_adjustment_factors = np.log(1. / 2.) * projections_on_local_basis ** 2

        #   > Adjust the scaling of the Gaussians
        gaussians._scaling[densification_mask] = gaussians._scaling[densification_mask] + log_adjustment_factors

    # Compute xyz of cloned Gaussians as same xyz minus a small multiple of the normal
    new_xyz = gaussians._xyz[densification_mask]  # (N_new_gaussians, 3)
    new_normals = - gaussian_normals[densification_mask]  # (N_new_gaussians, 3)
    normal_stds = get_gaussian_std_in_direction(
        directions=new_normals.unsqueeze(1),  # (N_new_gaussians, 1, 3)
        gaussian_scaling=gaussians.get_scaling_with_3D_filter[densification_mask].detach(),
        gaussian_rotation=gaussians._rotation[densification_mask].detach(),
        normalize_directions=False,
    )  # (N_gaussians, 1)
    delta = 0.1
    new_xyz = new_xyz + delta * normal_stds * new_normals

    # Compute normal features of cloned Gaussians to obtain the opposite normal
    new_gaussian_features = gaussians._gaussian_features[densification_mask]  # (N_new_gaussians, n_features)
    new_gaussian_features[:, -1:] = -new_gaussian_features[:, -1:]

    # Update xyz of densified Gaussians to be xyz plus a small multiple of the normal
    gaussians._xyz[densification_mask] = (
        gaussians._xyz[densification_mask]
        + delta * normal_stds * gaussian_normals[densification_mask]
    )

    # Densify Gaussians
    gaussians.densify_and_clone_from_mask(
        selected_pts_mask=densification_mask,
        new_xyz=new_xyz,
        new_gaussian_features=new_gaussian_features,
    )
