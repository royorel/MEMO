from typing import List, Optional
import numpy as np
import torch
from einops import einsum
import math
from diff_gaussian_rasterization_sof import (
    GaussianRasterizationSettings,
    ExtendedSettings,
    GaussianRasterizer,
    DebugVisualization,
    SortMode,
    GlobalSortOrder,
)
from memo.scene.cameras import Camera
from memo.utils.graphics_utils import fov2focal
from memo.utils.general_utils import quaternion_raw_multiply
from tqdm import tqdm


@torch.no_grad()
def get_frustum_mask(points: torch.Tensor, cameras: List[Camera], near: float = 0.02, far: float = 1e6):
    H, W = cameras[0].image_height, cameras[0].image_width

    intrinsics = torch.stack(
        [
            torch.Tensor(
                [[fov2focal(cam.FoVx, cam.image_width), 0, cam.cx if cam.cx is not None else cam.image_width / 2],
                 [0, fov2focal(cam.FoVy, cam.image_height), cam.cy if cam.cy is not None else cam.image_height / 2],
                 [0, 0, 1]]
            ) for cam in cameras
        ],
        dim=0
    ).to(points.device)

    # full_proj_matrices: (n_view, 4, 4)
    view_matrices = torch.stack(
        [cam.world_view_transform for cam in cameras], dim=0
    ).transpose(1, 2)

    ones = torch.ones_like(points[:, 0]).unsqueeze(-1)
    # homo_points: (N, 4)
    homo_points = torch.cat([points, ones], dim=-1)

    # uv_points: (n_view, N, 4, 4)
    # Apply batch matrix multiplication to get uv_points for all cameras
    view_points = einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
    view_points = view_points[:, :, :3]

    uv_points = einsum(intrinsics, view_points, "n_view b c, n_view N c -> n_view N b")

    z = uv_points[:, :, -1:]
    uv_points = uv_points[:, :, :2] / z
    u, v = uv_points[:, :, 0], uv_points[:, :, 1]

    # Apply near-far culling
    depth = view_points[:, :, -1]
    cull_near_fars = (depth >= near) & (depth <= far)

    # Apply frustum mask
    mask = torch.any(cull_near_fars & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1), dim=0)
    return mask


# Defines default SOF settings based on the config file from the SOF repository
def default_splat_args() -> ExtendedSettings:
    splat_args = ExtendedSettings()

    splat_args.culling_settings.hierarchical_4x4_culling = False
    splat_args.culling_settings.rect_bounding = True
    splat_args.culling_settings.tight_opacity_bounding = True
    splat_args.culling_settings.tile_based_culling = False

    splat_args.load_balancing = True
    splat_args.proper_ewa_scaling = False
    splat_args.exact_depth = True

    splat_args.sort_settings.queue_sizes.per_pixel = 4
    splat_args.sort_settings.queue_sizes.tile_2x2 = 8
    splat_args.sort_settings.queue_sizes.tile_4x4 = 64
    splat_args.sort_settings.sort_mode = SortMode(3)
    splat_args.sort_settings.sort_order = GlobalSortOrder(3)

    return splat_args


def get_full_proj_transform_inverse(viewpoint_camera: Camera) -> torch.Tensor:
    return (
        viewpoint_camera.world_view_transform.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0).inverse()


@torch.no_grad()
def compute_transmittance(
    points3D, viewpoint_camera, pc, bg_color : torch.Tensor,
    splat_args : ExtendedSettings,
    thin_structures : bool,
    t : Optional[float] = None,
):
    """Compute transmittance at `points3D` for the given viewpoint.

    Args:
        thin_structures (bool): Evaluate the Gaussians with the 3D-filtered
            scales but the trained opacities.
        t (float, optional): Time in [0, 1] at which to evaluate the 4D GPs.
            None reads the Gaussians as they are. Defaults to None.
    """
     # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=viewpoint_camera.cx if viewpoint_camera.cx is not None else viewpoint_camera.image_width / 2.0,
        cy=viewpoint_camera.cy if viewpoint_camera.cy is not None else viewpoint_camera.image_height / 2.0,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        inv_viewprojmatrix=get_full_proj_transform_inverse(viewpoint_camera=viewpoint_camera),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        settings=splat_args,
        debug_data=DebugVisualization(),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    rotations = pc.get_rotation

    if t is not None:
        means3D = means3D + pc.position_gp(t)
        d_quat = pc.rotation_gp(t)
        rotations = quaternion_raw_multiply(d_quat, rotations)

    if pc.use_unbounded_opacity:
        scales = pc.get_scaling_with_3D_filter
        opacity = pc.get_contribution(viewpoint_camera=viewpoint_camera)
    elif thin_structures:
        # The 3D filter scales a thin Gaussian's opacity down by its footprint
        # ratio, so thin structures never reach the 0.5 level. Keep the
        # filtered extent but the trained opacity.
        scales = pc.get_scaling_with_3D_filter
        opacity = pc.get_opacity
    else:
        scales, opacity = pc.get_scaling_n_opacity_with_3D_filter

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    transmittance, radii = rasterizer.compute_transmittance(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        view2gaussian_precomp = None
    )

    return {
        "transmittance": transmittance,
        "visibility_filter" : radii > 0,
        "radii": radii
    }


@torch.no_grad()
def evaluate_vacancy_sof_fast(
    points,
    views,
    gaussians,
    background,
    splat_args: ExtendedSettings,
    znear,
    zfar,
    t,
    thin_structures,
):
    """
    Evaluate if the vacancy at a point is greater than a threshold.
    Returns a boolean mask indicating which points have a vacancy value greater than the threshold in splat_args.meshing_settings.transmittance_threshold.

    Args:
        points (torch.Tensor): Points to evaluate the vacancy field at. (N, 3)
        views (List[Camera]): Views to evaluate the vacancy field at.
        gaussians (GaussianModel4D): Gaussian model to evaluate the vacancy field at.
        background (torch.Tensor): Background color.
        splat_args (ExtendedSettings): Rasterizer settings, with the transmittance threshold.
        znear (float): Near plane of the view frustums.
        zfar (float): Far plane of the view frustums.
        t (float): Time in [0, 1] at which to evaluate the 4D GPs.
        thin_structures (bool): Evaluate the Gaussians with the 3D-filtered
            scales but the trained opacities.

    Returns:
        torch.Tensor: Boolean mask indicating which points have a vacancy value greater than the threshold. Shape (N,).
    """
    vacancy = torch.zeros((points.shape[0]), dtype=torch.bool, device="cuda")
    update_idx = torch.arange(points.shape[0], device="cuda")

    # The Gaussians are moved to time t once for all views.
    orig_xyz = gaussians._xyz
    orig_rotation = gaussians._rotation
    gaussians._xyz = orig_xyz + gaussians.position_gp(t)
    d_quat = gaussians.rotation_gp(t)
    gaussians._rotation = quaternion_raw_multiply(d_quat, gaussians.get_rotation)

    # Permute the list of views
    permuted_indices = np.random.permutation(len(views))
    views_to_use = [views[i] for i in permuted_indices]

    for _, view in enumerate(tqdm(views_to_use, desc="Vacancy evaluation progress")):
        # Get current remaining points to update
        N_current = update_idx.shape[0]
        current_points = points[update_idx]  # (N_current, 3)

        # Get frustum mask for current view
        frustum_mask = get_frustum_mask(points=current_points, cameras=[view], near=znear, far=zfar)  # (N_current,)

        # Compute transmittance of points that are in the frustum
        ret = compute_transmittance(
            points3D=current_points[frustum_mask],  # (N_frustum, 3)
            viewpoint_camera=view,
            pc=gaussians,
            bg_color=background,
            splat_args=splat_args,
            thin_structures=thin_structures,
        )

        # Compute mask of points that pass the threshold
        passing_mask = torch.zeros((N_current), dtype=torch.bool, device="cuda")  # (N_current,)
        passing_mask[frustum_mask] = ret["transmittance"] > splat_args.meshing_settings.transmittance_threshold  # (N_frustum,)

        # Get indices of points that pass the threshold
        pass_idx = update_idx[passing_mask]  # (N_pass,)

        # Update vacancy of points that pass the threshold
        vacancy[pass_idx] = True  # (N_pass,)

        # Update update_idx to only include points that did not pass the threshold yet
        update_idx = update_idx[~passing_mask]  # (N_current - N_pass,)

    gaussians._xyz = orig_xyz
    gaussians._rotation = orig_rotation

    return vacancy
