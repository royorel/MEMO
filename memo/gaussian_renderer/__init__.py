#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from memo.scene.gaussian_model import GaussianModel
from memo.utils.sh_utils import eval_sh
from memo.utils.general_utils import quaternion_raw_multiply, build_rotation
from diff_gaussian_rasterization_memo import (
    GaussianRasterizationSettings as MemoRasterizationSettings,
    GaussianRasterizer as MemoRasterizer,
)


def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, extra_features = None, flag_max_count=True, culling=None, require_coord=False, require_depth=False, apply_4d=False):
    """Render with the MEMO rasterizer.

    With apply_4d=True, pc must be a GaussianModel4D and viewpoint_camera
    must carry normalized_t. Position and rotation GPs are evaluated at t
    before the rasterizer call, and `extra_features` (learned normals) are
    rotated by the same delta rotation. The extra features are alpha-blended
    as additional channels after the RGB.
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = MemoRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=viewpoint_camera.cx if viewpoint_camera.cx is not None else viewpoint_camera.image_width / 2.0,
        cy=viewpoint_camera.cy if viewpoint_camera.cy is not None else viewpoint_camera.image_height / 2.0,
        kernel_size=0.0,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_coord=require_coord,
        require_depth=require_depth,
        debug=False
    )

    rasterizer = MemoRasterizer(raster_settings=raster_settings)

    if apply_4d:
        assert viewpoint_camera.normalized_t is not None, "4D rendering requires normalized_t on the camera"
        t = viewpoint_camera.normalized_t
        pos_delta = pc.position_gp(t)
        means3D = pc.get_xyz + pos_delta
        d_quat = pc.rotation_gp(t)  # (N, 4) unit quaternion delta
        rotations = quaternion_raw_multiply(d_quat, pc.get_rotation)
        if extra_features is not None:
            R_delta = build_rotation(d_quat)
            extra_features = (R_delta @ extra_features.unsqueeze(-1)).squeeze(-1)
    else:
        means3D = pc.get_xyz
    means2D = screenspace_points
    if pc.use_unbounded_opacity:
        opacity = pc.get_contribution(viewpoint_camera=viewpoint_camera)
    else:
        opacity = pc.get_opacity_with_3D_filter

    scales = None
    cov3D_precomp = None
    if not apply_4d:
        rotations = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        if not apply_4d:
            rotations = pc.get_rotation

    # SH coefficients for RGB — always provided (extra_features go through
    # the separate extra_features parameter, not colors_precomp)
    dc = None
    shs = None
    colors_precomp = None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        dc, shs = pc.get_features_dc, pc.get_features_rest

    if culling is None:
        culling = torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    rendered_image, radii, rendered_expected_coord, rendered_median_coord, \
    rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal, \
    gidx, accum_max_count = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        culling=culling,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        extra_features=extra_features,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        flag_max_count=flag_max_count)

    # Split output: channels 0-2 are RGB, channels 3+ are extra feature maps
    if extra_features is not None:
        extra_channels = extra_features.shape[1]
        extra_maps = rendered_image[3:3+extra_channels]  # (K, H, W)
        rendered_image = rendered_image[:3]                # (3, H, W)
    else:
        extra_maps = None

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "area_max": accum_max_count,
            "mask": rendered_alpha,
            "normal": rendered_normal,
            "expected_depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "gidx": gidx,
            "extra_features": extra_maps,
            }


def render_simp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, culling=None, apply_4d=False):
    """
    Render the per-Gaussian importance statistics used by the culling events.

    Background tensor (bg_color) must be on GPU!

    When apply_4d=True, pc must be a GaussianModel4D and viewpoint_camera
    must carry a non-None normalized_t. Position and rotation GPs are
    evaluated at t before the rasterizer call.
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = MemoRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=viewpoint_camera.cx if viewpoint_camera.cx is not None else viewpoint_camera.image_width / 2.0,
        cy=viewpoint_camera.cy if viewpoint_camera.cy is not None else viewpoint_camera.image_height / 2.0,
        kernel_size=0.0,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_coord=False,
        require_depth=False,
        debug=False
    )

    rasterizer = MemoRasterizer(raster_settings=raster_settings)

    if apply_4d:
        assert viewpoint_camera.normalized_t is not None, "4D rendering requires normalized_t on the camera"
        t = viewpoint_camera.normalized_t
        pos_delta = pc.position_gp(t)
        means3D = pc.get_xyz + pos_delta
        d_quat = pc.rotation_gp(t)  # (N, 4) unit quaternion delta
        rotations = quaternion_raw_multiply(d_quat, pc.get_rotation)
    else:
        means3D = pc.get_xyz
    means2D = screenspace_points
    if pc.use_unbounded_opacity:
        opacity = pc.get_contribution(viewpoint_camera=viewpoint_camera)
    else:
        opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    cov3D_precomp = None
    if not apply_4d:
        rotations = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        if not apply_4d:
            rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling is None:
        culling = torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, \
    accum_weights_ptr, accum_weights_count, accum_max_count = rasterizer.render_simp(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        culling=culling,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
            }
