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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    extra_features,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    flag_max_count,
    culling,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        extra_features,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        flag_max_count,
        culling,
        raster_settings,
    )

def rasterize_gaussians_simp(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    culling,
    raster_settings,
):
    return _RasterizeGaussians.render_simp(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        culling,
        raster_settings,
    )



def rasterize_gaussians_depth(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    culling,
    raster_settings,
):
    return _RasterizeGaussians.render_depth(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        culling,
        raster_settings,
    )     

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        extra_features,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        flag_max_count,
        culling,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            extra_features,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.kernel_size,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.require_coord,
            raster_settings.require_depth,
            flag_max_count,
            culling,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, num_buckets, color, coord, mcoord, alpha, normal, depth, mdepth, gidx, accum_max_count, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_buckets, color, coord, mcoord, alpha, normal, depth, mdepth, gidx, accum_max_count, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(colors_precomp, extra_features, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)

        return color, radii, coord, mcoord, depth, mdepth, alpha, normal, gidx, accum_max_count
    
    def render_simp(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        culling,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            culling,
            raster_settings.debug,
        )


        num_rendered, color, accum_weights_ptr, accum_weights_count, accum_max_count, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians_simp(*args)


        return color, radii, accum_weights_ptr, accum_weights_count, accum_max_count

    def render_depth(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        culling,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            culling,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, out_pts, depth, accum_alpha, gidx, discriminants, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians_depth(*args)

        res =  {"render": color,
                "out_pts": out_pts,
                "rendered_depth": depth,
                "discriminants": discriminants,
                "gidx": gidx,
                "accum_alpha": accum_alpha,
                }

        return res
    
    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_coord, grad_mcoord, grad_depth, grad_mdepth, grad_alpha, grad_normal, grad_gidx, accum_max_count):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        colors_precomp, extra_features, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                extra_features,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.cx,
                raster_settings.cy,
                raster_settings.kernel_size,
                grad_out_color,
                grad_coord,
                grad_mcoord,
                grad_depth,
                grad_mdepth,
                grad_alpha,
                grad_normal,
                dc,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                num_buckets,
                sampleBuffer,
                raster_settings.require_coord,
                raster_settings.require_depth,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, \
                grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, \
                grad_extra_features = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, \
             grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations, \
             grad_extra_features = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_dc,
            grad_sh,
            grad_colors_precomp,
            grad_extra_features,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    cx : float
    cy : float
    kernel_size : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    require_coord : bool
    require_depth : bool
    debug : bool
    debug_grad: bool = False

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, culling, dc=None, shs=None, colors_precomp=None, extra_features=None, scales=None, rotations=None, cov3D_precomp=None, flag_max_count=False):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # Empty-placeholder tensors must be on CUDA — the backward pass passes
        # them to the CUDA kernel, which dereferences them as CUDA pointers
        # regardless of numel().
        if dc is None:
            dc = torch.empty(0, device='cuda')
        if shs is None:
            shs = torch.empty(0, device='cuda')
        if colors_precomp is None:
            colors_precomp = torch.empty(0, device='cuda')
        if extra_features is None:
            extra_features = torch.empty(0, device='cuda')

        if scales is None:
            scales = torch.empty(0, device='cuda')
        if rotations is None:
            rotations = torch.empty(0, device='cuda')
        if cov3D_precomp is None:
            cov3D_precomp = torch.empty(0, device='cuda')

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            extra_features,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            flag_max_count,
            culling,
            raster_settings,
        )
    

    def render_simp(self, means3D, means2D, opacities, culling, dc=None, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])            
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians_simp(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            culling,
            raster_settings,
        )
    

    
    def render_depth(self, means3D, means2D, opacities, culling, dc=None, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])            
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians_depth(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            culling,
            raster_settings,
        )

    def integrate(self, points3D, means3D, means2D, opacities, culling=None, dc=None, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None, view2gaussian_precomp=None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # TODO check and raise exception for precomputed view2gaussian
        if view2gaussian_precomp is None:
            view2gaussian_precomp = torch.Tensor([])
            
        subpixel_offset = torch.zeros((int(raster_settings.image_height), int(raster_settings.image_width), 2), dtype=torch.float32, device="cuda")
        # Invoke C++/CUDA rasterization routine
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            points3D,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            view2gaussian_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            0.0,
            subpixel_offset,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            shs,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            culling,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, alpha_integrated, color_integrated, point_sdf, point_depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.integrate_gaussians_to_points(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, alpha_integrated, color_integrated, point_coordinate, point_sdf, point_depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.integrate_gaussians_to_points(*args)

        return color, alpha_integrated, color_integrated, point_coordinate, point_sdf, point_depth, radii    
    

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None or torch.prod(torch.tensor(param.grad.shape))==0:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]

            # compensate lr for sparse adam, (1-b2**step)**0.5/(1-b1**step)
            state['step']+=1
            step=state['step']

            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr*(1-0.999**step)**0.5/(1-0.9**step), 0.9, 0.999, eps, N, M)



