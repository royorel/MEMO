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
import numpy as np
from memo.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from memo.utils.general_utils import strip_symmetric, build_scaling_rotation
from diff_gaussian_rasterization_memo import SparseGaussianAdam


# Learning rate of the learned Gaussian normals.
_GAUSSIAN_FEATURES_LR = 0.05 / 2.0


def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune
    else:
        non_prune_mask = torch.ones_like(importance).bool()

    return non_prune_mask


class GaussianModel:

    def setup_functions(self, use_unbounded_opacity: bool = False):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        if use_unbounded_opacity:
            self.opacity_activation = torch.exp
            self.inverse_opacity_activation = torch.log
        else:
            self.opacity_activation = torch.sigmoid
            self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(
        self, sh_degree : int,
        use_mip_filter : bool = False,
        n_gaussian_features : int = 0,
        use_unbounded_opacity : bool = False,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions(use_unbounded_opacity=use_unbounded_opacity)

        self.use_mip_filter = use_mip_filter

        self.use_gaussian_features = n_gaussian_features > 0
        self.n_gaussian_features = n_gaussian_features
        if self.use_gaussian_features:
            self._gaussian_features = torch.empty(0)

        self.use_unbounded_opacity = use_unbounded_opacity
        if self.use_unbounded_opacity:
            print("[INFO] Using unbounded opacity")

    def capture(self):
        to_return = (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        if self.use_gaussian_features:
            to_return += (self._gaussian_features,)
        return to_return

    def restore(self, model_args, training_args):
        start_idx = 12
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args[:start_idx]

        if self.use_gaussian_features:
            self._gaussian_features = model_args[start_idx]
            start_idx = start_idx + 1

        if start_idx != len(model_args):
            print(f"[ WARNING ] Restoring model with extra arguments: Only {start_idx} arguments expected, but {len(model_args)} provided.")

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        if self.use_mip_filter:
            scales = self.get_scaling
            scales = torch.square(scales) + torch.square(self.filter_3D)
            scales = torch.sqrt(scales)
            return scales
        else:
            return self.get_scaling

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_with_3D_filter(self):
        if self.use_mip_filter and (not self.use_unbounded_opacity):
            opacity = self.get_opacity
            scales = self.get_scaling

            # apply 3D filter
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)

            scales_after_square = scales_square + torch.square(self.filter_3D)
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            return opacity * coef[..., None]
        else:
            return self.get_opacity

    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        if self.use_mip_filter:
            opacity = self.opacity_activation(self._opacity)
            scales = self.get_scaling
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + torch.square(self.filter_3D)
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            scales = torch.sqrt(scales_after_square)
            return scales, opacity * coef[..., None]
        else:
            return self.get_scaling, self.get_opacity

    def get_contribution(self, viewpoint_camera, stable=True, use_3D_filter=True) -> torch.Tensor:
        """
        Compute the contribution of the Gaussian at its center to the image.
        The contribution is equal to the integral of the Gaussian density times the transmittance
        along the camera ray passing through the center of the Gaussian.

        For a single Gaussian, this is equal to transmittance_at_cam_center - transmittance_at_infinity.

        Args:
            self (GaussianModel): The Gaussian model.
            viewpoint_camera (Camera): The camera.

        Returns:
            torch.Tensor: The contribution of the Gaussian at its center to the image. Has shape (N_gaussians, 1).
        """
        cam_center = viewpoint_camera.camera_center.unsqueeze(0)  # (1, 3)
        rays = self.get_xyz - cam_center  # (N_gaussians, 3)
        rays = torch.nn.functional.normalize(rays, dim=-1)  # (N_gaussians, 3)

        # Equal to S @ R^T
        if stable:
            scaling_rotation = build_scaling_rotation(
                s=self.get_scaling_with_3D_filter if use_3D_filter else self.get_scaling,
                r=self._rotation
            ).transpose(-1, -2)  # (N_gaussians, 3, 3)

            # Equal to S @ R^T @ r
            transformed_rays = torch.bmm(
                scaling_rotation,  # (N_gaussians, 3, 3)
                rays.unsqueeze(-1)  # (N_gaussians, 3, 1)
            ).squeeze(-1)  # (N_gaussians, 3)

            # Equal to r^T @ Sigma @ r
            std2 = (transformed_rays * transformed_rays).sum(dim=-1, keepdim=True)  # (N_gaussians, 1)
        else:
            inv_scaling_rotation = build_scaling_rotation(
                s=1. / self.get_scaling_with_3D_filter if use_3D_filter else (1. / self.get_scaling).clamp(min=1e-8),
                r=self._rotation
            ).transpose(-1, -2)  # (N_gaussians, 3, 3)

            # Equal to S^-1 @ R^T @ r
            transformed_rays = torch.bmm(
                inv_scaling_rotation,  # (N_gaussians, 3, 3)
                rays.unsqueeze(-1)  # (N_gaussians, 3, 1)
            ).squeeze(-1)  # (N_gaussians, 3)

            # Equal to r^T @ Sigma^-1 @ r
            a = (transformed_rays * transformed_rays).sum(dim=-1, keepdim=True)  # (N_gaussians, 1)

            std2 = 1. / a

        # Transmittance of the Gaussian at infinity
        opacity = self.get_opacity_with_3D_filter if use_3D_filter else self.get_opacity  # (N_gaussians, 1)
        transmittance_at_infinity = torch.exp(
            - torch.sqrt(2. * np.pi * std2)  # (N_gaussians, 1)
            * opacity  # (N_gaussians, 1)
        )

        # Transmittance of the Gaussian at camera_center
        transmittance_at_cam_center = 1.

        # Contribution of the Gaussian at its center
        gaussian_contribution = transmittance_at_cam_center - transmittance_at_infinity  # (N_gaussians, 1)

        return gaussian_contribution

    @property
    def get_gaussian_features(self):
        if self.use_gaussian_features:
            return self._gaussian_features
        else:
            raise ValueError("Pivot features are not used")

    def convert_features_to_normals(self, normalize: bool = True):
        """Learned normals: 3 direction components times the tanh of a sign feature."""
        assert self.use_gaussian_features and self.n_gaussian_features == 4

        features = self.get_gaussian_features

        normal_directions = features[:, :3]  # (N_gaussians, 3)
        if normalize:
            normal_directions = torch.nn.functional.normalize(normal_directions, dim=-1)

        normal_signs = torch.tanh(features[:, -1:])  # (N_gaussians, 1)

        return normal_directions * normal_signs  # (N_gaussians, 3)

    @torch.no_grad()
    def reset_normal_features(self):
        """Reset the normal signs to 0 and the normal directions to the shortest Gaussian axis."""
        assert self.use_gaussian_features and self.n_gaussian_features == 4

        self._gaussian_features[:, -1:] = 0.0

        #   > Get min scale
        scale = self.get_scaling_with_3D_filter  # (N_gaussians, 3)
        min_scaling_idx = torch.argmin(scale, dim=-1, keepdim=True)  # (N_gaussians, 1)
        #   > Get rotation matrix
        rotation_matrices = build_rotation(self._rotation)  # (N_gaussians, 3, 3)
        #   > Get column of rotation matrix corresponding to min scale
        gaussian_shortest_axis = torch.gather(
            input=rotation_matrices,  # (N_gaussians, 3, 3)
            index=min_scaling_idx.unsqueeze(1).repeat(1, 3, 1),  # (N_gaussians, 3, 1)
            dim=2,
        ).squeeze(-1)  # (N_gaussians, 3)

        self._gaussian_features[:, :3] = gaussian_shortest_axis

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling_with_3D_filter, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_dc_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_rest_lr, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.use_gaussian_features:
            l.append({'params': [self._gaussian_features], 'lr': _GAUSSIAN_FEATURES_LR, "name": "gaussian_features"})

        self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_mip_filter:
            l.append('filter_3D')
        if self.use_gaussian_features:
            for i in range(self._gaussian_features.shape[1]):
                l.append('gaussian_features_{}'.format(i))
        return l

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
