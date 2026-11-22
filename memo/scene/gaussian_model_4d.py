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
# 4D extension: GP-based temporal model for dynamic Gaussian splatting.

import re
import math
import torch
import numpy as np
import os
from typing import Optional, Tuple
from torch import nn
from simple_knn._C import distCUDA2
from memo.utils.system_utils import mkdir_p
from memo.utils.general_utils import build_rotation, culling_schedule, cull_stride
from memo.utils.sh_utils import RGB2SH
from plyfile import PlyData, PlyElement
from memo.scene.gaussian_model import GaussianModel, init_cdf_mask


_POS_GP_ALPHA_M0_RE = re.compile(r"^property\s+\S+\s+pos_gp_alpha_(\d+)_0$")
_ROT_GP_INDUCING_T_RE = re.compile(r"^rot_gp_inducing_t_\d+$")

# The rotation GP outputs a raw quaternion offset from the identity.
_ROT_GP_ALPHA_DIM = 4


def inspect_4d_checkpoint(ply_path: str) -> Tuple[bool, Optional[int]]:
    """Detect 4D and infer M_total by header-only scan of the PLY.

    Why header-only: PlyData.read parses the entire binary body, which is
    wasteful when we only need the property list. We also count the
    pos_gp_alpha_<m>_0 properties to recover M_total (alpha is (N, M_total, 3)
    and the 3 d-channels share the same m index, so the _0 slice gives one
    entry per inducing point).

    Why catch OSError: bad / missing path, fall through to gaussians.load_ply
    which raises a clearer error than a header parse failure here. Real PLY
    format errors are not caught and propagate.
    """
    try:
        f = open(ply_path, "rb")
    except OSError:
        return False, None
    m_indices = set()
    is_4d = False
    with f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore").strip()
            if line == "end_header":
                break
            match = _POS_GP_ALPHA_M0_RE.match(line)
            if match:
                is_4d = True
                m_indices.add(int(match.group(1)))
    if not is_4d:
        return False, None
    return True, len(m_indices)


class GaussianModel4D(GaussianModel):
    """4D Gaussian model extending the static GaussianModel with GP-based temporal motion.

    Each Gaussian has:
    - Position GP: per-Gaussian inducing times, lengthscale, and weights -> position delta (N, 3)
    - Rotation GP: per-Gaussian inducing times, lengthscale, and weights -> quaternion delta (N, 4)

    Zero GP alphas reproduce static output identically.
    """

    def __init__(
        self,
        sh_degree: int,
        num_inducing: int = 10,
        num_inducing_rot: int = -1,
        boundary_inducing_points: int = 0,
        use_mip_filter: bool = False,
        n_gaussian_features: int = 0,
        use_unbounded_opacity: bool = False,
    ):
        super().__init__(
            sh_degree=sh_degree,
            use_mip_filter=use_mip_filter,
            n_gaussian_features=n_gaussian_features,
            use_unbounded_opacity=use_unbounded_opacity,
        )
        self.M = num_inducing
        # num_inducing_rot <= 0 means the rotation GP shares the position
        # count. load_ply / restore override M_rot from the stored tensor
        # widths, so the constructor value only matters for fresh models.
        self.M_rot = num_inducing_rot if num_inducing_rot > 0 else num_inducing
        self.boundary_inducing_points = boundary_inducing_points
        self.M_total = self.M + 2 * self.boundary_inducing_points
        self.M_total_rot = self.M_rot + 2 * self.boundary_inducing_points
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")

        # Culling-mask scaffolding. init_culling (below) populates _culling as
        # a packed uint8 (N, T_sub, ceil(N_views/8)) tensor from the start,
        # matching the shape that aggressive-clone events and subsequent
        # renders expect. Kept out of capture()/restore(), since the mask is
        # training-time scaffolding, not part of the model.
        self._culling = None

    # ------------------------------------------------------------------
    # GP evaluation
    # ------------------------------------------------------------------

    # Per-bin bounding for inducing times.
    #   inducing_t[m] = m/M + eps + (1/M - 2*eps) * sigmoid(raw_m)
    # Each inducing point m is confined to the bin [m/M + eps, (m+1)/M - eps],
    # so ordering is preserved, points can't collapse onto each other, and
    # neighboring points have at least 2*eps separation. At raw = 0
    # (sigmoid = 0.5), inducing_t[m] = (2m+1)/(2M) (midpoint rule).
    #
    # eps scales with 1/M so the bin margin stays a fixed fraction of raw bin
    # width across different M: 1% of the bin on each side, leaving 98% for
    # the actual movable range.
    _INDUCING_EPS_FRAC = 0.01

    def _bounded_inducing_t(self, raw):
        """Map raw parameter (N, M_total) to per-bin bounded inducing times (N, M_total).

        With boundary_inducing_points=N per side, the M_total = M + 2N bins span
        [-N/M, 1 + N/M] in steps of 1/M. The N-offset on the arange shifts bin 0
        from [0, 1/M] to [-N/M, -N/M + 1/M].

        The family's M is derived from the raw tensor's own width, so the same
        mapping serves the position and rotation GPs when their inducing
        counts differ.
        """
        m_total = raw.shape[-1]
        m = m_total - 2 * self.boundary_inducing_points
        eps = self._INDUCING_EPS_FRAC / m
        bin_base = (torch.arange(m_total, device=raw.device, dtype=raw.dtype)
                    - self.boundary_inducing_points) / m + eps
        bin_width = 1.0 / m - 2 * eps
        return bin_base + bin_width * torch.sigmoid(raw)

    def position_gp(self, t):
        """Position GP displacement at time t. Returns (N, 3)."""
        l = torch.sigmoid(self._pos_gp_raw_lengthscale)          # (N, 1)
        inducing_t = self._bounded_inducing_t(self._pos_gp_inducing_t)  # (N, M)
        k = torch.exp(-0.5 * (t - inducing_t)**2 / l**2)         # (N, M)
        return (k.unsqueeze(-1) * self._pos_gp_alpha).sum(dim=-2)       # (N, 3)

    def rotation_gp(self, t):
        """Rotation GP delta quaternion at time t. Returns (N, 4) unit quaternion.

        The GP outputs a raw 4-D vector; d_quat = normalize(raw + q_identity).
        At zero alphas this is the identity quaternion with well-defined
        gradients. The renderer composes rotations_t = d_quat * _rotation.
        """
        l = torch.sigmoid(self._rot_gp_raw_lengthscale)          # (N, 1)
        inducing_t = self._bounded_inducing_t(self._rot_gp_inducing_t)  # (N, M)
        k = torch.exp(-0.5 * (t - inducing_t)**2 / l**2)         # (N, M)
        raw = (k.unsqueeze(-1) * self._rot_gp_alpha).sum(dim=-2)        # (N, 4)
        return torch.nn.functional.normalize(raw + self._identity_quat, dim=-1)  # (N, 4)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def create_from_pcd_4d(self, gp_init: dict, spatial_lr_scale: float):
        """Initialize from GP regression output (track-based initialization).

        Args:
            gp_init: dict with keys:
                'xyz': (N, 3) numpy — base positions
                'rgb': (N, 3) numpy — colors
                'normals': (N, 3) numpy — zeros
                'pos_gp_alpha': (N, M, 3) numpy — GP displacement coefficients
                'pos_gp_inducing_t': (N, M) numpy — inducing times
                'pos_gp_lengthscale': (N, 1) numpy — lengthscale
            spatial_lr_scale: scene extent for learning rate scaling
        """
        xyz = torch.tensor(gp_init['xyz'], dtype=torch.float, device="cuda")
        rgb = torch.tensor(gp_init['rgb'], dtype=torch.float, device="cuda")
        N = xyz.shape[0]

        self.spatial_lr_scale = spatial_lr_scale

        # SH DC component from the init colors in [0, 1]
        shs = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        shs[:, :3] = RGB2SH(rgb)
        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device="cuda")
        features[:, :3, 0] = shs
        features[:, 3:, 1:] = 0.0

        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        # Scaling from KNN distances
        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        # Rotation: identity
        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        # Opacity
        opacities = torch.logit(0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # Position
        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        # Gaussian features (normals), reset when the normal field starts
        if self.use_gaussian_features:
            gf = torch.zeros((N, self.n_gaussian_features), device="cuda")
            self._gaussian_features = nn.Parameter(gf.requires_grad_(True))

        self.max_radii2D = torch.zeros((N), device="cuda")

        # Temporal params from GP regression.
        # gp_regression_from_tracks runs the regression at M_total bins with
        # midpoints (m - N + 0.5)/M, which corresponds to raw = 0 in the
        # bounded inducing-time parameterization. Lengthscale is stored
        # pre-sigmoid (logit).
        pos_alpha_init = torch.tensor(gp_init['pos_gp_alpha'], dtype=torch.float, device="cuda")
        assert pos_alpha_init.shape[1] == self.M_total, (
            f"pos_gp_alpha has shape[1]={pos_alpha_init.shape[1]}, expected M_total={self.M_total}")
        self._pos_gp_alpha = nn.Parameter(pos_alpha_init.requires_grad_(True))
        self._pos_gp_inducing_t = nn.Parameter(
            torch.zeros(N, self.M_total, device="cuda").requires_grad_(True))

        ls_init = torch.tensor(gp_init['pos_gp_lengthscale'], dtype=torch.float, device="cuda")
        self._pos_gp_raw_lengthscale = nn.Parameter(
            torch.logit(ls_init.clamp(1e-6, 1 - 1e-6)).requires_grad_(True))

        # Rotation GP: zero alphas, raw=0 inducing times (midpoint), default
        # lengthscale 1/M_rot (one inducing-point spacing). Anchor stays M_rot,
        # not M_total_rot.
        self._rot_gp_alpha = nn.Parameter(
            torch.zeros(N, self.M_total_rot, _ROT_GP_ALPHA_DIM, device="cuda").requires_grad_(True)
        )
        self._rot_gp_inducing_t = nn.Parameter(
            torch.zeros(N, self.M_total_rot, device="cuda").requires_grad_(True))
        default_ls_rot = 1.0 / self.M_rot
        raw_ls_rot = math.log(default_ls_rot / (1 - default_ls_rot))
        self._rot_gp_raw_lengthscale = nn.Parameter(
            torch.full((N, 1), raw_ls_rot, device="cuda").requires_grad_(True))

        print(f"GaussianModel4D.create_from_pcd_4d: {N} Gaussians, M={self.M}, "
              f"M_total={self.M_total}, M_rot={self.M_rot}, M_total_rot={self.M_total_rot}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    # Optimizer group name -> attribute, for the GP temporal tensors that a zero
    # learning rate freezes.
    _FREEZABLE_GP_GROUPS = {
        "pos_gp_inducing_t": "_pos_gp_inducing_t",
        "pos_gp_lengthscale": "_pos_gp_raw_lengthscale",
        "rot_gp_inducing_t": "_rot_gp_inducing_t",
        "rot_gp_lengthscale": "_rot_gp_raw_lengthscale",
    }
    _frozen_gp_groups = ()

    def training_setup(self, training_args):
        super().training_setup(training_args)

        # A zero learning rate alone would still cost a gradient and two Adam
        # moments per element, so a frozen tensor is held as a model buffer,
        # outside the optimizer and without gradient.
        # prune_points and densification_postfix carry it along explicitly.
        self._frozen_gp_groups = tuple(
            name for name in self._FREEZABLE_GP_GROUPS
            if getattr(training_args, f"{name}_lr") == 0
        )
        for name in self._frozen_gp_groups:
            attr = self._FREEZABLE_GP_GROUPS[name]
            setattr(self, attr, getattr(self, attr).detach().requires_grad_(False))

        # All GP params are per-Gaussian (N, ...) — safe for SparseGaussianAdam
        gp_param_groups = [
            {"params": [self._pos_gp_alpha], "lr": training_args.pos_gp_alpha_lr, "name": "pos_gp_alpha"},
            {"params": [self._pos_gp_inducing_t], "lr": training_args.pos_gp_inducing_t_lr, "name": "pos_gp_inducing_t"},
            {"params": [self._pos_gp_raw_lengthscale], "lr": training_args.pos_gp_lengthscale_lr, "name": "pos_gp_lengthscale"},
            {"params": [self._rot_gp_alpha], "lr": training_args.rot_gp_alpha_lr, "name": "rot_gp_alpha"},
            {"params": [self._rot_gp_inducing_t], "lr": training_args.rot_gp_inducing_t_lr, "name": "rot_gp_inducing_t"},
            {"params": [self._rot_gp_raw_lengthscale], "lr": training_args.rot_gp_lengthscale_lr, "name": "rot_gp_lengthscale"},
        ]
        for group in gp_param_groups:
            if group["name"] not in self._frozen_gp_groups:
                self.optimizer.add_param_group(group)

    # ------------------------------------------------------------------
    # PLY attribute list
    # ------------------------------------------------------------------

    def construct_list_of_attributes(self):
        l = super().construct_list_of_attributes()
        # Per-Gaussian position GP params
        for m in range(self.M_total):
            for d in range(3):
                l.append(f'pos_gp_alpha_{m}_{d}')
        for m in range(self.M_total):
            l.append(f'pos_gp_inducing_t_{m}')
        l.append('pos_gp_raw_lengthscale')
        # Per-Gaussian rotation GP params
        for m in range(self.M_total_rot):
            for d in range(_ROT_GP_ALPHA_DIM):
                l.append(f'rot_gp_alpha_{m}_{d}')
        for m in range(self.M_total_rot):
            l.append(f'rot_gp_inducing_t_{m}')
        l.append('rot_gp_raw_lengthscale')
        return l

    # ------------------------------------------------------------------
    # PLY save
    # ------------------------------------------------------------------

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # Base optional attributes
        if self.use_mip_filter:
            filter_3D = self.filter_3D.detach().cpu().numpy()
        if self.use_gaussian_features:
            gaussian_features = self._gaussian_features.detach().cpu().numpy()

        # Temporal per-Gaussian attributes
        pos_gp_alpha = self._pos_gp_alpha.detach().cpu().numpy().reshape(xyz.shape[0], -1)  # (N, M_total*3)
        pos_gp_inducing_t = self._pos_gp_inducing_t.detach().cpu().numpy()                  # (N, M_total)
        pos_gp_raw_lengthscale = self._pos_gp_raw_lengthscale.detach().cpu().numpy()         # (N, 1)
        rot_gp_alpha = self._rot_gp_alpha.detach().cpu().numpy().reshape(xyz.shape[0], -1)   # (N, M_total*D)
        rot_gp_inducing_t = self._rot_gp_inducing_t.detach().cpu().numpy()                  # (N, M_total)
        rot_gp_raw_lengthscale = self._rot_gp_raw_lengthscale.detach().cpu().numpy()         # (N, 1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        to_concatenate = (xyz, normals, f_dc, f_rest, opacities, scale, rotation)
        if self.use_mip_filter:
            to_concatenate = to_concatenate + (filter_3D,)
        if self.use_gaussian_features:
            to_concatenate = to_concatenate + (gaussian_features,)
        # Temporal per-Gaussian
        to_concatenate = to_concatenate + (
            pos_gp_alpha, pos_gp_inducing_t, pos_gp_raw_lengthscale,
            rot_gp_alpha, rot_gp_inducing_t, rot_gp_raw_lengthscale,
        )

        attributes = np.concatenate(to_concatenate, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # ------------------------------------------------------------------
    # PLY load
    # ------------------------------------------------------------------

    def load_ply(self, path):
        plydata = PlyData.read(path)

        # --- Base static params ---
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if "filter_3D" in plydata.elements[0]:
            self.use_mip_filter = True
            filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
        else:
            self.use_mip_filter = False

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        if self.max_sh_degree is None:
            self.max_sh_degree = int(np.sqrt(len(extra_f_names) / 3 + 1) - 1)
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Match only base rotation attributes (rot_0 .. rot_3), not rot_gp_alpha_* etc.
        rot_names = [p.name for p in plydata.elements[0].properties
                     if p.name.startswith("rot_") and not p.name.startswith("rot_gp_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Gaussian features
        gaussian_features_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("gaussian_features_")]
        if len(gaussian_features_names) > 0:
            self.use_gaussian_features = True
            self.n_gaussian_features = len(gaussian_features_names)
            gaussian_features_names = sorted(gaussian_features_names, key=lambda x: int(x.split('_')[-1]))
            gaussian_features = np.zeros((xyz.shape[0], len(gaussian_features_names)))
            for idx, attr_name in enumerate(gaussian_features_names):
                gaussian_features[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            self.use_gaussian_features = False

        # --- Set base static nn.Parameters ---
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.use_mip_filter:
            self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")
        if self.use_gaussian_features:
            self._gaussian_features = nn.Parameter(torch.tensor(gaussian_features, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        # --- Temporal per-Gaussian params from PLY ---
        N = xyz.shape[0]

        pos_gp_alpha = np.zeros((N, self.M_total * 3))
        for m in range(self.M_total):
            for d in range(3):
                attr_name = f'pos_gp_alpha_{m}_{d}'
                pos_gp_alpha[:, m * 3 + d] = np.asarray(plydata.elements[0][attr_name])
        self._pos_gp_alpha = nn.Parameter(
            torch.tensor(pos_gp_alpha.reshape(N, self.M_total, 3), dtype=torch.float, device="cuda")
        )

        pos_gp_inducing_t = np.zeros((N, self.M_total))
        for m in range(self.M_total):
            attr_name = f'pos_gp_inducing_t_{m}'
            pos_gp_inducing_t[:, m] = np.asarray(plydata.elements[0][attr_name])
        self._pos_gp_inducing_t = nn.Parameter(
            torch.tensor(pos_gp_inducing_t, dtype=torch.float, device="cuda")
        )

        pos_gp_raw_lengthscale = np.asarray(plydata.elements[0]["pos_gp_raw_lengthscale"])[..., np.newaxis]
        self._pos_gp_raw_lengthscale = nn.Parameter(
            torch.tensor(pos_gp_raw_lengthscale, dtype=torch.float, device="cuda")
        )

        # The rotation inducing count is self-describing through the stored
        # attribute list (it can differ from the position count).
        ply_attrs = {p.name for p in plydata.elements[0].properties}
        self.M_total_rot = sum(
            1 for a in ply_attrs if _ROT_GP_INDUCING_T_RE.match(a))
        self.M_rot = self.M_total_rot - 2 * self.boundary_inducing_points
        D = _ROT_GP_ALPHA_DIM

        rot_gp_alpha = np.zeros((N, self.M_total_rot * D))
        for m in range(self.M_total_rot):
            for d in range(D):
                attr_name = f'rot_gp_alpha_{m}_{d}'
                rot_gp_alpha[:, m * D + d] = np.asarray(plydata.elements[0][attr_name])
        self._rot_gp_alpha = nn.Parameter(
            torch.tensor(rot_gp_alpha.reshape(N, self.M_total_rot, D), dtype=torch.float, device="cuda")
        )

        rot_gp_inducing_t = np.zeros((N, self.M_total_rot))
        for m in range(self.M_total_rot):
            attr_name = f'rot_gp_inducing_t_{m}'
            rot_gp_inducing_t[:, m] = np.asarray(plydata.elements[0][attr_name])
        self._rot_gp_inducing_t = nn.Parameter(
            torch.tensor(rot_gp_inducing_t, dtype=torch.float, device="cuda")
        )

        rot_gp_raw_lengthscale = np.asarray(plydata.elements[0]["rot_gp_raw_lengthscale"])[..., np.newaxis]
        self._rot_gp_raw_lengthscale = nn.Parameter(
            torch.tensor(rot_gp_raw_lengthscale, dtype=torch.float, device="cuda")
        )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_points(self, mask):
        valid_points_mask = ~mask

        # Prune optimizer state for all per-Gaussian param groups
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_points_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_points_mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid_points_mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid_points_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        for name in self._frozen_gp_groups:
            optimizable_tensors[name] = getattr(self, self._FREEZABLE_GP_GROUPS[name])[valid_points_mask]

        # Reassign base static params
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_gaussian_features:
            self._gaussian_features = optimizable_tensors["gaussian_features"]

        # Reassign temporal per-Gaussian params
        self._pos_gp_alpha = optimizable_tensors["pos_gp_alpha"]
        self._pos_gp_inducing_t = optimizable_tensors["pos_gp_inducing_t"]
        self._pos_gp_raw_lengthscale = optimizable_tensors["pos_gp_lengthscale"]
        self._rot_gp_alpha = optimizable_tensors["rot_gp_alpha"]
        self._rot_gp_inducing_t = optimizable_tensors["rot_gp_inducing_t"]
        self._rot_gp_raw_lengthscale = optimizable_tensors["rot_gp_lengthscale"]

        # Prune non-optimizer tracking tensors
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self._culling is not None:
            self._culling = self._culling[valid_points_mask]

    # ------------------------------------------------------------------
    # Densification
    # ------------------------------------------------------------------

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _gp_params(self, selected_pts_mask, repeats=1):
        """The six GP tensors of the selected Gaussians, each repeated `repeats` times."""
        def take(t):
            sel = t[selected_pts_mask]
            return sel.repeat(repeats, *([1] * (sel.dim() - 1)))
        return dict(
            new_pos_gp_alpha=take(self._pos_gp_alpha),
            new_pos_gp_inducing_t=take(self._pos_gp_inducing_t),
            new_pos_gp_raw_lengthscale=take(self._pos_gp_raw_lengthscale),
            new_rot_gp_alpha=take(self._rot_gp_alpha),
            new_rot_gp_inducing_t=take(self._rot_gp_inducing_t),
            new_rot_gp_raw_lengthscale=take(self._rot_gp_raw_lengthscale),
        )

    def _extend_culling(self, selected_pts_mask, repeats=1):
        if self._culling is not None:
            new_culling = self._culling[selected_pts_mask].repeat(repeats, 1, 1)
            self._culling = torch.cat((self._culling, new_culling))

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
        new_gaussian_features,
        new_pos_gp_alpha, new_pos_gp_inducing_t, new_pos_gp_raw_lengthscale,
        new_rot_gp_alpha, new_rot_gp_inducing_t, new_rot_gp_raw_lengthscale,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "pos_gp_alpha": new_pos_gp_alpha,
            "pos_gp_inducing_t": new_pos_gp_inducing_t,
            "pos_gp_lengthscale": new_pos_gp_raw_lengthscale,
            "rot_gp_alpha": new_rot_gp_alpha,
            "rot_gp_inducing_t": new_rot_gp_inducing_t,
            "rot_gp_lengthscale": new_rot_gp_raw_lengthscale,
        }
        if self.use_gaussian_features:
            d["gaussian_features"] = new_gaussian_features

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        for name in self._frozen_gp_groups:
            optimizable_tensors[name] = torch.cat(
                (getattr(self, self._FREEZABLE_GP_GROUPS[name]), d[name].detach()), dim=0)

        # Reassign base static params
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_gaussian_features:
            self._gaussian_features = optimizable_tensors["gaussian_features"]

        # Reassign temporal per-Gaussian params
        self._pos_gp_alpha = optimizable_tensors["pos_gp_alpha"]
        self._pos_gp_inducing_t = optimizable_tensors["pos_gp_inducing_t"]
        self._pos_gp_raw_lengthscale = optimizable_tensors["pos_gp_lengthscale"]
        self._rot_gp_alpha = optimizable_tensors["rot_gp_alpha"]
        self._rot_gp_inducing_t = optimizable_tensors["rot_gp_inducing_t"]
        self._rot_gp_raw_lengthscale = optimizable_tensors["rot_gp_lengthscale"]

        # Reset tracking tensors
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone small Gaussians with high gradients. Inherits GP params from parent."""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        self.densification_postfix(
            self._xyz[selected_pts_mask],
            self._features_dc[selected_pts_mask],
            self._features_rest[selected_pts_mask],
            self._opacity[selected_pts_mask],
            self._scaling[selected_pts_mask],
            self._rotation[selected_pts_mask],
            self._gaussian_features[selected_pts_mask] if self.use_gaussian_features else None,
            **self._gp_params(selected_pts_mask),
        )
        self._extend_culling(selected_pts_mask)

    def densify_and_split(self, grads, grad_threshold, scene_extent, mask_split, N=2):
        """Split large Gaussians with high gradients. Inherits GP params from parent.

        The aggressive-cloning blur mask `mask_split` is OR'd into the
        gradient/scale-based selection so oversize Gaussians flagged by
        area_max are split too.
        """
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device="cuda")
        padded_mask[:mask_split.shape[0]] = mask_split
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)

        # Static params (with random offset for split)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))

        self.densification_postfix(
            new_xyz,
            self._features_dc[selected_pts_mask].repeat(N, 1, 1),
            self._features_rest[selected_pts_mask].repeat(N, 1, 1),
            self._opacity[selected_pts_mask].repeat(N, 1),
            new_scaling,
            self._rotation[selected_pts_mask].repeat(N, 1),
            self._gaussian_features[selected_pts_mask].repeat(N, 1) if self.use_gaussian_features else None,
            **self._gp_params(selected_pts_mask, repeats=N),
        )
        self._extend_culling(selected_pts_mask, repeats=N)

        # Prune the original selected points (they've been replaced by splits)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_prune_mask(self, max_grad, min_opacity, extent, mask_split):
        """Densify (clone + blur-mask augmented split), then prune low-opacity Gaussians.

        The low min_opacity lets opacity-split clone children survive long
        enough to accumulate gradient signal.
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, mask_split)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def clone(self, selected_pts_mask):
        """Opacity-splitting aggressive clone with GP parameter inheritance.

        Applies a volume-preserving opacity split and scaling adjustment,
        modifies parent opacity/scaling in place, then creates child Gaussians
        that inherit all six GP tensors from their parents.
        """
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        temp_opacity_old = self.get_opacity[selected_pts_mask]
        new_opacity = 1 - (1 - temp_opacity_old) ** 0.5

        temp_scale_old = self.get_scaling[selected_pts_mask]
        new_scaling = (temp_opacity_old / (2 * new_opacity - 0.5**0.5 * new_opacity**2)) * temp_scale_old

        new_opacity = torch.clamp(new_opacity, max=1.0 - torch.finfo(torch.float32).eps, min=0.0051)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling)

        # Volume-preserving split: parent opacity/scaling updated in place to
        # the split values; children get the same split values.
        self._opacity[selected_pts_mask] = new_opacity
        self._scaling[selected_pts_mask] = new_scaling

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
            self._rotation[selected_pts_mask],
            self._gaussian_features[selected_pts_mask] if self.use_gaussian_features else None,
            **self._gp_params(selected_pts_mask),
        )
        self._extend_culling(selected_pts_mask)

    def densify_and_clone_from_mask(self, selected_pts_mask, new_xyz, new_gaussian_features):
        """Clone the selected Gaussians at `new_xyz` with `new_gaussian_features`.

        Every other parameter, the six GP tensors included, is inherited from
        the parents. Used by the normal-aware densification.
        """
        self.densification_postfix(
            new_xyz,
            self._features_dc[selected_pts_mask],
            self._features_rest[selected_pts_mask],
            self._opacity[selected_pts_mask],
            self._scaling[selected_pts_mask],
            self._rotation[selected_pts_mask],
            new_gaussian_features,
            **self._gp_params(selected_pts_mask),
        )
        self._extend_culling(selected_pts_mask)

    # ------------------------------------------------------------------
    # 4D-aware 3D MIP filter
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_3D_filter(self, cameras, max_culling_images=None):
        """4D-aware MIP filter: projects x_i(t) = _xyz + position_gp(t) at each
        camera's frame time and aggregates the min depth and max focal length
        over the cameras.

        If `max_culling_images` is provided and the camera list has more
        (view, frame) pairs than that budget, a stride/offset subsampling gate
        derived from `max_culling_images` picks a representative subset (same
        subsampling convention used by `culling_with_clone`). Skipped cameras
        would only tighten the filter where included; smooth GP motion means
        adjacent-frame min-depth is a close proxy.
        """
        xyz = self.get_xyz  # (N, 3)
        device = xyz.device

        # Frame subsampling gate on the integer frame index t (matches the
        # culling_with_clone stride phase).
        sampled = list(cameras)
        if max_culling_images is not None:
            interval = cull_stride(len(sampled), max_culling_images)
            if interval > 1:
                offset = interval // 2
                sampled = [
                    cam for cam in sampled
                    if (cam.t - offset) % interval == 0
                ]

        distance = torch.full((xyz.shape[0],), 1e5, device=device)
        valid_points = torch.zeros((xyz.shape[0],), device=device, dtype=torch.bool)
        focal_length = 0.0

        for camera in sampled:
            W, H = camera.image_width, camera.image_height
            focal_x = W / (2 * math.tan(camera.FoVx / 2.0))
            focal_y = H / (2 * math.tan(camera.FoVy / 2.0))

            # Time-dependent world-space positions for this camera's frame.
            x_world = xyz + self.position_gp(camera.normalized_t)

            R = torch.tensor(camera.R, device=device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=device, dtype=torch.float32)
            # R is stored transposed due to 'glm' in CUDA code.
            xyz_cam = x_world @ R + T[None, :]

            valid_depth = xyz_cam[:, 2] > 0.2
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            x = x / z * focal_x + camera.image_width / 2.0
            y = y / z * focal_y + camera.image_height / 2.0

            # Tangent-space (looser) in-screen test.
            in_screen = torch.logical_and(
                torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15),
                torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height),
            )
            valid = torch.logical_and(valid_depth, in_screen)
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < focal_x:
                focal_length = focal_x

        # Any Gaussian not seen by any camera gets the worst observed distance.
        if valid_points.any():
            distance[~valid_points] = distance[valid_points].max()
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]

    # ------------------------------------------------------------------
    # Checkpoint save/restore
    # ------------------------------------------------------------------

    def capture(self):
        return super().capture() + (
            self._pos_gp_alpha,
            self._pos_gp_inducing_t,
            self._pos_gp_raw_lengthscale,
            self._rot_gp_alpha,
            self._rot_gp_inducing_t,
            self._rot_gp_raw_lengthscale,
            self.M,
            self.boundary_inducing_points,
        )

    def restore(self, model_args, training_args):
        # Restore temporal params BEFORE super().restore() so training_setup
        # registers the correct tensors in the optimizer and load_state_dict
        # keys state to them.
        (self._pos_gp_alpha,
         self._pos_gp_inducing_t,
         self._pos_gp_raw_lengthscale,
         self._rot_gp_alpha,
         self._rot_gp_inducing_t,
         self._rot_gp_raw_lengthscale,
         self.M,
         self.boundary_inducing_points) = model_args[-8:]
        self.M_total = self.M + 2 * self.boundary_inducing_points
        # Rotation count is not in the capture tuple; it is self-describing
        # through the restored tensor width.
        self.M_total_rot = self._rot_gp_inducing_t.shape[1]
        self.M_rot = self.M_total_rot - 2 * self.boundary_inducing_points

        super().restore(model_args[:-8], training_args)

    # ------------------------------------------------------------------
    # Packed-uint8 culling-mask helpers
    # ------------------------------------------------------------------
    #
    # Logical shape of the 4D mask is (N, T_sub, N_views) bool. PyTorch's bool
    # dtype is one byte per element, so we pack along the view axis into
    # (N, T_sub, ceil(N_views / 8)) uint8, 8 views per byte. Only per-view
    # slice access needs shift-and-mask.
    #
    # Convention: bit position `view_num % 8` inside byte column `view_num // 8`.
    # Higher bits are unused when N_views is not a multiple of 8.

    def init_culling(self, num_views, t_sub=1):
        """Initialize _culling as a packed uint8 (N, T_sub, ceil(N_views/8)) tensor.

        T_sub is the time-bucket axis used by culling_with_clone and subsequent
        per-(view, frame) mask lookups. The caller supplies it directly: with
        aggressive cloning enabled it should come from culling_schedule (one
        bucket per processed frame), otherwise 1 collapses the mask to a single
        time slot.

        Shape/dtype are fixed at init so all downstream code (densify_and_*,
        clone, get_culling_for_view) can assume a single representation.
        """
        N = self._xyz.shape[0]
        n_bytes = (num_views + 7) // 8
        self._culling = torch.zeros((N, t_sub, n_bytes), dtype=torch.uint8, device="cuda")

    @staticmethod
    def _unpack_view_slice(mask_packed, view_num):
        """(N, T_sub, ceil(N_views / 8)) uint8 -> (N, T_sub) bool for one view."""
        byte_col = view_num // 8
        bit_pos = view_num % 8
        return torch.bitwise_and(mask_packed[:, :, byte_col] >> bit_pos,
                                 torch.tensor(1, dtype=torch.uint8, device=mask_packed.device)).to(torch.bool)

    @staticmethod
    def _set_view_slice(mask_packed, view_num, frame, bool_vec):
        """In place: OR bool_vec into the bit for (frame, view_num) across all N.

        This accumulates culling bits at a specific (frame, view) location.
        """
        byte_col = view_num // 8
        bit_pos = view_num % 8
        bit_mask = (bool_vec.to(torch.uint8) << bit_pos)
        mask_packed[:, frame, byte_col] = torch.bitwise_or(
            mask_packed[:, frame, byte_col], bit_mask
        )

    def get_culling_for_view(self, view, num_frames, num_views, max_culling_images):
        """Return the (N,) bool culling vector for a given view.

        _culling is a packed uint8 (N, T_sub, ceil(N_views/8)) tensor, or
        None if init_culling has not been called yet. Returning None lets
        callers omit the `culling` kwarg so the renderer creates a fresh zero
        vector.
        """
        if self._culling is None:
            return None
        T_sub = self._culling.shape[1]
        interval, offset, _ = culling_schedule(num_views, num_frames, max_culling_images)
        culling_frame = max(0, round((view.t - 1 - offset) / num_frames * T_sub))
        culling_frame = min(culling_frame, T_sub - 1)
        return self._unpack_view_slice(self._culling, view.view_num)[:, culling_frame]

    # ------------------------------------------------------------------
    # Culling events: aggressive clone and the two simplification passes
    # ------------------------------------------------------------------

    def _accumulate_importance(self, scene, render_simp, args, pipe, background, area_normalized):
        """Render subsampled (view, frame) pairs and accumulate per-frame statistics.

        Returns (packed_mask, use_persist, imp_score, accum_area_max,
        count_vis, T_sub). imp_score is (N, T_sub); with `area_normalized`
        and the 'outdoor' importance metric it accumulates accum_weights /
        area_proj on Gaussians with a nonzero area_max. Each view's
        Level-1 (per-view CDF 0.99) culling bits are written into
        packed_mask.

        The culling mask is stored packed as uint8 along the view axis. A
        dynamic memory check picks between persist-on-model and
        compute-and-discard modes based on free GPU memory; the allocation is
        also wrapped in try/except for the OOM fallback.
        """
        N = self._xyz.shape[0]
        device = "cuda"

        views = scene.getTrainCameras().copy()
        N_views = scene.num_views
        N_frames = scene.num_frames
        interval, offset, T_sub = culling_schedule(N_views, N_frames, args.max_culling_images)
        n_bytes = (N_views + 7) // 8

        reserve_bytes = int(args.culling_mask_memory_reserve_gb * (2 ** 30))
        mask_bytes = N * T_sub * n_bytes
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        use_persist = mask_bytes <= free_bytes - reserve_bytes
        # The allocation is the same size in both modes; what differs is
        # whether it is persisted on self._culling at the end. If the budget
        # said "persist" but the allocation OOMs (stale free-memory read,
        # unexpected kernel scratch), clear the cache and retry once as
        # compute-and-discard so self._culling keeps its prior value. A second
        # failure propagates, max_culling_images then needs lowering.
        try:
            packed_mask = torch.zeros((N, T_sub, n_bytes), dtype=torch.uint8, device=device)
        except torch.cuda.OutOfMemoryError:
            print(f"[culling] OOM on first allocation ({mask_bytes / 2**20:.1f} MB); "
                  "clearing cache and retrying as compute-and-discard")
            torch.cuda.empty_cache()
            use_persist = False
            packed_mask = torch.zeros((N, T_sub, n_bytes), dtype=torch.uint8, device=device)

        imp_score = torch.zeros(N, T_sub, device=device)
        accum_area_max = torch.zeros(N, T_sub, device=device)
        count_vis = torch.zeros(N, T_sub, device=device)

        for view in views:
            if (view.t - offset) % interval != 0:
                continue

            culling_frame = max(0, round((view.t - 1 - offset) / N_frames * T_sub))
            view.to_device(device)

            view_slice = self._unpack_view_slice(packed_mask, view.view_num)[:, culling_frame]
            render_pkg = render_simp(view, self, pipe, background,
                                     culling=view_slice, apply_4d=True)

            accum_weights = render_pkg["accum_weights"]
            area_max = render_pkg["area_max"]

            accum_area_max[:, culling_frame] += area_max
            if area_normalized and args.imp_metric == 'outdoor':
                mask_t = area_max != 0
                temp = imp_score[:, culling_frame] + accum_weights / render_pkg["area_proj"]
                imp_score[:, culling_frame][mask_t] = temp[mask_t]
            else:
                imp_score[:, culling_frame] += accum_weights

            per_view_mask = init_cdf_mask(importance=accum_weights, thres=0.99)
            self._set_view_slice(packed_mask, view.view_num, culling_frame, ~per_view_mask)
            count_vis[:, culling_frame][per_view_mask] += 1

        return packed_mask, use_persist, imp_score, accum_area_max, count_vis, T_sub

    def culling_with_clone(self, scene, render_simp, args, pipe, background):
        """Aggressive clone with visibility culling, extended to 4D.

        Renders subsampled (view, frame) pairs with apply_4d=True, accumulates
        (N, T_sub) importance scores, applies three-level conservative pruning
        (per-view -> per-frame AND -> global OR), then clones high-importance
        Gaussians with GP parameter inheritance.
        """
        N = self._xyz.shape[0]
        packed_mask, use_persist, imp_score, accum_area_max, count_vis, T_sub = \
            self._accumulate_importance(scene, render_simp, args, pipe, background,
                                        area_normalized=False)

        # Level 2: per-frame AND across frames. A Gaussian is pruned only if,
        # in every frame, it is visible from at most one viewpoint or falls
        # outside that frame's top-99.9% importance.
        prune_mask = torch.ones(N, dtype=torch.bool, device="cuda")
        for frame in range(T_sub):
            frame_vis_mask = init_cdf_mask(importance=imp_score[:, frame], thres=0.999)
            frame_prune_mask = torch.logical_or(count_vis[:, frame] <= 1, ~frame_vis_mask)
            prune_mask = torch.logical_and(prune_mask, frame_prune_mask)

        # Level 3: global OR with ~top-99.99% across all frames
        non_prune_mask = init_cdf_mask(importance=imp_score.sum(dim=-1), thres=0.9999)
        prune_mask = torch.logical_or(prune_mask, ~non_prune_mask)

        # In persist mode: replace self._culling with the packed 3D mask so
        # downstream training renders can consume it and prune_points / clone
        # slice-and-extend it along dim 0 automatically. In compute-and-discard
        # mode: leave self._culling at whatever init_culling or a prior event
        # set it to.
        if use_persist:
            self._culling = packed_mask

        self.prune_points(prune_mask)

        # Clone candidate selection operates on pre-prune importance, masked
        # by ~prune_mask to align with the post-prune model size.
        imp_score[accum_area_max == 0] = 0
        agg_importance = imp_score.sum(dim=-1)                               # (N,) pre-prune
        intersection_pts_mask = init_cdf_mask(importance=agg_importance, thres=0.99)
        clone_mask = intersection_pts_mask[~prune_mask]                      # (N_post_prune,)
        self.clone(clone_mask)

    def culling_with_interesction_sampling(self, scene, render_simp, args, pipe, background):
        """Simplification pass 1.

        Level 2 is the per-frame AND of (count_vis <= 1). The survivor
        criterion is random sampling from the max-over-frames importance via
        `np.random.choice`, with `args.sampling_factor` as the sampling ratio.
        """
        N = self._xyz.shape[0]
        device = "cuda"
        packed_mask, use_persist, imp_score, accum_area_max, count_vis, T_sub = \
            self._accumulate_importance(scene, render_simp, args, pipe, background,
                                        area_normalized=True)
        imp_score[accum_area_max == 0] = 0

        prune_mask = torch.ones(N, dtype=torch.bool, device=device)
        for frame in range(T_sub):
            prune_mask = torch.logical_and(prune_mask, count_vis[:, frame] <= 1)

        agg_imp = imp_score.max(dim=-1).values
        prob = agg_imp / agg_imp.sum()
        prob_np = prob.cpu().numpy()
        num_nonzero = int((prob_np > 0).sum())
        num_sampled = int(N * args.sampling_factor * (num_nonzero / prob_np.shape[0]))
        indices = np.random.choice(prob_np.shape[0], size=num_sampled, p=prob_np, replace=False)
        sampled_mask = torch.zeros(N, dtype=torch.bool, device=device)
        sampled_mask[indices] = True

        prune_mask = torch.logical_or(prune_mask, ~sampled_mask)

        if use_persist:
            self._culling = packed_mask

        self.prune_points(prune_mask)

    def culling_with_intersection_preserving(self, scene, render_simp, args, pipe, background):
        """Simplification pass 2.

        Per-frame CDF 0.99 OR'd with per-frame (count_vis <= 1), AND across
        frames. A Gaussian survives iff it is in the top-CDF for at least one
        frame where count_vis > 1.
        """
        N = self._xyz.shape[0]
        packed_mask, use_persist, imp_score, accum_area_max, count_vis, T_sub = \
            self._accumulate_importance(scene, render_simp, args, pipe, background,
                                        area_normalized=True)
        imp_score[accum_area_max == 0] = 0

        prune_mask = torch.ones(N, dtype=torch.bool, device="cuda")
        for frame in range(T_sub):
            frame_non_prune_mask = init_cdf_mask(importance=imp_score[:, frame], thres=0.99)
            frame_prune_mask = torch.logical_or(count_vis[:, frame] <= 1, ~frame_non_prune_mask)
            prune_mask = torch.logical_and(prune_mask, frame_prune_mask)

        if use_persist:
            self._culling = packed_mask

        self.prune_points(prune_mask)
