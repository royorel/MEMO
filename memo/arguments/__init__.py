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

from argparse import ArgumentParser
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._white_background = False
        self.use_unbounded_opacity = False
        # Number of anchors for kalman_multi. -1 = one anchor per inducing
        # bin, so the count follows M. An explicit value decouples the two.
        self.kalman_num_anchors = -1
        # Static zone for the kalman inits: tracks whose median accepted
        # Euclidean distance from the rig camera centroid exceeds the threshold are
        # dropped from the GP fit (their apparent motion is reconstruction
        # noise) and the per-frame far-point union is appended as static
        # points in their place. -1 = off, positive = multiplier of the
        # rig extent, 0 = default multiplier 20x.
        self.static_zone_distance = -1.0
        # Background point cloud appended to the initialization with zero
        # motion. A relative path is resolved against source_path.
        self.static_background_ply = ""
        # With init_crop_bbox set, a positive value switches from cropping to
        # split-budget densification: the in-bbox foreground densifies to this
        # target and the rest of the scene to the default target.
        self.init_fg_target_points = 0
        super().__init__(parser, "Loading Parameters")
        # Registered after super() because ParamGroup infers neither choices
        # nor nargs. kalman tracks from frame 0; kalman_multi unions the tracks
        # of several anchors.
        self.init_method = "kalman"
        parser.add_argument("--init_method", default="kalman", type=str,
                            choices=["kalman", "kalman_multi"])
        # Foreground crop for the 4D init: [x_min, y_min, z_min, x_max, y_max,
        # z_max]. Tracks outside the box are dropped before densification (the
        # static background.ply covers that geometry). Empty = off.
        self.init_crop_bbox = []
        parser.add_argument("--init_crop_bbox", default=[], nargs=6, type=float)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_dc_lr = 0.0025
        self.feature_rest_lr = 0.0025 / 20.0
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        # Post-training camera-blocking floater prune (fraction of
        # cameras_extent; 0 disables). Applied once before the final save.
        self.near_camera_prune_extent = 0.0
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        # Depth-Normal consistency (L_DN) regularization.
        self.regularization_from_iter = 3_000
        self.lambda_depth_normal = 0.05
        # Detach L_DN's rasterized normal map so the regularizer only pulls
        # depth toward the smooth normal map (2DGS-style smoothing), never
        # geometry toward the noisy depth-derived normals. Off = the mutual
        # (two-way) formulation.
        self.detach_dn_normal = False

        # Aggressive cloning & simplification (MiniSplatting2).
        # Aggressive cloning's upper bound is densify_until_iter; no separate
        # until_iter knob.
        self.aggressive_clone_from_iter = -1
        self.aggressive_clone_interval = 250
        self.simp_iteration1 = -1
        self.simp_iteration2 = -1
        self.sampling_factor = 0.6
        self.imp_metric = "outdoor"
        # Budget of (view, frame) pairs rendered by each culling event.
        self.max_culling_images = 5000
        self.culling_mask_memory_reserve_gb = 8.0

        # Motion-GP learning rates.
        self.pos_gp_alpha_lr = 1e-5
        self.pos_gp_inducing_t_lr = 1e-5
        self.pos_gp_lengthscale_lr = 1e-5
        self.rot_gp_alpha_lr = 1e-5
        self.rot_gp_inducing_t_lr = 1e-5
        self.rot_gp_lengthscale_lr = 1e-5

        super().__init__(parser, "Optimization Parameters")
