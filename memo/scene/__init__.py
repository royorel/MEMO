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

import os
import pickle
import lmdb
from memo.utils.system_utils import searchForMaxIteration
from memo.scene.dataset_loaders import Views4D, ViewpointLoader
from memo.utils.init_4d import (
    set_initial_pcd_kalman_filter_gp,
    set_initial_pcd_kalman_multi_anchor,
    load_per_frame_point_clouds,
    append_static_background,
    rig_cameras_from_transforms,
)


class Scene:

    def __init__(self, args, gaussians, load_iteration=None, shuffle=True):
        """Scene read from `<source_path>/train_lmdb` and `<source_path>/test_lmdb`.

        Without `load_iteration`, the Gaussians are initialized from the
        per-frame point clouds `<source_path>/sfm_init/frame_<t>/colmap_points3D.ply`
        and the cameras of `<source_path>/transforms_train.json`.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        train_lmdb_path = os.path.join(args.source_path, "train_lmdb")
        test_lmdb_path = os.path.join(args.source_path, "test_lmdb")
        if not os.path.isdir(train_lmdb_path):
            raise FileNotFoundError(
                f"{train_lmdb_path} not found. Prepare the scene with memo.data.prepare, or build "
                f"its LMDBs with python -m memo.data.build_lmdb --source_path {args.source_path}")

        env = lmdb.open(train_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.num_frames = int(txn.get(b"num_frames"))
            self.num_views = int(txn.get(b"num_views"))
            self.cameras_extent = pickle.loads(txn.get(b"nerf_normalization"))["radius"]
        env.close()

        # Cameras are loaded lazily from the LMDBs. 16 persistent workers give
        # enough prefetch headroom to hide LMDB cold-cache disk reads behind
        # training-step compute.
        allowed_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
        num_workers = min(16, max(allowed_cpus - 1, 1))
        self.train_loader = ViewpointLoader(Views4D(train_lmdb_path), batch_size=1,
                                            shuffle=True, num_workers=num_workers,
                                            pin_memory=True, persistent_workers=True)
        self.test_loader = None
        if os.path.exists(test_lmdb_path):
            self.test_loader = ViewpointLoader(Views4D(test_lmdb_path), batch_size=1,
                                               shuffle=False, num_workers=num_workers,
                                               pin_memory=True, persistent_workers=True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.initialize_gaussians(args)

    def initialize_gaussians(self, args):
        """Track matching + GP regression + densification from the per-frame point clouds."""
        gaussians = self.gaussians
        static_zone_distance = args.static_zone_distance
        point_clouds, timestamps = load_per_frame_point_clouds(args.source_path)
        cam_centers = (rig_cameras_from_transforms(args.source_path)
                       if static_zone_distance >= 0 else None)

        if args.init_method == 'kalman':
            gp_init = set_initial_pcd_kalman_filter_gp(
                point_clouds, timestamps, M=gaussians.M,
                boundary_inducing_points=gaussians.boundary_inducing_points,
                anchor=0,
                static_zone_distance=static_zone_distance,
                cam_centers=cam_centers)
        else:
            gp_init = set_initial_pcd_kalman_multi_anchor(
                point_clouds, timestamps, M=gaussians.M,
                boundary_inducing_points=gaussians.boundary_inducing_points,
                n_anchors=args.kalman_num_anchors if args.kalman_num_anchors > 0 else None,
                static_zone_distance=static_zone_distance,
                cam_centers=cam_centers,
                crop_bbox=list(args.init_crop_bbox) or None,
                fg_target_points=args.init_fg_target_points)

        if args.static_background_ply:
            gp_init = append_static_background(
                gp_init, os.path.join(args.source_path, args.static_background_ply))
        gaussians.create_from_pcd_4d(gp_init, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_loader

    def getTestCameras(self):
        return self.test_loader
