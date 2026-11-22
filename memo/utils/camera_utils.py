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
from typing import List
from memo.scene.cameras import Camera


def get_cameras_spatial_extent(cameras:List[Camera]):
    cam_centers = torch.cat([camera.camera_center.view(1, 3) for camera in cameras], dim=0)

    avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)
    dist = torch.norm(cam_centers - avg_cam_center, dim=1, keepdim=True)

    half_diagonal = torch.max(dist)
    radius = half_diagonal * 1.1

    translate = -avg_cam_center

    return {"translate": translate, "radius": radius, "avg_cam_center": avg_cam_center}
