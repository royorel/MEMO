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

import math
import torch
from torch import nn
import numpy as np
from memo.utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 normalized_t=None, t=None, view_num=None, cx=None, cy=None,
                 image_width=None, image_height=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        # Default normalized_t to 0.0 so apply_4d=True rendering works on
        # static scenes without breakage. Time value doesn't matter on a
        # static scene (all frames are identical) — any fixed t would do.
        self.normalized_t = normalized_t if normalized_t is not None else 0.0
        self.t = t
        self.view_num = view_num
        self.cx = cx
        self.cy = cy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # Cached cameras pass a 1x1 dummy image plus the true resolution so the
        # projection is built at the real W/H (cx/W blows up otherwise for an
        # off-center principal point). Live cameras derive W/H from the image.
        self.image_width = image_width if image_width is not None else self.original_image.shape[2]
        self.image_height = image_height if image_height is not None else self.original_image.shape[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=self.cx, cy=self.cy, W=self.image_width, H=self.image_height).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        self.Fx = self.image_width / (2 * math.tan(self.FoVx / 2.))
        self.Fy = self.image_height / (2 * math.tan(self.FoVy / 2.))
        self.Cx = self.cx if self.cx is not None else float(self.image_width - 1) / 2
        self.Cy = self.cy if self.cy is not None else float(self.image_height - 1) / 2


    def to_device(self, device):
        self.data_device = torch.device(device)
        self.original_image = self.original_image.to(self.data_device)
        self.world_view_transform = self.world_view_transform.to(self.data_device)
        self.projection_matrix = self.projection_matrix.to(self.data_device)
        self.full_proj_transform = self.full_proj_transform.to(self.data_device)
        self.camera_center = self.camera_center.to(self.data_device)

    def load_original_image(self):
        """Load the real image from LMDB if original_image is a 1x1 dummy placeholder."""
        if self.original_image.shape[-1] != 1 or not hasattr(self, '_lmdb_path'):
            return
        import lmdb
        import pickle
        env = lmdb.open(self._lmdb_path, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            raw = pickle.loads(txn.get(self._lmdb_key.encode()))
        env.close()
        gt_image = (torch.from_numpy(raw['image']).float() / 255.0).permute(2, 0, 1).contiguous()
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)

