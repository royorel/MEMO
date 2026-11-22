from typing import List
import torch
from memo.scene.cameras import Camera
from memo.scene.mesh import (
    Meshes,
    MeshRasterizer,
    ScalableMeshRenderer,
)
from memo.utils.camera_utils import get_cameras_spatial_extent
from memo.utils.geometry_utils import (
    is_in_view_frustum,
    transform_points_to_pixel_space,
    transform_points_world_to_view,
)
from tqdm import tqdm


class AdaptiveTSDF:
    def __init__(
        self,
        points:torch.Tensor,
        trunc_margin:float,
        initial_sdf_value:float,
    ):
        """
        A TSDF field with colors at a set of points, fused from a collection of posed depth maps.

        Args:
            points (torch.Tensor): Points at which to compute the TSDF. Has shape (N, 3).
            trunc_margin (float): Truncation margin for the TSDF.
            initial_sdf_value (float): TSDF value of a point no observation reached.
        """
        self._n_points = points.shape[0]
        self._points = points
        self._trunc_margin = trunc_margin
        self.tsdf = initial_sdf_value * torch.ones(self._n_points, 1, device=points.device)
        self.weights = torch.zeros(self._n_points, 1, device=points.device)
        self.colors = torch.zeros(self._n_points, 3, device=points.device)

    def integrate(
        self,
        img:torch.Tensor,
        depth:torch.Tensor,
        camera:Camera,
    ):
        """
        Integrate a new observation into the TSDF, reading the depth and color of
        the pixel each point projects to.

        Args:
            img (torch.Tensor): Image. Has shape (H, W, 3) or (3, H, W).
            depth (torch.Tensor): Depth. Has shape (H, W), (H, W, 1) or (1, H, W).
            camera (Camera): Camera.
        """
        # Reshape image and depth to (H, W, 3) and (H, W) respectively
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        depth = depth.squeeze()
        H, W = depth.shape
        device = self._points.device

        # Transform points to view space
        view_points = transform_points_world_to_view(
            points=self._points.view(1, self._n_points, 3),
            cameras=[camera],
        )[0]  # (N, 3)

        # Project points to pixel space
        pix_points = transform_points_to_pixel_space(
            points=view_points.view(1, self._n_points, 3),
            cameras=[camera],
            points_are_already_in_view_space=True,
            keep_float=True,
        )[0]  # (N, 2)
        int_pix_points = pix_points.round().long()  # (N, 2)
        pix_x, pix_y, pix_z = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]
        int_pix_x, int_pix_y = int_pix_points[..., 0], int_pix_points[..., 1]

        # Remove points outside view frustum and outside depth range
        valid_mask = (
            (pix_x >= 0) & (pix_x <= W-1)
            & (pix_y >= 0) & (pix_y <= H-1)
            & (pix_z > camera.znear)
            & (pix_z < camera.zfar)
        )  # (N,)

        if valid_mask.sum() > 0:
            # Get depth and image values at pixel locations
            packed_values = torch.cat(
                [
                    -torch.ones(len(valid_mask), 1, device=device),  # Depth values
                    torch.zeros(len(valid_mask), 3, device=device)  # Image values
                ],
                dim=-1
            )  # (N, 4)
            packed_values[valid_mask] = torch.cat([depth.unsqueeze(-1), img], dim=-1)[int_pix_y[valid_mask], int_pix_x[valid_mask]]  # (N_valid, 4)
            depth_values = packed_values[..., :1]  # (N, 1)
            img_values = packed_values[..., 1:]  # (N, 3)
            valid_mask = valid_mask & (depth_values[..., 0] > 0.)  # (N,)

            # Compute distance
            sdf = ((depth_values - pix_z.unsqueeze(-1)) / self._trunc_margin).clamp_max(1.)  # (N, 1)
            valid_mask = valid_mask & (sdf[..., 0] >= -1.)

            # Update field values, each observation with unit weight
            new_weights = self.weights + 1.0  # (N, 1)
            new_tsdf = (self.tsdf * self.weights + sdf) / new_weights  # (N, 1)
            new_colors = (self.colors * self.weights + img_values) / new_weights  # (N, 3)
            new_colors = new_colors.clamp(min=0., max=1.)  # (N, 3)

            self.weights = torch.where(valid_mask.unsqueeze(-1), new_weights, self.weights).detach()  # (N, 1)
            self.tsdf = torch.where(valid_mask.unsqueeze(-1), new_tsdf, self.tsdf).detach()  # (N, 1)
            self.colors = torch.where(valid_mask.unsqueeze(-1), new_colors, self.colors).detach()  # (N, 3)


def evaluate_mesh_colors(
    views:List[Camera],
    mesh:Meshes,
    trunc_margin:float=None,
    override_points:torch.Tensor=None,
) -> torch.Tensor:
    """Evaluate vertex colors using Depth Fusion.
    Vertices that are not visible won't be assigned a color.

    Args:
        views (List[Camera]): List of cameras.
        mesh (Meshes): Mesh.
        trunc_margin (float, optional): Truncation margin. Defaults to None.
        override_points (torch.Tensor, optional): Points to color in place of the mesh vertices. Defaults to None.

    Returns:
        colors (torch.Tensor): Colors. Has shape (N, 3).
        hit_mask (torch.Tensor): Mask of the points some view observed. Has shape (N,).
    """
    if trunc_margin is None:
        trunc_margin = 2e-3 * get_cameras_spatial_extent(views)["radius"]

    points = mesh.verts if override_points is None else override_points

    tsdf_volume = AdaptiveTSDF(
        points=points,
        trunc_margin=trunc_margin,
        initial_sdf_value=-1.1,
    )

    mesh_renderer = ScalableMeshRenderer(MeshRasterizer(cameras=views))

    for cam_id, view in enumerate(tqdm(views, desc="Computing vertex colors")):
        faces_mask = is_in_view_frustum(mesh.verts, view)[mesh.faces].any(axis=1)
        render_pkg = mesh_renderer(
            Meshes(verts=mesh.verts, faces=mesh.faces[faces_mask]),
            cam_idx=cam_id,
            return_depth=True,
            use_antialiasing=False,
        )
        tsdf_volume.integrate(
            img=view.original_image.to(points.device),
            depth=render_pkg["depth"],
            camera=view,
        )

    return tsdf_volume.colors, tsdf_volume.tsdf[..., 0] > -1.1


def evaluate_mesh_colors_all_vertices(
    views:List[Camera],
    mesh:Meshes,
) -> torch.Tensor:
    """Evaluate vertex colors for all vertices of the mesh, including the ones that are not visible.
    Useful for avoiding artifacts when rendering the mesh.

    Args:
        views (List[Camera]): List of cameras.
        mesh (Meshes): Mesh.

    Returns:
        vertex_colors (torch.Tensor): Colors. Has shape (N, 3).
    """
    # First pass : evaluate colors for visible vertices
    print("[INFO] Computing colors for visible vertices.")
    vertex_colors, color_mask = evaluate_mesh_colors(
        views=views,
        mesh=mesh,
    )

    # Second pass : evaluate colors for non-visible vertices by using a larger truncation margin
    print("[INFO] Estimating colors for non-visible vertices.")
    vertex_colors[~color_mask], _ = evaluate_mesh_colors(
        views=views,
        mesh=mesh,
        trunc_margin=1.0 * get_cameras_spatial_extent(views)["radius"],
        override_points=mesh.verts[~color_mask],
    )
    return vertex_colors
