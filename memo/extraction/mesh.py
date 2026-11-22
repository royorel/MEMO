from typing import Callable, Tuple
import gc
import torch
from memo.scene.mesh import Meshes
from memo.utils.tetmesh import marching_tetrahedra


def extract_mesh(
    delaunay_tets: torch.Tensor,
    pivots: torch.Tensor,
    pivots_sdf: torch.Tensor,
    pivots_scale: torch.Tensor,
    filter_large_edges: bool,
    large_edge_mult: float,
) -> Tuple[Meshes, torch.Tensor, torch.Tensor]:
    """
    Extract a mesh from a set of pivots, their SDF values and the Delaunay triangulation.

    Args:
        delaunay_tets (torch.Tensor): The Delaunay tetrahedra. Shape: (N_tets, 4).
        pivots (torch.Tensor): The pivots. Shape: (N_pivots, 3).
        pivots_sdf (torch.Tensor): The SDF values for the pivots. Shape: (N_pivots,).
        pivots_scale (torch.Tensor): The extent of each pivot's Gaussian, std_factor times its
            largest scale. Shape: (N_pivots,).
        filter_large_edges (bool): Drop the triangles of a surface-crossing edge longer than
            large_edge_mult times the sum of its two endpoints' extents.
        large_edge_mult (float): Multiplier on the large-edge threshold, >1 keeps more triangles.

    Returns:
        Tuple[Meshes, torch.Tensor, torch.Tensor]: The extracted mesh, and the end points
            (N_verts, 2, 3) and end SDF values (N_verts, 2, 1) of each vertex's crossing edge.
    """
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(
        vertices=pivots[None],
        tets=delaunay_tets,
        sdf=pivots_sdf.reshape(1, -1),
        scales=pivots_scale.reshape(-1, 1)[None],
    )
    end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
    end_scales = scale_list[0]  # (N_verts, 2, 1)

    # Linear interpolation of the SDF values along each crossing edge
    norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)  # (N_verts, 2, 1)
    verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]
    faces = faces_list[0]  # (N_faces, 3)

    if filter_large_edges:
        edge_length = (end_points[:, 1, :] - end_points[:, 0, :]).norm(dim=-1)  # (N_verts,)
        short_edge = edge_length <= large_edge_mult * (end_scales[:, 0, 0] + end_scales[:, 1, 0])  # (N_verts,)
        faces = faces[short_edge[faces].all(axis=1)]

    return Meshes(verts=verts, faces=faces), end_points, end_sdf


def refine_intersections_with_binary_search(
    end_points: torch.Tensor,
    end_sdf: torch.Tensor,
    sdf_function: Callable,
    n_binary_steps: int,
) -> torch.Tensor:
    """
    Refine the intersected isosurface points with binary search.

    Args:
        end_points (torch.Tensor): The end points. (N_verts, 2, 3)
        end_sdf (torch.Tensor): The SDF values at the end points. (N_verts, 2, 1)
        sdf_function (Callable): The SDF function. Takes a tensor of points and returns the SDF values.
        n_binary_steps (int): The number of binary steps.

    Returns:
        refined_points (torch.Tensor): The refined points. (N_verts, 3)
    """
    left_points = end_points[:, 0, :].clone()  # (N_verts, 3)
    right_points = end_points[:, 1, :].clone()  # (N_verts, 3)
    left_sdf = end_sdf[:, 0, :].clone()  # (N_verts, 1)
    right_sdf = end_sdf[:, 1, :].clone()  # (N_verts, 1)
    points = (left_points + right_points) / 2  # (N_verts, 3)

    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2

        mid_sdf = sdf_function(mid_points)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

        points = (left_points + right_points) / 2  # (N_verts, 3)

        torch.cuda.empty_cache()
        gc.collect()

    return points
