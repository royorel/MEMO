from typing import Tuple
import torch
from memo.regularization.normal_field import get_gaussian_std_in_direction


def get_intersecting_pivots_from_normals(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    normals: torch.Tensor,
    std_factor: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two pivots per Gaussian, the point `std_factor` standard deviations along
    its learned normal and its center.

    Args:
        xyz (torch.Tensor): Gaussian centers, of shape (N_gaussians, 3).
        scaling (torch.Tensor): Gaussian scales, of shape (N_gaussians, 3).
        rotation (torch.Tensor): Gaussian rotations as quaternions, of shape (N_gaussians, 4).
        normals (torch.Tensor): Learned normals, of shape (N_gaussians, 3), not necessarily unit length.
        std_factor (float, optional): Distance of the normal pivot from the center, in standard
            deviations along the normal. Defaults to 3.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - pivots (torch.Tensor): Of shape (N_kept, 2, 3), the normal pivot first and the center last.
            - pivots_scale (torch.Tensor): Of shape (N_kept, 2), the largest scale of each pivot's Gaussian.
            Gaussians with a zero normal are left out.
    """
    # The two pivots of a zero normal coincide, which crashes gdel3d.
    valid_mask = normals.norm(dim=-1) > 0
    if (~valid_mask).any():
        print(f"[INFO] Discarding {(~valid_mask).sum().item()} Gaussians with zero normals")
        xyz = xyz[valid_mask]
        scaling = scaling[valid_mask]
        rotation = rotation[valid_mask]
        normals = normals[valid_mask]

    normals = torch.nn.functional.normalize(normals, dim=-1)  # (N_gaussians, 3)

    # Clamped so that flat Gaussians do not give coincident pivots either.
    normal_stds = get_gaussian_std_in_direction(
        directions=normals.unsqueeze(1),  # (N_gaussians, 1, 3)
        gaussian_scaling=scaling,
        gaussian_rotation=rotation,
        normalize_directions=False,
    ).clamp(min=1e-6)  # (N_gaussians, 1)

    pivots = torch.stack([xyz + std_factor * normal_stds * normals, xyz], dim=1)  # (N_gaussians, 2, 3)
    pivots_scale = scaling.detach().max(dim=-1).values[:, None].expand(-1, 2).contiguous()  # (N_gaussians, 2)
    return pivots, pivots_scale
