import torch
from memo.utils.general_utils import build_scaling_rotation


def get_gaussian_std_in_direction(
    directions: torch.Tensor,
    gaussian_scaling: torch.Tensor,
    gaussian_rotation: torch.Tensor,
    normalize_directions: bool = True,
) -> torch.Tensor:
    """
    Get the standard deviation of the Gaussian in the given directions.

    Args:
        directions (torch.Tensor): A vector of shape (N_gaussians, n_directions, 3).
        gaussian_scaling (torch.Tensor): The scaling of the Gaussians. Has shape (N_gaussians, 3).
        gaussian_rotation (torch.Tensor): The rotation of the Gaussians as quaternions. Has shape (N_gaussians, 4).
        normalize_directions (bool): Whether to normalize the directions.

    Returns:
        torch.Tensor: The standard deviation of the Gaussian in the direction of the given vector, of shape (N_gaussians, n_directions).
    """
    transposed_scaled_rotation = build_scaling_rotation(
        s=gaussian_scaling,  # (N_gaussians, 3)
        r=gaussian_rotation,  # (N_gaussians, 4)
    ).transpose(-1, -2)  # (N_gaussians, 3, 3)

    if normalize_directions:
        directions_to_use = torch.nn.functional.normalize(directions, dim=-1)
    else:
        directions_to_use = directions

    scaled_directions = torch.bmm(
        transposed_scaled_rotation,  # (N_gaussians, 3, 3)
        directions_to_use.permute(0, 2, 1),  # (N_gaussians, 3, n_directions)
    ).permute(0, 2, 1)  # (N_gaussians, n_directions, 3)

    direction_stds = scaled_directions.norm(dim=-1)  # (N_gaussians, n_directions)
    return direction_stds  # (N_gaussians, n_directions)
