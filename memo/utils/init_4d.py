"""4D initialization: Kalman track matching, GP regression, and densification.

Per-frame point clouds are tracked across the sequence with a Kalman filter,
each track's motion is fit with a closed-form GP regression, and the tracked
cloud is densified by importance sampling. Scene.__init__ calls
set_initial_pcd_kalman_filter_gp or set_initial_pcd_kalman_multi_anchor.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch3d.ops import knn_points as knn
from simple_knn._C import distCUDA2
from plyfile import PlyData

from memo.utils.graphics_utils import BasicPointCloud


# ---------------------------------------------------------------------------
# GP regression from tracks
# ---------------------------------------------------------------------------

def _se_kernel(t1, t2, lengthscale):
    """Squared exponential kernel. t1: (..., N), t2: (..., M) -> (..., N, M)."""
    return torch.exp(-0.5 * (t1.unsqueeze(-1) - t2.unsqueeze(-2)) ** 2 / lengthscale ** 2)


def gp_regression_from_tracks(
    track_positions,    # (N_tracks, L_max, 3) — padded positions
    track_times,        # (N_tracks, L_max) — normalized times
    track_lengths,      # (N_tracks,) — actual track length per point
    M=10,               # number of inducing points inside [0, 1]
    lengthscale=None,   # initial GP lengthscale; default 1/M (one inducing-point spacing)
    noise_var=1e-3,     # observation noise variance
    boundary_inducing_points=0,  # extra inducing points per side outside [0, 1]
):
    """Compute GP initialization from tracked point trajectories.

    Returns:
        dict with keys:
            'xyz': (N_tracks, 3) — base positions (mean of tracked positions)
            'pos_gp_alpha': (N_tracks, M_total, 3) — GP displacement coefficients
            'pos_gp_inducing_t': (N_tracks, M_total) — inducing times (bin midpoints)
            'pos_gp_lengthscale': (N_tracks, 1) — uniform lengthscale

    With boundary_inducing_points = N, the regression runs at
    M_total = M + 2N inducing points placed at bin midpoints
    (m - N + 0.5)/M for m in 0..M_total-1, which matches
    GaussianModel4D._bounded_inducing_t at raw=0.
    """
    if lengthscale is None:
        lengthscale = 1.0 / M
    M_total = M + 2 * boundary_inducing_points
    N_tracks = track_positions.shape[0]
    L_max = track_positions.shape[1]
    device = track_positions.device
    print(f"GP regression: {N_tracks} tracks, L_max={L_max}, M={M}, M_total={M_total}, "
          f"lengthscale={lengthscale}, noise_var={noise_var}")

    # Mask for valid (non-padded) entries
    mask = torch.arange(L_max, device=device).unsqueeze(0) < track_lengths.unsqueeze(1)  # (N, L_max)
    valid_counts = mask.sum(dim=1).clamp(min=1)  # (N,)

    # Base position: mean of valid tracked positions
    masked_positions = track_positions * mask.unsqueeze(-1).float()
    base_xyz = masked_positions.sum(dim=1) / valid_counts.unsqueeze(-1).float()  # (N, 3)

    # Displacements from base position
    delta_Y = track_positions - base_xyz.unsqueeze(1)  # (N, L_max, 3)
    delta_Y = delta_Y * mask.unsqueeze(-1).float()  # zero out padded entries

    # Inducing times: bin midpoints (m - N + 0.5)/M for m in 0..M_total-1.
    # Matches GaussianModel4D._bounded_inducing_t at raw=0. At
    # boundary_inducing_points=0 this collapses to (arange(M) + 0.5)/M.
    inducing_t = ((torch.arange(M_total, dtype=torch.float, device=device)
                   - boundary_inducing_points + 0.5) / M)
    inducing_t = inducing_t.unsqueeze(0).expand(N_tracks, -1)

    # Kernel matrices
    K_MM = _se_kernel(inducing_t, inducing_t, lengthscale)  # (N, M_total, M_total)
    K_LM = _se_kernel(track_times, inducing_t, lengthscale)  # (N, L_max, M_total)

    # Zero out kernel entries for padded positions
    K_LM = K_LM * mask.unsqueeze(-1).float()  # (N, L_max, M_total)

    K_ML = K_LM.transpose(-1, -2)  # (N, M_total, L_max)

    # Closed-form alpha solve: A @ alpha = K_ML @ delta_Y
    A = noise_var * K_MM + K_ML @ K_LM  # (N, M_total, M_total)
    A = A + 1e-6 * torch.eye(M_total, device=device).unsqueeze(0)  # jitter
    rhs = K_ML @ delta_Y  # (N, M_total, 3)

    alpha = torch.linalg.solve(A, rhs)  # (N, M_total, 3)

    # Uniform lengthscale for all Gaussians
    ls = torch.full((N_tracks, 1), lengthscale, device=device)

    return {
        'xyz': base_xyz,
        'pos_gp_alpha': alpha,
        'pos_gp_inducing_t': inducing_t,
        'pos_gp_lengthscale': ls,
    }


# ---------------------------------------------------------------------------
# Kalman filter tracking
# ---------------------------------------------------------------------------

def _kalman_filter_forward(point_clouds, timestamps, n_sigma=2.0, K_local=8):
    """Run forward Kalman over a sequence with a position Mahalanobis gate
    and in-loop local-motion fill from K nearest accepted neighbors.

    Reads `point_clouds[0]` and `point_clouds[1]` for the two-frame init,
    then loops from index 2. The "anchor" concept lives outside this function:
    callers pass forward or reversed sub-sequences as appropriate.

    Gating is a 3-DOF Mahalanobis ellipsoid on the position innovation
    against `S = H P H^T + R`; accept if maha_pos < n_sigma**2 * 3.
    Position is the only gate. Color is deliberately excluded: a
    per-channel color z-gate calibrates its sigma on static-dominated
    accepted residuals and systematically rejects correctly tracked
    movers whose color legitimately changes under lighting.

    For tracks whose update is rejected at frame t, the (position, velocity, P)
    triple is replaced with values derived from the K nearest accepted tracks
    at the same frame via position-only KNN. Displacement is one-step (t-1 → t).

    Args:
        point_clouds: list of (N_i, 6) tensors on CUDA (xyz + weighted rgb).
            All N_i must equal N (caller's responsibility).
        timestamps: (T,) tensor on CUDA, monotonically increasing.
        n_sigma: per-dim sigma multiplier for the position ellipsoid
            threshold n_sigma**2 * 3. Default 2.0.
        K_local: number of nearest accepted neighbors used for the local-motion
            fill of gated tracks. Default 8.

    Returns:
        positions: (N_tracks, T, 3) Kalman positions; gated entries are filled
            from local motion (no separate post-pass needed).
        colors: (N_tracks, T, 3) per-frame colors at accepted associations
            (zero on rejected frames).
        accepted: (N_tracks, T) bool, True where the measurement update was
            applied; the anchor frame (index 0) is always True by construction.
    """
    K = 1
    base_dim = 3
    num_states = 2 * base_dim
    T_seq = len(point_clouds)
    N = point_clouds[0].shape[0]
    pos_threshold = (n_sigma ** 2) * base_dim  # 3-DOF Mahalanobis ellipsoid for position

    kalman_states = torch.zeros(T_seq, N, num_states, 1, device='cuda')
    kalman_colors = torch.zeros(T_seq, N, 3, 1, device='cuda')
    accepted = torch.zeros(T_seq, N, dtype=torch.bool, device='cuda')

    H = torch.eye(base_dim, num_states, device='cuda')[None]

    dist2_list = [torch.clamp_min(distCUDA2(pcd[:, :3]).float().cuda(), 1e-7) for pcd in point_clouds]
    dist2 = torch.cat(dist2_list).mean()
    R = (dist2 * torch.eye(base_dim, device='cuda'))[None]

    position_var = 10 * dist2 * torch.ones(3, 1, device='cuda')
    velocity_var = 40 * dist2 * torch.ones(3, 1, device='cuda')
    process_vars = torch.cat([position_var, velocity_var], dim=0)
    Q = (process_vars * torch.eye(num_states, device='cuda'))[None]

    def make_transition(dt):
        A = torch.eye(num_states, device='cuda')[None]
        for i in range(base_dim):
            A[0, i, base_dim + i] = dt
        return A

    kalman_states[0, :, :base_dim, 0] = point_clouds[0][:, :base_dim]
    kalman_colors[0] = point_clouds[0][:, 3:, None]
    accepted[0] = True

    if T_seq < 2:
        positions = kalman_states[:, :, :3, 0].permute(1, 0, 2).contiguous()
        colors = kalman_colors[:, :, :, 0].permute(1, 0, 2).contiguous()
        return positions, colors, accepted.permute(1, 0).contiguous()

    _, nn_idx, _ = knn(point_clouds[0][None, :, :base_dim],
                       point_clouds[1][None, :, :base_dim], K=K)
    nn_idx = nn_idx[0, :, 0]
    dt01 = timestamps[1] - timestamps[0]
    kalman_states[1, :, :base_dim, 0] = point_clouds[1][nn_idx, :base_dim]
    kalman_states[1, :, base_dim:, 0] = (point_clouds[1][nn_idx, :base_dim]
                                          - kalman_states[0, :, :base_dim, 0]) / dt01
    kalman_colors[1] = point_clouds[1][nn_idx, 3:, None]
    accepted[1] = True

    P = Q.clone().expand(N, num_states, num_states).contiguous()

    for t in tqdm(range(2, T_seq), desc='Kalman filter'):
        dt = timestamps[t] - timestamps[t - 1]
        prev_state = kalman_states[t - 1]

        A = make_transition(dt)
        predicted_state = torch.matmul(A, prev_state)
        P = torch.matmul(torch.matmul(A, P), A.permute(0, 2, 1)) + Q

        PHt = torch.matmul(P, H.permute(0, 2, 1))
        S = torch.matmul(H, PHt) + R
        S_inv = torch.linalg.inv(S)
        K_gain = torch.matmul(PHt, S_inv)

        _, nn_idx, _ = knn(predicted_state[None, :, :base_dim],
                           point_clouds[t][None, :, :base_dim], K=K)
        nn_idx = nn_idx[0, :, 0]

        measurement = point_clouds[t][nn_idx, :base_dim, None]
        innovation = measurement - torch.matmul(H, predicted_state)

        # 3-DOF Mahalanobis ellipsoid gate on position.
        maha_pos = torch.matmul(torch.matmul(innovation.permute(0, 2, 1), S_inv),
                                innovation).squeeze(-1).squeeze(-1)        # (N,)
        accept_mask = maha_pos < pos_threshold

        updated_state = predicted_state + torch.matmul(K_gain, innovation)
        updated_P = P - torch.matmul(K_gain, torch.matmul(H, P))

        am = accept_mask[:, None, None]
        kalman_states[t] = torch.where(am, updated_state, predicted_state)
        P = torch.where(am, updated_P, P)
        kalman_colors[t] = torch.where(
            am, point_clouds[t][nn_idx, 3:, None], torch.zeros_like(kalman_colors[t]))
        accepted[t] = accept_mask

        # Local-motion fill for gated tracks: query the K nearest accepted
        # neighbors at frame t, transfer their one-step displacement +
        # instantaneous velocity + averaged covariance to the gated state.
        gated_idx = (~accept_mask).nonzero(as_tuple=True)[0]
        if gated_idx.numel() > 0:
            acc_idx = accept_mask.nonzero(as_tuple=True)[0]
            if acc_idx.numel() > 0:
                K_use = min(K_local, acc_idx.numel())
                query = kalman_states[t, gated_idx, :base_dim, 0]
                target = kalman_states[t, acc_idx, :base_dim, 0]
                _, nn_local, _ = knn(query[None], target[None], K=K_use)
                nn_global = acc_idx[nn_local[0]]  # (n_gated, K_use)

                disp = (kalman_states[t,     nn_global, :base_dim, 0]
                        - kalman_states[t - 1, nn_global, :base_dim, 0]).mean(dim=1)
                dt_step = timestamps[t] - timestamps[t - 1]

                kalman_states[t, gated_idx, :base_dim, 0] = (
                    kalman_states[t - 1, gated_idx, :base_dim, 0] + disp
                )
                kalman_states[t, gated_idx, base_dim:, 0] = disp / dt_step
                P[gated_idx] = P[nn_global].mean(dim=1)

    positions = kalman_states[:, :, :3, 0].permute(1, 0, 2).contiguous()
    colors = kalman_colors[:, :, :, 0].permute(1, 0, 2).contiguous()
    return positions, colors, accepted.permute(1, 0).contiguous()


def _kalman_filter_bidirectional(point_clouds, timestamps, anchor, n_sigma=2.0, K_local=8):
    """Run forward + backward Kalman from an anchor frame, splice into full trajectories.

    Args:
        point_clouds: list of (N_i, 6) tensors. The anchor's cloud seeds tracks.
        timestamps: (T,) tensor on CUDA, monotonically increasing.
        anchor: integer in [0, T-1]. anchor=0 is a single forward pass.
        n_sigma: see _kalman_filter_forward.
        K_local: K for the in-loop local-motion fill (see _kalman_filter_forward).

    Returns:
        positions: (N_tracks, T, 3)
        colors: (N_tracks, T, 3)
        accepted: (N_tracks, T) bool
    """
    T = len(point_clouds)
    if not 0 <= anchor < T:
        raise ValueError(f"anchor={anchor} out of range [0, {T})")

    fwd_pcs = point_clouds[anchor:]
    fwd_ts = timestamps[anchor:].contiguous()
    fwd_pos, fwd_col, fwd_acc = _kalman_filter_forward(fwd_pcs, fwd_ts, n_sigma, K_local)

    if anchor == 0:
        return fwd_pos, fwd_col, fwd_acc

    bwd_pcs = list(reversed(point_clouds[:anchor + 1]))
    t_anchor = timestamps[anchor]
    bwd_ts = (t_anchor + (t_anchor - timestamps[:anchor + 1])).flip(0).contiguous()
    bwd_pos, bwd_col, bwd_acc = _kalman_filter_forward(bwd_pcs, bwd_ts, n_sigma, K_local)

    if anchor == T - 1:
        return bwd_pos.flip(1).contiguous(), bwd_col.flip(1).contiguous(), bwd_acc.flip(1).contiguous()

    N = fwd_pos.shape[0]
    positions = torch.zeros(N, T, 3, device='cuda')
    colors = torch.zeros(N, T, 3, device='cuda')
    accepted = torch.zeros(N, T, dtype=torch.bool, device='cuda')

    positions[:, anchor:] = fwd_pos
    colors[:, anchor:] = fwd_col
    accepted[:, anchor:] = fwd_acc

    # Backward owns [0, anchor-1]. Skip its index-0 (the anchor frame), since
    # the forward pass already covers it.
    positions[:, :anchor] = bwd_pos[:, 1:].flip(1)
    colors[:, :anchor] = bwd_col[:, 1:].flip(1)
    accepted[:, :anchor] = bwd_acc[:, 1:].flip(1)

    return positions, colors, accepted


# ---------------------------------------------------------------------------
# Point cloud densification via P(z|x,y) importance sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sample_pdf(weights, num_samples, bounds_min, bounds_max):
    """Sample new positions via inverse CDF of P(z|x,y) from an occupancy volume.

    Adapted from SplatFields/scene/splatsdf_model.py.

    Args:
        weights: (N, N, N) bool or float tensor
        num_samples: number of samples to generate
        bounds_min: (3,) scene bounds min
        bounds_max: (3,) scene bounds max

    Returns:
        samples: (num_samples, 3) world-space positions
    """
    device = weights.device
    N = weights.shape[0]
    uniform_samples = torch.rand(num_samples, 3, device=device)
    bins = torch.linspace(0.0, 1.0, steps=N + 1, device=device)

    # Compute conditional distributions
    pdf_xyz = weights.float() / (weights.float().sum() + 1e-15)
    pdf_xy = pdf_xyz.sum(dim=-1, keepdim=True)
    pdf_x = pdf_xy.sum(dim=-2, keepdim=True)
    cdf_x = pdf_x[:, 0, 0].cumsum(dim=-1)
    pdf_y_given_x = pdf_xy[..., 0] / (pdf_x[..., 0] + 1e-15)
    cdf_y_given_x = pdf_y_given_x.cumsum(dim=-1)
    pdf_z_given_xy = pdf_xyz / (pdf_xy + 1e-15)
    cdf_z_given_xy = pdf_z_given_xy.cumsum(dim=-1)

    uniform_grid_pos = -1 + 2 * torch.linspace(0.5 / N, 1.0 - 0.5 / N, steps=N, device=device)

    samples = []
    for i in range(3):
        curr_uniform = uniform_samples[:, i].contiguous()

        if i == 0:
            curr_cdf = torch.cat([torch.zeros_like(cdf_x[:1]), cdf_x], dim=-1).contiguous()
        elif i == 1:
            normalized_samples_x = -1 + 2 * samples[0]
            grid = torch.stack([
                uniform_grid_pos[None].expand(samples[0].shape[0], -1),
                normalized_samples_x[:, None].expand(-1, uniform_grid_pos.shape[0])
            ], dim=-1)
            curr_cdf = F.grid_sample(cdf_y_given_x[None, None], grid[None], mode='nearest')[0, 0]
            curr_cdf = torch.cat([torch.zeros_like(curr_cdf[:, :1]), curr_cdf], dim=-1).contiguous()
        else:
            normalized_samples_y = -1 + 2 * samples[1]
            grid = torch.stack([
                uniform_grid_pos[None].expand(samples[0].shape[0], -1),
                normalized_samples_y[:, None].expand(-1, uniform_grid_pos.shape[0]),
                normalized_samples_x[:, None].expand(-1, uniform_grid_pos.shape[0])
            ], dim=-1)
            curr_cdf = F.grid_sample(cdf_z_given_xy[None, None], grid[None, None],
                                      mode='nearest')[0, 0, 0]
            curr_cdf = torch.cat([torch.zeros_like(curr_cdf[:, :1]), curr_cdf], dim=-1).contiguous()

        inds = torch.searchsorted(curr_cdf, curr_uniform[..., None], right=True)
        below = torch.max(torch.zeros_like(inds), inds - 1)
        above = torch.min((curr_cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.cat([below, above], dim=-1)

        matched_shape = [inds_g.shape[0], curr_cdf.shape[-1]]
        if curr_cdf.dim() == 1:
            curr_cdf = curr_cdf[None]

        cdf_g = torch.gather(curr_cdf.expand(matched_shape), 1, inds_g)
        bins_g = torch.gather(bins[None].expand(matched_shape), 1, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (curr_uniform - cdf_g[..., 0]) / denom
        curr_samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        samples.append(curr_samples)

    # Normalized [0,1] -> world space
    samples = torch.stack(samples, dim=-1)  # (num_samples, 3)
    samples = bounds_min + samples * (bounds_max - bounds_min)

    return samples


def _build_distance_field(tracked_xyz, voxel_resolution=64, margin=0.1):
    """Build a distance-based weight volume from tracked base positions.

    For each voxel, computes the distance to the nearest tracked point.
    Voxels within a threshold distance are weighted for sampling, producing
    a smooth shell around the tracked surface.

    Args:
        tracked_xyz: (N, 3) tensor of base positions
        voxel_resolution: grid size N (creates N^3 voxels)
        margin: extra space around point cloud bounds (fraction of extent)

    Returns:
        weights: (N, N, N) float tensor (1 inside shell, 0 outside)
        bounds_min: (3,) tensor
        bounds_max: (3,) tensor
    """
    device = tracked_xyz.device
    bounds_min = tracked_xyz.min(dim=0).values
    bounds_max = tracked_xyz.max(dim=0).values
    extent = bounds_max - bounds_min
    bounds_min = bounds_min - margin * extent
    bounds_max = bounds_max + margin * extent

    R = voxel_resolution
    # Create voxel center coordinates
    half_step = 0.5 / R
    axes = [torch.linspace(half_step, 1.0 - half_step, R, device=device) for _ in range(3)]
    gx, gy, gz = torch.meshgrid(*axes, indexing='ij')
    voxel_centers_norm = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (R^3, 3)
    voxel_centers = bounds_min + voxel_centers_norm * (bounds_max - bounds_min)

    # Distance from each voxel to nearest tracked point
    sq_dists, _, _ = knn(voxel_centers[None], tracked_xyz[None], K=1)
    dists = sq_dists[0, :, 0].sqrt()  # (R^3,)

    # Threshold: 95th percentile of NN distances between tracked points
    tracked_nn_dists, _, _ = knn(tracked_xyz[None], tracked_xyz[None], K=2)
    tracked_nn_dist = tracked_nn_dists[0, :, 1].sqrt()  # K=2, skip self
    threshold = tracked_nn_dist.quantile(0.95).item()

    # Binary shell
    weights = (dists <= threshold).float().view(R, R, R)
    occupied = (weights > 0).sum().item()
    print(f"Distance field: threshold={threshold:.4f}, "
          f"{occupied}/{R**3} voxels in shell ({100*occupied/R**3:.1f}%)")

    return weights, bounds_min, bounds_max


def densify_point_cloud(gp_init, target_num_points=50000, voxel_resolution=64,
                        field_xyz=None):
    """Augment a sparse GP-initialized point cloud via importance sampling.

    Args:
        gp_init: dict from gp_regression_from_tracks
        target_num_points: target total point count
        voxel_resolution: grid resolution for the distance field
        field_xyz: points defining the sampling field and its bounds
            (default gp_init['xyz']). Far outliers stretch the field's
            bounding box and coarsen every voxel, so callers with a static
            zone pass the near-field subset to keep the grid resolving the
            room instead of the background tail.

    Returns:
        Updated gp_init dict with additional points
    """
    N_tracked = gp_init['xyz'].shape[0]
    if N_tracked >= target_num_points:
        print(f"Skipping densification: {N_tracked} tracked >= {target_num_points} target")
        return gp_init

    num_new = target_num_points - N_tracked
    if field_xyz is None or field_xyz.shape[0] == 0:
        field_xyz = gp_init['xyz']
    print(f"Densification: sampling {num_new} new points (resolution={voxel_resolution}^3)...")

    weights, bounds_min, bounds_max = _build_distance_field(
        field_xyz, voxel_resolution=voxel_resolution)

    if (weights > 0).sum().item() == 0:
        print("Warning: no occupied voxels, skipping densification")
        return gp_init

    # Sample new base positions
    new_xyz = _sample_pdf(weights, num_new, bounds_min, bounds_max)  # (num_new, 3)

    # Inherit GP params from nearest tracked Gaussian
    _, nn_idx, _ = knn(new_xyz[None], gp_init['xyz'][None], K=1)
    nn_idx = nn_idx[0, :, 0]  # (num_new,)

    new_alpha = gp_init['pos_gp_alpha'][nn_idx]  # (num_new, M, 3)
    new_inducing_t = gp_init['pos_gp_inducing_t'][nn_idx]  # (num_new, M)
    new_ls = gp_init['pos_gp_lengthscale'][nn_idx]  # (num_new, 1)

    # Concatenate
    gp_init['xyz'] = torch.cat([gp_init['xyz'], new_xyz], dim=0)
    gp_init['pos_gp_alpha'] = torch.cat([gp_init['pos_gp_alpha'], new_alpha], dim=0)
    gp_init['pos_gp_inducing_t'] = torch.cat([gp_init['pos_gp_inducing_t'], new_inducing_t], dim=0)
    gp_init['pos_gp_lengthscale'] = torch.cat([gp_init['pos_gp_lengthscale'], new_ls], dim=0)

    if 'rgb' in gp_init:
        new_rgb = gp_init['rgb'][nn_idx]
        gp_init['rgb'] = torch.cat([gp_init['rgb'], new_rgb], dim=0)

    print(f"Densified: {N_tracked} tracked + {num_new} sampled = {gp_init['xyz'].shape[0]} total")
    return gp_init


# ---------------------------------------------------------------------------
# Static zone
# ---------------------------------------------------------------------------

def rig_cameras_from_transforms(dataset_path):
    """Unique camera centers from transforms_train.json, as (V, 3) numpy."""
    with open(os.path.join(dataset_path, "transforms_train.json")) as f:
        frames = json.load(f)["frames"]
    seen = {}
    for fr in frames:
        view = fr["file_path"].split("view_")[-1]
        if view not in seen:
            seen[view] = np.array(fr["transform_matrix"], dtype=np.float64)[:3, 3]
    return np.stack(list(seen.values())).astype(np.float32)


def _resolve_static_zone(cam_centers, static_zone_distance, device):
    """(centroid tensor, d_star) for the static zone.

    static_zone_distance is a multiplier of the getNerfppNorm-style rig
    extent (1.1 x max camera distance from the centroid); d_star is
    multiplier x extent. Non-positive values fall back to 20x.
    """
    cams = torch.as_tensor(np.asarray(cam_centers), dtype=torch.float32,
                           device=device)
    centroid = cams.mean(dim=0)
    extent = 1.1 * (cams - centroid).norm(dim=-1).max()
    mult = static_zone_distance if static_zone_distance > 0 else 20.0
    return centroid, float(mult * extent)


def _harvest_far_points(point_clouds, centroid, d_star, device):
    """Union of per-frame points beyond d_star, exact-deduped.

    Collection bypasses tracking entirely, coverage of the far field must not
    depend on anchor-frame visibility or association survival.
    """
    xyz_parts, rgb_parts = [], []
    for pcd in point_clouds:
        pts = torch.from_numpy(pcd.points).float().to(device)
        far = (pts - centroid).norm(dim=-1) > d_star
        if far.any():
            xyz_parts.append(pts[far])
            rgb_parts.append(torch.from_numpy(pcd.colors).float().to(device)[far])
    if not xyz_parts:
        return None
    xyz = torch.cat(xyz_parts)
    rgb = torch.cat(rgb_parts)
    uniq, inverse = torch.unique(xyz, dim=0, return_inverse=True)
    first = torch.full((uniq.shape[0],), xyz.shape[0], dtype=torch.long,
                       device=device)
    first.scatter_reduce_(0, inverse,
                          torch.arange(xyz.shape[0], device=device),
                          reduce="amin", include_self=True)
    return {"xyz": xyz[first], "rgb": rgb[first]}


def _append_static_rows(gp_init, static_append):
    """Append harvested static points to a regression output dict."""
    if static_append is None or static_append['xyz'].shape[0] == 0:
        return gp_init
    n_s = static_append['xyz'].shape[0]
    m_total = gp_init['pos_gp_alpha'].shape[1]
    gp_init['xyz'] = torch.cat([gp_init['xyz'], static_append['xyz']], dim=0)
    gp_init['rgb'] = torch.cat([gp_init['rgb'], static_append['rgb']], dim=0)
    gp_init['pos_gp_alpha'] = torch.cat(
        [gp_init['pos_gp_alpha'],
         torch.zeros(n_s, m_total, 3, device='cuda')], dim=0)
    gp_init['pos_gp_inducing_t'] = torch.cat(
        [gp_init['pos_gp_inducing_t'],
         gp_init['pos_gp_inducing_t'][:1].expand(n_s, -1)], dim=0)
    gp_init['pos_gp_lengthscale'] = torch.cat(
        [gp_init['pos_gp_lengthscale'],
         gp_init['pos_gp_lengthscale'][:1].expand(n_s, -1)], dim=0)
    return gp_init


def _gate_static_zone(positions, accepted, point_clouds, cam_centers, static_zone_distance):
    """Static-zone gate shared by both Kalman initializations.

    Tracks whose median accepted-observation distance from the rig camera
    centroid exceeds d_star are dropped from the fit, since their apparent
    motion out there is reconstruction noise. The union of per-frame points
    beyond d_star is harvested to be appended as static points in their
    place (exact positional duplicates removed, since identical coordinates
    give degenerate simple-knn initial scales).

    Returns (static_mask, static_append, centroid, d_star).
    """
    centroid, d_star = _resolve_static_zone(cam_centers, static_zone_distance,
                                            positions.device)
    obs_dist = (positions - centroid.view(1, 1, 3)).norm(dim=-1)
    obs_dist = obs_dist.masked_fill(~accepted, float('nan'))
    static_mask = obs_dist.nanmedian(dim=1).values > d_star
    print(f"  static zone (distance > {d_star:g}): "
          f"{int(static_mask.sum())} tracks gated")
    static_append = _harvest_far_points(point_clouds, centroid, d_star,
                                        positions.device)
    if static_append is not None:
        print(f"  harvest: {static_append['xyz'].shape[0]} static "
              f"points from the per-frame far union")
    return static_mask, static_append, centroid, d_star


def _to_numpy_result(gp_init):
    result = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
              for k, v in gp_init.items()}
    result['normals'] = np.zeros((result['xyz'].shape[0], 3), dtype=np.float32)
    return result


# ---------------------------------------------------------------------------
# Initialization entry points
# ---------------------------------------------------------------------------

def set_initial_pcd_kalman_filter_gp(
    point_clouds,
    timestamps,
    M=10,
    lengthscale=None,
    noise_var=1e-3,
    target_num_points=50000,
    voxel_resolution=64,
    boundary_inducing_points=0,
    anchor=0,
    n_sigma=2.0,
    min_accepted_frac=0.1,
    K_local=8,
    static_zone_distance=-1.0,
    cam_centers=None,
):
    """Single-anchor Kalman tracking + GP regression initialization.

    Args:
        point_clouds: list of BasicPointCloud objects (one per frame).
        timestamps: numpy array of normalized timestamps [0, 1].
        M: number of GP inducing points.
        lengthscale: initial GP lengthscale; default 1/M (one inducing-point spacing).
        noise_var: observation noise variance.
        target_num_points: target point count after densification.
        voxel_resolution: grid resolution for densification.
        boundary_inducing_points: extra inducing points per side outside [0, 1].
        anchor: index of the anchor frame; 0 is a single forward pass.
        n_sigma: per-dim sigma multiplier for the position gate, a 3-DOF
            Mahalanobis ellipsoid (threshold = n_sigma**2 * 3). Default 2.0.
        min_accepted_frac: drop tracks whose accepted-frame count falls below
            max(2, min_accepted_frac * T_sub).
        K_local: number of nearest accepted neighbors used for the in-loop
            local-motion fill of gated tracks (default 8).
        static_zone_distance: static-zone threshold (see _gate_static_zone).
            -1 disables; positive values are multipliers of the getNerfppNorm
            rig extent; 0 uses the default multiplier 20x.
        cam_centers: (V, 3) unique camera centers; required when the static
            zone is enabled (static_zone_distance >= 0).

    Returns:
        dict with keys: xyz, rgb, normals, pos_gp_alpha, pos_gp_inducing_t, pos_gp_lengthscale.
    """
    color_weight = 1.0
    nn_point_clouds = []
    ts = torch.from_numpy(timestamps).float().cuda()

    for pcd in point_clouds:
        nn_point_clouds.append(torch.cat([
            torch.from_numpy(pcd.points).float().cuda(),
            color_weight * torch.from_numpy(pcd.colors).float().cuda()
        ], dim=-1))

    positions, colors_per_frame, accepted = _kalman_filter_bidirectional(
        nn_point_clouds, ts, anchor=anchor, n_sigma=n_sigma, K_local=K_local,
    )
    T_sub = positions.shape[1]
    print(f"Kalman (anchor={anchor}, K_local={K_local}): "
          f"{positions.shape[0]} tracks, {T_sub} frames, "
          f"mean accepted/track = {accepted.float().mean(dim=1).mean().item():.2f}")

    min_accepted_count = max(2, int(min_accepted_frac * T_sub))
    accepted_count = accepted.sum(dim=1)
    keep = accepted_count >= min_accepted_count
    n_dropped = int((~keep).sum().item())
    if n_dropped > 0:
        print(f"  dropped {n_dropped} tracks below {min_accepted_count}-frame floor")
    positions = positions[keep]
    accepted = accepted[keep]
    colors_per_frame = colors_per_frame[keep]

    # Average track colors over accepted frames only
    acc_f = accepted[..., None].float()
    color_sum = (colors_per_frame * acc_f).sum(dim=1)
    color_count = acc_f.sum(dim=1).clamp(min=1.0)
    colors_avg = color_sum / color_count

    static_append = None
    if static_zone_distance >= 0:
        static_mask, static_append, centroid, d_star = _gate_static_zone(
            positions, accepted, point_clouds, cam_centers, static_zone_distance)
        positions = positions[~static_mask]
        colors_avg = colors_avg[~static_mask]

    N_tracks = positions.shape[0]
    track_times = ts.unsqueeze(0).expand(N_tracks, -1)
    track_lengths = torch.full((N_tracks,), T_sub, device='cuda')

    gp_init = gp_regression_from_tracks(
        positions, track_times, track_lengths,
        M=M, lengthscale=lengthscale, noise_var=noise_var,
        boundary_inducing_points=boundary_inducing_points,
    )
    gp_init['rgb'] = colors_avg / color_weight

    # Densify before appending the harvest, the statics must not consume the
    # room's target budget, and clamp the sampling field to the near zone so
    # its grid resolves the room instead of the background tail.
    field_xyz = None
    if static_zone_distance >= 0:
        near = (gp_init['xyz'] - centroid.view(1, 3)).norm(dim=-1) <= d_star
        field_xyz = gp_init['xyz'][near]
    gp_init = densify_point_cloud(gp_init, target_num_points=target_num_points,
                                  voxel_resolution=voxel_resolution,
                                  field_xyz=field_xyz)
    gp_init = _append_static_rows(gp_init, static_append)

    return _to_numpy_result(gp_init)


def set_initial_pcd_kalman_multi_anchor(
    point_clouds,
    timestamps,
    M=10,
    lengthscale=None,
    noise_var=1e-3,
    target_num_points=50000,
    voxel_resolution=64,
    boundary_inducing_points=0,
    n_anchors=None,
    static_zone_distance=-1.0,
    cam_centers=None,
    n_sigma=2.0,
    min_accepted_frac=0.1,
    K_local=8,
    crop_bbox=None,
    fg_target_points=0,
):
    """Union of bidirectional Kalman tracks from multiple anchors + one GP fit.

    The anchors are the frames nearest each bin midpoint over `n_anchors`
    bins, and `n_anchors` itself defaults to M, so the anchor count follows
    the inducing-point count unless decoupled. Each anchor contributes one
    full-span track per point of its anchor cloud; the union is fit jointly,
    so static geometry appears once per anchor and densification is typically
    skipped.

    crop_bbox ([x_min, y_min, z_min, x_max, y_max, z_max]) restricts the
    densification to the tracks inside the box. With fg_target_points > 0
    the box instead splits the budget: the inside densifies to
    fg_target_points and the outside to target_num_points.

    Same return format as set_initial_pcd_kalman_filter_gp.
    """
    color_weight = 1.0
    nn_point_clouds = [torch.cat([
        torch.from_numpy(pcd.points).float().cuda(),
        color_weight * torch.from_numpy(pcd.colors).float().cuda()
    ], dim=-1) for pcd in point_clouds]
    ts = torch.from_numpy(timestamps).float().cuda()
    T = len(nn_point_clouds)

    n = n_anchors if n_anchors else M
    anchors = sorted({round((i + 0.5) / n * (T - 1)) for i in range(n)})

    all_pos, all_col, all_acc = [], [], []
    for a in anchors:
        positions, colors_per_frame, accepted = _kalman_filter_bidirectional(
            nn_point_clouds, ts, anchor=a, n_sigma=n_sigma, K_local=K_local)
        min_accepted_count = max(2, int(min_accepted_frac * T))
        keep = accepted.sum(dim=1) >= min_accepted_count
        positions = positions[keep]
        accepted = accepted[keep]
        colors_per_frame = colors_per_frame[keep]
        acc_f = accepted[..., None].float()
        colors_avg = (colors_per_frame * acc_f).sum(dim=1) \
            / acc_f.sum(dim=1).clamp(min=1.0)
        all_pos.append(positions)
        all_col.append(colors_avg)
        all_acc.append(accepted)
        print(f"Multi-anchor Kalman: anchor {a}, "
              f"{positions.shape[0]} tracks kept")

    positions = torch.cat(all_pos)
    colors_avg = torch.cat(all_col)
    accepted = torch.cat(all_acc)
    print(f"Multi-anchor Kalman: {positions.shape[0]} tracks total "
          f"from {len(anchors)} anchors {anchors}")

    static_append = None
    if static_zone_distance >= 0:
        static_mask, static_append, centroid, d_star = _gate_static_zone(
            positions, accepted, point_clouds, cam_centers, static_zone_distance)
        positions = positions[~static_mask]
        colors_avg = colors_avg[~static_mask]

    N_tracks = positions.shape[0]

    gp_init = gp_regression_from_tracks(
        positions, ts.unsqueeze(0).expand(N_tracks, -1),
        torch.full((N_tracks,), T, device='cuda'),
        M=M, lengthscale=lengthscale, noise_var=noise_var,
        boundary_inducing_points=boundary_inducing_points,
    )
    gp_init['rgb'] = colors_avg / color_weight

    if crop_bbox is not None:
        lo = torch.as_tensor(crop_bbox[:3], device=gp_init['xyz'].device)
        hi = torch.as_tensor(crop_bbox[3:], device=gp_init['xyz'].device)
        inside = ((gp_init['xyz'] >= lo) & (gp_init['xyz'] <= hi)).all(dim=-1)
        n_before = gp_init['xyz'].shape[0]
        subset = lambda m: {k: (v[m] if isinstance(v, torch.Tensor)
                                and v.shape[:1] == (n_before,) else v)
                            for k, v in gp_init.items()}
        if fg_target_points > 0:
            # Split-budget densification: the sparse foreground tracks (fast
            # movers defeat COLMAP matching) get a dedicated budget instead of
            # competing with the whole scene for one target count.
            fg = densify_point_cloud(subset(inside),
                                     target_num_points=fg_target_points,
                                     voxel_resolution=voxel_resolution)
            bg = densify_point_cloud(subset(~inside),
                                     target_num_points=target_num_points,
                                     voxel_resolution=voxel_resolution)
            n_fg = fg['xyz'].shape[0]
            gp_init = {k: (torch.cat([v, bg[k]]) if isinstance(v, torch.Tensor)
                           and v.shape[:1] == (n_fg,) else v)
                       for k, v in fg.items()}
            print(f"Split densify: fg {int(inside.sum())} -> {n_fg}, "
                  f"bg {int((~inside).sum())} -> {bg['xyz'].shape[0]}")
            gp_init = _append_static_rows(gp_init, static_append)
            return _to_numpy_result(gp_init)
        # Foreground crop: the static background is supplied separately (the
        # dense background.ply), so points outside the action bbox only spend
        # densification budget on geometry the background already covers.
        gp_init = subset(inside)
        print(f"Foreground crop: {int(inside.sum())} of {n_before} tracks "
              f"inside bbox")

    # Same ordering as set_initial_pcd_kalman_filter_gp, densify on the fit
    # tracks with a near-field sampling grid, then append the harvest.
    field_xyz = None
    if static_zone_distance >= 0:
        near = (gp_init['xyz'] - centroid.view(1, 3)).norm(dim=-1) <= d_star
        field_xyz = gp_init['xyz'][near]
    gp_init = densify_point_cloud(gp_init, target_num_points=target_num_points,
                                  voxel_resolution=voxel_resolution,
                                  field_xyz=field_xyz)
    gp_init = _append_static_rows(gp_init, static_append)

    return _to_numpy_result(gp_init)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_per_frame_point_clouds(dataset_path):
    """Load the per-frame point clouds `sfm_init/frame_<t>/colmap_points3D.ply`.

    A frame without a point cloud is skipped, along with its timestamp.

    Returns:
        point_clouds: list of BasicPointCloud
        timestamps: numpy array of normalized timestamps [0, 1]
    """
    with open(os.path.join(dataset_path, "transforms_train.json")) as f:
        frames = json.load(f)["frames"]
    num_frames = max(frame['t'] for frame in frames)
    normalized_timestamps = np.linspace(0, 1, num_frames, dtype=np.float32)

    point_clouds = []
    frame_timestamps = []

    for curr_frame in tqdm(range(1, num_frames + 1), desc="Loading per-frame point clouds"):
        ply_path = os.path.join(dataset_path, 'sfm_init', f'frame_{curr_frame}',
                                'colmap_points3D.ply')
        if os.path.isfile(ply_path):
            vertex = PlyData.read(ply_path)['vertex']
            xyz = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)
            rgb = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T.astype(np.float32) / 255.0
            normals = np.zeros_like(xyz)
            point_clouds.append(BasicPointCloud(points=xyz, colors=rgb, normals=normals))
            frame_timestamps.append(normalized_timestamps[curr_frame - 1])

    timestamps = np.array(frame_timestamps, dtype=np.float32)
    print(f"Loaded {len(point_clouds)} per-frame point clouds")
    return point_clouds, timestamps


def append_static_background(gp_init, ply_path):
    """Append background points (from a PLY) onto a Kalman gp_init dict.

    Background points carry no motion track, so they get a zero-motion GP init
    (pos_gp_alpha zero); inducing times and lengthscale are tiled from a
    foreground row to give valid shapes (both are inert at zero alpha). The
    points are then optimized exactly like every other Gaussian. All six
    row-aligned arrays grow by the background-point count so the dict stays
    consistent for create_from_pcd_4d.
    """
    v = PlyData.read(ply_path)["vertex"]
    xyz_bg = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    rgb_bg = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.float32) / 255.0
    n_bg = xyz_bg.shape[0]
    M_total = gp_init["pos_gp_alpha"].shape[1]

    it = gp_init["pos_gp_inducing_t"]
    ls = gp_init["pos_gp_lengthscale"]
    out = dict(gp_init)
    out["xyz"] = np.vstack([gp_init["xyz"], xyz_bg])
    out["rgb"] = np.vstack([gp_init["rgb"], rgb_bg])
    out["normals"] = np.vstack([gp_init["normals"], np.zeros_like(xyz_bg)])
    out["pos_gp_alpha"] = np.concatenate(
        [gp_init["pos_gp_alpha"], np.zeros((n_bg, M_total, 3), np.float32)], axis=0)
    out["pos_gp_inducing_t"] = np.concatenate([it, np.repeat(it[:1], n_bg, axis=0)], axis=0)
    out["pos_gp_lengthscale"] = np.concatenate([ls, np.repeat(ls[:1], n_bg, axis=0)], axis=0)
    print(f"Appended {n_bg} background points from {ply_path}")
    return out
