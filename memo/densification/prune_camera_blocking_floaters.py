import math

import torch


def camera_blocking_mask(gaussians, rig_cameras, cameras_extent, extent_fraction, sigma=3.0,
                         positions=None, extra_cameras=None):
    """Boolean mask of camera-blocking floaters (True = prune candidate).

    See prune_camera_blocking_floaters for the rule. `rig_cameras` are the
    train and test cameras, and `cameras_extent` the scene's camera extent.
    Returns (mask, radius, n_cameras); computes only, so render-time callers
    can apply the mask without an optimizer. `positions` overrides the base
    centers (e.g. GP-deformed positions at a specific frame time).
    `extra_cameras` extends the rig set, e.g. with a video's orbit camera,
    whose path floaters can block without being near any rig camera.
    """
    views = list(rig_cameras)
    if extra_cameras:
        views = views + list(extra_cameras)
    seen = set()
    cameras = []
    for v in views:
        key = tuple(round(float(x), 6) for x in v.camera_center.detach().cpu())
        if key not in seen:
            seen.add(key)
            cameras.append(v)
    radius = extent_fraction * cameras_extent

    xyz = (positions if positions is not None else gaussians.get_xyz).detach()
    s_max = gaussians.get_scaling.detach().max(dim=1).values
    ones = torch.ones(xyz.shape[0], 1, device=xyz.device)
    p_hom = torch.cat([xyz, ones], dim=1)
    mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    for v in cameras:
        center = v.camera_center.detach().to(xyz.device)
        near = torch.linalg.norm(xyz - center, dim=1) < radius
        if not near.any():
            continue
        p_view = p_hom[near] @ v.world_view_transform.detach().to(xyz.device)
        z = p_view[:, 2]
        slack = sigma * s_max[near]
        reaches = (
            (z > 0.0)
            & (p_view[:, 0].abs() <= z * math.tan(v.FoVx * 0.5) + slack)
            & (p_view[:, 1].abs() <= z * math.tan(v.FoVy * 0.5) + slack)
        )
        mask[near.nonzero(as_tuple=True)[0][reaches]] = True
    return mask, radius, len(cameras)


def snapshot_full_state(gaussians):
    """All N-length tensors of the model, for apply_near_camera_prune."""
    n = gaussians._xyz.shape[0]
    return {name: val for name, val in vars(gaussians).items()
            if torch.is_tensor(val) and val.shape[:1] == (n,)}


@torch.no_grad()
def apply_near_camera_prune(gaussians, rig_cameras, cameras_extent, extent_fraction,
                            full_state, t, sigma=3.0, extra_cameras=None,
                            clear_view=None, clear_depth=0.0):
    """Render-time twin of the training-side prune, applied per frame, with
    the rig of camera_blocking_mask.

    Restores the full tensors from `full_state` first, so consecutive frames
    each mask the complete model rather than compounding slices, then masks at
    this frame's GP-deformed positions and slices by direct tensor assignment
    (there is no optimizer at render time). Opacity zeroing is not an option
    because the vacancy field has no min-opacity gate, so masked Gaussians
    must leave the model before mesh extraction. With clear_view and a
    positive clear_depth, everything inside that camera's frustum closer than
    clear_depth is also pruned: on a camera-arc orbit the rig hardware stands
    on the path itself, in the orbit's cone but outside every proximity ball.
    Returns (n_pruned, radius, n_cameras).
    """
    for name, val in full_state.items():
        setattr(gaussians, name, val)
    xyz_t = gaussians.get_xyz.detach() + gaussians.position_gp(t).detach()
    mask, radius, n_cameras = camera_blocking_mask(
        gaussians, rig_cameras, cameras_extent, extent_fraction, sigma=sigma, positions=xyz_t,
        extra_cameras=extra_cameras)
    if clear_view is not None and clear_depth > 0:
        ones = torch.ones(xyz_t.shape[0], 1, device=xyz_t.device)
        p_view = torch.cat([xyz_t, ones], dim=1) @ clear_view.world_view_transform.detach()
        z = p_view[:, 2]
        # Same sigma slack as the rig-camera rule: near-frustum centers just
        # outside the cone still smear their splat tails into the image.
        slack = sigma * gaussians.get_scaling.detach().max(dim=1).values
        in_cone = (
            (z > 0.0)
            & (p_view[:, 0].abs() <= z * math.tan(clear_view.FoVx * 0.5) + slack)
            & (p_view[:, 1].abs() <= z * math.tan(clear_view.FoVy * 0.5) + slack)
        )
        mask = torch.logical_or(mask, in_cone & (z < clear_depth))
    keep = ~mask
    for name, val in full_state.items():
        sliced = val[keep].contiguous()
        if isinstance(val, torch.nn.Parameter):
            sliced = torch.nn.Parameter(sliced, requires_grad=False)
        setattr(gaussians, name, sliced)
    return int(mask.sum()), radius, n_cameras


def prune_camera_blocking_floaters(gaussians, scene, extent_fraction, sigma=3.0):
    """Delete Gaussians parked in the empty shell around the camera rig.

    Camouflaged floater clusters settle in front of a rig camera, in the
    space between the camera and the scene content. They match the
    background color from the training views (so training loss never
    penalizes them) but veil any camera that looks through them, and the
    vacancy field meshes them. A Gaussian is pruned when it lies within
    extent_fraction * cameras_extent of some camera center, in front of
    that camera (view-space z > 0), AND within 3 sigma of that camera's
    view cone (|x| <= z tan(FoVx/2) + 3 s_max, same for y). The 3 sigma
    slack matters because the floaters' centers sit far off-axis and veil
    the image through their projected splat tails; a strict center-in-frustum gate keeps exactly the
    Gaussians this prune targets. Content behind a camera, or whose
    splat cannot reach the image, survives.

    Runs once, post-training, before the final save. Base positions are
    used; the floaters are time-fixed (GP displacements do not move them
    across the sequence).
    """
    rig_cameras = list(scene.getTrainCameras().copy()) + list(scene.getTestCameras().copy())
    mask, radius, n_cameras = camera_blocking_mask(
        gaussians, rig_cameras, scene.cameras_extent, extent_fraction, sigma=sigma)
    n_pruned = int(mask.sum())
    if n_pruned > 0:
        gaussians.prune_points(mask)
        # prune_points leaves filter_3D untouched (densification flows always
        # recompute it afterwards); slice it here so the final save stays
        # consistent without a full mip-filter recompute.
        if gaussians.use_mip_filter:
            gaussians.filter_3D = gaussians.filter_3D[~mask]
    print(f"[prune_camera_blocking_floaters] removed {n_pruned} Gaussians within "
          f"{radius:.3f} ({extent_fraction} x extent), in front and within "
          f"{sigma:g} sigma of the view cone, over {n_cameras} cameras")
    return n_pruned
