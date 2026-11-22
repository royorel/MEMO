"""Camera paths of the videos.

`orbit` circles the world origin at a radius taken from the training rig.
`camera_arc` sweeps an arc through the training rig itself, for captures
that observe the scene from one side only. Both give one camera position
per scene frame, at that frame's normalized time.
"""
import math

import numpy as np
import torch

from memo.scene.cameras import Camera


def estimate_orbit_radius(cameras):
    """Mean L2 norm of training cameras' world-to-view translation vectors."""
    ts = np.stack([np.asarray(cam.T, dtype=np.float64) for cam in cameras])
    return float(np.linalg.norm(ts, axis=-1).mean())


def lookat_extrinsics(eye, target, world_up):
    """Build $(R_{c \\to w}, T_{w \\to c})$ for a camera at `eye` looking at
    `target` with image-up aligned to `world_up`.

    Right-handed OpenCV basis (right=$\\hat{x}$, down=$\\hat{y}$,
    forward=$\\hat{z}$). Rows of $R_{w \\to c}$ are the camera basis in
    world coordinates; `Camera` expects $R_{c \\to w} = R_{w \\to c}^T$ and
    $T_{w \\to c} = -R_{w \\to c} \\cdot \\text{eye}$ (see `getWorld2View2`).
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_w2c = np.stack([x, y, z], axis=0)
    T_w2c = -R_w2c @ eye
    return R_w2c.T.astype(np.float32), T_w2c.astype(np.float32)


def build_orbit_trajectory(world_up, radius, num_frames, arc, elevation_deg,
                           start_angle_deg, direction):
    """List of (eye, normalized_t) tuples for the orbit around the origin."""
    elevation = math.radians(elevation_deg)
    start_angle = math.radians(start_angle_deg)
    theta_max = 2 * math.pi if arc == "full" else math.pi
    s = +1.0 if direction == "ccw" else -1.0

    u = world_up / np.linalg.norm(world_up)
    seed = np.array([1.0, 0.0, 0.0]) if abs(u @ np.array([1.0, 0.0, 0.0])) < 0.9 \
        else np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    denom = max(num_frames - 1, 1)
    out = []
    for i in range(num_frames):
        theta = start_angle + s * i * theta_max / denom
        eye = radius * (math.cos(elevation) * (math.cos(theta) * e1 + math.sin(theta) * e2)
                        + math.sin(elevation) * u)
        out.append((eye, i / denom))
    return out


def build_camera_arc_trajectory(cameras, num_frames, coverage=0.9, mode="sweep"):
    """Orbit derived from the scene's own training rig, for forward-facing /
    partial-surround captures (n3d, selfcap) where a full or half circle
    would leave the observed region. Returns (trajectory, center, up):

      center : least-squares intersection of the cameras' forward rays
      up     : mean camera up (-R[:,1]), so the rig's own world-up is recovered
               regardless of the dataset's axis convention
      radius : median camera->center distance
      elevation : mean camera elevation above the up-plane (held fixed)
      azimuth   : swept across `coverage` of the rig's angular span, centered on
                  it (largest-gap method handles wrap-around)

    mode="sweep" goes start->end once; "oscillate" goes start->end->start for a
    seamless loop. The caller looks each eye at `center` with up `up`.
    """
    centers = np.stack([-np.asarray(c.R, np.float64) @ np.asarray(c.T, np.float64) for c in cameras])
    forwards = np.stack([np.asarray(c.R, np.float64)[:, 2] for c in cameras])
    ups = np.stack([-np.asarray(c.R, np.float64)[:, 1] for c in cameras])

    forwards = forwards / np.linalg.norm(forwards, axis=1, keepdims=True)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(centers, forwards):
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    center = np.linalg.solve(A, b)

    up = ups.mean(0); up /= np.linalg.norm(up)
    seed = np.array([1.0, 0.0, 0.0]) if abs(up @ np.array([1.0, 0.0, 0.0])) < 0.9 \
        else np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ up) * up; e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    rel = centers - center
    dist = np.linalg.norm(rel, axis=1)
    radius = float(np.median(dist))
    elevation = math.radians(float(np.degrees(np.arcsin((rel @ up) / dist)).mean()))
    proj = rel - np.outer(rel @ up, up)
    az = np.arctan2(proj @ e2, proj @ e1)
    a = np.sort(az % (2 * math.pi))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
    g = int(np.argmax(gaps))
    span = 2 * math.pi - gaps[g]
    az_center = (a[(g + 1) % len(a)] + span / 2) % (2 * math.pi)

    inset = coverage * span
    a0 = az_center - inset / 2
    denom = max(num_frames - 1, 1)
    out = []
    for i in range(num_frames):
        frac = i / denom
        sweep = inset * (1 - abs(2 * frac - 1)) if mode == "oscillate" else inset * frac
        theta = a0 + sweep
        eye = center + radius * (math.cos(elevation) * (math.cos(theta) * e1 + math.sin(theta) * e2)
                                 + math.sin(elevation) * up)
        out.append((eye, i / denom))
    return out, center, up


def make_synthetic_camera(R, T, FoVx, FoVy, height, width, normalized_t):
    """Camera with intrinsics borrowed from a training camera and a 1x1
    placeholder image, which the 4D render path never reads."""
    return Camera(colmap_id=0, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                  image=torch.zeros(3, 1, 1),
                  image_name="path", uid=0,
                  normalized_t=normalized_t, data_device="cpu",
                  image_width=width, image_height=height)
