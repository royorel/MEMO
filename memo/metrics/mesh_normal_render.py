"""Rendered normal metrics on the test views, for scenes with ground-truth meshes.

Each test view of a ground-truth frame rasterizes the ground-truth mesh and the
predicted mesh with nvdiffrast, which pairs the two surfaces along every camera
ray. Only surface a test camera sees enters, weighted by its image area. At a
pixel both meshes cover, the error is the unsigned angle between their face
normals, unsigned because mesh orientation conventions differ between methods.
Missing pixels (ground-truth foreground the method does not cover) and spurious
pixels (method coverage of ground-truth background) score the worst unsigned
angle, 90 degrees, in the penalized Chamfer. A pixel is correct at angle tau
when both cover it, its angle is below tau, and the two depths along the ray
differ by less than depth_tau of the scene diagonal. Precision is over the
method's foreground, recall over the ground-truth foreground.
"""

import numpy as np
import torch
import trimesh

from memo.utils.graphics_utils import fov2focal


NEAR, FAR = 0.01, 1000.0
MISSING_PENALTY_DEG = 90.0


def _clip_transform(cam):
    """OpenGL clip transform (float32, CUDA) and center (float64) of a Camera.

    The transform is built in float64 from the camera's R and T rather than
    from its float32 world_view_transform, so it rounds to float32 only once.
    Camera stores the COLMAP world-to-camera (y down, z forward), and negating
    its y and z rows gives the OpenGL one (y up, looking down -z).
    """
    w2c = np.eye(4)
    w2c[:3, :3] = np.transpose(cam.R)
    w2c[:3, 3] = cam.T
    w2c[1:3] *= -1
    w, h = cam.image_width, cam.image_height
    fl_x, fl_y = fov2focal(cam.FoVx, w), fov2focal(cam.FoVy, h)
    # No principal point means the image center, as in the training projection.
    cx = cam.cx if cam.cx is not None else w / 2
    cy = cam.cy if cam.cy is not None else h / 2
    proj = np.array([
        [2 * fl_x / w, 0, 1 - 2 * cx / w, 0],
        [0, 2 * fl_y / h, 2 * cy / h - 1, 0],
        [0, 0, -(FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)],
        [0, 0, -1, 0],
    ])
    return (torch.tensor(proj @ w2c, dtype=torch.float32, device="cuda"),
            np.linalg.inv(w2c)[:3, 3])


def mesh_tensors(ply):
    """(vertices, int32 faces, unit face normals) on CUDA."""
    mesh = trimesh.load(str(ply), process=False, force="mesh")
    return (torch.as_tensor(mesh.vertices, dtype=torch.float32, device="cuda"),
            torch.as_tensor(mesh.faces, dtype=torch.int32, device="cuda"),
            torch.as_tensor(np.array(mesh.face_normals), dtype=torch.float32, device="cuda"))


def render_view(glctx, mesh, cam):
    """Per-pixel face normal, distance to the camera center, and coverage mask,
    with the first row at the top of the image."""
    import nvdiffrast.torch as dr

    verts, faces, face_normals = mesh
    full_proj, cam_center = _clip_transform(cam)
    clip = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=1) @ full_proj.T
    rast, _ = dr.rasterize(glctx, clip.unsqueeze(0).contiguous(), faces,
                           resolution=[int(cam.image_height), int(cam.image_width)])
    rast = rast[0].flip(0)
    face = rast[..., 3].long() - 1
    hit = face >= 0
    tri = verts[faces.long()[face[hit]]]
    u, v = rast[..., 0][hit, None], rast[..., 1][hit, None]
    points = u * tri[:, 0] + v * tri[:, 1] + (1 - u - v) * tri[:, 2]
    center = torch.tensor(cam_center, dtype=torch.float32, device="cuda")
    normals = torch.zeros(*face.shape, 3, device="cuda")
    depth = torch.zeros(face.shape, device="cuda")
    normals[hit] = face_normals[face[hit]]
    depth[hit] = (points - center).norm(dim=1)
    return normals, depth, hit


def compare_view(gt, pred, depth_tau_abs, angle_taus):
    """Per-pixel angle map and the counts the metrics aggregate, for one view."""
    gt_n, gt_d, gt_hit = gt
    pred_n, pred_d, pred_hit = pred
    both = gt_hit & pred_hit
    angle = torch.rad2deg(torch.arccos((gt_n * pred_n).sum(dim=-1).abs().clamp(max=1.0)))
    near = both & ((gt_d - pred_d).abs() < depth_tau_abs)
    counts = {"angle_sum": float(angle[both].sum()), "both": int(both.sum()),
              "gt_fg": int(gt_hit.sum()), "pred_fg": int(pred_hit.sum()),
              "missing": int((gt_hit & ~pred_hit).sum()), "spurious": int((pred_hit & ~gt_hit).sum()),
              "correct": {f"{float(t)}": int((near & (angle < t)).sum()) for t in angle_taus}}
    return angle, counts


def add_counts(total, counts):
    for k, v in counts.items():
        if isinstance(v, dict):
            sub = total.setdefault(k, {})
            for t, c in v.items():
                sub[t] = sub.get(t, 0) + c
        else:
            total[k] = total.get(k, 0) + v
    return total


def finalize(counts):
    """Metrics from summed counts."""
    union = counts["both"] + counts["missing"] + counts["spurious"]
    f_score = {}
    for t, c in counts["correct"].items():
        p = c / max(counts["pred_fg"], 1)
        r = c / max(counts["gt_fg"], 1)
        f_score[t] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {
        "angle_mean": counts["angle_sum"] / max(counts["both"], 1),
        "chamfer": (counts["angle_sum"] + MISSING_PENALTY_DEG * (counts["missing"] + counts["spurious"]))
                   / max(union, 1),
        "missing": counts["missing"] / max(counts["gt_fg"], 1),
        "spurious": counts["spurious"] / max(counts["gt_fg"], 1),
        "f_score": f_score,
    }


def frame_render_normal_quality(glctx, cameras, gt_ply, pred_ply, diagonal,
                                angle_taus=(10.0, 15.0, 20.0), depth_tau=0.01):
    """The frame's metrics, summed over its test cameras in the given order."""
    gt_mesh, pred_mesh = mesh_tensors(gt_ply), mesh_tensors(pred_ply)
    total = {}
    for cam in cameras:
        _angle, counts = compare_view(render_view(glctx, gt_mesh, cam),
                                      render_view(glctx, pred_mesh, cam),
                                      depth_tau * diagonal, angle_taus)
        add_counts(total, counts)
    return finalize(total)
