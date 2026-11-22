"""Point-sampled mesh metrics, Chamfer distance, F-score, and normal scores.

All functions take and return CUDA tensors. The sampling follows kaolin's
sample_points draw for draw, so a seed gives kaolin's samples, and the
Chamfer and F-score definitions are kaolin's, on exact nearest neighbors.
"""

import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.ops import knn_points


def load_mesh_as_tensors(ply_path):
    """(verts (V, 3) float32, faces (F, 3) long) of a PLY, on CUDA."""
    mesh = trimesh.load(str(ply_path), force="mesh")
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.long, device="cuda")
    return verts, faces


def sample_surface(verts, faces, num_points=200000, return_normals=False):
    """Uniform area-weighted samples on the triangles, (num_points, 3), plus
    each sample's unit face normal when return_normals is set."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    x1, x2, x3 = (v0 - v1).unbind(-1)
    y1, y2, y3 = (v1 - v2).unbind(-1)
    areas = torch.sqrt((x2 * y3 - x3 * y2) ** 2 + (x3 * y1 - x1 * y3) ** 2
                       + (x1 * y2 - x2 * y1) ** 2) * 0.5
    # The random draws mirror kaolin's sample_points: a face per sample, then
    # u with density 2u and v uniform for the barycentric weights.
    face_idx = torch.distributions.Categorical(areas[None]).sample([num_points]).transpose(0, 1)[0]
    u = torch.sqrt(torch.rand((num_points, 1), device=verts.device, dtype=verts.dtype))
    v = torch.rand((num_points, 1), device=verts.device, dtype=verts.dtype)
    samples = (1 - u) * v0[face_idx] + u * (1 - v) * v1[face_idx] + u * v * v2[face_idx]
    if not return_normals:
        return samples
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-10)
    return samples, normals[face_idx]


def sided_distance(src, dst):
    """(squared distance, index) of each src point's nearest dst point."""
    nn = knn_points(src[None], dst[None], K=1)
    return nn.dists[0, :, 0], nn.idx[0, :, 0]


def chamfer_and_f_score(pred_samples, gt_samples, taus=(0.005, 0.01, 0.02, 0.05), eps=1e-8):
    """(Chamfer distance, {float(tau): F-score}) from one nearest-neighbor
    pass in each direction.

    The Chamfer distance is bidirectional L2 (not squared), the mean of the
    two directions' mean nearest-neighbor distances. In the F-score, true
    positives count predicted samples within tau of the ground truth, and
    recall's misses count ground-truth samples farther than tau, the formula
    the reported scores used."""
    pred_to_gt = sided_distance(pred_samples, gt_samples)[0].sqrt()
    gt_to_pred = sided_distance(gt_samples, pred_samples)[0].sqrt()
    chamfer = (0.5 * pred_to_gt.mean() + 0.5 * gt_to_pred.mean()).item()
    f_scores = {}
    for tau in taus:
        fn = (gt_to_pred > tau).sum().float()
        fp = (pred_to_gt > tau).sum().float()
        tp = pred_to_gt.shape[0] - fp
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        f_scores[float(tau)] = (2 * precision * recall / (precision + recall + eps)).item()
    return chamfer, f_scores


def _f_from_counts(precision, recall):
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def normal_scores(pred_samples, pred_normals, gt_samples, gt_normals,
                  angle_taus=(10.0, 15.0, 20.0), dist_tau=0.01):
    """Normal Chamfer and normal F-scores from nearest-neighbor matching.

    Each sample is matched to its nearest sample on the other surface and
    scored by the unsigned angle between their normals, unsigned because mesh
    orientation conventions differ between methods. Normal Chamfer is the mean
    of the two directions' mean angles, in degrees. A sample is correct at
    angle tau when its angle is below tau, and for the joint score also when
    its matched distance is below dist_tau. Precision is over the predicted
    samples, recall over the ground-truth samples.

    Returns {"normal_chamfer": float, "normal_f_score": {"joint", "normal_only"} -> {tau: f}}.
    """
    def matched(src, src_n, dst, dst_n):
        dist, idx = sided_distance(src, dst)
        cos = (src_n * dst_n[idx]).sum(dim=1).abs().clamp(max=1.0)
        return dist.sqrt(), torch.rad2deg(torch.arccos(cos))

    acc_dist, acc_angle = matched(pred_samples, pred_normals, gt_samples, gt_normals)
    comp_dist, comp_angle = matched(gt_samples, gt_normals, pred_samples, pred_normals)
    scores = {"joint": {}, "normal_only": {}}
    for tau in angle_taus:
        for kind, gate in (("normal_only", (None, None)),
                           ("joint", (acc_dist < dist_tau, comp_dist < dist_tau))):
            acc_ok, comp_ok = acc_angle < tau, comp_angle < tau
            if gate[0] is not None:
                acc_ok, comp_ok = acc_ok & gate[0], comp_ok & gate[1]
            scores[kind][f"{float(tau)}"] = _f_from_counts(acc_ok.float().mean().item(),
                                                           comp_ok.float().mean().item())
    return {"normal_chamfer": 0.5 * (acc_angle.mean().item() + comp_angle.mean().item()),
            "normal_f_score": scores}


def load_scene_normalization(path):
    """The scene diagonal of a scene_normalization.json, which divides every
    distance so thresholds are fractions of the scene size."""
    data = json.loads(Path(path).read_text())
    return float(data["diagonal"])
