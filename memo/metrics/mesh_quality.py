"""Mesh metrics of per-frame meshes against ground-truth meshes.

Chamfer distance and F-score on surface samples, normalized by the scene
diagonal, plus two normal blocks. `normal` holds the sample-based normal
Chamfer and normal F-scores, joint and normal-only, and `render_normal` the
test-view metrics of `mesh_normal_render`.
"""

from pathlib import Path

import torch
from tqdm import tqdm

from memo.metrics.mesh_normal_render import frame_render_normal_quality
from memo.metrics.mesh_quality_metrics import (
    chamfer_and_f_score,
    load_mesh_as_tensors,
    load_scene_normalization,
    normal_scores,
    sample_surface,
)
from memo.scene.cameras_source import load_cameras


HEADLINE_TAU = 0.01
F_SCORE_TAUS = (0.005, 0.01, 0.02, 0.05)
NORMAL_ANGLE_TAUS = (10.0, 15.0, 20.0)


def _mean_tree(trees):
    """Leafwise mean of nested dicts of floats with identical structure."""
    first = trees[0]
    if isinstance(first, dict):
        return {k: _mean_tree([t[k] for t in trees]) for k in first}
    return sum(trees) / len(trees)


@torch.no_grad()
def score_scene(mesh_dir: Path, gt_dir: Path, cameras_source: Path, frames=None) -> dict:
    """Mesh metrics of the meshes in `mesh_dir` against the ground truth in `gt_dir`.

    Both folders hold `frame_<NNNN>.ply` with 1-based frame numbers, and
    `gt_dir` also holds `scene_normalization.json`. `cameras_source` is an LMDB
    scene folder or a folder with `transforms_test.json`, whose test cameras
    render the normal maps. `frames` defaults to every ground-truth frame that
    has a mesh.

    Returns {"chamfer", "f_score": {tau: f}, "f_score_headline_tau", "normal",
    "render_normal", "per_frame": {"<NNNN>": {...}}}, each top-level value the
    mean over frames.
    """
    mesh_dir, gt_dir = Path(mesh_dir), Path(gt_dir)
    if frames is None:
        frames = sorted(int(p.stem.split("_")[1]) for p in gt_dir.glob("frame_*.ply")
                        if (mesh_dir / p.name).exists())
    if not frames:
        raise FileNotFoundError(f"no meshes in {mesh_dir} match a ground-truth frame in {gt_dir}")
    diagonal = load_scene_normalization(gt_dir / "scene_normalization.json")

    # The distance pass samples every frame before the normal pass resamples
    # them, so a fixed seed gives the same samples, and numbers, as scoring
    # distances and normals in two separate runs over the frames.
    per_frame = {}
    for idx in tqdm(frames, desc="Mesh distances"):
        gt_verts, gt_faces = load_mesh_as_tensors(gt_dir / f"frame_{idx:04d}.ply")
        pred_verts, pred_faces = load_mesh_as_tensors(mesh_dir / f"frame_{idx:04d}.ply")
        gt_samples = sample_surface(gt_verts, gt_faces) / diagonal
        pred_samples = sample_surface(pred_verts, pred_faces) / diagonal
        chamfer, f_scores = chamfer_and_f_score(pred_samples, gt_samples, taus=F_SCORE_TAUS)
        per_frame[f"{idx:04d}"] = {
            "chamfer": chamfer,
            "f_score": {f"{tau}": float(v) for tau, v in f_scores.items()},
        }

    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()
    cameras = load_cameras(cameras_source, "test")

    for idx in tqdm(frames, desc="Mesh normals"):
        gt_ply = gt_dir / f"frame_{idx:04d}.ply"
        pred_ply = mesh_dir / f"frame_{idx:04d}.ply"
        gt_samples, gt_normals = sample_surface(*load_mesh_as_tensors(gt_ply), return_normals=True)
        pred_samples, pred_normals = sample_surface(*load_mesh_as_tensors(pred_ply),
                                                    return_normals=True)
        scores = normal_scores(pred_samples / diagonal, pred_normals, gt_samples / diagonal,
                               gt_normals, angle_taus=NORMAL_ANGLE_TAUS, dist_tau=HEADLINE_TAU)
        frame_cameras = sorted((c for c in cameras if int(c.t) == idx), key=lambda c: c.view_num)
        per_frame[f"{idx:04d}"].update({
            "normal": {"chamfer": scores["normal_chamfer"], "f_score": scores["normal_f_score"]},
            "render_normal": frame_render_normal_quality(glctx, frame_cameras, gt_ply, pred_ply,
                                                         diagonal, angle_taus=NORMAL_ANGLE_TAUS,
                                                         depth_tau=HEADLINE_TAU),
        })

    entries = list(per_frame.values())
    return {
        "chamfer": sum(v["chamfer"] for v in entries) / len(entries),
        "f_score": {
            f"{tau}": sum(v["f_score"][f"{tau}"] for v in entries) / len(entries)
            for tau in F_SCORE_TAUS
        },
        "f_score_headline_tau": HEADLINE_TAU,
        "normal": _mean_tree([v["normal"] for v in entries]),
        "render_normal": _mean_tree([v["render_normal"] for v in entries]),
        "per_frame": per_frame,
    }
