"""Cameras and trained Gaussians for rendering and extraction.

A model is a training output directory or a bare `point_cloud.ply`, and a
source is a scene folder with `train_lmdb/` and `test_lmdb/`, or one with
`transforms_train.json` and `transforms_test.json` and no images.
"""
import json
import pickle
from argparse import Namespace
from pathlib import Path

import lmdb

from memo.config import load_config
from memo.data.build_lmdb import (
    _MiniCamInfo, _nerf_normalization, frame_camera, numbered_frames, timeline)
from memo.scene.cameras import Camera
from memo.scene.dataset_loaders import Views4D, cached_cameras, camera_from_metadata
from memo.scene.gaussian_model_4d import GaussianModel4D, inspect_4d_checkpoint
from memo.utils.system_utils import searchForMaxIteration


def load_cameras(source: Path, split: str) -> list[Camera]:
    """The cameras of `split` in LMDB key order, on the CPU, with 1x1 dummy
    images. LMDB cameras load their images with load_original_image."""
    lmdb_path = source / f"{split}_lmdb"
    if lmdb_path.exists():
        return cached_cameras(Views4D(str(lmdb_path)))
    transforms_path = source / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise FileNotFoundError(
            f"{source} holds neither {split}_lmdb nor transforms_{split}.json")
    contents = json.loads(transforms_path.read_text())
    frames = contents["frames"]
    _, normalized_timestamps = timeline(frames)
    # The image size the LMDB reads from each image. Some transforms files
    # store it as a float.
    width, height = int(contents["w"]), int(contents["h"])
    cameras = sorted((frame_camera(uid, frame, contents["camera_angle_x"], normalized_timestamps,
                                   width, height, training=split == "train")
                      for uid, frame in numbered_frames(frames)),
                     key=lambda key_camera: key_camera[0])
    return [camera_from_metadata(camera) for _, camera in cameras]


def cameras_extent(source: Path, train_cameras: list[Camera]) -> float:
    """The camera extent of the scene, as training's Scene gives it, from the
    train LMDB or else from the training cameras."""
    lmdb_path = source / "train_lmdb"
    if not lmdb_path.exists():
        return _nerf_normalization([_MiniCamInfo(R=c.R, T=c.T, fovx=c.FoVx, width=c.image_width)
                                    for c in train_cameras])["radius"]
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        radius = pickle.loads(txn.get(b"nerf_normalization"))["radius"]
    env.close()
    return radius


def model_dir(model: Path) -> Path:
    """The training output directory, or the folder that holds the PLY."""
    return model.parent if model.suffix == ".ply" else model


def _config_path(model: Path, config: Path | None) -> Path:
    if config is not None:
        return config
    if model.suffix == ".ply":
        raise ValueError(f"{model} is a PLY, so its scene config must be given")
    return model / "training_config.yaml"


def load_model(model: Path, config: Path | None, iteration: int) -> GaussianModel4D:
    """The Gaussians of a training output directory at `iteration` (-1 for
    the latest), or of a PLY. The inducing-point counts come from `config`,
    by default the output directory's training_config.yaml."""
    config = _config_path(model, config)
    cfg = load_config(config)
    if model.suffix == ".ply":
        ply_path = model
    else:
        if iteration == -1:
            iteration = searchForMaxIteration(str(model / "point_cloud"))
        ply_path = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    # A config key that is absent took train.py's default.
    gaussians = GaussianModel4D(
        sh_degree=cfg.get("sh_degree", 3),
        num_inducing=cfg.get("num_inducing", 10),
        num_inducing_rot=cfg.get("num_inducing_rot", -1),
        boundary_inducing_points=cfg.get("boundary_inducing_points", 2),
        use_unbounded_opacity=cfg.get("use_unbounded_opacity", False),
    )
    # load_ply reads as many position inducing points as the config gives,
    # so a mismatch would load a truncated motion model without an error. An
    # unreadable PLY is left to load_ply, whose error names the problem.
    _, m_total_in_ply = inspect_4d_checkpoint(str(ply_path))
    if m_total_in_ply is not None and m_total_in_ply != gaussians.M_total:
        boundary = gaussians.boundary_inducing_points
        raise ValueError(
            f"{ply_path} has {m_total_in_ply} position inducing points, num_inducing "
            f"{m_total_in_ply - 2 * boundary} with boundary_inducing_points {boundary}, but "
            f"{config} gives num_inducing {gaussians.M} ({gaussians.M_total} in all).")
    gaussians.load_ply(str(ply_path))
    print(f"[INFO] Loaded {gaussians.get_xyz.shape[0]} Gaussians from {ply_path}")
    return gaussians


def add_model_args(parser):
    """Register the model and source flags that resolve_model_args completes."""
    parser.add_argument("-m", "--model_path", type=Path, required=True,
                        help="Training output directory, or a model's point_cloud.ply.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Scene config, <model_path>/training_config.yaml by default. "
                             "Required with a .ply.")
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Saved iteration to load, -1 for the latest. Unused with a .ply.")
    parser.add_argument("-s", "--source_path", type=Path, default=None,
                        help="Scene folder with train_lmdb and test_lmdb, or with "
                             "transforms_train.json and transforms_test.json. By default the "
                             "folder the model was trained on, or the .ply's folder.")


def resolve_model_args(parser, args):
    """`args` with the defaults of --config and -s/--source_path filled in."""
    try:
        args.config = _config_path(args.model_path, args.config)
    except ValueError as e:
        parser.error(str(e))
    if args.source_path is None:
        if args.model_path.suffix == ".ply":
            args.source_path = args.model_path.parent
        else:
            saved = eval((args.model_path / "cfg_args").read_text(), {"Namespace": Namespace})
            args.source_path = Path(saved.source_path)
    return args
