#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""Render the held-out test views of a trained MEMO model.

Each view's render goes to <output>/test/renders/frame_<t>_view_<v>.png, the
name of its LMDB record.

Usage:
    python -m memo.render_images -m <model_path>
    python -m memo.render_images -m <point_cloud.ply> --config <scene config>
"""
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

from memo.arguments import PipelineParams
from memo.config import load_config
from memo.densification.prune_camera_blocking_floaters import (
    apply_near_camera_prune, snapshot_full_state)
from memo.gaussian_renderer import render
from memo.scene.cameras_source import (
    add_model_args, cameras_extent, load_cameras, load_model, model_dir, resolve_model_args)
from memo.utils.general_utils import safe_state


def render_set(output, split, views, gaussians, pipeline, background, near_prune=None):
    render_path = output / split / "renders"
    render_path.mkdir(parents=True, exist_ok=True)

    for view in tqdm(views, desc="Rendering progress"):
        view.to_device("cuda")
        if near_prune is not None:
            rig_cameras, extent, prune_extent, full_state = near_prune
            apply_near_camera_prune(gaussians, rig_cameras, extent, prune_extent, full_state,
                                    view.normalized_t)
        rendering = render(view, gaussians, pipeline, background, apply_4d=True)["render"]
        name = f"frame_{view.t:04d}_view_{view.view_num + 1:04d}.png"
        torchvision.utils.save_image(rendering, render_path / name)


@torch.no_grad()
def render_sets(args, pipeline):
    config = load_config(args.config)
    gaussians = load_model(args.model_path, args.config, args.iteration)
    background = torch.tensor([1, 1, 1] if config.get("white_background", False) else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    output = args.output or model_dir(args.model_path) / "renders"

    splits = [split for split, skip in (("train", args.skip_train), ("test", args.skip_test))
              if not skip]
    # The near-camera prune reads the whole rig.
    cameras = {split: load_cameras(args.source_path, split)
               for split in (("train", "test") if args.near_camera_prune else splits)}
    near_prune = None
    if args.near_camera_prune:
        near_prune = (cameras["train"] + cameras["test"],
                      cameras_extent(args.source_path, cameras["train"]),
                      args.near_camera_prune_extent, snapshot_full_state(gaussians))

    for split in splits:
        render_set(output, split, cameras[split], gaussians, pipeline, background,
                   near_prune=near_prune)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render the held-out test views of a trained MEMO model")
    add_model_args(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output", default=None, type=Path,
                        help="Output directory, renders/ beside the model by default.")
    parser.add_argument("--skip_train", action=BooleanOptionalAction, default=True,
                        help="Skip the training views, on by default.")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip the held-out test views.")
    parser.add_argument("--near_camera_prune", action="store_true",
                        help="Remove the Gaussians that block a camera of the rig at each "
                             "view's time before rendering it.")
    parser.add_argument("--near_camera_prune_extent", default=0.15, type=float,
                        help="Prune radius as a fraction of the camera extent.")
    args = resolve_model_args(parser, parser.parse_args())
    print(f"Rendering {args.model_path}")
    safe_state()
    render_sets(args, pipeline.extract(args))
