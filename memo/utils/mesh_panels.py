"""Blender rendering of the videos' mesh panels. The renderer,
memo/video/blender_render.py, runs as a subprocess in the memo-blender env,
since its bpy build needs numpy below 2."""

import json
import os
import subprocess

import numpy as np
from PIL import Image


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER_ENV = "memo-blender"
# Ambient-occlusion distance as a fraction of the eye-to-subject distance.
AO_FRACTION = 0.01
# Frames per Blender subprocess. Bounds the per-frame PLYs on disk and the
# panels in RAM; each launch costs a few seconds of Blender startup.
BLENDER_CHUNK = 25
MATCAP_SAMPLES = 16


def blender_job(view, ply: str, out: str, clip_start: float, **extra) -> dict:
    """One blender_render job of a frame's mesh from `view`, at its size."""
    return {
        "ply": ply, "out": out,
        "R_c2w": np.asarray(view.R, dtype=np.float64).tolist(),
        "T_w2c": np.asarray(view.T, dtype=np.float64).tolist(),
        "FoVx": float(view.FoVx),
        "height": int(view.image_height), "width": int(view.image_width),
        "clip_start": float(clip_start),
        **extra,
    }


def run_blender(jobs, samples: int, work_dir: str) -> None:
    """Render the jobs with Blender in the memo-blender env.

    `--no-capture-output` streams the subprocess's output live. The inherited
    library path breaks bpy (Embree's rtcIsSYCLDeviceSupported), so the
    subprocess runs without it.
    """
    spec = os.path.join(work_dir, "jobs.json")
    with open(spec, "w") as f:
        json.dump({"samples": int(samples), "jobs": jobs}, f)
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    subprocess.run(["conda", "run", "-n", BLENDER_ENV, "--no-capture-output", "python",
                    "-m", "memo.video.blender_render", spec],
                   check=True, cwd=REPO_ROOT, env=env)


def load_on_background(png: str, background_rgb) -> np.ndarray:
    """An RGBA render composited over the video background, as uint8 RGB."""
    rgba = np.asarray(Image.open(png).convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + np.asarray(background_rgb, dtype=np.float32) * (1.0 - alpha)
    return (rgb.clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
