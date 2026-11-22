"""Train, extract, render, and score one scene, as the paper's tables do.

Each run trains in <output>/run_<k>/ with the scene config, renders the
held-out views for PSNR, SSIM, and LPIPS, and on Blender-Moments extracts the
ground-truth frames' meshes for the Chamfer, F-score, and normal metrics. Each
run's metrics go to run_<k>/result.json, and <output>/result.json gathers the
runs with their mean, standard deviation, and median-PSNR run. The paper used
nine runs per scene. To save disk, only the median run keeps its model,
checkpoints, and meshes once every run is scored; the others keep their
result.json, train.log, and configs, so rescoring them in another --mode
trains them again.

Finished work is reused. A run with a result.json is read, not repeated. A run
holding the trained model at the config's iteration count is scored without
training. A run stopped at another iteration resumes from the latest
chkpnt<iter>.pth in its folder when there is one (checkpoints exist only when
the run was trained with train.py's --checkpoint_iterations), and trains from
scratch otherwise. A run whose saved training_config.yaml differs from the
scene config stops the script, naming the differing keys.

--model scores a given model, such as a released point_cloud.ply, without
training, into <output>/model_result.json, apart from the runs.

--mode paper extracts the meshes with the settings of the paper's tables, which
its mesh metrics reproduce. The default release mode extracts them as
extract_mesh does by default. A run scored in the other mode is rescored, its
trained model reused.

Usage:
    python -m evaluation.evaluate_scene --config configs/n3d/sear_steak.yaml \
        --data_root data --output output/sear_steak --n_runs 9
"""

import json
import re
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from memo.arguments import PipelineParams
from memo.config import load_config
from memo.data.scenes import SCENES
from memo.gaussian_renderer import render
from memo.metrics.image_quality import score_views
from memo.metrics.mesh_quality import score_scene
from memo.scene.cameras_source import load_cameras, load_model
from memo.utils.general_utils import safe_state


IMAGE_METRICS = ("psnr", "ssim", "lpips")
PAPER_ANGLE = "20.0"


def scene_of(config: Path):
    """(dataset, scene) of a configs/<dataset>/<scene>.yaml path, one of the
    paper's scenes."""
    dataset, scene = config.parent.name, config.stem
    if scene not in SCENES.get(dataset, []):
        raise SystemExit(f"{config} is not a scene config, configs/<dataset>/<scene>.yaml with "
                         f"a dataset among {', '.join(SCENES)} and one of its scenes")
    return dataset, scene


def _num(v):
    """Values compared as numbers when they are numbers, so 5e-3 written as a
    string and as a float compare equal."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_num(x) for x in v]
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


# Sections that do not change a trained model: extraction settings only
# change its meshes, and video settings nothing evaluate_scene scores.
NOT_TRAINING = {"extraction", "video"}
# Written by train.py whether or not the config sets them.
RECORDED_DEFAULTS = {"num_inducing": 10, "num_inducing_rot": -1, "boundary_inducing_points": 2}


def config_differences(config: dict, saved: dict, keys=None) -> list:
    """Keys whose value differs between the scene config and a run's saved
    config, in either direction, over `keys` or every training key. A key
    absent from the scene config takes its recorded default."""
    if keys is None:
        keys = (set(config) | set(saved)) - NOT_TRAINING
    return sorted(k for k in keys
                  if _num(config.get(k, RECORDED_DEFAULTS.get(k))) != _num(saved.get(k)))


def checkpoint_to_resume(run: Path):
    """The run's latest chkpnt<iter>.pth, or None."""
    found = [(int(m.group(1)), p) for p in run.glob("chkpnt*.pth")
             if (m := re.fullmatch(r"chkpnt(\d+)\.pth", p.name))]
    return max(found)[1] if found else None


def train(config_path: Path, source: Path, run: Path) -> float:
    """Train one run and return its wall-clock seconds."""
    cmd = [sys.executable, "-m", "memo.train", "-s", str(source), "-m", str(run),
           "--config", str(config_path)]
    resume = checkpoint_to_resume(run)
    if resume is not None:
        cmd += ["--start_checkpoint", str(resume)]
        print(f"Training {run.name}, resuming from {resume.name}", flush=True)
    else:
        print(f"Training {run.name}", flush=True)
    start = time.time()
    # The training output goes both to the screen and to the run's train.log.
    with open(run / "train.log", "ab") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while chunk := process.stdout.read1(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    return time.time() - start


def image_metrics(gaussians, source: Path, config: dict) -> dict:
    parser = ArgumentParser()
    pipe = PipelineParams(parser).extract(parser.parse_args([]))
    background = torch.tensor([1, 1, 1] if config.get("white_background", False) else [0, 0, 0],
                              dtype=torch.float32, device="cuda")

    def pairs():
        for view in tqdm(load_cameras(source, "test"), desc="Image metrics"):
            view.to_device("cuda")
            view.load_original_image()
            with torch.no_grad():
                pred = render(view, gaussians, pipe, background, apply_4d=True)["render"]
            yield pred, view.original_image[0:3], int(view.t), int(view.view_num)
            view.original_image = torch.zeros(3, 1, 1)

    return score_views(pairs(), torch.device("cuda"))


def mesh_metrics(model: Path, config_path: Path, source: Path, out: Path, mode: str) -> dict:
    """Extract the ground-truth frames' meshes into `out` in extract_mesh's
    `mode` and score them."""
    gt = source / "gt_meshes"
    frames = sorted(int(p.stem.split("_")[1]) for p in gt.glob("frame_*.ply"))
    subprocess.run([sys.executable, "-m", "memo.extract_mesh", "-m", str(model),
                    "--config", str(config_path), "-s", str(source), "--skip_colors",
                    "--mode", mode, "--output", str(out), "--frames", *map(str, frames)],
                   check=True)
    return score_scene(out, gt, source)


def score(model: Path, config_path: Path, source: Path, dataset: str, mesh_dir: Path,
          iteration: int, mode: str) -> dict:
    config = load_config(config_path)
    gaussians = load_model(model, config_path, iteration)
    start = time.time()
    result = {"image_quality": image_metrics(gaussians, source, config)}
    result["render_seconds"] = time.time() - start
    del gaussians
    torch.cuda.empty_cache()
    if dataset == "blender_moments":
        result["mesh_quality"] = mesh_metrics(model, config_path, source, mesh_dir, mode)
    return result


def evaluate_run(run: Path, config_path: Path, config: dict, source: Path, dataset: str,
                 mode: str) -> dict:
    result_path = run / "result.json"
    saved_path = run / "training_config.yaml"
    if saved_path.exists():
        diff = config_differences(config, yaml.safe_load(saved_path.read_text()))
        if diff:
            raise SystemExit(f"{run}: training_config.yaml differs from {config_path} in "
                             f"{', '.join(diff)}; restore the config or use another --output")
    extraction = {**config.get("extraction", {}), "mode": mode}
    train_seconds = None
    if result_path.exists():
        result = json.loads(result_path.read_text())
        # A result scored with other extraction settings is rescored, the
        # trained model reused.
        if _num(result.get("extraction", {})) == _num(extraction):
            print(f"{run.name}: already scored", flush=True)
            return result
        print(f"{run.name}: extraction settings changed, rescoring", flush=True)
        train_seconds = result.get("train_seconds")
    run.mkdir(parents=True, exist_ok=True)
    iteration = int(config["iterations"])
    ply = run / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        train_seconds = train(config_path, source, run)
    result = score(run, config_path, source, dataset, run / "meshes", iteration, mode)
    result["train_seconds"] = train_seconds
    result["extraction"] = extraction
    result_path.write_text(json.dumps(result, indent=1))
    # Read back, so fresh and reused runs carry the same JSON key types.
    return json.loads(result_path.read_text())


def _metrics(result: dict) -> dict:
    """The paper's table metrics of one run: PSNR, SSIM, and LPIPS, and on
    Blender-Moments CD (Chamfer distance x 100), NC (normal Chamfer distance
    in degrees), and the joint F-scores at 20 degrees on surface samples
    (F_S) and on normals rendered at the test views (F_R)."""
    out = {m: result["image_quality"][m] for m in IMAGE_METRICS}
    mq = result.get("mesh_quality")
    if mq:
        out["cd"] = 100 * mq["chamfer"]
        out["nc"] = mq["normal"]["chamfer"]
        out["f_s"] = mq["normal"]["f_score"]["joint"][PAPER_ANGLE]
        out["f_r"] = mq["render_normal"]["f_score"][PAPER_ANGLE]
    return out


def summarize(results: list) -> dict:
    """Per-run metrics with their mean, standard deviation, and the median-PSNR
    run, as the paper reports them."""
    per_run = [_metrics(r) for r in results]
    keys = per_run[0].keys()
    psnrs = [m["psnr"] for m in per_run]
    median = int(np.argsort(psnrs)[(len(psnrs) - 1) // 2]) + 1
    return {"runs": [{"run": k + 1, **r} for k, r in enumerate(results)],
            "mean": {k: float(np.mean([m[k] for m in per_run])) for k in keys},
            "std": {k: float(np.std([m[k] for m in per_run], ddof=1)) if len(per_run) > 1 else 0.0
                    for k in keys},
            "median_run": median}


def free_run(run: Path) -> None:
    """Delete a scored run's model, checkpoints, and meshes, keeping its
    result.json, train.log, and configs."""
    for sub in ("point_cloud", "meshes"):
        shutil.rmtree(run / sub, ignore_errors=True)
    for p in run.glob("chkpnt*.pth"):
        p.unlink()


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True,
                        help="the scene config, configs/<dataset>/<scene>.yaml")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="the folder memo.data.prepare wrote")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--model", type=Path, default=None,
                        help="score this model, such as a released point_cloud.ply, and nothing else")
    parser.add_argument("--mode", choices=["release", "paper"], default="release",
                        help="extract_mesh's mode for the mesh metrics, paper for the settings "
                             "of the paper's tables")
    args = parser.parse_args()
    dataset, scene = scene_of(args.config)
    source = args.data_root / dataset / scene
    safe_state()

    if args.model is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        result = score(args.model, args.config, source, dataset, args.output / "model_meshes", -1,
                       args.mode)
        (args.output / "model_result.json").write_text(json.dumps(result, indent=1))
        print(f"Scored {args.model}: PSNR {result['image_quality']['psnr']:.2f}", flush=True)
        return

    config = load_config(args.config)
    results = [evaluate_run(args.output / f"run_{k}", args.config, config, source, dataset,
                            args.mode)
               for k in range(1, args.n_runs + 1)]
    summary = summarize(results)
    (args.output / "result.json").write_text(json.dumps(summary, indent=1))
    for k in range(1, args.n_runs + 1):
        if k != summary["median_run"]:
            free_run(args.output / f"run_{k}")
    print(f"{dataset}/{scene}: PSNR {summary['mean']['psnr']:.2f} over {args.n_runs} runs, "
          f"median run {summary['median_run']}", flush=True)


if __name__ == "__main__":
    main()
