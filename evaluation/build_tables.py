"""Build per-dataset Markdown tables from evaluate_scene results.

Reads <results>/<dataset>/<scene>/result.json, as written by
evaluation.evaluate_scene with --output <results>/<dataset>/<scene>, and writes
one table per dataset with each scene's mean over its runs, the standard
deviation when there are several runs, and the dataset average. --models
tabulates the released models' scores instead, <results>/<dataset>/<scene>/
model_result.json from evaluate_scene --model.

Usage:
    python -m evaluation.build_tables --results output --out tables.md
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from memo.data.scenes import SCENES


# The paper's columns. CD is the Chamfer distance x 100, NC the normal Chamfer
# distance in degrees, and F_S and F_R the joint F-scores at 20 degrees on
# surface samples and on normals rendered at the test views.
COLUMNS = [("psnr", "PSNR ↑", 2), ("ssim", "SSIM ↑", 3), ("lpips", "LPIPS ↓", 3),
           ("cd", "CD ↓", 2), ("nc", "NC ↓", 1), ("f_s", "F<sub>S</sub> ↑", 3),
           ("f_r", "F<sub>R</sub> ↑", 3)]
DATASET_NAMES = {"blender_moments": "Blender-Moments", "n3d": "Neural 3D Video",
                 "selfcap": "SelfCap"}


def cell(summary: dict, key: str, digits: int) -> str:
    mean = summary["mean"][key]
    if len(summary["runs"]) > 1:
        return f"{mean:.{digits}f} ± {summary['std'][key]:.{digits}f}"
    return f"{mean:.{digits}f}"


def dataset_table(dataset: str, summaries: dict) -> list:
    columns = [c for c in COLUMNS if all(c[0] in s["mean"] for s in summaries.values())]
    lines = [f"## {DATASET_NAMES[dataset]}", "",
             "| Scene | " + " | ".join(c[1] for c in columns) + " |",
             "| --- |" + " --- |" * len(columns)]
    for scene, s in summaries.items():
        lines.append(f"| {scene} | " + " | ".join(cell(s, k, d) for k, _, d in columns) + " |")
    avg = {k: sum(s["mean"][k] for s in summaries.values()) / len(summaries) for k, _, _ in columns}
    lines.append("| **Average** | " + " | ".join(f"{avg[k]:.{d}f}" for k, _, d in columns) + " |")
    return lines + [""]


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", action="store_true",
                        help="tabulate the released-model scores, model_result.json, written by "
                             "evaluate_scene --model, instead of the runs' result.json")
    args = parser.parse_args()
    name = "model_result.json" if args.models else "result.json"
    if args.models:
        # evaluate_scene imports torch, which the runs' tables do not need.
        from evaluation.evaluate_scene import summarize

    lines = []
    for dataset, scenes in SCENES.items():
        paths = {s: args.results / dataset / s / name for s in scenes}
        summaries = {s: json.loads(p.read_text()) for s, p in paths.items() if p.exists()}
        if args.models:
            summaries = {s: summarize([r]) for s, r in summaries.items()}
        if summaries:
            lines += dataset_table(dataset, summaries)
    if not lines:
        raise SystemExit(f"no {name} under {args.results}/<dataset>/<scene>/")
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
