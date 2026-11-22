"""Download the trained MEMO models of the paper's scenes.

Each model is a `point_cloud.ply` with its scene's `transforms_train.json` and
`transforms_test.json`, written to <out>/<dataset>/<scene>/ and checked against
the artifact repository's published checksums. The cameras in the transforms
files let rendering and mesh extraction run without the scene's images.

Usage:
    python -m scripts.download_models --out models                     # all 22 models
    python -m scripts.download_models --out models --scenes lego_4d    # one model
"""

import argparse
import shutil
from pathlib import Path

from memo.data import hf_download
from memo.data.scenes import SCENES, select


REPO = "royorel/MEMO-artifacts"
FILES = ["point_cloud.ply", "transforms_train.json", "transforms_test.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None, metavar="DATASET",
                        help=f"datasets (default all): {', '.join(SCENES)}")
    parser.add_argument("--scenes", nargs="+", default=None, metavar="SCENE",
                        help="scenes (default all of the selected datasets)")
    parser.add_argument("--revision", default=None,
                        help="repository revision (branch, tag, or commit); default latest")
    args = parser.parse_args()
    try:
        chosen = select(args.datasets, args.scenes)
    except ValueError as e:
        parser.error(str(e))

    with hf_download.work_dir_for(args.out) as tmp:
        work_dir = Path(tmp)
        sums = hf_download.load_checksums(REPO, work_dir, args.revision)
        for dataset, scene in chosen:
            for name in FILES:
                path = f"models/{dataset}/{scene}/{name}"
                dest = args.out / dataset / scene / name
                digest = hf_download.published_digest(sums, REPO, path)
                if dest.is_file() and hf_download.sha256_file(dest) == digest:
                    continue
                src = hf_download.fetch_verified(REPO, path, digest, work_dir, args.revision)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
            print(f"{dataset}/{scene} -> {args.out / dataset / scene}", flush=True)


if __name__ == "__main__":
    main()
