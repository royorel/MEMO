"""Download the processed data of the selected scenes and build the LMDBs
MEMO trains on.

Blender-Moments comes from its own HuggingFace dataset, and the processed N3D
and SelfCap scenes from the MEMO artifact repository. Each archive is checked
against its published checksum and unpacked into <data_root>/<dataset>/<scene>/,
where the scene's LMDBs are then built. A rerun skips archives whose files are
all still in place and scenes whose LMDBs are built, so an interrupted run
resumes where it stopped. The unpacked data stays on disk.

Usage:
    python -m memo.data.prepare --data_root data                       # everything
    python -m memo.data.prepare --data_root data --datasets n3d        # one dataset
    python -m memo.data.prepare --data_root data --scenes lego_4d      # one scene
    python -m memo.data.prepare --data_root data --dry-run             # list the archives
"""

import argparse
import os
import shutil
from pathlib import Path

from memo.data import hf_download
from memo.data.scenes import SCENES, WHITE_BACKGROUND, select


BLENDER_MOMENTS_REPO = "royorel/Blender-Moments"
ARTIFACT_REPO = "royorel/MEMO-artifacts"
BLENDER_MOMENTS_ARCHIVES = ["renders_train.zip", "renders_test.zip", "sfm_init.zip",
                            "gt_meshes.zip"]
ARTIFACT_ARCHIVES = ["frames.zip", "scene.zip"]
LMDB_DONE = ".lmdb_built"


def archives(dataset: str, scene: str) -> list:
    """[(repo id, repo path)] of a scene's archives."""
    if dataset == "blender_moments":
        return [(BLENDER_MOMENTS_REPO, f"{scene}/{a}") for a in BLENDER_MOMENTS_ARCHIVES]
    return [(ARTIFACT_REPO, f"{dataset}/{scene}/{a}") for a in ARTIFACT_ARCHIVES]


def member_rename(dataset: str, scene: str):
    """Maps an archive member to its path under <data_root>/<dataset>/<scene>/.
    Blender-Moments archives hold the dataset's own layout, the artifact
    archives a scene folder."""
    if dataset != "blender_moments":
        return lambda name: f"{dataset}/{scene}/{name}"
    renders, gt = f"processed/{scene}/", f"synthetic_gt/{scene}/"

    def rename(name):
        if name.startswith(renders):
            return f"{dataset}/{scene}/{name[len(renders):]}"
        if name.startswith(gt):
            return f"{dataset}/{scene}/gt_meshes/{name[len(gt):]}"
        raise ValueError(f"unexpected member {name}")
    return rename


def build_scene_lmdbs(scene_dir: Path, white_background: bool) -> None:
    """train_lmdb/ and test_lmdb/ from the scene's two transforms files."""
    from memo.data.build_lmdb import build_lmdb
    # The CPUs this process may run on, which a scheduler's allocation limits.
    workers = len(os.sched_getaffinity(0))
    for split in ("train", "test"):
        out = scene_dir / f"{split}_lmdb"
        if out.exists():
            shutil.rmtree(out)
        build_lmdb(str(scene_dir), f"transforms_{split}.json", str(out),
                   white_background, num_workers=workers, training=split == "train")
    (scene_dir / LMDB_DONE).touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None, metavar="DATASET",
                        help=f"datasets to prepare (default all): {', '.join(SCENES)}")
    parser.add_argument("--scenes", nargs="+", default=None, metavar="SCENE",
                        help="scenes to prepare (default all of the selected datasets)")
    parser.add_argument("--revision", default=None,
                        help="revision (branch, tag, or commit) of both repositories; "
                             "default latest")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the archives that would be fetched and exit")
    args = parser.parse_args()
    try:
        chosen = select(args.datasets, args.scenes)
    except ValueError as e:
        parser.error(str(e))

    if args.dry_run:
        print("\n".join(f"{repo}:{path}" for d, s in chosen for repo, path in archives(d, s)))
        return

    root = args.data_root
    with hf_download.work_dir_for(root) as tmp:
        work_dir = Path(tmp)
        sums = {}
        for dataset, scene in chosen:
            scene_dir = root / dataset / scene
            for repo, path in archives(dataset, scene):
                if repo not in sums:
                    sums[repo] = hf_download.load_checksums(repo, work_dir / repo, args.revision)
                digest = hf_download.published_digest(sums[repo], repo, path)
                if hf_download.already_unpacked(root, repo, path, digest):
                    print(f"{repo}:{path}: already downloaded", flush=True)
                    continue
                archive = hf_download.fetch_verified(repo, path, digest, work_dir / repo,
                                                     args.revision)
                sizes = hf_download.unpack(archive, root, member_rename(dataset, scene))
                archive.unlink()
                hf_download.write_record(root, repo, path, digest, sizes)
                (scene_dir / LMDB_DONE).unlink(missing_ok=True)
                print(f"{repo}:{path}: {len(sizes)} files", flush=True)
            built = all((scene_dir / f"{split}_lmdb" / "data.mdb").is_file()
                        for split in ("train", "test"))
            if (scene_dir / LMDB_DONE).exists() and built:
                print(f"{dataset}/{scene}: LMDBs already built", flush=True)
                continue
            build_scene_lmdbs(scene_dir, WHITE_BACKGROUND[dataset])
            print(f"{dataset}/{scene}: LMDBs built", flush=True)


if __name__ == "__main__":
    main()
