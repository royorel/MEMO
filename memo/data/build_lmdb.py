"""LMDB writer for 4D datasets.

Two entry points:
  - `build_lmdb(source_path, transforms_file, output_path, ...)` reads frames
    from a `transforms.json` on disk and writes the LMDB.
  - `build_lmdb_from_frames(source_path, fovx, frames, output_path, ...)`
    accepts an in-memory frames list (each entry may override the rig-level
    `camera_angle_x` via a per-entry `camera_angle_x` field) and writes the
    LMDB. PNG files referenced by `frame["file_path"]` must exist on disk
    under `source_path`.

LMDB key format:  "frame_{t:04d}_view_{v:04d}"
LMDB value:       pickle of a compact dict (see _process_frame)

Metadata keys stored in each LMDB:
    "num_frames"        -- str(int)
    "num_views"         -- str(int)
    "nerf_normalization" -- pickle of dict {"translate": ..., "radius": ..., "d_min": ...}
"""

import os
import json
import math
import lmdb
import pickle
import argparse
import multiprocessing
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import NamedTuple

from memo.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


COMMIT_EVERY = 256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MiniCamInfo(NamedTuple):
    R: np.ndarray
    T: np.ndarray
    fovx: float
    width: int


def _nerf_normalization(cam_infos):
    centers = []
    for c in cam_infos:
        W2C = getWorld2View2(c.R, c.T)
        C2W = np.linalg.inv(W2C)
        centers.append(C2W[:3, 3:4])
    centers = np.hstack(centers)
    center = np.mean(centers, axis=1, keepdims=True)
    diagonal = np.max(np.linalg.norm(centers - center, axis=0))

    d_min_vals = [abs(float(c.T[2])) / (0.5 * c.width / math.tan(0.5 * c.fovx))
                  for c in cam_infos]
    d_min = max(min(d_min_vals), 1e-6)

    return {"translate": -center.flatten(), "radius": diagonal * 1.1, "d_min": d_min}


def _extract_view_number(frame):
    return int(frame["file_path"].split('_')[-1].split('.')[0])


def timeline(frames):
    """The frame count of `frames` and the normalized time of each 1-indexed
    frame number `t`."""
    num_frames = max(frame.get('t', 1) for frame in frames)
    return num_frames, np.linspace(0, 1, num_frames, dtype=np.float32)


def numbered_frames(frames):
    """(uid, frame) pairs, the uids numbering the frames in file-path order."""
    return list(enumerate(sorted(frames, key=lambda f: f["file_path"])))


def frame_camera(uid, frame, fovx_default, normalized_timestamps, im_w, im_h, training):
    """(LMDB key, camera) of a frame whose image is im_w x im_h pixels, the
    camera being the frame's record without its image.

    Per-frame `camera_angle_x` on the frame entry overrides `fovx_default`.
    A `training` view adds the frame's `sync_residual_frames` to its time;
    test views do not.
    """
    fovx = float(frame.get("camera_angle_x", fovx_default))
    fovy_override = frame.get("camera_angle_y")
    # Default t=1 for static (non-4D) scenes that have no 't' field.
    t = frame.get('t', 1)
    normalized_t = float(normalized_timestamps[t - 1])
    # Sub-frame capture-time residual in source frames (rolling multi-camera
    # rigs whose sync offsets are quantized to whole frames, e.g. selfcap).
    # The temporal model is continuous, so each training view supervises at
    # its true instant; values may push normalized_t slightly outside [0, 1].
    # Test views are scored at whole frames, as the paper's tables were.
    residual = float(frame.get('sync_residual_frames', 0.0)) if training else 0.0
    if residual and len(normalized_timestamps) > 1:
        spacing = float(normalized_timestamps[1]) - float(normalized_timestamps[0])
        normalized_t += residual * spacing
    view_num = _extract_view_number(frame)

    transform_matrix = np.array(frame["transform_matrix"])
    # OpenGL/Blender (Y up, Z back) -> COLMAP (Y down, Z forward)
    transform_matrix[:3, 1:3] *= -1

    w2c = np.linalg.inv(transform_matrix)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]

    # Prefer a real per-camera vertical FOV (camera_angle_y, derived from the
    # sensor's fy) when the frame supplies one -- real rigs have fx != fy. Only
    # fall back to deriving FoVy from FoVx + aspect (which assumes fx == fy);
    # that fallback is exact for single-focal LLFF (n3d) and square renders
    # (blender_moments). Note the WIDTH/HEIGHT order: the focal comes from the
    # WIDTH, the vertical fov from the HEIGHT.
    if fovy_override is not None:
        fovy = float(fovy_override)
    else:
        fovy = focal2fov(fov2focal(fovx, im_w), im_h)

    camera = {
        'uid':          uid,
        'colmap_id':    uid,
        'R':            R,
        'T':            T,
        'FoVx':         fovx,
        'FoVy':         float(fovy),
        'image_name':   Path(frame["file_path"]).stem,
        'image_width':  im_w,
        'image_height': im_h,
        't':            t,
        'normalized_t': normalized_t,
        'view_num':     view_num - 1,  # 0-indexed
    }
    # Per-camera principal point (real rigs, e.g. selfcap). Absent -> loaders default
    # to the image center.
    if "cx_cam" in frame:
        camera['cx'] = float(frame['cx_cam'])
        camera['cy'] = float(frame['cy_cam'])
    return f"frame_{t:04d}_view_{view_num:04d}", camera


def _process_frame(indexed_frame, path, fovx_default, normalized_timestamps, white_background,
                   training):
    """
    Worker function (runs in subprocess).

    Returns (lmdb_key: str, value: bytes, mini_info: _MiniCamInfo)
    storing images as uint8 HWC numpy arrays to keep the LMDB compact.
    """
    uid, frame = indexed_frame
    image_path = os.path.join(path, frame["file_path"])
    image_name = Path(image_path).stem

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        # Blender NeRF renames the files on disk to NNNN.ext while transforms
        # still reference r_NNNNN.ext. Parse the actual frame number out of
        # the transforms file_path so this works even when the frames array
        # has been subset (e.g. llff-style train/test split).
        base = os.path.basename(frame["file_path"])
        _, ext = os.path.splitext(base)
        num = int(base.split('_')[-1].split('.')[0])
        cam_name = os.path.join(os.path.dirname(frame["file_path"]),
                                f"{num:04d}{ext}")
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

    im_w, im_h = image.size
    im_data = np.array(image.convert("RGBA"))
    bg = np.array([1, 1, 1], dtype=np.float32) if white_background else np.array([0, 0, 0], dtype=np.float32)
    norm = im_data.astype(np.float32) / 255.0
    mask = norm[..., 3:4]
    rgb = norm[..., :3] * mask + bg * (1 - mask)
    # Store as uint8 HWC -- 4x smaller than float32
    image_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    key, camera = frame_camera(uid, frame, fovx_default, normalized_timestamps, im_w, im_h,
                               training)
    # The key order of the record fixes its pickled bytes, so the image goes
    # in right after FoVy.
    record = {k: camera[k] for k in ('uid', 'colmap_id', 'R', 'T', 'FoVx', 'FoVy')}
    record['image'] = image_uint8   # uint8 HWC numpy array
    record['image_name'] = image_name
    record.update((k, v) for k, v in camera.items() if k not in record)
    value = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
    # Metadata for camera cache (no image data)
    meta = {k: record[k] for k in (
        'colmap_id', 'R', 'T', 'FoVx', 'FoVy', 'image_name',
        'uid', 'normalized_t', 't', 'view_num', 'image_width', 'image_height',
    )}
    # Per-camera principal point must ride the cache too: loaders take a cache
    # shortcut and never re-read the record, so omitting it here silently drops
    # the off-center pp at train time (centered render vs off-center-BA poses).
    if 'cx' in record:
        meta['cx'], meta['cy'] = record['cx'], record['cy']
    return key, value, _MiniCamInfo(R=record['R'], T=record['T'], fovx=record['FoVx'], width=im_w), meta


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_lmdb(source_path, transforms_file, output_path, white_background, num_workers,
               training):
    """Read frames from a `transforms.json` and write the LMDB."""
    transforms_path = os.path.join(source_path, transforms_file)
    with open(transforms_path) as f:
        contents = json.load(f)
    return build_lmdb_from_frames(
        source_path=source_path,
        fovx=contents["camera_angle_x"],
        frames=contents["frames"],
        output_path=output_path,
        white_background=white_background,
        num_workers=num_workers,
        training=training,
    )


def build_lmdb_from_frames(source_path, fovx, frames, output_path, white_background, num_workers,
                           training):
    """Write the LMDB from an in-memory frames list.

    Each frame entry is a dict with `file_path` (relative to `source_path`),
    `transform_matrix` (4x4 list), `t` (1-indexed integer, default 1), and an
    optional `camera_angle_x` that overrides the rig-level `fovx`.
    `training` is true for a training split, see frame_camera.
    """
    num_frames, normalized_timestamps = timeline(frames)
    num_views  = max(_extract_view_number(frame) for frame in frames)
    # Training cameras index the culling masks by their 0-based number, so a
    # camera numbered 0 would silently write into the last camera's slot.
    if training and min(_extract_view_number(frame) for frame in frames) < 1:
        raise ValueError(f"{output_path}: training camera numbers in file names "
                         f"(..._view_<NNNN>.png) must start at 1")

    indexed_frames = numbered_frames(frames)

    worker_fn = partial(
        _process_frame,
        path=source_path,
        fovx_default=fovx,
        normalized_timestamps=normalized_timestamps,
        white_background=white_background,
        training=training,
    )

    os.makedirs(output_path, exist_ok=True)
    # map_size=1 TiB -- sparse file, only actual data is allocated on disk
    env = lmdb.open(output_path, map_size=1024 ** 4, readahead=False)

    mini_infos = []
    cache_data = []
    # Every commit syncs to disk, which on a parallel filesystem costs far more
    # than decoding a frame, so frames are committed in batches.
    with multiprocessing.Pool(num_workers) as pool:
        txn = env.begin(write=True)
        for i, (key, value, mini_info, meta) in enumerate(tqdm(
            pool.imap_unordered(worker_fn, indexed_frames),
            total=len(indexed_frames),
            desc=f"Writing {os.path.basename(output_path)}",
        ), 1):
            txn.put(key.encode(), value)
            if i % COMMIT_EVERY == 0:
                txn.commit()
                txn = env.begin(write=True)
            mini_infos.append(mini_info)
            cache_data.append((key, meta))
        txn.commit()

    nerf_norm = _nerf_normalization(mini_infos)

    with env.begin(write=True) as txn:
        txn.put(b"num_frames",        str(num_frames).encode())
        txn.put(b"num_views",         str(num_views).encode())
        txn.put(b"nerf_normalization", pickle.dumps(nerf_norm, protocol=pickle.HIGHEST_PROTOCOL))
        txn.put(b"length",            str(len(indexed_frames)).encode())

    env.close()

    # Save lightweight camera metadata cache for fast ViewpointLoader init
    cache_data.sort(key=lambda x: x[0])  # sort by key for deterministic order
    cache_file = os.path.join(output_path, "camera_cache.pt")
    torch.save(cache_data, cache_file)

    print(f"Done. {len(indexed_frames)} views written to {output_path}")
    print(f"  num_frames={num_frames}, num_views={num_views}")
    print(f"  camera_cache.pt saved ({len(cache_data)} entries)")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _allocated_cpus():
    """CPUs available to this process, respects slurm cgroup."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(description="Build LMDB dataset for MEMO training")
    parser.add_argument("--source_path",     required=True,  help="Path to dataset folder (contains transforms_train.json)")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--num_workers",     type=int, default=_allocated_cpus())
    parser.add_argument("--skip_train",      action="store_true", help="Skip building train LMDB")
    parser.add_argument("--skip_test",       action="store_true", help="Skip building test LMDB")
    args = parser.parse_args()

    if not args.skip_train:
        train_lmdb = build_lmdb(
            source_path=args.source_path,
            transforms_file="transforms_train.json",
            output_path=os.path.join(args.source_path, "train_lmdb"),
            white_background=args.white_background,
            num_workers=args.num_workers,
            training=True,
        )
        print(f"Train LMDB: {train_lmdb}")

    if not args.skip_test:
        test_lmdb = build_lmdb(
            source_path=args.source_path,
            transforms_file="transforms_test.json",
            output_path=os.path.join(args.source_path, "test_lmdb"),
            white_background=args.white_background,
            num_workers=args.num_workers,
            training=False,
        )
        print(f"Test LMDB:  {test_lmdb}")


if __name__ == "__main__":
    main()
