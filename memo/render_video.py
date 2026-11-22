"""Render a trained MEMO model along a camera path into a video.

At each scene frame the camera moves one step along the path and the motion
model is evaluated at the frame's time. The Gaussian render gives the color
panel. The frame's mesh, extracted with extract_mesh.py's setup, gives the
shaded panel, a Lambertian shading of the mesh, and the normal panel, a
matcap of its view-space normals, both rendered with Blender's Cycles in the
memo-blender env or with nvdiffrast. The camera path, the frame rate, and the
scene's mesh settings come from the `video` section of the scene config, and
the path flags override it.

Usage:
    python -m memo.render_video -m <model_path> [--mode full] [--renderer blender]
    python -m memo.render_video -m <point_cloud.ply> --config <scene config>
"""
import os
import shutil
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import torch

from memo.arguments import PipelineParams
from memo.config import load_config
from memo.densification.prune_camera_blocking_floaters import (
    apply_near_camera_prune, snapshot_full_state)
from memo.extract_mesh import add_extraction_args, marching_tetrahedra_with_binary_search
from memo.gaussian_renderer import render
from memo.scene.cameras_source import (
    add_model_args, cameras_extent, load_cameras, load_model, model_dir, resolve_model_args)
from memo.scene.mesh import MeshRasterizer, MeshRenderer
from memo.utils.general_utils import safe_state
from memo.utils.mesh_panels import (
    AO_FRACTION, BLENDER_CHUNK, MATCAP_SAMPLES, blender_job, load_on_background, run_blender)
from memo.video.camera_path import (
    build_camera_arc_trajectory, build_orbit_trajectory, estimate_orbit_radius,
    lookat_extrinsics, make_synthetic_camera)
from memo.video.encode import CODECS, DEFAULT_CRF, write_video
from memo.video.panels import compose_full, mesh_normals, set_near, shade_lambertian, shade_matcap


# The settings of the video section that have no flag, and their defaults.
#   path_mode         camera_arc only, `sweep` crosses the arc once and
#                     `oscillate` crosses it and returns
#   coverage          camera_arc only, the fraction of the rig's azimuth span
#                     the arc crosses
#   near_camera_prune remove, at each frame, the Gaussians in front of a rig
#                     camera or the path camera within near_camera_prune_extent
#                     times the camera extent, which floaters parked in front
#                     of the cameras would otherwise veil the view with
#   orbit_clear_fraction  with near_camera_prune, also remove everything in
#                     the path camera's view closer than this fraction of its
#                     distance to the scene center, and clip the mesh panels
#                     there, since a path through the rig passes the rig's
#                     hardware
# A key of the video section that names an extraction setting, such as
# crop_max_edge_mult or min_component_faces, overrides it for the video.
VIDEO_SETTINGS = {
    "path_mode": "sweep",
    "coverage": 0.9,
    "near_camera_prune": False,
    "near_camera_prune_extent": 0.15,
    "orbit_clear_fraction": 0.0,
}
# The keys of the video section that set a flag's default.
VIDEO_FLAGS = ("trajectory", "arc", "elevation", "start_angle", "direction", "world_up",
               "fixed_camera", "fps")


def parse_args():
    """The command line, the resolved config, and the extraction settings."""
    parser = ArgumentParser(description="Render a trained MEMO model along a camera path into a video")
    add_model_args(parser)
    parser.add_argument("--mode", choices=["full", "color", "mesh", "normals"], default="full",
                        help="color, the Gaussian render. mesh, the shaded mesh. normals, the "
                             "mesh's normals as a matcap. full, the Gaussian render beside the "
                             "mesh and normal panels stacked at half size.")
    parser.add_argument("--renderer", choices=["blender", "nvdiffrast"], default="blender",
                        help="Renderer of the mesh and normal panels. blender renders them with "
                             "Cycles in the memo-blender env and is slow. nvdiffrast is fast.")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Frames to render, numbered from 1, every frame by default.")
    parser.add_argument("--output", default=None,
                        help="Output video, videos/<mode>.mp4 beside the model by default.")
    parser.add_argument("--codec", choices=list(CODECS), default="av1",
                        help="av1 (libaom) compresses best and encodes slowly, h264 and h265 "
                             "encode fast.")
    parser.add_argument("--crf", type=int, default=None,
                        help="Encoding quality, lower is better. 30 for av1 and 10 for h264 "
                             "and h265 by default.")
    parser.add_argument("--fps", type=int, default=25,
                        help="Frame rate, the scene config's by default.")
    parser.add_argument("--cycles_samples", type=int, default=128,
                        help="Cycles samples per pixel of the shaded panel, Blender only.")
    path = parser.add_argument_group("Camera path, the scene config's by default")
    path.add_argument("--trajectory", choices=["orbit", "camera_arc"], default="orbit",
                      help="orbit circles the world origin. camera_arc sweeps an arc through "
                           "the capture rig, and ignores the other path flags but "
                           "--fixed_camera.")
    path.add_argument("--arc", choices=["full", "half"], default="half",
                      help="Orbit angle, a full or half circle.")
    path.add_argument("--elevation", type=float, default=30.0,
                      help="Orbit elevation above the horizontal plane, in degrees.")
    path.add_argument("--start_angle", type=float, default=0.0,
                      help="Starting orbit azimuth in degrees, 0 on the +x axis.")
    path.add_argument("--direction", choices=["cw", "ccw"], default="cw",
                      help="Orbit direction seen from above.")
    path.add_argument("--world_up", choices=["z-up", "y-up"], default="z-up",
                      help="World up axis of the orbit.")
    path.add_argument("--fixed_camera", action=BooleanOptionalAction, default=False,
                      help="Keep the camera at the path's start.")

    config_path = resolve_model_args(parser, parser.parse_known_args()[0]).config
    config = load_config(config_path)

    extraction_parser = ArgumentParser()
    add_extraction_args(extraction_parser)
    extraction_keys = {action.dest for action in extraction_parser._actions} - {"help"}
    extraction_config = config.get("extraction", {})
    video = config.get("video", {})
    unknown = set(extraction_config) - extraction_keys
    unknown |= set(video) - extraction_keys - set(VIDEO_SETTINGS) - set(VIDEO_FLAGS)
    if unknown:
        parser.error(f"unknown keys in the extraction or video section of {config_path}: "
                     f"{sorted(unknown)}")
    extraction_parser.set_defaults(**{**extraction_config,
                                      **{k: v for k, v in video.items() if k in extraction_keys}})
    extraction = extraction_parser.parse_args([])
    # The panels shade the geometry alone.
    extraction.skip_colors = True

    parser.set_defaults(**{k: v for k, v in video.items() if k in VIDEO_FLAGS})
    args = resolve_model_args(parser, parser.parse_args())
    if args.crf is None:
        args.crf = DEFAULT_CRF[args.codec]
    for key, default in VIDEO_SETTINGS.items():
        setattr(args, key, video.get(key, default))
    return args, config, extraction


def camera_path(args, cameras, num_frames):
    """(eye, normalized_t) per scene frame, and the point and up direction
    every eye looks at and with."""
    if args.trajectory == "camera_arc":
        path, look_center, look_up = build_camera_arc_trajectory(
            cameras, num_frames, coverage=args.coverage, mode=args.path_mode)
    else:
        look_center = np.zeros(3)
        look_up = np.array([0.0, 0.0, 1.0]) if args.world_up == "z-up" else np.array([0.0, 1.0, 0.0])
        path = build_orbit_trajectory(
            world_up=look_up, radius=estimate_orbit_radius(cameras), num_frames=num_frames,
            arc=args.arc, elevation_deg=args.elevation, start_angle_deg=args.start_angle,
            direction=args.direction)
    if args.fixed_camera:
        path = [(path[0][0], t) for _, t in path]
    return path, look_center, look_up


@torch.no_grad()
def video_frames(args, extraction, gaussians, raw_views, background_rgb, frames, work_dir):
    """The composed frames of `frames`, numbered from 1, in order, with the
    training cameras `raw_views`."""
    num_frames = max(v.t for v in raw_views)
    ref = raw_views[0]
    path, look_center, look_up = camera_path(args, raw_views, num_frames)

    pipeline_parser = ArgumentParser()
    pipeline = PipelineParams(pipeline_parser).extract(pipeline_parser.parse_args([]))
    background = torch.tensor(background_rgb, dtype=torch.float32, device="cuda")
    full_state = None
    if args.near_camera_prune:
        full_state = snapshot_full_state(gaussians)
        rig_cameras = raw_views + load_cameras(args.source_path, "test")
        extent = cameras_extent(args.source_path, raw_views)
    needs_color = args.mode in ("full", "color")
    needs_shaded = args.mode in ("full", "mesh")
    needs_normals = args.mode in ("full", "normals")
    mesh_renderer = None
    if args.renderer == "nvdiffrast" and (needs_shaded or needs_normals):
        mesh_renderer = MeshRenderer(MeshRasterizer())

    def compose(color, shaded, normals):
        if args.mode == "full":
            return compose_full(color, shaded, normals, background_rgb)
        return {"color": color, "mesh": shaded, "normals": normals}[args.mode]

    # (color panel, mesh PLY, shaded PNG, normals PNG) of the frames awaiting Blender.
    pending, jobs = [], []

    def flush_blender():
        if not pending:
            return
        run_blender(jobs, args.cycles_samples, work_dir)
        for color, ply, shaded_png, normals_png in pending:
            shaded = load_on_background(shaded_png, background_rgb) if shaded_png else None
            normals = load_on_background(normals_png, background_rgb) if normals_png else None
            yield compose(color, shaded, normals)
            for file in (ply, shaded_png, normals_png):
                if file:
                    os.remove(file)
        pending.clear()
        jobs.clear()

    for frame in frames:
        eye, t = path[frame - 1]
        print(f"[INFO] Frame {frame}/{num_frames} (t={t:.4f})", flush=True)
        R, T = lookat_extrinsics(eye, look_center, look_up)
        camera = make_synthetic_camera(R, T, ref.FoVx, ref.FoVy, ref.image_height,
                                       ref.image_width, normalized_t=t)
        camera.to_device("cuda")
        distance = float(np.linalg.norm(np.asarray(eye) - look_center))
        clear_depth = args.orbit_clear_fraction * distance

        if full_state is not None:
            # The path camera joins the rig, since floaters on its way block
            # the video without being near any rig camera.
            n_pruned, radius, n_cameras = apply_near_camera_prune(
                gaussians, rig_cameras, extent, args.near_camera_prune_extent, full_state, t,
                extra_cameras=[camera],
                clear_view=camera if clear_depth > 0 else None, clear_depth=clear_depth)
            print(f"[INFO] Near-camera prune: removed {n_pruned} Gaussians within "
                  f"{radius:.3f} of {n_cameras} cameras")

        color = None
        if needs_color:
            rendering = render(camera, gaussians, pipeline, background,
                               apply_4d=True)["render"]
            color = (rendering.clamp(0.0, 1.0) * 255.0).to(torch.uint8) \
                .permute(1, 2, 0).contiguous().cpu().numpy()
        if not (needs_shaded or needs_normals):
            yield compose(color, None, None)
            continue

        # Extraction reads the cameras alone, since the vertex colors are
        # skipped, so the frame's images stay unloaded.
        views = [v for v in raw_views
                 if int(round(v.normalized_t * (num_frames - 1))) == frame - 1]
        for v in views:
            v.to_device("cuda")
        mesh = marching_tetrahedra_with_binary_search(gaussians, views, t, background, extraction)

        if mesh_renderer is not None:
            if clear_depth > 0:
                set_near(camera, clear_depth)
            world_normals = mesh_normals(mesh, camera, mesh_renderer)
            shaded = shade_lambertian(world_normals, camera, background_rgb) if needs_shaded else None
            normals = shade_matcap(world_normals, camera, background_rgb) if needs_normals else None
            yield compose(color, shaded, normals)
        else:
            ply = os.path.join(work_dir, f"frame_{frame:04d}.ply")
            mesh.export(ply)
            # Blender's near plane clips the mesh where the path camera's view
            # was cleared, or at Blender's usual 0.01 without a clear.
            clip_start = clear_depth if clear_depth > 0 else 0.01
            shaded_png = os.path.join(work_dir, f"shaded_{frame:04d}.png") if needs_shaded else None
            normals_png = os.path.join(work_dir, f"normals_{frame:04d}.png") if needs_normals else None
            if shaded_png:
                extra = {"matcap_out": normals_png, "matcap_samples": MATCAP_SAMPLES} \
                    if normals_png else {}
                jobs.append(blender_job(camera, ply, shaded_png, clip_start,
                                        ao_distance=AO_FRACTION * distance, **extra))
            else:
                jobs.append(blender_job(camera, ply, normals_png, clip_start,
                                        shading="matcap", samples=MATCAP_SAMPLES))
            pending.append((color, ply, shaded_png, normals_png))
            if len(pending) >= BLENDER_CHUNK:
                yield from flush_blender()
        del mesh
        torch.cuda.empty_cache()
    yield from flush_blender()


def main():
    args, config, extraction = parse_args()
    print(f"Rendering a video of {args.model_path}")
    safe_state()
    gaussians = load_model(args.model_path, args.config, args.iteration)
    background_rgb = [1.0, 1.0, 1.0] if config.get("white_background", False) else [0.0, 0.0, 0.0]
    raw_views = load_cameras(args.source_path, "train")

    num_frames = max(v.t for v in raw_views)
    frames = args.frames or list(range(1, num_frames + 1))
    for frame in frames:
        if not 1 <= frame <= num_frames:
            raise ValueError(f"--frames {frame} out of range [1, {num_frames}]")
    if len(set(frames)) != len(frames):
        raise ValueError("--frames lists a frame more than once")
    frames = sorted(frames)

    output = args.output or os.path.join(model_dir(args.model_path), "videos", f"{args.mode}.mp4")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="memo_video_")
    try:
        write_video(video_frames(args, extraction, gaussians, raw_views, background_rgb, frames,
                                 work_dir),
                    output, args.fps, args.codec, args.crf)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[INFO] Wrote {len(frames)} frames at {args.fps} fps to {output}")


if __name__ == "__main__":
    main()
