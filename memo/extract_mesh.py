"""Per-frame mesh extraction from a trained MEMO model.

At a frame's time, each Gaussian gives two pivots, its center and a point along
its learned normal. The vacancy of every pivot is evaluated through the SOF
rasterizer from the cameras of that frame, marching tetrahedra on the pivots'
Delaunay triangulation places the surface, a binary search refines each vertex
onto the 0.5 level set, and the mesh is cleaned up. Settings default to the
`extraction` section of the scene config, by default the model's
training_config.yaml, and command-line flags override it. `--mode paper`
then switches to the settings of the paper's tables, PAPER_SETTINGS.

Usage:
    python -m memo.extract_mesh -m <model_path> --evenly_spaced 10
    python -m memo.extract_mesh -m <point_cloud.ply> --config <scene config> --evenly_spaced 10 --skip_colors
"""
#adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
import os
import time
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from memo.config import load_config
from memo.extraction.mesh import extract_mesh, refine_intersections_with_binary_search
from memo.extraction.pivots import get_intersecting_pivots_from_normals
from memo.extraction.vertex_colors import evaluate_mesh_colors_all_vertices
from memo.gaussian_renderer.sof import (
    GlobalSortOrder,
    compute_transmittance,
    default_splat_args,
    evaluate_vacancy_sof_fast,
    get_frustum_mask,
)
from memo.scene.cameras_source import (
    add_model_args, load_cameras, load_model, model_dir, resolve_model_args)
from memo.scene.gaussian_model_4d import GaussianModel4D
from memo.scene.mesh import return_delaunay_tets
from memo.utils.camera_utils import get_cameras_spatial_extent
from memo.utils.general_utils import build_rotation, quaternion_raw_multiply, safe_state


@torch.no_grad()
def face_component_sizes(faces):
    """Size of each face's connected component, on the GPU.

    Matches trimesh's connected_component_labels over face_adjacency: two
    faces are adjacent when they share an edge that exactly two faces use, so
    edges of three or more faces connect nothing. Components are labeled by
    min-label propagation with pointer jumping.
    """
    f = torch.as_tensor(np.asarray(faces), dtype=torch.int64, device="cuda")
    n_faces = f.shape[0]
    edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]).sort(dim=1).values
    owner = torch.arange(n_faces, device=f.device).repeat(3)
    key, order = (edges[:, 0] * (int(f.max()) + 1) + edges[:, 1]).sort()
    owner = owner[order]
    _, counts = torch.unique_consecutive(key, return_counts=True)
    first = (torch.cumsum(counts, 0) - counts)[counts == 2]
    a, b = owner[first], owner[first + 1]
    a, b = a[a != b], b[a != b]

    label = torch.arange(n_faces, device=f.device)
    while True:
        prev = label
        low = torch.minimum(label[a], label[b])
        label = label.scatter_reduce(0, a, low, reduce="amin").scatter_reduce(0, b, low, reduce="amin")
        while not torch.equal(label[label], label):
            label = label[label]
        if torch.equal(label, prev):
            break
    return torch.bincount(label, minlength=n_faces)[label].cpu().numpy()


def apply_mesh_cleanup(mesh, views, args):
    """Distance crop and absolute max-edge filter, both as multiples of the
    rig extent of the views' camera centers, then a connected-component
    face-count floor. Thresholds come from args (crop_distance_mult,
    crop_max_edge_mult, min_component_faces), and 0 disables a filter.
    """
    crop_mult = args.crop_distance_mult
    edge_mult = args.crop_max_edge_mult
    min_faces = args.min_component_faces
    if crop_mult <= 0 and edge_mult <= 0 and min_faces <= 0:
        return mesh
    centers = np.stack([-np.asarray(v.R, np.float64) @ np.asarray(v.T, np.float64)
                        for v in views])
    center = centers.mean(0)
    extent = 1.1 * np.linalg.norm(centers - center, axis=1).max()
    n0 = len(mesh.faces)
    if crop_mult > 0:
        keep = np.linalg.norm(mesh.vertices - center, axis=1) <= crop_mult * extent
        mesh.update_faces(keep[mesh.faces].all(axis=1))
        mesh.remove_unreferenced_vertices()
    if edge_mult > 0:
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        longest = np.linalg.norm(v[f[:, [0, 1, 2]]] - v[f[:, [1, 2, 0]]],
                                 axis=2).max(axis=1)
        mesh.update_faces(longest <= edge_mult * extent)
        mesh.remove_unreferenced_vertices()
    if min_faces > 0 and len(mesh.faces):
        mesh.update_faces(face_component_sizes(mesh.faces) >= min_faces)
        mesh.remove_unreferenced_vertices()
    print(f"[INFO] mesh cleanup: kept {len(mesh.faces)}/{n0} faces "
          f"(crop {crop_mult:g}x, max edge {edge_mult:g}x extent {extent:.2f}, "
          f"min component {min_faces} faces)")
    return mesh


@torch.no_grad()
def max_transmittance(points, views, gaussians, background, znear, zfar, t, thin_structures):
    """The vacancy field at each point, the largest transmittance over the
    views whose frustum contains it.

    evaluate_vacancy_sof_fast only reports whether this maximum exceeds the
    threshold and stops at the first view that does. This returns the value
    itself, without alpha early stopping.
    """
    splat_args = default_splat_args()
    splat_args.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
    splat_args.meshing_settings.alpha_early_stop = False
    best = torch.zeros(points.shape[0], device="cuda")
    for view in tqdm(views, desc="Vacancy field"):
        mask = get_frustum_mask(points=points, cameras=[view], near=znear, far=zfar)
        if not mask.any():
            continue
        ret = compute_transmittance(
            points3D=points[mask], viewpoint_camera=view, pc=gaussians,
            bg_color=background, splat_args=splat_args,
            thin_structures=thin_structures, t=t)
        best[mask] = torch.maximum(best[mask], ret["transmittance"].view(-1))
    return best


@torch.no_grad()
def marching_tetrahedra_with_binary_search(
    gaussians: GaussianModel4D,
    views: list,
    t: float,
    background: torch.Tensor,
    args,
) -> trimesh.Trimesh:
    """The mesh of the Gaussians at time `t` in [0, 1].

    `views` are the cameras of that frame, on the GPU, and need their images
    loaded unless `args.skip_colors` is set. `args` carries the settings that
    `add_extraction_args` registers.
    """
    # The core timer covers the motion model, pivots, tetrahedralization, the
    # vacancy evaluation, marching tetrahedra, and cleanup.
    torch.cuda.synchronize()
    core_start = time.perf_counter()

    xyz_t = gaussians.get_xyz + gaussians.position_gp(t)
    d_quat = gaussians.rotation_gp(t)
    rotation_t = quaternion_raw_multiply(d_quat, gaussians.get_rotation)
    normals_t = (build_rotation(d_quat) @ gaussians.convert_features_to_normals().unsqueeze(-1)).squeeze(-1)

    scene_radius = get_cameras_spatial_extent(cameras=views)['radius'].item()
    frustum_near = 0.02 * scene_radius / 6.
    frustum_far = 1e6 * scene_radius / 6.

    def sdf_function(points):
        """+0.5 at vacant points and -0.5 at occupied ones."""
        all_sdf = []
        for batch_points in torch.split(points, args.n_points_per_sdf_evaluation):
            splat_args = default_splat_args()
            splat_args.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
            splat_args.meshing_settings.alpha_early_stop = True
            splat_args.meshing_settings.transmittance_threshold = 0.5 + args.isosurface_value
            is_vacant = evaluate_vacancy_sof_fast(
                points=batch_points,
                views=views,
                gaussians=gaussians,
                background=background,
                splat_args=splat_args,
                znear=frustum_near,
                zfar=frustum_far,
                t=t,
                thin_structures=args.thin_structures,
            )  # (N,)
            all_sdf.append(is_vacant.float() - 0.5)
        return torch.cat(all_sdf, dim=0)

    pivots, pivots_scale = get_intersecting_pivots_from_normals(
        xyz=xyz_t,
        scaling=gaussians.get_scaling_with_3D_filter,
        rotation=rotation_t,
        normals=normals_t,
        std_factor=args.std_factor,
    )
    pivots = pivots.view(-1, 3)
    pivots_scale = (args.std_factor * pivots_scale).reshape(-1)
    pivot_sdfs = sdf_function(pivots)

    t0 = time.time()
    tets = return_delaunay_tets(pivots)
    print(f"[INFO] Computed {tets.shape[0]} tets with Delaunay triangulation: {time.time() - t0}s")

    mesh, end_points, end_sdf = extract_mesh(
        delaunay_tets=tets,
        pivots=pivots,
        pivots_sdf=pivot_sdfs,
        pivots_scale=pivots_scale,
        filter_large_edges=args.filter_large_edges,
        large_edge_mult=args.filter_large_edges_mult,
    )
    torch.cuda.empty_cache()

    if args.n_binary_steps > 0:
        mesh.verts = refine_intersections_with_binary_search(
            end_points=end_points,
            end_sdf=end_sdf,
            sdf_function=sdf_function,
            n_binary_steps=args.n_binary_steps,
        )

    if args.skip_colors:
        mesh = trimesh.Trimesh(
            vertices=mesh.verts.cpu().numpy(),
            faces=mesh.faces.cpu().numpy(),
            process=False
        )
    else:
        verts_colors = evaluate_mesh_colors_all_vertices(views=views, mesh=mesh).view(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=mesh.verts.cpu().numpy(),
            faces=mesh.faces.cpu().numpy(),
            vertex_colors=(verts_colors.cpu().numpy() * 255).astype(np.uint8),
            process=False
        )

    mesh = apply_mesh_cleanup(mesh, views, args)
    torch.cuda.synchronize()
    print(f"[INFO] core_seconds={time.perf_counter() - core_start:.4f}")

    # Marching tetrahedra connects the occupied pivots of a tetrahedron through
    # its interior, so a face can bridge empty space between nearby parts or
    # across a depth edge. Its barycenter then lies in air, far above the 0.5
    # level a face on the surface has.
    if len(mesh.faces) == 0:
        raise RuntimeError(
            f"no surface was extracted at t={t:.4f}. A model trained for fewer iterations than "
            f"normal_field.start_iter has no learned normals to place pivots with.")
    if args.filter_face_vacancy is None:
        return mesh
    # trimesh hands out its cached triangle centers read-only.
    barycenters = torch.as_tensor(np.array(mesh.triangles_center), dtype=torch.float32,
                                  device="cuda")
    face_vacancy = max_transmittance(barycenters, views, gaussians, background, frustum_near,
                                     frustum_far, t, args.thin_structures).cpu().numpy()
    quantile, floor = args.filter_face_vacancy
    threshold = max(float(np.quantile(face_vacancy, quantile)), floor)
    keep = face_vacancy <= threshold
    print(f"[INFO] filter_face_vacancy: removed {(~keep).sum()}/{len(keep)} faces "
          f"above vacancy {threshold:.3f}")
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh


# The extraction of the paper's tables: the large-edge filter at multiplier 1.0
# in place of the barycenter vacancy filter, and no thin-structures mode.
PAPER_SETTINGS = {
    "filter_large_edges": True,
    "filter_large_edges_mult": 1.0,
    "filter_face_vacancy": None,
    "thin_structures": False,
}


def add_extraction_args(parser):
    """Register the settings that marching_tetrahedra_with_binary_search reads."""
    parser.add_argument("--std_factor", default=3.0, type=float,
                        help="Distance of each Gaussian's normal pivot from its center, in "
                             "standard deviations along the normal.")
    parser.add_argument("--n_points_per_sdf_evaluation", default=999_999_999, type=int,
                        help="Points per vacancy evaluation batch.")
    parser.add_argument("--isosurface_value", default=0.0, type=float,
                        help="Offset of the vacancy level from 0.5.")
    parser.add_argument("--n_binary_steps", default=10, type=int,
                        help="Binary search steps refining each vertex onto the level set.")
    parser.add_argument("--thin_structures", action=BooleanOptionalAction, default=False,
                        help="Evaluate the vacancy field with the 3D-filtered scales but the "
                             "unfiltered opacities. Pivots keep the filtered scales.")
    parser.add_argument("--filter_large_edges", action=BooleanOptionalAction, default=False,
                        help="Drop the triangles of a surface-crossing edge longer than the sum "
                             "of its two endpoints' extents, each std_factor times the largest "
                             "scale of the endpoint's Gaussian.")
    parser.add_argument("--filter_large_edges_mult", default=3.0, type=float,
                        help="Multiplier on the large-edge threshold, >1 keeps more triangles.")
    parser.add_argument("--filter_face_vacancy", type=float, nargs=2, default=[0.99, 0.8],
                        metavar=("QUANTILE", "FLOOR"),
                        help="Remove faces whose barycenter vacancy exceeds the larger of its "
                             "QUANTILE over the mesh's faces and FLOOR.")
    parser.add_argument("--crop_distance_mult", default=5.0, type=float,
                        help="Drop mesh faces farther than this multiple of the "
                             "rig extent from the mean camera center (0 = off).")
    parser.add_argument("--crop_max_edge_mult", default=0.2, type=float,
                        help="Drop faces whose longest edge exceeds this multiple "
                             "of the rig extent (0 = off).")
    parser.add_argument("--min_component_faces", default=5000, type=int,
                        help="Drop connected components with fewer faces than "
                             "this (0 = off).")
    parser.add_argument("--skip_colors", action=BooleanOptionalAction, default=False,
                        help="Skip baking vertex colors from the frame's images.")


@torch.no_grad()
def main():
    parser = ArgumentParser(description="Per-frame mesh extraction from a trained MEMO model")
    add_model_args(parser)
    frame_selection = parser.add_mutually_exclusive_group(required=True)
    frame_selection.add_argument("--frames", nargs="+", type=int,
                                 help="Frames to extract, numbered from 1.")
    frame_selection.add_argument("--evenly_spaced", type=int, metavar="N",
                                 help="Extract N frames at the centers of N equal spans of the "
                                      "sequence, the frames the paper evaluates with N=10.")
    frame_selection.add_argument("--all_frames", action="store_true",
                                 help="Extract every frame.")
    frame_selection.add_argument("--times", nargs="+", type=float,
                                 help="Times in [0, 1] to extract, between frames as well. The "
                                      "vertex colors come from the nearest frame's images.")
    parser.add_argument("--output", default=None, type=str,
                        help="Output directory, meshes/ beside the model by default.")
    parser.add_argument("--mode", choices=["release", "paper"], default="release",
                        help="release: the settings below, recommended. paper: the settings "
                             "of the paper's tables, which override them.")
    add_extraction_args(parser)

    config_path = resolve_model_args(parser, parser.parse_known_args()[0]).config
    config = load_config(config_path)
    extraction_config = config.get("extraction", {})
    unknown = set(extraction_config) - {action.dest for action in parser._actions}
    if unknown:
        parser.error(f"unknown keys in the extraction section of {config_path}: {sorted(unknown)}")
    parser.set_defaults(**extraction_config)

    args = resolve_model_args(parser, parser.parse_args())
    if args.mode == "paper":
        vars(args).update(PAPER_SETTINGS)
    if not args.skip_colors and not (args.source_path / "train_lmdb").exists():
        parser.error(f"the vertex colors need the training images of {args.source_path}/train_lmdb, "
                     "pass --skip_colors to extract without them")
    print(f"Extracting meshes from {args.model_path}")
    safe_state()
    gaussians = load_model(args.model_path, args.config, args.iteration)
    background = torch.tensor([1, 1, 1] if config.get("white_background", False) else [0, 0, 0],
                              dtype=torch.float32, device="cuda")

    # Lightweight cameras of every frame, with 1x1 dummy images.
    raw_views = load_cameras(args.source_path, "train")
    num_frames = max(v.t for v in raw_views)
    # Each target is (output name, time, frame whose cameras are used). The rig
    # is static, so every frame has the same cameras, and the frame only
    # decides which images the vertex colors come from.
    if args.times is not None:
        for t in args.times:
            if not 0.0 <= t <= 1.0:
                raise ValueError(f"--times {t} out of range [0, 1]")
        targets = [(f"time_{t:.4f}", t, round(t * (num_frames - 1)) + 1) for t in args.times]
    else:
        if args.all_frames:
            frames = list(range(1, num_frames + 1))
        elif args.evenly_spaced:
            frames = [int((i + 0.5) * num_frames / args.evenly_spaced)
                      for i in range(args.evenly_spaced)]
        else:
            frames = args.frames
        for frame in frames:
            if not 1 <= frame <= num_frames:
                raise ValueError(f"--frames {frame} out of range [1, {num_frames}]")
        # t matches the normalized_t of the frame's views, so the geometry and
        # the vertex colors sample the same time.
        targets = [(f"frame_{frame:04d}", (frame - 1) / (num_frames - 1) if num_frames > 1 else 0.0,
                    frame) for frame in frames]
    if len({name for name, _, _ in targets}) != len(targets):
        raise ValueError("the selection lists a frame or time more than once")

    output_dir = args.output or os.path.join(model_dir(args.model_path), "meshes")
    os.makedirs(output_dir, exist_ok=True)

    for name, t, frame in targets:
        frame_idx = frame - 1
        views = [v for v in raw_views
                 if int(round(v.normalized_t * (num_frames - 1))) == frame_idx]
        for v in views:
            v.to_device("cuda")
            if not args.skip_colors:
                v.load_original_image()

        mesh_path = os.path.join(output_dir, f"{name}.ply")
        print(f"[INFO] Extracting {name} (t={t:.4f}, {len(views)} views of frame {frame}) -> {mesh_path}")
        mesh = marching_tetrahedra_with_binary_search(gaussians, views, t, background, args)
        mesh.export(mesh_path)

        # The cameras persist across frames, so each frame's images are
        # released once its colors are baked.
        if not args.skip_colors:
            for v in views:
                v.original_image = torch.zeros(3, 1, 1)


if __name__ == "__main__":
    main()
