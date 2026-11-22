#
# MEMO training: GaussianModel4D on multi-frame datasets with GP-based temporal motion.
#

import gc
import os
import sys
import time
import uuid
import yaml
from functools import partial
from argparse import ArgumentParser, Namespace

gc.disable()  # Python GC causes progressive slowdown; manual gc.collect() at safe points instead

import torch
from tqdm import tqdm
from fused_ssim import fused_ssim
from memo.config import load_config
from memo.utils.loss_utils import l1_loss
from memo.utils.general_utils import safe_state, culling_schedule
from memo.gaussian_renderer import render, render_simp
from memo.scene import Scene
from memo.scene.gaussian_model_4d import GaussianModel4D
from memo.densification.prune_camera_blocking_floaters import prune_camera_blocking_floaters
from memo.arguments import ModelParams, PipelineParams, OptimizationParams
from memo.utils.log_utils import log_training_progress
from memo.utils.geometry_utils import depth_to_normal_with_mask
from memo.regularization.regularizer.normal_field import (
    compute_normal_field_regularization,
    densify_normal_field,
)


# Config sections, which hold settings rather than command-line flags.
CONFIG_SECTIONS = {"normal_field", "extraction", "video"}


def prepare_output_dir(dataset, config):
    """Create the output folder and record the run's resolved config in it as
    training_config.yaml, which rendering, extraction, and evaluation read."""
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))
    with open(os.path.join(dataset.model_path, "training_config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def training(
    dataset, opt, pipe,
    testing_iterations, saving_iterations,
    checkpoint_iterations, checkpoint,
    args,
    config,
    log_interval,
):
    prepare_output_dir(dataset, config)
    normal_field_config = config["normal_field"]

    # --- Initialize 4D model and scene ---
    first_iter = 0

    gaussians = GaussianModel4D(
        sh_degree=dataset.sh_degree,
        num_inducing=args.num_inducing,
        num_inducing_rot=args.num_inducing_rot,
        boundary_inducing_points=args.boundary_inducing_points,
        use_mip_filter=True,
        # Learned normals: 3 direction components + 1 sign.
        n_gaussian_features=4,
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    print(f"[INFO] MEMO training with M={args.num_inducing} inducing points"
          + (f", M_rot={args.num_inducing_rot}" if args.num_inducing_rot > 0 else ""))
    print(f"[INFO] Number of Gaussians: {gaussians.get_xyz.shape[0]}")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[INFO] Using normal field regularization (start_iter={normal_field_config['start_iter']}).")

    # Initialize culling stats. Pre-compute T_sub so _culling is shaped
    # identically to what culling_with_clone will allocate, avoiding a
    # mid-training shape/dtype switch. When aggressive cloning is off,
    # the single-slot T_sub=1 degenerate still plays nicely with
    # _unpack_view_slice.
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    if opt.aggressive_clone_from_iter > 0:
        _, _, t_sub = culling_schedule(scene.num_views, scene.num_frames, opt.max_culling_images)
    else:
        t_sub = 1
    gaussians.init_culling(scene.num_views, t_sub=t_sub)

    # Initialize 3D Mip filter
    gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy(),
                                max_culling_images=opt.max_culling_images)

    # partial wraps apply_4d=True so training_report can call renderFunc(viewpoint, gaussians, pipe, bg)
    render_fn = partial(render, apply_4d=True)

    # --- Training loop ---
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    viewpoints_loader = scene.getTrainCameras()

    # Render-time culling cutoff. The _culling mask is built up by aggressive-
    # clone events (<= densify_until_iter) and inherited through subsequent
    # clones (normal-aware densification, etc.), but no event reverses an
    # exclusion: a Gaussian flagged at iter 3K stays culled at render time
    # even after thousands of iters of further training would have made it
    # useful. After the last simplification event the mask is effectively
    # frozen exclusion data; stop applying it 5K iters later so late-stage
    # training renders the full population. The internal _culling tensor is
    # left intact so densify_and_clone_from_mask's bookkeeping still works.
    disable_culling_after_iter = max(opt.simp_iteration1, opt.simp_iteration2) + 5000

    # EMA tracking for active losses
    ema = {"loss": 0.0, "depth_normal": 0.0, "normal_field_alignment": 0.0}
    postfix_dict = {}

    for iteration in range(first_iter + 1, opt.iterations + 1):
        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Periodically increase SH degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Sample viewpoint via DataLoader iterator (workers prefetch from LMDB)
        viewpoint_cam = next(viewpoints_loader)
        viewpoint_cam.to_device("cuda")
        viewpoint_idx = viewpoint_cam.uid

        # Regularizer kick-on flags (gates use `>` so configs can use round
        # numbers — regularizer fires the iteration AFTER any cloning /
        # pruning / simp event scheduled at that exact iter).
        reg_kick_on = iteration > opt.regularization_from_iter
        normal_field_kick_on = iteration > normal_field_config["start_iter"]

        # Body-frame normals for the alpha-blended path. The renderer's
        # apply_4d block rotates extra_features by R(d_quat) per iteration.
        # Reset normal features on the first L_N iteration before the autograd
        # graph is built (in-place reset on _gaussian_features after the graph
        # references it would invalidate the graph).
        if normal_field_kick_on:
            # Skip reset only when resuming from a checkpoint already past
            # start_iter (its normal features are L_N-trained; a re-reset
            # would destroy them). A checkpoint from before start_iter has
            # never had the L_N reset, so it behaves like a fresh run here.
            if iteration == normal_field_config["start_iter"] + 1 and (
                    checkpoint is None
                    or first_iter <= normal_field_config["start_iter"]):
                gaussians.reset_normal_features()
            base_normals = gaussians.convert_features_to_normals(normalize=True)
        else:
            base_normals = None

        # Render with 4D. Thread the culling mask through until
        # disable_culling_after_iter, then bypass it so late-stage training
        # sees the full Gaussian population.
        if iteration > disable_culling_after_iter:
            culling_vec = None
        else:
            culling_vec = gaussians.get_culling_for_view(
                viewpoint_cam,
                num_frames=scene.num_frames,
                num_views=scene.num_views,
                max_culling_images=opt.max_culling_images,
            )
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background,
            extra_features=base_normals,
            apply_4d=True,
            require_depth=(reg_kick_on or normal_field_kick_on),
            culling=culling_vec,
        )

        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor = render_pkg["viewspace_points"]

        # Compute loss
        Ll1 = l1_loss(image, gt_image)
        ssim_val = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        # --- Depth-Normal Consistency Regularization (L_DN) ---
        # Masked to the pixels with valid depth; the depth is the 0.4 / 0.6
        # expected/median blend.
        depth_normal_loss = 0
        if reg_kick_on:
            reg_depth_ratio = 0.6
            dn_median = render_pkg["median_depth"]
            dn_expected = render_pkg["expected_depth"]
            depth_blend = torch.where(
                dn_median > 0,
                (1. - reg_depth_ratio) * dn_expected
                + reg_depth_ratio * dn_median,
                dn_median,
            )
            depth_normal, valid_points = depth_to_normal_with_mask(viewpoint_cam, depth_blend)
            dn_normal = render_pkg["normal"]
            if opt.detach_dn_normal:
                dn_normal = dn_normal.detach()
            normal_error_map = 1 - torch.linalg.vecdot(
                dn_normal, depth_normal, dim=0
            )
            depth_normal_loss = opt.lambda_depth_normal * torch.where(
                valid_points.squeeze(),
                normal_error_map,
                torch.zeros_like(normal_error_map),
            ).mean()
            loss = loss + depth_normal_loss

        # --- Normal Field Regularization (L_N) ---
        normal_field_alignment_loss = 0
        if normal_field_kick_on:
            normal_field_render_pkg = compute_normal_field_regularization(
                render_pkg=render_pkg,
                viewpoint_cam=viewpoint_cam,
                config=normal_field_config,
            )
            loss = loss + normal_field_render_pkg["normal_field_loss"]
            normal_field_alignment_loss = normal_field_render_pkg["normal_field_alignment_loss"]

        loss.backward()

        with torch.no_grad():
            # Optimizer step (radii > 0 gives bool mask, visibility_filter
            # holds indices; the optimizer needs bool)
            visible = radii > 0
            gaussians.optimizer.step(visible, radii.shape[0])
            gaussians.optimizer.zero_grad(set_to_none=True)

            # The densification phase includes iter == densify_until_iter,
            # one extra densify + aggressive-clone burst immediately before
            # simp1.
            densify_phase_active = iteration <= opt.densify_until_iter
            aggressive_clone_active = opt.aggressive_clone_from_iter > 0
            if densify_phase_active:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if aggressive_clone_active:
                    area_max = render_pkg["area_max"]
                    mask_blur = torch.logical_or(
                        mask_blur,
                        area_max > (image.shape[1] * image.shape[2] / 5000),
                    )

                if (iteration >= opt.densify_from_iter
                        and iteration % opt.densification_interval == 0):
                    # Low 0.005 opacity threshold + blur mask split, so
                    # opacity-split clone children at o' = 1/(N+1) are not
                    # pruned before they accumulate gradient signal.
                    gaussians.densify_and_prune_mask(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, mask_blur,
                    )
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    # Update mip filter after Gaussian count changed
                    gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy(),
                                                max_culling_images=opt.max_culling_images)
                    torch.cuda.empty_cache()

            # Aggressive cloning schedule, bounded like the densification phase.
            if (aggressive_clone_active
                    and iteration >= opt.aggressive_clone_from_iter
                    and densify_phase_active
                    and iteration % opt.aggressive_clone_interval == 0):
                print(f"\n[ITER {iteration}] Aggressive clone event")
                gaussians.culling_with_clone(scene, render_simp, opt, pipe, background)
                gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy(),
                                            max_culling_images=opt.max_culling_images)
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                torch.cuda.empty_cache()

            # Simp 1 at simp_iteration1.
            if opt.simp_iteration1 > 0 and iteration == opt.simp_iteration1:
                print(f"\n[ITER {iteration}] Simp 1 (intersection sampling)")
                gaussians.culling_with_interesction_sampling(scene, render_simp, opt, pipe, background)
                gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy(),
                                            max_culling_images=opt.max_culling_images)
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                torch.cuda.empty_cache()

            # Simp 2 at simp_iteration2.
            if opt.simp_iteration2 > 0 and iteration == opt.simp_iteration2:
                print(f"\n[ITER {iteration}] Simp 2 (intersection preserving)")
                gaussians.culling_with_intersection_preserving(scene, render_simp, opt, pipe, background)
                gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy(),
                                            max_culling_images=opt.max_culling_images)
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                torch.cuda.empty_cache()

            # --- Normal-aware densification event ---
            # Fires every densify_every_n_iterations between
            # start_iter_densification and end_iter_densification (inclusive).
            # densify_and_clone_from_mask extends _culling along N inside the
            # call; we refresh the MIP filter and reset mask_blur here.
            if (normal_field_kick_on
                    and normal_field_config["start_iter_densification"] <= iteration <= normal_field_config["end_iter_densification"]
                    and (iteration - normal_field_config["start_iter_densification"]) % normal_field_config["densify_every_n_iterations"] == 0):
                print(f"\n[ITER {iteration}] Normal-aware densification")
                n_before = gaussians._xyz.shape[0]
                densify_normal_field(
                    gaussians=gaussians,
                    cameras=scene.getTrainCameras().copy(),
                    pipe=pipe,
                    background=background,
                    config=normal_field_config,
                    max_culling_images=opt.max_culling_images,
                )
                print(f"  > Gaussians {n_before} -> {gaussians._xyz.shape[0]}")
                gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy(),
                                            max_culling_images=opt.max_culling_images)
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                torch.cuda.empty_cache()

            # Midway between simp 2 and the end of training, clear the culling mask.
            if opt.simp_iteration2 > 0 and iteration == (opt.simp_iteration2 + opt.iterations) // 2:
                print(f"\n[ITER {iteration}] Midway init_culling")
                gaussians.init_culling(scene.num_views, t_sub=t_sub)

            # Update 3D Mip filter periodically after densification phase.
            if iteration > opt.densify_until_iter \
                    and iteration % args.update_mip_filter_every == 0:
                if iteration < opt.iterations - args.update_mip_filter_every:
                    gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy(),
                                                max_culling_images=opt.max_culling_images)
                    torch.cuda.empty_cache()
                else:
                    print(f"[INFO] Skipping 3D Mip Filter update at iteration {iteration}")

            # Post-training floater prune, before the final save/checkpoint so
            # the saved model and its meshes exclude camera-blocking floaters.
            if iteration == opt.iterations and opt.near_camera_prune_extent > 0:
                prune_camera_blocking_floaters(
                    gaussians, scene, opt.near_camera_prune_extent)

            # Logging, test evaluation, and saving
            postfix_dict = log_training_progress(
                dataset.model_path, iteration, log_interval, progress_bar,
                scene, gaussians, background, pipe, opt,
                viewpoint_idx, render_pkg,
                reg_kick_on=reg_kick_on,
                normal_field_kick_on=normal_field_kick_on,
                loss=loss,
                depth_normal_loss=depth_normal_loss,
                normal_field_alignment_loss=normal_field_alignment_loss,
                ema=ema,
                postfix_dict=postfix_dict,
                testing_iterations=testing_iterations,
                saving_iterations=saving_iterations,
                render_func=render_fn,
            )

            # Manual GC at test evaluation points (GC is disabled for training speed)
            if iteration in testing_iterations:
                gc.collect()
                torch.cuda.empty_cache()

            # Save checkpoint
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

    print(f"\nTraining complete. Final iteration: {opt.iterations}")
    print(f"Final number of Gaussians: {gaussians.get_xyz.shape[0]}")


if __name__ == "__main__":
    parser = ArgumentParser(description="MEMO training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", type=str, required=True,
                        help="Path to a scene config. Its top-level keys set the defaults "
                             "of the flags below, and its normal_field section configures "
                             "the normal field.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # GP motion
    parser.add_argument("--num_inducing", type=int, default=10,
                        help="M: number of GP inducing points")
    parser.add_argument("--num_inducing_rot", type=int, default=-1,
                        help="Rotation-GP inducing count; -1 shares --num_inducing. "
                             "Boundary IPs apply per family, so the rotation GP "
                             "carries num_inducing_rot + 2N total points.")
    parser.add_argument("--boundary_inducing_points", type=int, default=2,
                        help="N: extra GP inducing points on each side of [0, 1]. "
                             "Total IPs become M + 2N; bins of width 1/M extend the "
                             "inducing-time range to [-N/M, 1 + N/M], giving the "
                             "boundary frames two-sided kernel support. Lengthscale "
                             "default stays anchored to M.")

    # Mip filter
    parser.add_argument("--update_mip_filter_every", type=int, default=100)

    # Logging
    parser.add_argument("--log_interval", type=int, default=None)

    # First pass: load the config to set argparse defaults. Dict-valued
    # sections are not flags.
    pre_args, _ = parser.parse_known_args(sys.argv[1:])
    unified_config = load_config(pre_args.config)
    print(f"[INFO] Loading config: {pre_args.config}")
    known = {action.dest for action in parser._actions} | CONFIG_SECTIONS
    unknown = sorted(k for k in unified_config if k not in known)
    if unknown:
        parser.error(f"{pre_args.config} sets unknown keys: {', '.join(unknown)}")
    parser.set_defaults(**{k: v for k, v in unified_config.items()
                           if not isinstance(v, dict)})

    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    if -1 not in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state()

    torch.cuda.synchronize()
    time_start = time.time()

    # The config as this run used it, command-line overrides included.
    resolved_config = {k: (v if isinstance(v, dict) else getattr(args, k, v))
                       for k, v in unified_config.items()}
    # Loading the model needs its inducing-point counts, which may come from
    # the command line alone.
    for k in ("num_inducing", "num_inducing_rot", "boundary_inducing_points"):
        resolved_config[k] = getattr(args, k)

    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args,
        resolved_config,
        args.log_interval,
    )

    torch.cuda.synchronize()
    time_end = time.time()
    print(f"\nTotal training time: {time_end - time_start:.1f}s")
