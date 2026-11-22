import os
from typing import List, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from memo.utils.image_utils import psnr
from memo.utils.loss_utils import ssim
from memo.lpipsPyTorch import lpips


def make_log_figure(
    images:List[torch.Tensor],
    nrows:Union[int, None]=None,
    ncols:Union[int, None]=None,
    titles:Union[List[str], None]=None,
    figsize:int=30,
    cmap:Union[str, List[str]]="Spectral",
    save_path:Union[str, None]=None,
):
    log_images = []
    for image in images:
        image_to_add = image
        if (image.ndim == 3) and (image.shape[0] == 1 or image.shape[0] == 3):
            image_to_add = image.permute(1, 2, 0)
        log_images.append(image_to_add.detach().cpu())

    n_images = len(log_images)
    if nrows is None and ncols is None:
        raise ValueError("nrows and ncols cannot be None at the same time")
    elif nrows is None:
        nrows = round(np.ceil(float(n_images) / float(ncols)))
    elif ncols is None:
        ncols = round(np.ceil(float(n_images) / float(nrows)))

    height, width = log_images[0].shape[:2]
    plt.figure(figsize=(figsize, figsize * height / width * nrows / ncols))

    for i, log_image in enumerate(log_images):
        plt.subplot(nrows, ncols, i+1)
        if log_image.shape[-1] == 3:
            plt.imshow(log_image.clamp(min=0, max=1))
        else:
            if type(cmap) == list:
                plt.imshow(log_image, cmap=cmap[i])
            else:
                if log_image.dtype == torch.bool:
                    plt.imshow(log_image, cmap="gray")
                else:
                    plt.imshow(log_image, cmap=cmap)
            plt.colorbar()
        if titles is not None:
            plt.title(titles[i])

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def _lpips_or_none(image, gt_image):
    """LPIPS of one view, or None when its network weights cannot be fetched,
    for example on a node without internet access."""
    try:
        return lpips(image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1,
                     net_type='vgg').mean()
    except OSError as e:
        print(f"[WARNING] LPIPS skipped, its network weights could not be loaded: {e}")
        return None


def training_report(iteration, testing_iterations, scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint.to_device("cuda")
                    viewpoint.load_original_image()
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    if lpipss is not None:
                        value = _lpips_or_none(image, gt_image)
                        lpipss = None if value is None else lpipss + [value]

                psnr_test /= len(config['cameras'])

                ssims_test=torch.tensor(ssims).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean()))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean()))
                if lpipss is not None:
                    print("  LPIPS : {:>12.7f}".format(torch.tensor(lpipss).mean()))
                print("")

        torch.cuda.empty_cache()


def log_training_progress(
    model_path, iteration:int, log_interval:int, progress_bar,
    scene, gaussians, background, pipe, opt,
    viewpoint_idx, render_pkg,
    reg_kick_on:bool, normal_field_kick_on:bool,
    loss, depth_normal_loss, normal_field_alignment_loss,
    ema:dict, postfix_dict:dict,
    testing_iterations:List[int], saving_iterations:List[int], render_func,
):
    """Progress bar, optional log figures, test evaluation, and saving.

    `ema` holds the running loss averages and is updated in place. Returns
    the progress-bar postfix.
    """
    # ---Progress bar---
    ema["loss"] = 0.4 * loss.item() + 0.6 * ema["loss"]
    if reg_kick_on:
        ema["depth_normal"] = 0.4 * depth_normal_loss.item() + 0.6 * ema["depth_normal"]
    if normal_field_kick_on:
        ema["normal_field_alignment"] = 0.4 * normal_field_alignment_loss.item() + 0.6 * ema["normal_field_alignment"]

    if iteration % 10 == 0:
        postfix_dict = {"Loss": f"{ema['loss']:.{5}f}"}
        if reg_kick_on:
            postfix_dict["DNLoss"] = f"{ema['depth_normal']:.{4}f}"
        if normal_field_kick_on:
            postfix_dict["NFLoss"] = f"{ema['normal_field_alignment']:.{4}f}"
        postfix_dict["N_Gauss"] = f"{gaussians._xyz.shape[0]}"
        progress_bar.set_postfix(postfix_dict)
        progress_bar.update(10)

    if iteration == opt.iterations:
        progress_bar.close()

    # ---Logging---
    if (log_interval is not None) and (iteration % log_interval == 0):
        images_to_log, titles_to_log = [], []

        images_to_log.append(render_pkg["render"])
        titles_to_log.append(f"Rendered RGB {viewpoint_idx}")

        if reg_kick_on or normal_field_kick_on:
            images_to_log.append(render_pkg["median_depth"])
            titles_to_log.append(f"Rendered Depth {viewpoint_idx}")

            images_to_log.append((1. - render_pkg["normal"]) / 2.)
            titles_to_log.append(f"Rendered Normals {viewpoint_idx}")

        make_log_figure(
            images=images_to_log,
            titles=titles_to_log,
            cmap='Spectral',
            ncols=3,
            figsize=30,
            save_path=os.path.join(model_path, f"iter_{iteration}.png"),
        )

    # The model is saved before it is evaluated, so a failing evaluation
    # cannot lose it.
    if (iteration in saving_iterations):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)
    training_report(iteration, testing_iterations, scene, render_func, (pipe, background))

    return postfix_dict
