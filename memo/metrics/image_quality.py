"""Image metrics on held-out views, PSNR, SSIM, and LPIPS.

Every metric takes (B, 3, H, W) tensors in [0, 1]. LPIPS is torchmetrics' VGG
variant with normalize=True, which maps the [0, 1] inputs to [-1, 1] itself.
"""

from collections import defaultdict

import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


_LPIPS = None


def lpips_model(device):
    """The VGG LPIPS module on `device`, built once and reused."""
    global _LPIPS
    if _LPIPS is None or next(_LPIPS.parameters()).device != device:
        _LPIPS = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to(device)
        _LPIPS.eval()
    return _LPIPS


def psnr(img1, img2):
    """Per-image PSNR, shape (B, 1)."""
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian_kernel_1d(window_size, sigma):
    half = window_size // 2
    coords = torch.arange(window_size, dtype=torch.float32)
    kernel = torch.exp(-(coords - half) ** 2 / float(2 * sigma ** 2))
    return kernel / kernel.sum()


def _ssim_window(window_size, channel):
    k1 = _gaussian_kernel_1d(window_size, 1.5).unsqueeze(1)
    k2 = k1.mm(k1.t()).unsqueeze(0).unsqueeze(0)
    return k2.expand(channel, 1, window_size, window_size).contiguous()


_SSIM_WINDOW = 11
_SSIM_C1 = 0.01 ** 2
_SSIM_C2 = 0.03 ** 2


def ssim(img1, img2):
    """SSIM with an 11-pixel Gaussian window (sigma 1.5), averaged over the batch."""
    channel = img1.size(-3)
    window = _ssim_window(_SSIM_WINDOW, channel).type_as(img1).to(img1.device)
    pad = _SSIM_WINDOW // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    ssim_map = (
        ((2 * mu1_mu2 + _SSIM_C1) * (2 * sigma12 + _SSIM_C2))
        / ((mu1_sq + mu2_sq + _SSIM_C1) * (sigma1_sq + sigma2_sq + _SSIM_C2))
    )
    return ssim_map.mean()


@torch.no_grad()
def score_views(pairs, device) -> dict:
    """PSNR, SSIM, and LPIPS of rendered views against ground truth.

    `pairs` yields (pred, gt, t, view_num), with (3, H, W) images and the
    integer frame number t. Each metric is averaged over a frame's views, then
    over frames, so every frame weighs the same whatever its view count.
    Returns {"psnr", "ssim", "lpips", "per_frame": {"<t:04d>": {...}}}.
    """
    lpips = lpips_model(device)
    psnr_acc = defaultdict(list)
    ssim_acc = defaultdict(list)
    lpips_acc = defaultdict(list)

    for pred, gt, t, _view_num in pairs:
        # LPIPS needs inputs in [0, 1], and a render can overshoot slightly
        # (saturated highlights, alpha blending).
        pred = pred.to(device).clamp(0.0, 1.0)
        gt = gt[:3].to(device)
        frame_key = f"{int(t):04d}"
        psnr_acc[frame_key].append(psnr(pred.unsqueeze(0), gt.unsqueeze(0)).item())
        ssim_acc[frame_key].append(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item())
        lpips_acc[frame_key].append(lpips(pred.unsqueeze(0), gt.unsqueeze(0)).item())

    if not psnr_acc:
        raise ValueError("score_views: no views to score")

    per_frame = {}
    for f in sorted(psnr_acc):
        per_frame[f] = {
            "psnr": sum(psnr_acc[f]) / len(psnr_acc[f]),
            "ssim": sum(ssim_acc[f]) / len(ssim_acc[f]),
            "lpips": sum(lpips_acc[f]) / len(lpips_acc[f]),
        }

    return {
        "psnr": sum(v["psnr"] for v in per_frame.values()) / len(per_frame),
        "ssim": sum(v["ssim"] for v in per_frame.values()) / len(per_frame),
        "lpips": sum(v["lpips"] for v in per_frame.values()) / len(per_frame),
        "per_frame": per_frame,
    }
