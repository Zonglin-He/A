"""Label-preserving video corruptions and test-time augmentations.

All functions accept TubeDETR-normalized videos in ``C x T x H x W`` format.
No geometric transformation or horizontal flip is used, so referring expressions
such as "the person on the left" retain their meaning and boxes remain aligned.
"""

from __future__ import annotations

import torch
from torchvision.transforms.functional import gaussian_blur


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _stats(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = video.new_tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = video.new_tensor(IMAGENET_STD).view(3, 1, 1, 1)
    return mean, std


def denormalize(video: torch.Tensor) -> torch.Tensor:
    _validate_video(video)
    mean, std = _stats(video)
    return (video * std + mean).clamp(0.0, 1.0)


def normalize(video: torch.Tensor) -> torch.Tensor:
    _validate_video(video)
    mean, std = _stats(video)
    return (video - mean) / std


def _validate_video(video: torch.Tensor) -> None:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"expected CxTxHxW video with C=3, got {tuple(video.shape)}")


def _blur_rgb(video: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    # torchvision treats all leading dimensions as batch dimensions.
    frames = video.permute(1, 0, 2, 3)
    frames = gaussian_blur(frames, [kernel_size, kernel_size], [sigma, sigma])
    return frames.permute(1, 0, 2, 3)


def apply_condition(
    video: torch.Tensor,
    condition: str,
    *,
    severity: int = 3,
) -> torch.Tensor:
    """Apply one deterministic controlled target-domain shift.

    Severity is defined on a five-level scale. The first feasibility run uses
    severity 3 for ``low_light`` and ``blur``.
    """

    if not 1 <= severity <= 5:
        raise ValueError("severity must be in [1, 5]")
    if condition == "clean":
        return video.clone()

    rgb = denormalize(video)
    if condition == "low_light":
        factors = (0.70, 0.55, 0.40, 0.28, 0.18)
        rgb = rgb * factors[severity - 1]
    elif condition == "blur":
        kernels = (3, 5, 7, 9, 11)
        sigmas = (0.6, 1.0, 1.6, 2.2, 3.0)
        rgb = _blur_rgb(rgb, kernels[severity - 1], sigmas[severity - 1])
    else:
        raise ValueError(f"unsupported condition: {condition}")
    return normalize(rgb.clamp(0.0, 1.0))


def make_strong_view(
    video: torch.Tensor,
    *,
    seed: int,
    blur_probability: float = 0.5,
) -> torch.Tensor:
    """Create a deterministic, non-geometric strong view for consistency TTA."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    brightness = 0.75 + 0.50 * torch.rand((), generator=generator).item()
    contrast = 0.80 + 0.40 * torch.rand((), generator=generator).item()
    use_blur = torch.rand((), generator=generator).item() < blur_probability

    rgb = denormalize(video)
    frame_mean = rgb.mean(dim=(-2, -1), keepdim=True)
    rgb = (rgb - frame_mean) * contrast + frame_mean
    rgb = rgb * brightness
    if use_blur:
        rgb = _blur_rgb(rgb, kernel_size=5, sigma=1.0)
    return normalize(rgb.clamp(0.0, 1.0))

