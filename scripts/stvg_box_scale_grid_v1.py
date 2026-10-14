"""CPU-only full-context window-grid inputs for box-scale evidence tests."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont


IMAGE_SIZE = 448
GRID_SCHEMA = "stvg-box-scale-global-grid-v1"
MASK_SCHEMA = "stvg-box-scale-full-context-mask-v1"
MARK_SCHEMA = "stvg-box-scale-full-context-mark-v1"


def _box4(box: Any, label: str) -> tuple[float, float, float, float]:
    if torch.is_tensor(box):
        if box.device.type != "cpu":
            raise ValueError(f"{label} must already be on CPU")
        values = box.detach().reshape(-1).tolist()
    else:
        values = list(box)
    if len(values) != 4:
        raise ValueError(f"{label} must contain four cxcywh values")
    values = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must be finite")
    return values


def _axis_starts(span: float) -> list[float]:
    if not math.isfinite(span) or not 0.0 < span <= 1.0:
        raise ValueError("window scale must lie in (0, 1]")
    last = 1.0 - span
    if last == 0.0:
        return [0.0]
    starts = []
    index = 0
    while True:
        position = index * span
        if position > last:
            break
        starts.append(position)
        index += 1
    starts.append(last)
    return sorted(set(starts))


def full_scale_grid(student_cxcywh: Any) -> dict[str, Any]:
    """Tile the full normalized image using only the student's width and height.

    Coordinates advance by exactly one window width/height. The final start
    1-span is always included, so the last row and column reach the image edges.
    Windows use stable row-major order (y, then x).
    """
    cx, cy, width, height = _box4(student_cxcywh, "student_cxcywh")
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise ValueError("student center coordinates must be normalized to [0, 1]")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError("student width and height must be in (0, 1]")

    xs, ys = _axis_starts(width), _axis_starts(height)
    windows = []
    for row, y0 in enumerate(ys):
        for column, x0 in enumerate(xs):
            windows.append({
                "window_id": len(windows) + 1,
                "row": row,
                "column": column,
                "x0": x0,
                "y0": y0,
                "w": width,
                "h": height,
                "cx": x0 + width / 2.0,
                "cy": y0 + height / 2.0,
            })

    return {
        "schema": GRID_SCHEMA,
        "scale_wh": [width, height],
        "x_starts": xs,
        "y_starts": ys,
        "windows": windows,
        "window_count": len(windows),
        "row_major_id_rule": "1-based; y rows outer, x columns inner",
        "complete_normalized_image_coverage": True,
        "coarse_stride_wh": [width, height],
        "half_step_or_refinement": False,
        "truncated": False,
        "student_baseline": {
            "kind": "separate_student_original_box",
            "window_id": None,
            "cxcywh": [cx, cy, width, height],
            "center_used_for_grid": False,
        },
    }


def _rgb_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    return image.convert("RGB")


def _window_pixel_box(image: Image.Image, window: Mapping[str, Any]) -> tuple[int, int, int, int]:
    x0, y0 = float(window["x0"]), float(window["y0"])
    width, height = float(window["w"]), float(window["h"])
    if not all(math.isfinite(v) for v in (x0, y0, width, height)):
        raise ValueError("window coordinates and scale must be finite")
    if width <= 0 or height <= 0 or x0 < 0 or y0 < 0 or x0 + width > 1.0 + 1e-12 or y0 + height > 1.0 + 1e-12:
        raise ValueError("window must be a legal normalized image region")
    frame_width, frame_height = image.size
    left = max(0, min(frame_width, int(round(x0 * frame_width))))
    top = max(0, min(frame_height, int(round(y0 * frame_height))))
    right = max(0, min(frame_width, int(round((x0 + width) * frame_width))))
    bottom = max(0, min(frame_height, int(round((y0 + height) * frame_height))))
    if right <= left or bottom <= top:
        raise ValueError("window is smaller than one pixel at the source resolution")
    return left, top, right, bottom


def _pixel_hash(image: Image.Image) -> str:
    canonical = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{canonical.width}x{canonical.height}:".encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _letterbox_448(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    source = _rgb_image(image)
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, min(IMAGE_SIZE, int(round(width * scale))))
    resized_height = max(1, min(IMAGE_SIZE, int(round(height * scale))))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized = source.resize((resized_width, resized_height), resample=resampling)
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    canvas.paste(resized, (left, top))
    return canvas, {
        "size": IMAGE_SIZE,
        "resampling": "bicubic",
        "scale": scale,
        "resized_wh": [resized_width, resized_height],
        "padding_ltrb": [left, top, IMAGE_SIZE - resized_width - left,
                          IMAGE_SIZE - resized_height - top],
        "source_wh": [width, height],
    }


def _mask_full_frame(image: Image.Image, window: Mapping[str, Any], *,
                     mode: str, blur_radius: float = 8.0,
                     hard_fill: tuple[int, int, int] = (127, 127, 127)) -> tuple[Image.Image, dict[str, Any]]:
    source = _rgb_image(image)
    left, top, right, bottom = _window_pixel_box(source, window)
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle((left, top, right - 1, bottom - 1), fill=255)
    if mode == "gaussian_blur":
        if float(blur_radius) != 8.0:
            raise ValueError("Gaussian blur radius is fixed at 8 pixels")
        replacement = source.filter(ImageFilter.GaussianBlur(radius=8))
        variant = Image.composite(replacement, source, mask)
        replacement_description = "blur_full_original_frame_then_replace_window_region"
        effective_radius = 8.0
    elif mode == "hard_mask":
        if len(hard_fill) != 3 or any(isinstance(v, bool) or not 0 <= int(v) <= 255 for v in hard_fill):
            raise ValueError("hard_fill must be an RGB byte triple")
        replacement = Image.new("RGB", source.size, tuple(int(v) for v in hard_fill))
        variant = Image.composite(replacement, source, mask)
        replacement_description = "solid_fill_window_region_on_full_original_frame"
        effective_radius = None
    else:
        raise ValueError("mode must be gaussian_blur or hard_mask")

    mask_pixels = sum(mask.histogram()[1:])
    fraction = mask_pixels / float(source.width * source.height)
    metadata = {
        "schema": MASK_SCHEMA,
        "window_id": int(window["window_id"]),
        "window_xyxy_pixels_half_open": [left, top, right, bottom],
        "window_area_fraction_normalized": float(window["w"]) * float(window["h"]),
        "mask_pixel_count": mask_pixels,
        "mask_fraction": fraction,
        "mode": mode,
        "blur_radius_px": effective_radius,
        "hard_fill_rgb": list(hard_fill) if mode == "hard_mask" else None,
        "replacement_rule": replacement_description,
        "full_frame_preserved": True,
        "crop": False,
        "source_wh": [source.width, source.height],
        "variant_pixel_sha256_before_letterbox": _pixel_hash(variant),
    }
    return variant, metadata


def masked_window_input(image: Image.Image, window: Mapping[str, Any], *,
                        mode: str = "gaussian_blur", blur_radius: float = 8.0,
                        hard_fill: tuple[int, int, int] = (127, 127, 127)) -> dict[str, Any]:
    """Return a full-frame masked input, letterboxed to 448x448, and receipts."""
    variant, metadata = _mask_full_frame(image, window, mode=mode,
                                          blur_radius=blur_radius, hard_fill=hard_fill)
    output, letterbox = _letterbox_448(variant)
    metadata.update({
        "letterbox": letterbox,
        "pixel_sha256": _pixel_hash(output),
        "output_wh": [IMAGE_SIZE, IMAGE_SIZE],
    })
    return {"image": output, "metadata": metadata}


def marked_window_input(image: Image.Image, window: Mapping[str, Any]) -> dict[str, Any]:
    """Return the full frame with one red labeled rectangle, letterboxed 448x448."""
    source = _rgb_image(image)
    left, top, right, bottom = _window_pixel_box(source, window)
    output, letterbox = _letterbox_448(source)
    resized_width, resized_height = letterbox["resized_wh"]
    pad_left, pad_top = letterbox["padding_ltrb"][:2]
    box_left = pad_left + int(round(left / source.width * resized_width))
    box_top = pad_top + int(round(top / source.height * resized_height))
    box_right = pad_left + int(round(right / source.width * resized_width))
    box_bottom = pad_top + int(round(bottom / source.height * resized_height))
    box_left = max(pad_left, min(IMAGE_SIZE, box_left))
    box_top = max(pad_top, min(IMAGE_SIZE, box_top))
    box_right = max(box_left + 1, min(IMAGE_SIZE - letterbox["padding_ltrb"][2], box_right))
    box_bottom = max(box_top + 1, min(IMAGE_SIZE - letterbox["padding_ltrb"][3], box_bottom))

    drawer = ImageDraw.Draw(output)
    drawer.rectangle((box_left, box_top, box_right - 1, box_bottom - 1),
                     outline=(255, 0, 0), width=2)
    font = ImageFont.load_default()
    # Every candidate is a single marked-window prompt; the visible token is
    # constant to avoid exposing row-major candidate identity to the model.
    label = "1"
    bbox = drawer.textbbox((0, 0), label, font=font)
    label_width = bbox[2] - bbox[0] + 4
    label_height = bbox[3] - bbox[1] + 4
    label_x = min(box_left + 2, IMAGE_SIZE - label_width)
    label_y = min(box_top + 2, IMAGE_SIZE - label_height)
    drawer.rectangle((label_x, label_y, label_x + label_width, label_y + label_height),
                     fill=(255, 255, 255))
    drawer.text((label_x + 2, label_y + 2), label, font=font, fill=(255, 0, 0))

    pixel_count = (right - left) * (bottom - top)
    metadata = {
        "schema": MARK_SCHEMA,
        "window_id": int(window["window_id"]),
        "visual_label": "1",
        "window_xyxy_pixels_half_open": [left, top, right, bottom],
        "window_area_fraction_normalized": float(window["w"]) * float(window["h"]),
        "marked_source_pixel_fraction": pixel_count / float(source.width * source.height),
        "marked_window_count": 1,
        "marker": "single_red_rectangle_and_red_window_id_on_full_frame",
        "full_frame_preserved": True,
        "crop": False,
        "letterbox": letterbox,
        "pixel_sha256": _pixel_hash(output),
        "output_wh": [IMAGE_SIZE, IMAGE_SIZE],
    }
    return {"image": output, "metadata": metadata}


def _validate_cxcywh_numpy(box: Any, label: str) -> np.ndarray:
    value = np.asarray(box, dtype=np.float64)
    if value.shape[-1:] != (4,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must have finite cxcywh values")
    if (value[..., 2:] < 0).any():
        raise ValueError(f"{label} width and height must be nonnegative")
    return value


def _validate_cxcywh_torch(box: Any, label: str) -> torch.Tensor:
    if torch.is_tensor(box) and box.device.type != "cpu":
        raise ValueError(f"{label} must already be on CPU")
    value = torch.as_tensor(box, dtype=torch.float64, device="cpu")
    if value.shape[-1:] != (4,) or not torch.isfinite(value).all():
        raise ValueError(f"{label} must have finite cxcywh values")
    if (value[..., 2:] < 0).any():
        raise ValueError(f"{label} width and height must be nonnegative")
    return value


def _cxcywh_to_xyxy_numpy(box: np.ndarray) -> np.ndarray:
    center, size = box[..., :2], box[..., 2:]
    return np.concatenate((center - size / 2.0, center + size / 2.0), axis=-1)


def _cxcywh_to_xyxy_torch(box: torch.Tensor) -> torch.Tensor:
    center, size = box[..., :2], box[..., 2:]
    return torch.cat((center - size / 2.0, center + size / 2.0), dim=-1)


def _overlap_components_numpy(pred_cxcywh: Any, target_cxcywh: Any) -> dict[str, Any]:
    pred = _cxcywh_to_xyxy_numpy(_validate_cxcywh_numpy(pred_cxcywh, "pred_cxcywh"))
    target = _cxcywh_to_xyxy_numpy(_validate_cxcywh_numpy(target_cxcywh, "target_cxcywh"))
    left_top = np.maximum(pred[..., :2], target[..., :2])
    right_bottom = np.minimum(pred[..., 2:], target[..., 2:])
    intersection = np.maximum(right_bottom - left_top, 0.0).prod(axis=-1)
    pred_area = np.maximum(pred[..., 2:] - pred[..., :2], 0.0).prod(axis=-1)
    target_area = np.maximum(target[..., 2:] - target[..., :2], 0.0).prod(axis=-1)
    union = pred_area + target_area - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)
    enclosing_left_top = np.minimum(pred[..., :2], target[..., :2])
    enclosing_right_bottom = np.maximum(pred[..., 2:], target[..., 2:])
    enclosing_diag_sq = np.square(np.maximum(enclosing_right_bottom - enclosing_left_top, 0.0)).sum(axis=-1)
    center_distance_sq = np.square(pred[..., :2] + pred[..., 2:] - target[..., :2] - target[..., 2:]).sum(axis=-1) / 4.0
    center_penalty = np.divide(center_distance_sq, enclosing_diag_sq,
                               out=np.zeros_like(center_distance_sq, dtype=np.float64),
                               where=enclosing_diag_sq > 0)
    return {"iou": iou, "center_distance_sq": center_distance_sq,
            "enclosing_diagonal_sq": enclosing_diag_sq,
            "center_penalty": center_penalty, "diou_reward_lambda_center_1": iou - center_penalty}


def _overlap_components_torch(pred_cxcywh: Any, target_cxcywh: Any) -> dict[str, torch.Tensor]:
    pred = _cxcywh_to_xyxy_torch(_validate_cxcywh_torch(pred_cxcywh, "pred_cxcywh"))
    target = _cxcywh_to_xyxy_torch(_validate_cxcywh_torch(target_cxcywh, "target_cxcywh"))
    left_top = torch.maximum(pred[..., :2], target[..., :2])
    right_bottom = torch.minimum(pred[..., 2:], target[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    pred_area = (pred[..., 2:] - pred[..., :2]).clamp_min(0).prod(dim=-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp_min(0).prod(dim=-1)
    union = pred_area + target_area - intersection
    iou = torch.where(union > 0, intersection / union.clamp_min(torch.finfo(union.dtype).tiny),
                      torch.zeros_like(union))
    enclosing_left_top = torch.minimum(pred[..., :2], target[..., :2])
    enclosing_right_bottom = torch.maximum(pred[..., 2:], target[..., 2:])
    enclosing_diag_sq = (enclosing_right_bottom - enclosing_left_top).clamp_min(0).square().sum(dim=-1)
    center_distance_sq = (pred[..., :2] + pred[..., 2:] - target[..., :2] - target[..., 2:]).square().sum(dim=-1) / 4.0
    center_penalty = torch.where(enclosing_diag_sq > 0,
                                 center_distance_sq / enclosing_diag_sq.clamp_min(torch.finfo(enclosing_diag_sq.dtype).tiny),
                                 torch.zeros_like(enclosing_diag_sq))
    return {"iou": iou, "center_distance_sq": center_distance_sq,
            "enclosing_diagonal_sq": enclosing_diag_sq,
            "center_penalty": center_penalty, "diou_reward_lambda_center_1": iou - center_penalty}


def iou_cxcywh_numpy(pred_cxcywh: Any, target_cxcywh: Any):
    return _overlap_components_numpy(pred_cxcywh, target_cxcywh)["iou"]


def diou_reward_cxcywh_numpy(pred_cxcywh: Any, target_cxcywh: Any):
    return _overlap_components_numpy(pred_cxcywh, target_cxcywh)["diou_reward_lambda_center_1"]


def iou_cxcywh_torch(pred_cxcywh: Any, target_cxcywh: Any) -> torch.Tensor:
    return _overlap_components_torch(pred_cxcywh, target_cxcywh)["iou"]


def diou_reward_cxcywh_torch(pred_cxcywh: Any, target_cxcywh: Any) -> torch.Tensor:
    return _overlap_components_torch(pred_cxcywh, target_cxcywh)["diou_reward_lambda_center_1"]


def contracts() -> dict[str, Any]:
    """Run deterministic CPU-only contracts on synthetic boxes and pixels."""
    box = [0.45, 0.40, 0.4, 0.3]
    grid = full_scale_grid(box)
    shifted_grid = full_scale_grid([0.7, 0.6, 0.4, 0.3])
    assert grid["windows"] == shifted_grid["windows"]
    assert grid["student_baseline"]["cxcywh"] != shifted_grid["student_baseline"]["cxcywh"]
    assert grid["student_baseline"]["window_id"] is None
    xs, ys = grid["x_starts"], grid["y_starts"]
    assert xs[0] == ys[0] == 0.0
    assert xs[-1] == 1.0 - box[2] and ys[-1] == 1.0 - box[3]
    assert all(b - a <= box[2] for a, b in zip(xs, xs[1:]))
    assert all(b - a <= box[3] for a, b in zip(ys, ys[1:]))
    assert [row["window_id"] for row in grid["windows"]] == list(range(1, grid["window_count"] + 1))
    for window in grid["windows"]:
        assert window["w"] == box[2] and window["h"] == box[3]
        assert 0.0 <= window["x0"] <= 1.0 - window["w"]
        assert 0.0 <= window["y0"] <= 1.0 - window["h"]
        assert window["x0"] + window["w"] <= 1.0 + 1e-15
        assert window["y0"] + window["h"] <= 1.0 + 1e-15
    small = full_scale_grid([0.5, 0.5, 0.1, 0.1])
    assert small["window_count"] == 100 and small["truncated"] is False
    single = full_scale_grid([0.5, 0.5, 1.0, 1.0])
    assert single["window_count"] == 1 and single["windows"][0]["window_id"] == 1

    source = Image.new("RGB", (64, 48))
    pixels = source.load()
    for y in range(source.height):
        for x in range(source.width):
            pixels[x, y] = (x * 3 % 256, y * 5 % 256, (x + y) * 2 % 256)
    first = grid["windows"][0]
    hard_pre, hard_meta = _mask_full_frame(source, first, mode="hard_mask")
    left, top, right, bottom = hard_meta["window_xyxy_pixels_half_open"]
    outside_sample = (right, bottom)
    assert hard_pre.getpixel(outside_sample) == source.getpixel(outside_sample)
    blur_pre, blur_meta = _mask_full_frame(source, first, mode="gaussian_blur", blur_radius=8.0)
    assert blur_meta["blur_radius_px"] == 8.0
    assert blur_pre.size == source.size and blur_pre.getpixel(outside_sample) == source.getpixel(outside_sample)
    try:
        _mask_full_frame(source, first, mode="gaussian_blur", blur_radius=7.0)
    except ValueError:
        fixed_radius_enforced = True
    else:
        fixed_radius_enforced = False
    assert fixed_radius_enforced
    masked = masked_window_input(source, first, mode="gaussian_blur")
    marked = marked_window_input(source, first)
    marked_second = marked_window_input(source, grid["windows"][1])
    assert masked["image"].size == marked["image"].size == (448, 448)
    assert masked["metadata"]["crop"] is False and marked["metadata"]["crop"] is False
    assert masked["metadata"]["window_id"] == marked["metadata"]["window_id"] == 1
    assert masked["metadata"]["mask_fraction"] == hard_meta["mask_fraction"]
    assert masked["metadata"]["pixel_sha256"] == _pixel_hash(masked["image"])
    assert marked["metadata"]["pixel_sha256"] == _pixel_hash(marked["image"])
    assert marked["metadata"]["marked_window_count"] == 1
    assert marked["metadata"]["visual_label"] == marked_second["metadata"]["visual_label"] == "1"
    assert marked_second["metadata"]["window_id"] == 2
    assert masked["metadata"]["letterbox"]["size"] == IMAGE_SIZE

    same_iou_np = float(iou_cxcywh_numpy([0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]))
    same_diou_np = float(diou_reward_cxcywh_numpy([0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]))
    pred, target = [0.5, 0.5, 0.2, 0.2], [0.6, 0.5, 0.2, 0.2]
    overlap_np = _overlap_components_numpy(pred, target)
    overlap_t = _overlap_components_torch(torch.tensor(pred, dtype=torch.float64),
                                          torch.tensor(target, dtype=torch.float64))
    assert same_iou_np == same_diou_np == 1.0
    assert 0.0 < float(overlap_np["diou_reward_lambda_center_1"]) < float(overlap_np["iou"]) < 1.0
    assert np.isclose(float(overlap_np["iou"]), float(overlap_t["iou"]), atol=1e-12, rtol=0)
    assert np.isclose(float(overlap_np["diou_reward_lambda_center_1"]),
                      float(overlap_t["diou_reward_lambda_center_1"]), atol=1e-12, rtol=0)
    expected_penalty = float(overlap_np["center_distance_sq"] / overlap_np["enclosing_diagonal_sq"])
    assert np.isclose(float(overlap_np["diou_reward_lambda_center_1"]),
                      float(overlap_np["iou"]) - expected_penalty, atol=1e-12, rtol=0)

    return {
        "status": "pass",
        "CPU_only": True,
        "synthetic_only": True,
        "GT_read": False,
        "model_run": False,
        "GPU_run": False,
        "check_groups": [
            "fixed width-height, center-independent grid and separate baseline",
            "complete axis coverage, exact scale, legal boundaries and row-major IDs",
            "final edge starts, exact deduplication, no truncation and one full-frame window",
            "hard-mask and full-frame Gaussian-blur region replacement",
            "fixed blur radius 8 and non-crop full-frame context",
            "448 bicubic letterbox and reproducible output pixel hashes",
            "single marked window with fixed visible ID 1 and separate row-major receipt ID",
            "lambda-center 1 DIoU formula and IoU comparison",
            "NumPy and Torch CPU reward agreement",
        ],
        "check_group_count": 9,
        "grid": {
            "example_scale_wh": [0.4, 0.3],
            "x_starts": xs,
            "y_starts": ys,
            "window_count": grid["window_count"],
            "small_scale_window_count_no_truncation": small["window_count"],
            "student_center_changes_grid": False,
            "row_major_ids_stable": True,
            "full_normalized_coverage": True,
            "coarse_full_step_only": True,
        },
        "pil": {
            "output_wh": [448, 448],
            "blur_radius_px": 8,
            "blur_on_full_frame_before_region_replacement": True,
            "full_frame_context_preserved": True,
            "crop": False,
            "single_marked_window_id": 1,
            "marked_single_visual_ID": "1",
            "masked_pixel_sha256": masked["metadata"]["pixel_sha256"],
            "marked_pixel_sha256": marked["metadata"]["pixel_sha256"],
            "mask_fraction": masked["metadata"]["mask_fraction"],
        },
        "overlap_reward": {
            "lambda_center": 1.0,
            "identity_iou": same_iou_np,
            "identity_diou_reward": same_diou_np,
            "shifted_iou": float(overlap_np["iou"]),
            "shifted_diou_reward": float(overlap_np["diou_reward_lambda_center_1"]),
            "numpy_torch_max_abs_error": max(
                abs(float(overlap_np["iou"]) - float(overlap_t["iou"])),
                abs(float(overlap_np["diou_reward_lambda_center_1"]) - float(overlap_t["diou_reward_lambda_center_1"])),
            ),
        },
        "scope_note": "Single-anchor window evidence only; no multi-frame DP, tube, re-identification, training, GT candidate selection, half-step search, or multiscale grid.",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(contracts(), ensure_ascii=False, indent=2))
