"""Deterministic raw-pixel corruption primitives for bounded shift studies.

This module is deliberately data-only.  It accepts caller-supplied RGB
``uint8`` frames and never opens a video, reads annotations, constructs a
model, or touches CUDA.  Each operation returns a :class:`CorruptionResult`
with both the transformed pixels and an auditable physical-frame mapping.

The seven operations are intentionally not aliases:

``low_light``
    Per-pixel multiplicative attenuation followed by uint8 rounding.
``gaussian_blur``
    Per-frame Gaussian convolution with reflective boundaries.
``motion_blur``
    Per-frame normalized line-kernel convolution.  Its line kernel is not a
    Gaussian kernel.
``jpeg_compression``
    Per-frame JPEG encode/decode at a fixed quality level.
``gaussian_noise``
    Zero-mean additive Gaussian noise, with a local query/seed RNG.
``temporal_subsampling``
    Deterministic regular-stride retention, with first/last/minimum guards.
``frame_dropping``
    Random interior-frame removal using a local query/seed RNG, with the same
    first/last/minimum guards.  It is intentionally distinct from regular
    subsampling.

The parameter table is part of the API and should be pinned by a downstream
protocol.  Severity values are exactly 1 through 5.  For very short clips,
the physical constraints can make a requested temporal intensity
unattainable; the result metadata records the requested and actual counts
instead of silently claiming the requested rate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import math
from typing import Any

import numpy as np


VERSION = "shift_corruptions_v2"
# Blur severities are specified in pixels at this reference resolution.  The
# realized values are scaled by the input video's short side and recorded in
# every blur result, so a 224-pixel and a 448-pixel input receive the same
# normalized severity rather than the same raw kernel.
REFERENCE_SHORT_SIDE = 224
SEVERITIES = (1, 2, 3, 4, 5)
CORRUPTION_TYPES = (
    "low_light",
    "gaussian_blur",
    "motion_blur",
    "jpeg_compression",
    "gaussian_noise",
    "temporal_subsampling",
    "frame_dropping",
)

# These are explicit, conservative development settings.  A runner should
# pin this object (or its canonical hash) before looking at any labels.
PARAMETER_TABLE: dict[str, dict[int, dict[str, Any]]] = {
    "low_light": {
        1: {"factor": 0.90, "rounding": "nearest_uint8"},
        2: {"factor": 0.75, "rounding": "nearest_uint8"},
        3: {"factor": 0.60, "rounding": "nearest_uint8"},
        4: {"factor": 0.40, "rounding": "nearest_uint8"},
        5: {"factor": 0.25, "rounding": "nearest_uint8"},
    },
    "gaussian_blur": {
        1: {"kernel_size": 3, "sigma": 0.60, "border": "BORDER_REFLECT_101"},
        2: {"kernel_size": 5, "sigma": 1.00, "border": "BORDER_REFLECT_101"},
        3: {"kernel_size": 7, "sigma": 1.60, "border": "BORDER_REFLECT_101"},
        4: {"kernel_size": 9, "sigma": 2.20, "border": "BORDER_REFLECT_101"},
        5: {"kernel_size": 11, "sigma": 3.00, "border": "BORDER_REFLECT_101"},
    },
    "motion_blur": {
        1: {"kernel_size": 3, "angle_deg": 0.0, "kernel": "normalized_line"},
        2: {"kernel_size": 5, "angle_deg": 0.0, "kernel": "normalized_line"},
        3: {"kernel_size": 7, "angle_deg": 0.0, "kernel": "normalized_line"},
        4: {"kernel_size": 9, "angle_deg": 0.0, "kernel": "normalized_line"},
        5: {"kernel_size": 11, "angle_deg": 0.0, "kernel": "normalized_line"},
    },
    "jpeg_compression": {
        1: {"quality": 95, "codec": "jpeg", "color_conversion": "RGB_to_BGR_to_RGB"},
        2: {"quality": 80, "codec": "jpeg", "color_conversion": "RGB_to_BGR_to_RGB"},
        3: {"quality": 60, "codec": "jpeg", "color_conversion": "RGB_to_BGR_to_RGB"},
        4: {"quality": 40, "codec": "jpeg", "color_conversion": "RGB_to_BGR_to_RGB"},
        5: {"quality": 20, "codec": "jpeg", "color_conversion": "RGB_to_BGR_to_RGB"},
    },
    "gaussian_noise": {
        1: {"sigma": 2.0, "mean": 0.0, "distribution": "normal"},
        2: {"sigma": 5.0, "mean": 0.0, "distribution": "normal"},
        3: {"sigma": 10.0, "mean": 0.0, "distribution": "normal"},
        4: {"sigma": 20.0, "mean": 0.0, "distribution": "normal"},
        5: {"sigma": 30.0, "mean": 0.0, "distribution": "normal"},
    },
    "temporal_subsampling": {
        1: {"stride": 2, "selection": "regular_stride"},
        2: {"stride": 3, "selection": "regular_stride"},
        3: {"stride": 4, "selection": "regular_stride"},
        4: {"stride": 6, "selection": "regular_stride"},
        5: {"stride": 8, "selection": "regular_stride"},
    },
    "frame_dropping": {
        1: {"drop_fraction": 0.10, "selection": "random_interior"},
        2: {"drop_fraction": 0.20, "selection": "random_interior"},
        3: {"drop_fraction": 0.35, "selection": "random_interior"},
        4: {"drop_fraction": 0.50, "selection": "random_interior"},
        5: {"drop_fraction": 0.65, "selection": "random_interior"},
    },
}


class CorruptionContractError(ValueError):
    """Raised when a raw-pixel corruption contract is violated."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parameter_table() -> dict[str, dict[int, dict[str, Any]]]:
    """Return a deep copy of the pinned severity table."""

    return copy.deepcopy(PARAMETER_TABLE)


def parameter_table_sha256() -> str:
    """Return the canonical SHA-256 suitable for a protocol lock."""

    payload = repr(_canonical(PARAMETER_TABLE)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_raw(raw_rgb: Any) -> np.ndarray:
    if not isinstance(raw_rgb, np.ndarray):
        raise TypeError("raw_rgb must be a numpy array")
    if raw_rgb.ndim != 4 or raw_rgb.shape[-1] != 3 or raw_rgb.dtype != np.uint8:
        raise CorruptionContractError(
            "raw_rgb must have dtype uint8 and shape [T,H,W,3], "
            f"got dtype={raw_rgb.dtype} shape={tuple(raw_rgb.shape)}"
        )
    if raw_rgb.shape[0] < 1 or raw_rgb.shape[1] < 1 or raw_rgb.shape[2] < 1:
        raise CorruptionContractError("raw_rgb has an empty temporal or spatial dimension")
    return np.ascontiguousarray(raw_rgb)


def _validate_frame_ids(frame_ids: Sequence[int] | None, frame_count: int) -> list[int]:
    if frame_ids is None:
        return list(range(frame_count))
    if isinstance(frame_ids, (str, bytes)) or not isinstance(frame_ids, Sequence):
        raise TypeError("frame_ids must be a sequence of integers")
    if len(frame_ids) != frame_count:
        raise CorruptionContractError("frame_ids must have one physical id per input frame")
    result: list[int] = []
    for value in frame_ids:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise CorruptionContractError("frame_ids must contain integer physical ids")
        result.append(int(value))
    if result != sorted(set(result)):
        raise CorruptionContractError("frame_ids must be strictly increasing")
    return result


def _validate_request(corruption: str, severity: int) -> tuple[str, dict[str, Any]]:
    if not isinstance(corruption, str) or corruption not in CORRUPTION_TYPES:
        raise ValueError(f"unsupported corruption {corruption!r}; expected one of {CORRUPTION_TYPES}")
    if isinstance(severity, bool) or not isinstance(severity, (int, np.integer)):
        raise TypeError("severity must be an integer in [1, 5]")
    severity = int(severity)
    if severity not in SEVERITIES:
        raise ValueError("severity must be an integer in [1, 5]")
    return corruption, copy.deepcopy(PARAMETER_TABLE[corruption][severity])


def _derived_seed(query_id: str, seed: int, corruption: str, severity: int) -> int:
    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise TypeError("seed must be integer-like") from exc
    payload = f"{seed_int}\0{query_id}\0{corruption}\0{severity}".encode("utf-8")
    # NumPy's default_rng accepts uint32-range seeds across supported versions.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "shift_corruptions_v2 requires OpenCV for blur/JPEG operations; "
            "the module does not install or download it"
        ) from exc
    return cv2


def _clip_uint8(values: np.ndarray) -> np.ndarray:
    return np.rint(values).clip(0.0, 255.0).astype(np.uint8, copy=False)


def _low_light(raw: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    return _clip_uint8(raw.astype(np.float32) * float(params["factor"]))


def _short_side_realization(raw: np.ndarray) -> tuple[int, float]:
    short_side = int(min(raw.shape[1:3]))
    if short_side < 1:
        raise CorruptionContractError("blur realization requires a positive spatial short side")
    return short_side, short_side / float(REFERENCE_SHORT_SIDE)


def _realized_gaussian_parameters(raw: np.ndarray, params: Mapping[str, Any]) -> dict[str, Any]:
    short_side, scale = _short_side_realization(raw)
    sigma = float(params["sigma"]) * scale
    # Keep the same ceil(6*sigma), minimum-3, odd-kernel convention as the
    # established res224 corruption contract.
    kernel_size = max(3, int(math.ceil(sigma * 6.0)) | 1)
    return {
        "reference_short_side": REFERENCE_SHORT_SIDE,
        "input_short_side": short_side,
        "short_side_scale": float(scale),
        "realized_sigma": float(sigma),
        "realized_kernel_size": int(kernel_size),
    }


def _gaussian_blur(
    raw: np.ndarray,
    params: Mapping[str, Any],
    *,
    realized: Mapping[str, Any] | None = None,
) -> np.ndarray:
    cv2 = _cv2()
    if realized is None:
        realized = _realized_gaussian_parameters(raw, params)
    kernel = int(realized["realized_kernel_size"])
    sigma = float(realized["realized_sigma"])
    return np.stack(
        [cv2.GaussianBlur(frame, (kernel, kernel), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101) for frame in raw],
        axis=0,
    ).astype(np.uint8, copy=False)


def _realized_motion_parameters(raw: np.ndarray, params: Mapping[str, Any]) -> dict[str, Any]:
    short_side, scale = _short_side_realization(raw)
    requested_length = float(params["kernel_size"]) * scale
    kernel_size = max(3, int(round(requested_length)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return {
        "reference_short_side": REFERENCE_SHORT_SIDE,
        "input_short_side": short_side,
        "short_side_scale": float(scale),
        "requested_kernel_length": float(requested_length),
        "realized_kernel_size": int(kernel_size),
        "realized_kernel_length": int(kernel_size),
    }


def _motion_kernel(params: Mapping[str, Any], *, kernel_size: int | None = None) -> np.ndarray:
    cv2 = _cv2()
    kernel_size = int(params["kernel_size"] if kernel_size is None else kernel_size)
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.line(kernel, (0, center), (kernel_size - 1, center), 1.0, 1)
    angle = float(params["angle_deg"])
    if angle:
        matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size), flags=cv2.INTER_LINEAR)
    total = float(kernel.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise CorruptionContractError("motion kernel normalization failed")
    return kernel / total


def _motion_blur(
    raw: np.ndarray,
    params: Mapping[str, Any],
    *,
    realized: Mapping[str, Any] | None = None,
) -> np.ndarray:
    cv2 = _cv2()
    if realized is None:
        realized = _realized_motion_parameters(raw, params)
    kernel = _motion_kernel(params, kernel_size=int(realized["realized_kernel_size"]))
    return np.stack(
        [cv2.filter2D(frame, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101) for frame in raw],
        axis=0,
    ).astype(np.uint8, copy=False)


def _jpeg_compression(raw: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    cv2 = _cv2()
    quality = int(params["quality"])
    flags = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    frames: list[np.ndarray] = []
    for index, frame in enumerate(raw):
        # OpenCV's color codec convention is BGR, while this module's public
        # contract is RGB.  Convert on both sides so JPEG does not silently
        # swap red and blue channels in the returned RGB tensor.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", frame_bgr, flags)
        if not ok:
            raise CorruptionContractError(f"JPEG encoding failed at frame {index}")
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded_bgr is None or decoded_bgr.shape != frame.shape:
            raise CorruptionContractError(f"JPEG decoding changed frame geometry at frame {index}")
        decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        frames.append(np.asarray(decoded_rgb, dtype=np.uint8))
    return np.stack(frames, axis=0)


def _gaussian_noise(raw: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(
        loc=float(params["mean"]),
        scale=float(params["sigma"]),
        size=raw.shape,
    ).astype(np.float32)
    return _clip_uint8(raw.astype(np.float32) + noise)


def _minimum_retained_count(frame_count: int) -> int:
    return min(frame_count, 4)


def _with_temporal_guards(candidate: Sequence[int], frame_count: int) -> tuple[list[int], dict[str, Any]]:
    """Add endpoint/minimum guards and report any short-clip adjustment."""

    if frame_count < 1:
        raise CorruptionContractError("cannot retain frames from an empty clip")
    requested = sorted(set(int(value) for value in candidate))
    if any(value < 0 or value >= frame_count for value in requested):
        raise CorruptionContractError("temporal candidate is outside the physical frame range")
    guarded = set(requested)
    guarded.update((0, frame_count - 1))
    minimum = _minimum_retained_count(frame_count)
    if len(guarded) < minimum:
        # Evenly spaced additions make the fallback deterministic and do not
        # depend on a global RNG.  For T<4, retaining all T is the only valid
        # interpretation of the minimum-frame contract.
        evenly_spaced = np.rint(np.linspace(0, frame_count - 1, minimum)).astype(int).tolist()
        guarded.update(evenly_spaced)
    result = sorted(guarded)
    return result, {
        "requested_retained_count": len(requested),
        "actual_retained_count": len(result),
        "minimum_retained_frames": minimum,
        "minimum_four_possible": frame_count >= 4,
        "minimum_retained_frames_satisfied": len(result) >= minimum,
        "endpoint_guard_added": not {0, frame_count - 1}.issubset(set(requested)),
        "minimum_guard_added": len(result) > len(set(requested) | {0, frame_count - 1}),
        "short_clip_adjustment": frame_count < 4,
    }


def _regular_subsample(frame_count: int, stride: int) -> tuple[list[int], dict[str, Any]]:
    requested = list(range(0, frame_count, int(stride)))
    positions, guard = _with_temporal_guards(requested, frame_count)
    guard.update({"requested_stride": int(stride), "selection": "regular_stride"})
    return positions, guard


def _random_drop(frame_count: int, drop_fraction: float, rng: np.random.Generator) -> tuple[list[int], dict[str, Any]]:
    requested_drop = int(round(frame_count * float(drop_fraction)))
    max_drop = max(0, frame_count - _minimum_retained_count(frame_count))
    actual_drop = min(requested_drop, max_drop)
    interior = np.arange(1, max(1, frame_count - 1), dtype=np.int64)
    if actual_drop > len(interior):
        raise CorruptionContractError("frame-drop interior selection is impossible")
    if actual_drop:
        dropped = set(int(value) for value in rng.choice(interior, size=actual_drop, replace=False).tolist())
    else:
        dropped = set()
    requested = [index for index in range(frame_count) if index not in dropped]
    positions, guard = _with_temporal_guards(requested, frame_count)
    guard.update({
        "requested_drop_fraction": float(drop_fraction),
        "requested_drop_count": requested_drop,
        "actual_drop_count": frame_count - len(positions),
        "actual_drop_fraction": (frame_count - len(positions)) / float(frame_count),
        "selection": "random_interior",
    })
    return positions, guard


@dataclass(frozen=True)
class CorruptionResult:
    """Transformed raw pixels plus an auditable physical-grid receipt."""

    frames: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def retained_positions(self) -> list[int]:
        return list(self.metadata["retained_positions"])

    @property
    def retained_frame_ids(self) -> list[int]:
        return list(self.metadata["retained_frame_ids"])

    def as_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        result = dict(self.metadata)
        if include_frames:
            result["frames"] = self.frames
        return result


def apply_corruption(
    raw_rgb: np.ndarray,
    corruption: str,
    severity: int,
    *,
    query_id: str = "",
    seed: int = 0,
    frame_ids: Sequence[int] | None = None,
) -> CorruptionResult:
    """Apply one deterministic severity to raw RGB frames.

    Parameters
    ----------
    raw_rgb:
        Caller-supplied, unnormalized RGB array with shape ``[T,H,W,3]`` and
        dtype ``uint8``.  It is copied logically; the input is never mutated.
    corruption, severity:
        One of :data:`CORRUPTION_TYPES` and an integer from 1 through 5.
    query_id, seed:
        Together with operation/severity determine the local RNG stream for
        noise and frame dropping.  No global NumPy or Python RNG is touched.
    frame_ids:
        Optional strictly increasing physical IDs.  If omitted, positions
        ``0..T-1`` are used as physical IDs.
    """

    raw = _validate_raw(raw_rgb)
    corruption, params = _validate_request(corruption, severity)
    ids = _validate_frame_ids(frame_ids, int(raw.shape[0]))
    severity = int(severity)
    derived_seed = _derived_seed(str(query_id), int(seed), corruption, severity)
    rng = np.random.default_rng(derived_seed)
    frame_count = int(raw.shape[0])
    realized_parameters: dict[str, Any] = {}

    if corruption == "low_light":
        transformed = _low_light(raw, params)
        positions = list(range(frame_count))
        temporal_audit: dict[str, Any] = {"selection": "all_frames"}
    elif corruption == "gaussian_blur":
        realized_parameters = _realized_gaussian_parameters(raw, params)
        transformed = _gaussian_blur(raw, params, realized=realized_parameters)
        positions = list(range(frame_count))
        temporal_audit = {"selection": "all_frames"}
    elif corruption == "motion_blur":
        realized_parameters = _realized_motion_parameters(raw, params)
        transformed = _motion_blur(raw, params, realized=realized_parameters)
        positions = list(range(frame_count))
        temporal_audit = {"selection": "all_frames"}
    elif corruption == "jpeg_compression":
        transformed = _jpeg_compression(raw, params)
        positions = list(range(frame_count))
        temporal_audit = {"selection": "all_frames"}
    elif corruption == "gaussian_noise":
        transformed = _gaussian_noise(raw, params, rng)
        positions = list(range(frame_count))
        temporal_audit = {"selection": "all_frames"}
    elif corruption == "temporal_subsampling":
        positions, temporal_audit = _regular_subsample(frame_count, int(params["stride"]))
        transformed = raw[positions].copy()
    elif corruption == "frame_dropping":
        positions, temporal_audit = _random_drop(frame_count, float(params["drop_fraction"]), rng)
        transformed = raw[positions].copy()
    else:  # pragma: no cover - guarded by _validate_request
        raise AssertionError(corruption)

    transformed = np.ascontiguousarray(transformed, dtype=np.uint8)
    if transformed.ndim != 4 or transformed.shape[-1] != 3:
        raise CorruptionContractError("corruption changed raw frame rank/channel geometry")
    if not np.isfinite(transformed.astype(np.float32)).all():
        raise CorruptionContractError("corruption produced non-finite pixels")
    if positions[0] != 0 or positions[-1] != frame_count - 1:
        raise CorruptionContractError("temporal corruption failed first/last physical-frame guard")

    temporal_audit = dict(temporal_audit)
    temporal_audit.setdefault("requested_retained_count", frame_count)
    temporal_audit.setdefault("actual_retained_count", len(positions))
    temporal_audit.setdefault("minimum_retained_frames", _minimum_retained_count(frame_count))
    temporal_audit.setdefault("minimum_four_possible", frame_count >= 4)
    temporal_audit.setdefault("minimum_retained_frames_satisfied", len(positions) >= _minimum_retained_count(frame_count))
    temporal_audit.setdefault("endpoint_guard_added", False)
    temporal_audit.setdefault("minimum_guard_added", False)
    temporal_audit.setdefault("short_clip_adjustment", False)

    metadata: dict[str, Any] = {
        "schema_version": VERSION,
        "corruption": corruption,
        "severity": severity,
        "query_id": str(query_id),
        "seed": int(seed),
        "derived_seed": int(derived_seed),
        "parameters": {**copy.deepcopy(params), **copy.deepcopy(realized_parameters)},
        "realized_parameters": copy.deepcopy(realized_parameters),
        "input_dtype": str(raw.dtype),
        "output_dtype": str(transformed.dtype),
        "input_shape": list(raw.shape),
        "output_shape": list(transformed.shape),
        "input_sha256": hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
        "output_sha256": hashlib.sha256(transformed.tobytes(order="C")).hexdigest(),
        "retained_positions": list(positions),
        "retained_frame_ids": [ids[position] for position in positions],
        "input_frame_ids": list(ids),
        "physical_grid_preserved": True,
        "first_frame_preserved": positions[0] == 0,
        "last_frame_preserved": positions[-1] == frame_count - 1,
        "temporal_audit": temporal_audit,
        "rng": {
            "local_only": True,
            "operation_uses_rng": corruption in {"gaussian_noise", "frame_dropping"},
            "global_rng_mutated": False,
        },
        "labels_used": False,
        "gt_used": False,
        "decoded_video": False,
        "gpu_used": False,
        "input_contract": "caller-supplied raw RGB uint8; this module does not decode video",
    }
    return CorruptionResult(frames=transformed, metadata=metadata)


def apply(*args: Any, **kwargs: Any) -> CorruptionResult:
    """Short alias for :func:`apply_corruption`."""

    return apply_corruption(*args, **kwargs)


__all__ = [
    "VERSION",
    "REFERENCE_SHORT_SIDE",
    "SEVERITIES",
    "CORRUPTION_TYPES",
    "PARAMETER_TABLE",
    "CorruptionContractError",
    "CorruptionResult",
    "parameter_table",
    "parameter_table_sha256",
    "apply_corruption",
    "apply",
]
