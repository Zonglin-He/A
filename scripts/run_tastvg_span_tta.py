#!/usr/bin/env python3
"""Device-safe TA-STVG runtime and temporal-head-only adaptation runner.

This file is an integration shim around the untouched checkout in
``external/TA-STVG``.  It deliberately keeps all generated dataset links,
converted annotations, model-zoo links, reports, and caches below
``artifacts/tastvg_runtime``.  It never edits the official checkout and never
touches TubeDETR or ``vg_tta`` artifacts.

The official TA-STVG code has two small integration hazards for the current
workspace:

* ``models/language_model/__init__.py`` imports its optional LSTM module even
  when ``MODEL.USE_LSTM=False``.  The LSTM module imports torchtext, so this
  runner installs a package-shaped lazy shim before importing TA-STVG.  It is
  intentionally an error to request ``USE_LSTM=True`` through this runner.
* The official model uses paths relative to its current working directory for
  RoBERTa and Video-Swin.  ``prepare_runtime`` links the locally audited
  assets into a private ``model_zoo`` directory and ``load_cpu_model`` changes
  directory only for the duration of model construction/loading.

The adaptation API is intentionally explicit:

``frozen``
    Query inference without an optimizer or parameter update.
``ttaug``
    Query-time, no-parameter four-view ensemble: both official temporal
    offsets, with and without a horizontal flip.  This is the TTAug baseline,
    not an adaptation objective.
``ttaug-consistency``
    Optional ablation that uses the original/flipped support pair to minimize
    temporal boundary-logit consistency.  Only ``temp_embed.*`` is trainable.
``boundary-entropy``
    Minimize normalized independent start/end entropy on support clips.
    Only ``temp_embed.*`` is trainable.
``trajcal``
    Build a detached trajectory-identity temporal target from the captured
    query-conditioned decoder features, then update only ``temp_embed.*``
    with a temporal-target KL objective.
``span-tta``
    Minimize normalized entropy over strict legal spans ``start < end`` on
    support, then evaluate disjoint queries with the adapted head.

Every episode snapshots the complete model state (parameters and buffers),
creates a fresh optimizer, and restores the complete state in ``finally`` by
default.  This makes repeated episodes independent and gives the caller an
explicit ``reset_verified`` result.  The default is fail-safe CPU.  CUDA is
available only when the caller explicitly supplies ``--device cuda``; model
construction and checkpoint loading always happen on CPU before an explicit
device transfer.

Examples (run from the project root or with an absolute script path)::

  .conda/tubedetr/bin/python scripts/run_tastvg_span_tta.py \
      --dataset hcstvg2 --prepare-runtime --loader-smoke
  .conda/tubedetr/bin/python scripts/run_tastvg_span_tta.py \
      --dataset vidstg --prepare-runtime --loader-smoke
  .conda/tubedetr/bin/python scripts/run_tastvg_span_tta.py \
      --dataset hcstvg2 --load-model

The last command loads the released checkpoint on CPU only; it does not run a
model forward.  A CUDA run must opt in explicitly, for example
``--device cuda --run-mode trajcal``.  This file itself never starts an
experiment or chooses CUDA implicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = PROJECT_ROOT / "external" / "TA-STVG"
RUNTIME_ROOT = PROJECT_ROOT / "artifacts" / "tastvg_runtime"
RUNNER_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "tastvg_runner"
TORCH_HOME = PROJECT_ROOT / ".cache" / "torch"
HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
SWIN_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "tastvg_model_zoo"
    / "swin_tiny_patch244_window877_kinetics400_1k.pth"
)
ROBERTA_CACHE = HF_HOME / "hub" / "models--roberta-base"
RESNET_CACHE = TORCH_HOME / "hub" / "checkpoints" / "resnet101-cd907fc2.pth"

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "hcstvg2": {
        "name": "HC-STVG",
        "yaml": OFFICIAL_REPO / "experiments" / "hcstvg2.yaml",
        "source_root": PROJECT_ROOT / "data" / "hcstvg2_confirm512",
        "runtime_dirname": "hc-stvg2",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "TASTVG_HCSTVG2.pth",
        "video_subdir": "v2_video",
        "source_annotation": "annotations/valv2_proc.json",
        "loader_split": "test",
    },
    "vidstg": {
        "name": "VidSTG",
        "yaml": OFFICIAL_REPO / "experiments" / "vidstg.yaml",
        "source_root": PROJECT_ROOT / "data" / "vidstg_phase3_confirmation_A",
        "runtime_dirname": "vidstg",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "TASTVG_VidSTG.pth",
        "video_subdir": "videos",
        "source_annotation": "annotations/test.json",
        "loader_split": "test",
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading a released checkpoint into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(path: Path) -> str:
    return sha256_file(path)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_device(device: str | Any) -> Any:
    """Resolve an explicit CPU/CUDA request without implicit CUDA fallback.

    Keeping this check in one place is important for the preparation runner:
    the default CLI path must remain CPU-only, while an explicitly requested
    CUDA path should fail loudly when CUDA is unavailable instead of silently
    moving work back to CPU.  ``torch`` is imported lazily so loader and
    manifest-only commands do not initialize a CUDA runtime.
    """

    import torch

    if isinstance(device, torch.device):
        resolved = device
        requested = str(resolved)
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError(
                f"unsupported TA-STVG device {requested!r}; use cpu or cuda"
            )
    else:
        requested = str(device)
        if requested not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported TA-STVG device {requested!r}; use cpu or cuda")
        resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was explicitly requested, but torch.cuda.is_available() "
            "is false; no CPU fallback is performed"
        )
    if resolved.type == "cuda" and resolved.index is None:
        # Normalize ``cuda`` to the actual current device so equality checks
        # against tensors/parameters that report ``cuda:0`` are reliable.
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _ensure_cpu_device(device: str) -> str:
    """Backward-compatible guard for commands whose contract is CPU-only."""

    if str(device) != "cpu":
        raise RuntimeError(
            "this TA-STVG helper is CPU-only; pass the device-aware model/episode "
            "entry point for an explicit --device cuda run"
        )
    return "cpu"


def _model_device(model: Any) -> Any:
    """Return the single parameter device used by the complete model."""

    import torch

    devices = {parameter.device for parameter in model.parameters()}
    if not devices:
        return torch.device("cpu")
    if len(devices) != 1:
        raise RuntimeError(f"TA-STVG model spans multiple parameter devices: {sorted(map(str, devices))}")
    return next(iter(devices))


def _autocast_dtype(name: str | None, device: Any) -> Any | None:
    """Resolve an autocast dtype; CPU defaults to disabled regardless of name."""

    import torch

    if device.type != "cuda" or name in (None, "none", "off"):
        return None
    values = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return values[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported autocast dtype {name!r}; use float16, bfloat16, or none"
        ) from exc


@contextlib.contextmanager
def _autocast_context(
    device: Any,
    *,
    enabled: bool | None = None,
    dtype: str | None = "float16",
) -> Iterator[None]:
    """Use autocast only for an explicitly requested CUDA execution."""

    import torch

    use_autocast = bool(device.type == "cuda" if enabled is None else enabled)
    resolved_dtype = _autocast_dtype(dtype, device)
    if not use_autocast:
        yield
        return
    if resolved_dtype is None and str(dtype).lower() in {"none", "off"}:
        yield
        return
    if device.type != "cuda" or resolved_dtype is None:
        raise ValueError("autocast is enabled but no supported CUDA dtype was selected")
    with torch.autocast(device_type="cuda", dtype=resolved_dtype, enabled=True):
        yield


def _make_grad_scaler(device: Any, *, enabled: bool, dtype: Any | None) -> Any | None:
    """Create a CUDA GradScaler only for fp16 CUDA adaptation."""

    if not enabled or device.type != "cuda" or dtype is None:
        return None
    import torch

    if dtype is not torch.float16:
        return None
    # torch.amp.GradScaler is the current API; retain the old spelling for
    # compatibility with the isolated torch 2.0.1 audit environment.
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _move_to_device(value: Any, device: Any) -> Any:
    """Recursively move tensors and official tensor-like containers.

    Text strings, ids, durations, and scalar metadata intentionally stay as
    Python values.  ``NestedTensor`` and ``BoxList`` both expose ``to`` and
    are handled without importing their implementation classes.
    """

    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return type(value)((key, _move_to_device(item, device)) for key, item in value.items())
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if hasattr(value, "to") and callable(value.to):
        try:
            return value.to(device)
        except (AttributeError, TypeError, RuntimeError):
            # Non-tensor objects with an incidental ``to`` method are metadata;
            # do not hide a true tensor/device error from normal torch values.
            if value.__class__.__module__.startswith(("torch", "utils")):
                raise
    return value


def _batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Move model inputs while retaining expensive metadata on host memory.

    The official target stores ``orignal_frame`` (the unresized decoded clip)
    for evaluation/debugging.  It is not consumed by ``TASTVGNet.forward``;
    copying it to a CUDA device for every one of K=64 support clips would
    needlessly exhaust VRAM.  Small target tensors used by the official
    forward are moved, while ids, frame lists, original frames, and size
    metadata remain host-side by design.
    """

    if not isinstance(batch, Mapping):
        raise TypeError("TA-STVG collated batch must be a mapping")
    moved = dict(batch)
    if "videos" in moved:
        moved["videos"] = _move_to_device(moved["videos"], device)
    targets = moved.get("targets")
    if isinstance(targets, list):
        host_metadata = {
            "orignal_frame",
            "frame_ids",
            "vid",
            "item_id",
            "ori_size",
            "img_size",
        }
        moved_targets: list[Any] = []
        for target in targets:
            if not isinstance(target, Mapping):
                moved_targets.append(target)
                continue
            moved_targets.append(
                {
                    key: value
                    if key in host_metadata
                    else _move_to_device(value, device)
                    for key, value in target.items()
                }
            )
        moved["targets"] = moved_targets
    # Text, durations, and all other collator metadata are already valid Python
    # values.  Avoid a broad recursive pass that would move hidden debug data.
    return moved


def _prepend_path(path: Path) -> None:
    entries = os.environ.get("PATH", "").split(os.pathsep)
    value = str(path.resolve())
    if value not in entries:
        os.environ["PATH"] = os.pathsep.join([value, *[x for x in entries if x]])


def install_optional_torchtext_shim() -> dict[str, Any]:
    """Install the import-only torchtext shim required by TA-STVG's package.

    ``models.language_model.__init__`` imports ``RNNEncoder`` regardless of
    the configured text model.  ``lstm.py`` only needs torchtext when an LSTM
    is actually instantiated, so an empty package module is safe for the
    default RoBERTa path.  Returning provenance makes this workaround visible
    in every runtime report.
    """

    shimmed = False
    module = sys.modules.get("torchtext")
    if module is None:
        module = types.ModuleType("torchtext")
        module.__path__ = []  # type: ignore[attr-defined]
        module.__package__ = "torchtext"
        module.__spec__ = None
        sys.modules["torchtext"] = module
        shimmed = True
    return {
        "module": "torchtext",
        "shimmed": shimmed,
        "supported_use_lstm": False,
        "reason": (
            "TA-STVG imports models.language_model.lstm eagerly; the default "
            "RoBERTa path never dereferences torchtext."
        ),
    }


def official_imports() -> dict[str, Any]:
    """Import official modules after installing the optional dependency shim."""

    if not OFFICIAL_REPO.is_dir():
        raise FileNotFoundError(f"official TA-STVG checkout missing: {OFFICIAL_REPO}")
    install_optional_torchtext_shim()
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    # These imports are intentionally lazy.  The staging/annotation commands
    # can run in an environment that has no Torch installation at all.
    from config import cfg as upstream_cfg  # type: ignore[import-not-found]
    from datasets.build import build_dataset, build_transforms  # type: ignore[import-not-found]
    from datasets.collate_batch import collate_fn  # type: ignore[import-not-found]
    from models import build_model  # type: ignore[import-not-found]
    from models.post_processor import PostProcess  # type: ignore[import-not-found]
    from utils.checkpoint import VSTGCheckpointer  # type: ignore[import-not-found]

    return {
        "cfg": upstream_cfg,
        "build_dataset": build_dataset,
        "build_transforms": build_transforms,
        "collate_fn": collate_fn,
        "build_model": build_model,
        "postprocessor": PostProcess,
        "checkpointer": VSTGCheckpointer,
    }


def _ensure_link(destination: Path, source: Path) -> None:
    """Create a non-destructive private symlink to an existing source."""

    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise FileExistsError(
            f"private runtime link points elsewhere: {destination} -> "
            f"{destination.resolve()} (wanted {source})"
        )
    if destination.exists():
        if destination.resolve() == source:
            return
        raise FileExistsError(
            f"refusing to overwrite existing runtime path: {destination}"
        )
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextlib.contextmanager
def _trusted_runtime_torch_load() -> Iterator[None]:
    """Keep upstream cache reads working with PyTorch >=2.6.

    TA-STVG calls ``torch.load(cache_path)`` without ``weights_only``.  Newer
    PyTorch releases default that argument to true, which rejects the NumPy
    arrays in the cache generated by the official loader.  The cache lives in
    our private runtime directory and is generated only from the audited
    source annotations, so explicitly opting into the legacy unpickler is
    scoped to dataset construction and is recorded by the runner instead of
    changing the global torch installation.
    """

    import torch

    original_load = torch.load

    def _load_with_legacy_cache_support(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        try:
            return original_load(*args, **kwargs)
        except TypeError as exc:
            # ``weights_only`` was added after the isolated torch 2.0.1
            # audit prefix.  Keep that prefix usable for dataset-only checks
            # without weakening the current torch 2.7 path above.
            if "weights_only" not in str(exc):
                raise
            kwargs.pop("weights_only", None)
            return original_load(*args, **kwargs)

    torch.load = _load_with_legacy_cache_support  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_load  # type: ignore[assignment]


def _normalize_hc_trajectory(
    record: Mapping[str, Any]
) -> tuple[list[list[float]], str, dict[str, int]]:
    """Normalize mixed legacy HC boxes to TA's ``xywh`` convention.

    The supplied HC-STVG2 confirmation archive contains both legacy forms in
    its trajectory arrays: most clips use ``[xmin, ymin, width, height]``,
    while some high-resolution clips use ``[xmin, ymin, xmax, ymax]``.  The
    official TA loader assumes the former and asserts image bounds.  Infer a
    single form per source video from the number of impossible boxes, convert
    to a bounded ``xywh`` list, and record the inference rather than silently
    allowing the upstream assertion to fail.
    """

    width, height = int(record["width"]), int(record["height"])
    raw = [list(map(float, box)) for box in record["trajectory"]]

    def wh_valid(box: Sequence[float]) -> bool:
        x, y, w, h = box
        return (
            w >= 0
            and h >= 0
            and x <= width
            and y <= height
            and x + w >= 0
            and y + h >= 0
            and x + w <= width
            and y + h <= height
        )

    def xyxy_valid(box: Sequence[float]) -> bool:
        x1, y1, x2, y2 = box
        return (
            x1 <= width
            and y1 <= height
            and x2 >= 0
            and y2 >= 0
            and x2 >= x1
            and y2 >= y1
            and x2 <= width
            and y2 <= height
        )

    wh_invalid = sum(not wh_valid(box) for box in raw)
    xyxy_invalid = sum(not xyxy_valid(box) for box in raw)
    if wh_invalid < xyxy_invalid:
        source_format = "xywh"
    elif xyxy_invalid < wh_invalid:
        source_format = "xyxy"
    else:
        # The public HC-STVG schema and the official TA/TubeDETR readers name
        # this field xywh.  Use that documented convention for an exact tie.
        source_format = "xywh"

    normalized: list[list[float]] = []
    for box in raw:
        x, y, a, b = box
        if source_format == "xyxy":
            x2, y2 = a, b
        else:
            x2, y2 = x + a, y + b
        x1 = min(float(width), max(0.0, x))
        y1 = min(float(height), max(0.0, y))
        x2 = min(float(width), max(x1, x2))
        y2 = min(float(height), max(y1, y2))
        normalized.append([x1, y1, x2 - x1, y2 - y1])
    return normalized, source_format, {
        "raw_box_count": len(raw),
        "xywh_invalid_count": wh_invalid,
        "xyxy_invalid_count": xyxy_invalid,
    }


def _hc_annotation_item(record: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, int]]:
    """Convert the existing TubeDETR HC record to TA-STVG's source schema."""

    frame_count = int(record["frame_count"])
    tube_start = int(record["tube_start_frame"])
    source_tube_end_exclusive = int(record["tube_end_frame"])
    # The pinned official HC loader constructs ``frame_ids`` with
    # ``range(0, frame_count - 1)``.  Its effective timeline therefore ends
    # at ``frame_count - 2`` even though the source annotation can contain a
    # final box at ``frame_count - 1``.  Keeping that last box produces one
    # more bbox than active actionness entry for every tube that reaches the
    # physical final frame.  Match the loader's actual timeline explicitly
    # and record the trim in the runtime manifest.
    official_frame_end_exclusive = max(frame_count - 1, 0)
    tube_end_exclusive = min(
        source_tube_end_exclusive,
        official_frame_end_exclusive,
    )
    trajectory, source_bbox_format, bbox_stats = _normalize_hc_trajectory(record)
    if not (0 <= tube_start < tube_end_exclusive <= frame_count):
        raise ValueError(
            f"HC tube interval is outside the source video: {record.get('video_path')} "
            f"[{tube_start}, {source_tube_end_exclusive}) vs frame_count={frame_count}"
        )
    # A small number of existing HC proc records contain one extra trajectory
    # box at frame_count.  TA's loader clamps the inclusive end to
    # frame_count-1, so trim the corresponding one-past box here and record
    # the source boundary in the manifest instead of allowing its assertion to
    # fail.
    trajectory = trajectory[: tube_end_exclusive - tube_start]
    if len(trajectory) != tube_end_exclusive - tube_start:
        raise ValueError(
            "HC annotation trajectory length does not match its exclusive "
            f"interval for {record.get('video_path')}: "
            f"{len(trajectory)} != {tube_end_exclusive - tube_start}"
        )
    width, height = int(record["width"]), int(record["height"])
    return {
        "vid": str(record["video_path"]),
        "width": width,
        "height": height,
        "img_size": [height, width, 3],
        "img_num": frame_count,
        # TA's HCSTVGDataset subtracts one from st_frame.  The TubeDETR
        # record is zero-based, therefore this conversion is +1 exactly.
        "st_frame": tube_start + 1,
        "st_time": float(record.get("tube_start_time", 0.0)),
        "ed_time": float(record.get("tube_end_time", 0.0)),
        "ed_offset": float(record.get("tube_end_time", 0.0)),
        "English": str(record.get("caption", "")),
        "caption": str(record.get("caption", "")),
        "bbox": trajectory,
        "ta_bbox_source_format": source_bbox_format,
        "source_tube_end_frame": source_tube_end_exclusive,
        "ta_tube_end_frame_exclusive": tube_end_exclusive,
        "ta_official_frame_end_exclusive": official_frame_end_exclusive,
        # The official model indexes these fields during forward even in
        # eval.  The prepared HC source has no verb/adjective labels, so use
        # explicit empty label lists rather than fabricating class ids.
        "sub": "person",
        "verb_index_list": [],
        "adj_index_list": [],
    }, source_bbox_format, bbox_stats


def prepare_hc_runtime(source_root: Path, runtime_dir: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    source_annotation = source_root / "annotations" / "valv2_proc.json"
    if not source_annotation.is_file():
        raise FileNotFoundError(f"HC source annotation missing: {source_annotation}")
    records = json.loads(source_annotation.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("HC source annotation must be a non-empty list")

    source_video_dir = source_root / "video"
    if not source_video_dir.is_dir():
        raise FileNotFoundError(f"HC source video directory missing: {source_video_dir}")
    converted: dict[str, dict[str, Any]] = {}
    format_counts: dict[str, int] = {"xywh": 0, "xyxy": 0}
    format_stats: dict[str, dict[str, int]] = {}
    trimmed_tube_count = 0
    missing_videos: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("HC source annotation contains a non-object record")
        video_name = str(record["video_path"])
        if not (source_video_dir / video_name).is_file():
            missing_videos.append(video_name)
        item, source_bbox_format, bbox_stats = _hc_annotation_item(record)
        if item["source_tube_end_frame"] != item["ta_tube_end_frame_exclusive"]:
            trimmed_tube_count += 1
        format_counts[source_bbox_format] = format_counts.get(source_bbox_format, 0) + 1
        format_stats[video_name] = {
            "source_format": source_bbox_format,
            **bbox_stats,
        }
        converted[video_name] = item
    if missing_videos:
        raise FileNotFoundError(
            f"HC source is missing {len(missing_videos)} videos, first={missing_videos[:3]}"
        )

    runtime_dir.mkdir(parents=True, exist_ok=True)
    _ensure_link(runtime_dir / "v2_video", source_video_dir)
    train_path = runtime_dir / "annos" / "train.json"
    test_path = runtime_dir / "annos" / "test.json"
    _write_json(train_path, converted)
    _write_json(test_path, converted)
    manifest = {
        "schema_version": 1,
        "dataset": "hcstvg2",
        "ta_dataset_name": "HC-STVG",
        "source_root": str(source_root),
        "source_annotation": str(source_annotation),
        "source_annotation_sha256": _json_sha256(source_annotation),
        "source_record_count": len(records),
        "runtime_dir": str(runtime_dir.resolve()),
        "runtime_annotation": str(test_path.resolve()),
        "runtime_annotation_sha256": _json_sha256(test_path),
        "runtime_video_link": str((runtime_dir / "v2_video").resolve()),
        "coordinate_semantics": {
            "source": "TubeDETR HC-STVG2 proc records",
            "frame_origin": "zero_based",
            "source_tube_end": "exclusive",
            "ta_effective_frame_end": (
                "exclusive frame_count-1 because pinned official HC loader "
                "uses range(0, frame_count-1)"
            ),
            "ta_st_frame": "one_based; TA preprocess subtracts one",
            "bbox": "[xmin, ymin, width, height] per tube frame",
        },
        "label_semantics": (
            "sub=person and empty verb_index_list/adj_index_list are explicit "
            "runtime placeholders; they are not evaluation labels."
        ),
        "bbox_format_inference": {
            "source_format_counts": format_counts,
            "per_video": format_stats,
            "trimmed_source_tube_end_count": trimmed_tube_count,
            "normalization": (
                "Each source trajectory is inferred as one legacy form by "
                "minimizing impossible image-bound boxes, converted to xywh, "
                "then clipped to [0,width]x[0,height]."
            ),
        },
    }
    _write_json(runtime_dir / "manifest.json", manifest)
    return manifest


def _vid_bbox_for_frame(
    trajectory: Mapping[str, Any], frame_id: int, width: int, height: int
) -> dict[str, float]:
    value = trajectory.get(str(frame_id))
    if not isinstance(value, dict) or "bbox" not in value:
        raise ValueError(f"VidSTG trajectory missing target bbox at frame {frame_id}")
    x, y, box_width, box_height = (float(v) for v in value["bbox"])
    if x < 0 or y < 0 or box_width < 0 or box_height < 0:
        raise ValueError(f"negative VidSTG bbox at frame {frame_id}: {value}")
    if x + box_width > width + 1e-4 or y + box_height > height + 1e-4:
        raise ValueError(
            f"VidSTG bbox exceeds image at frame {frame_id}: {value} "
            f"for {width}x{height}"
        )
    return {
        "xmin": x,
        "ymin": y,
        "xmax": x + box_width,
        "ymax": y + box_height,
    }


def _vid_annotation_item(record: Mapping[str, Any], trajectories: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a VidSTG confirmation record to TA's already-merged schema."""

    original_video_id = str(record["original_video_id"])
    target_id = str(int(record["target_id"]))
    target_trajectories = trajectories.get(original_video_id, {}).get(target_id)
    if not isinstance(target_trajectories, dict):
        raise ValueError(
            f"missing trajectory for video={original_video_id}, target={target_id}"
        )

    # The confirmation/TubesDETR records carry boundary fields produced from
    # VidSTG's ``*_fid`` annotations.  The upstream TubeDETR path treats
    # these as an end boundary, while TA's direct merged format and loader use
    # inclusive ends.  Some source clips end at ``frame_count - 1`` and some
    # source temporal tubes end at ``frame_count``; resolve both safely to a
    # valid inclusive TA interval below.
    segment_start = int(record["start_frame"])
    source_segment_end = int(record["end_frame"])
    tube_start = int(record["tube_start_frame"])
    source_tube_end = int(record["tube_end_frame"])
    source_frame_count = int(record["frame_count"])
    segment_end = min(source_segment_end, source_frame_count - 1)
    tube_end = min(source_tube_end - 1, segment_end)
    if not (0 <= segment_start <= segment_end):
        raise ValueError(f"invalid VidSTG segment: {record}")
    if not (segment_start <= tube_start <= tube_end):
        raise ValueError(
            "VidSTG tube must be inside its segment for this runtime conversion: "
            f"{record}"
        )
    width, height = int(record["width"]), int(record["height"])
    target_bboxs = [
        _vid_bbox_for_frame(target_trajectories, frame_id, width, height)
        for frame_id in range(tube_start, tube_end + 1)
    ]
    qtype = str(record.get("qtype", "declarative"))
    qtype = "declar" if qtype in {"declarative", "caption", "declar"} else "inter"
    video_path = Path(str(record["video_path"]))
    video_stem = video_path.with_suffix("").as_posix()
    return {
        "vid": video_stem,
        "fps": float(record.get("fps", 0.0)),
        "used_segment": {"begin_fid": segment_start, "end_fid": segment_end},
        "width": width,
        "height": height,
        "frame_count": segment_end - segment_start + 1,
        "ori_temp_gt": {"begin_fid": tube_start, "end_fid": tube_end},
        "temp_gt": {
            "begin_fid": tube_start - segment_start,
            "end_fid": tube_end - segment_start,
        },
        "id": int(record["video_id"]),
        "qtype": qtype,
        "sentence": {
            "description": str(record["caption"]),
            "target_id": int(record["target_id"]),
        },
        "target_category": "person",
        "target_bboxs": target_bboxs,
        # These are read by TASTVGNet to prepend a target subject.  The source
        # confirmation file has no noun-class annotation, so keep this
        # conservative and record it in the manifest as a placeholder.
        "sub": "person",
        "verb_index_list": [],
        "adj_index_list": [],
        "source_video_path": str(record["video_path"]),
        "source_original_video_id": original_video_id,
    }


def prepare_vid_runtime(source_root: Path, runtime_dir: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    source_annotation = source_root / "annotations" / "test.json"
    if not source_annotation.is_file():
        raise FileNotFoundError(f"VidSTG source annotation missing: {source_annotation}")
    source = json.loads(source_annotation.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("videos"), list):
        raise ValueError("VidSTG source annotation must contain a videos list")
    trajectories = source.get("trajectories")
    if not isinstance(trajectories, dict):
        raise ValueError("VidSTG source annotation must contain trajectories")

    source_video_dir = source_root / "video"
    if not source_video_dir.is_dir():
        raise FileNotFoundError(f"VidSTG source video directory missing: {source_video_dir}")
    converted: dict[str, dict[str, Any]] = {}
    missing_videos: list[str] = []
    for record in source["videos"]:
        if not isinstance(record, dict):
            raise TypeError("VidSTG source annotation contains a non-object record")
        relative_video = Path(str(record["video_path"]))
        if not (source_video_dir / relative_video).is_file():
            missing_videos.append(str(relative_video))
        item = _vid_annotation_item(record, trajectories)
        converted[str(item["id"])] = item
    if missing_videos:
        raise FileNotFoundError(
            f"VidSTG source is missing {len(missing_videos)} videos, "
            f"first={missing_videos[:3]}"
        )

    runtime_dir.mkdir(parents=True, exist_ok=True)
    _ensure_link(runtime_dir / "videos", source_video_dir)
    train_path = runtime_dir / "annos" / "train.json"
    test_path = runtime_dir / "annos" / "test.json"
    _write_json(train_path, converted)
    _write_json(test_path, converted)
    segment_lengths = [int(x["frame_count"]) for x in converted.values()]
    manifest = {
        "schema_version": 1,
        "dataset": "vidstg",
        "ta_dataset_name": "VidSTG",
        "source_root": str(source_root),
        "source_annotation": str(source_annotation),
        "source_annotation_sha256": _json_sha256(source_annotation),
        "source_record_count": len(source["videos"]),
        "runtime_dir": str(runtime_dir.resolve()),
        "runtime_annotation": str(test_path.resolve()),
        "runtime_annotation_sha256": _json_sha256(test_path),
        "runtime_video_link": str((runtime_dir / "videos").resolve()),
        "segment_frame_count_min": min(segment_lengths),
        "segment_frame_count_max": max(segment_lengths),
        "coordinate_semantics": {
            "source": "VidSTG confirmation/TubesDETR merged records",
            "source_segment_end": (
                "TubeDETR boundary field; source annotations use both the "
                "last valid frame and a one-past boundary"
            ),
            "source_tube_end": "one-past boundary for the trajectory list",
            "ta_used_segment_end": "inclusive min(source end, frame_count - 1)",
            "ta_ori_temp_gt_end": "inclusive min(source tube end - 1, segment end)",
            "bbox_source": "[xmin, ymin, width, height]",
            "bbox_ta": "{xmin, ymin, xmax, ymax}",
        },
        "label_semantics": (
            "sub=person and empty verb_index_list/adj_index_list are explicit "
            "runtime placeholders; they are not evaluation labels."
        ),
    }
    _write_json(runtime_dir / "manifest.json", manifest)
    return manifest


def prepare_model_zoo(runtime_root: Path) -> dict[str, Any]:
    """Link audited model assets into a private cwd-relative model_zoo."""

    if not SWIN_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Video-Swin checkpoint missing: {SWIN_CHECKPOINT}")
    if not ROBERTA_CACHE.is_dir():
        raise FileNotFoundError(f"RoBERTa cache missing: {ROBERTA_CACHE}")
    ref_path = ROBERTA_CACHE / "refs" / "main"
    if not ref_path.is_file():
        raise FileNotFoundError(f"RoBERTa cache revision marker missing: {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    roberta_snapshot = ROBERTA_CACHE / "snapshots" / revision
    required_roberta = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    missing = [name for name in required_roberta if not (roberta_snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"RoBERTa snapshot {roberta_snapshot} is missing {missing}"
        )
    model_zoo = runtime_root / "model_zoo"
    model_zoo.mkdir(parents=True, exist_ok=True)
    _ensure_link(
        model_zoo / "swin_tiny_patch244_window877_kinetics400_1k.pth",
        SWIN_CHECKPOINT,
    )
    _ensure_link(model_zoo / "roberta-base", roberta_snapshot)
    _ensure_link(model_zoo / "roberta", roberta_snapshot)
    return {
        "runtime_model_zoo": str(model_zoo.resolve()),
        "swin": {
            "path": _relative(SWIN_CHECKPOINT),
            "size_bytes": SWIN_CHECKPOINT.stat().st_size,
            "sha256": sha256_file(SWIN_CHECKPOINT),
            "url": (
                "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/"
                "swin_tiny_patch244_window877_kinetics400_1k.pth"
            ),
        },
        "roberta": {
            "cache": _relative(ROBERTA_CACHE),
            "snapshot": str(roberta_snapshot.resolve()),
            "revision": revision,
            "required_files": required_roberta,
        },
        "resnet101": {
            "path": _relative(RESNET_CACHE) if RESNET_CACHE.is_file() else None,
            "size_bytes": RESNET_CACHE.stat().st_size if RESNET_CACHE.is_file() else None,
            "sha256": sha256_file(RESNET_CACHE) if RESNET_CACHE.is_file() else None,
            "url": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            "status": "cached" if RESNET_CACHE.is_file() else "missing; torchvision may download",
        },
    }


def _dataset_spec(dataset: str) -> dict[str, Any]:
    try:
        return DATASET_DEFAULTS[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown TA-STVG dataset: {dataset}") from exc


def prepare_runtime(
    dataset: str,
    source_root: Path | None = None,
    runtime_root: Path = RUNTIME_ROOT,
) -> dict[str, Any]:
    spec = _dataset_spec(dataset)
    source = (source_root or spec["source_root"]).resolve()
    runtime_dir = runtime_root.resolve() / str(spec["runtime_dirname"])
    if dataset == "hcstvg2":
        manifest = prepare_hc_runtime(source, runtime_dir)
    else:
        manifest = prepare_vid_runtime(source, runtime_dir)
    manifest["official_repository"] = str(OFFICIAL_REPO.resolve())
    manifest["official_repository_url"] = "https://github.com/HengLan/TA-STVG"
    manifest["official_repository_commit"] = _git_commit(OFFICIAL_REPO)
    _write_json(runtime_dir / "manifest.json", manifest)
    return manifest


def _git_commit(path: Path) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def runtime_path(dataset: str, runtime_root: Path = RUNTIME_ROOT) -> Path:
    return runtime_root.resolve() / str(_dataset_spec(dataset)["runtime_dirname"])


def _set_random_seed(seed: int, device: Any | None = None) -> None:
    """Seed CPU RNGs and, only for an explicit CUDA run, CUDA RNGs."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if device is not None and getattr(device, "type", str(device)) == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_runtime_config(
    dataset: str,
    data_dir: Path,
    *,
    resolution: int,
    sample_frames: int,
    use_model_defaults: bool = False,
) -> Any:
    """Build a frozen official config with only private runtime overrides."""

    imports = official_imports()
    cfg = imports["cfg"].clone()
    spec = _dataset_spec(dataset)
    cfg.merge_from_file(str(spec["yaml"]))
    cfg.DATA_DIR = str(data_dir.resolve())
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.USE_LSTM = False
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.BATCH_SIZE = 1
    cfg.INPUT.RESOLUTION = int(resolution)
    if not use_model_defaults:
        if dataset == "vidstg":
            cfg.INPUT.TRAIN_SAMPLE_NUM = int(sample_frames)
        else:
            # HC-STVG's test loader samples at 2*SAMPLE_FPS over its fixed
            # 20-second clips.  Expose the same small-smoke control as
            # VidSTG: SAMPLE_FPS=sample_frames/40 yields approximately
            # ``sample_frames`` frames while preserving the official loader.
            cfg.INPUT.SAMPLE_FPS = max(float(sample_frames) / 40.0, 0.05)
    cfg.OUTPUT_DIR = str((RUNNER_ARTIFACT_ROOT / dataset).resolve())
    cfg.TENSORBOARD_DIR = str((RUNNER_ARTIFACT_ROOT / dataset / "tensorboard").resolve())
    cfg.freeze()
    return cfg


def _load_manifest(runtime_dir: Path) -> dict[str, Any]:
    path = runtime_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"runtime manifest missing: {path}; run --prepare-runtime first"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"runtime manifest must be an object: {path}")
    return value


def _batch_for_dataset_item(
    dataset_object: Any,
    index: int,
    collate_fn: Any,
) -> dict[str, Any]:
    import torch

    if not (0 <= index < len(dataset_object)):
        raise IndexError(f"dataset index {index} outside [0, {len(dataset_object)})")
    subset = torch.utils.data.Subset(dataset_object, [index])
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))
    # Official visualization-only unresized RGB frames are never consumed by
    # the model, temporal postprocessor, or corrected metric implementation.
    for target in batch["targets"]:
        target.pop("orignal_frame", None)
    # Preserve the un-subsampled official record for metric postprocessing.
    # The model never sees this private runner field; it lets corrected vIoU
    # use the official dense frame timeline and ground-truth tube after the
    # model input has been temporally sampled to 2*SAMPLE_NUM frames.
    batch["_tastvg_full_record"] = dataset_object.all_gt_data[index]
    return batch


def _shortest_distinct_indices(dataset_object: Any, count: int = 2) -> list[int]:
    """Select deterministic short clips from real loader records for smoke."""

    rows: list[tuple[int, int, str]] = []
    for index, item in enumerate(dataset_object.all_gt_data):
        rows.append(
            (
                len(item.get("frame_ids", [])),
                index,
                str(item.get("vid", "")),
            )
        )
    rows.sort()
    selected: list[int] = []
    selected_vids: set[str] = set()
    for _length, index, vid in rows:
        if vid in selected_vids:
            continue
        selected.append(index)
        selected_vids.add(vid)
        if len(selected) >= count:
            break
    if len(selected) < count:
        selected = [row[1] for row in rows[:count]]
    if not selected:
        raise RuntimeError("TA-STVG runtime dataset is empty")
    return selected


def loader_smoke(
    dataset: str,
    runtime_dir: Path,
    *,
    resolution: int = 64,
    sample_frames: int = 8,
    seed: int = 20260904,
) -> dict[str, Any]:
    """Exercise the official real-video dataset and collate path on CPU."""

    _ensure_cpu_device("cpu")
    # The current project prefix intentionally has no ffmpeg CLI.  Reuse the
    # private conda runtime's ffmpeg binary without changing either conda env.
    _prepend_path(PROJECT_ROOT / ".conda" / "tastvg" / "bin")
    _set_random_seed(seed)
    imports = official_imports()
    cfg = build_runtime_config(
        dataset,
        runtime_dir,
        resolution=resolution,
        sample_frames=sample_frames,
    )
    transforms = imports["build_transforms"](cfg, is_train=False)
    with _trusted_runtime_torch_load():
        dataset_object = imports["build_dataset"](cfg, "test", transforms)
    selected = _shortest_distinct_indices(dataset_object, count=2)
    batch = _batch_for_dataset_item(dataset_object, selected[0], imports["collate_fn"])
    videos = batch["videos"]
    target = batch["targets"][0]
    return {
        "status": "passed",
        "device": "cpu",
        "official_dataset_name": str(cfg.DATASET.NAME),
        "runtime_dir": str(runtime_dir.resolve()),
        "dataset_length": len(dataset_object),
        "selected_indices": selected,
        "selected_item_ids": [
            int(dataset_object.all_gt_data[i]["item_id"]) if str(dataset_object.all_gt_data[i]["item_id"]).isdigit() else str(dataset_object.all_gt_data[i]["item_id"])
            for i in selected
        ],
        "selected_videos": [str(dataset_object.all_gt_data[i]["vid"]) for i in selected],
        "batch_video_tensor_shape": list(videos.tensors.shape),
        "batch_video_mask_shape": list(videos.mask.shape),
        "durations": list(batch["durations"]),
        "text": batch["texts"][0],
        "target_keys": sorted(target.keys()),
        "target_actioness_length": int(target["actioness"].numel()),
        "config": {
            "data_dir": str(cfg.DATA_DIR),
            "resolution": int(cfg.INPUT.RESOLUTION),
            "train_sample_num": int(cfg.INPUT.TRAIN_SAMPLE_NUM),
            "num_workers": int(cfg.DATALOADER.NUM_WORKERS),
            "model_device": str(cfg.MODEL.DEVICE),
        },
        "torchtext_shim": install_optional_torchtext_shim(),
        "ffmpeg_cli": shutil_which("ffmpeg"),
    }


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


@contextlib.contextmanager
def _temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _cpu_only_torch_guard() -> Iterator[None]:
    """Neutralize the official constructor's unconditional CUDA cleanup call.

    ``models/vidswin/video_swin_transformer.py`` calls
    ``torch.cuda.empty_cache()`` after loading its CPU checkpoint.  That call
    is unnecessary for CPU-first construction and can initialize a CUDA
    context on some PyTorch builds, so replace it with a scoped no-op while
    constructing and loading the model.  Explicit CUDA transfer/forward code
    runs only after this guard is restored.
    """

    import torch

    original_empty_cache = torch.cuda.empty_cache
    torch.cuda.empty_cache = lambda: None  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.cuda.empty_cache = original_empty_cache  # type: ignore[assignment]


def _checkpoint_for(dataset: str, checkpoint: Path | None) -> Path:
    path = (checkpoint or _dataset_spec(dataset)["checkpoint"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TA-STVG checkpoint missing: {path}")
    return path


def _infer_checkpoint_source(checkpoint_path: Path) -> str:
    """Identify a released source dataset when the path names one.

    An explicit arbitrary checkpoint is reported as ``custom`` rather than
    guessing its architecture.  The caller can still provide
    ``--source-dataset`` to make the provenance unambiguous.
    """

    resolved = checkpoint_path.resolve()
    for name, spec in DATASET_DEFAULTS.items():
        if resolved == Path(spec["checkpoint"]).resolve():
            return name
    lowered = resolved.name.lower()
    if "hcstvg" in lowered or "hc-stvg" in lowered:
        return "hcstvg2"
    if "vidstg" in lowered:
        return "vidstg"
    return "custom"


def _load_checkpoint_cpu_compat(
    model: Any,
    checkpoint_path: Path,
    *,
    source_dataset: str | None = None,
    target_dataset: str | None = None,
) -> dict[str, Any]:
    """Strictly load a released TA checkpoint with one known HF shim.

    A released checkpoint must be loaded into the architecture it was
    trained with.  In particular, HC-STVG2 and VidSTG have different
    ``APP_NUM``/``MOT_NUM`` spatial classifier dimensions; constructing the
    model from the *target* YAML and skipping those tensors would silently
    random-initialize the spatial/query guidance heads and invalidate a
    natural-shift comparison.  The caller therefore builds this model from
    ``source_dataset`` and this function rejects every missing key, unexpected
    key, and shape mismatch except the deterministic
    ``RobertaModel.embeddings.position_ids`` persistence difference introduced
    by modern Transformers.
    """

    import torch

    with _trusted_runtime_torch_load():
        checkpoint = torch.load(
            str(checkpoint_path), map_location=torch.device("cpu"), weights_only=False
        )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"TA checkpoint must be a mapping: {checkpoint_path}")
    model_key = "model_ema" if "model_ema" in checkpoint else "model"
    if model_key not in checkpoint or not isinstance(checkpoint[model_key], Mapping):
        raise KeyError(f"TA checkpoint has no mapping under {model_key!r}: {checkpoint_path}")
    state_dict = checkpoint[model_key]
    current = model.state_dict()
    current_keys = set(current)
    checkpoint_keys = set(state_dict)
    unexpected = sorted(checkpoint_keys - current_keys)
    missing = sorted(current_keys - checkpoint_keys)
    allowed_deterministic_extras = {
        "text_encoder.body.embeddings.position_ids",
    }
    ignored: list[str] = []
    if unexpected:
        if not set(unexpected).issubset(allowed_deterministic_extras):
            raise RuntimeError(
                "TA checkpoint has unexpected state keys outside the known "
                f"Transformers compatibility shim: {unexpected}"
            )
        ignored = unexpected
    if missing:
        raise RuntimeError(
            "TA checkpoint is missing model state keys; refusing partial load: "
            f"{missing[:20]} (total={len(missing)})"
        )

    shape_mismatches: list[str] = []
    for key in sorted(current_keys & checkpoint_keys):
        current_value = current[key]
        checkpoint_value = state_dict[key]
        current_shape = tuple(getattr(current_value, "shape", ()))
        checkpoint_shape = tuple(getattr(checkpoint_value, "shape", ()))
        if current_shape != checkpoint_shape:
            shape_mismatches.append(
                f"{key}: checkpoint{checkpoint_shape} != model{current_shape}"
            )
    if shape_mismatches:
        raise RuntimeError(
            "TA checkpoint/model tensor shapes differ; build the model from "
            "the source dataset architecture instead of partially loading "
            "target-shaped heads: "
            f"{shape_mismatches[:20]}"
        )
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current_keys
    }
    result = model.load_state_dict(filtered, strict=False)
    unexpected_after = list(result.unexpected_keys)
    missing_after = list(result.missing_keys)
    if missing_after or unexpected_after:
        raise RuntimeError(
            "strict TA checkpoint load returned incompatible keys: "
            f"missing={missing_after}, unexpected={unexpected_after}"
        )
    # Release the second copy of the ~2 GB checkpoint before the caller
    # starts an optional loader/episode smoke.
    del checkpoint, state_dict, filtered
    return {
        "loader": "strict_source_architecture_with_known_transformers_extra_filter",
        "model_key": model_key,
        "checkpoint_state_key_count": len(checkpoint_keys),
        "model_state_key_count": len(current_keys),
        "ignored_deterministic_extra_keys": ignored,
        "missing_keys": [],
        "unexpected_keys_before_filter": unexpected,
        "shape_mismatch_keys": shape_mismatches,
        "skipped_cross_dataset_shape_keys": [],
        "cross_dataset_partial_spatial_load": False,
        "strict_all_model_weights": True,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
    }


def _model_parameter_device_report(
    model: Any, *, expected_device: Any | None = None
) -> dict[str, Any]:
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    expected = None if expected_device is None else str(expected_device)
    return {
        "devices": devices,
        "expected_device": expected,
        "all_cpu": devices == ["cpu"],
        "all_on_expected_device": expected is None or devices == [expected],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def exact_temporal_head_parameters(
    model: Any, *, include_actionness: bool = False
) -> tuple[list[str], int]:
    """Return the only parameter names permitted by strict head-only TTA."""

    expected_boundary = {
        "temp_embed.layers.0.weight",
        "temp_embed.layers.0.bias",
        "temp_embed.layers.1.weight",
        "temp_embed.layers.1.bias",
    }
    expected_action = {
        "action_embed.layers.0.weight",
        "action_embed.layers.0.bias",
        "action_embed.layers.1.weight",
        "action_embed.layers.1.bias",
    }
    names = {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("temp_embed.")
        or (include_actionness and name.startswith("action_embed."))
    }
    expected = expected_boundary | (expected_action if include_actionness else set())
    if names != expected:
        raise RuntimeError(
            "TA-STVG temporal head parameter layout changed; refusing a loose "
            f"optimizer allowlist. expected={sorted(expected)}, found={sorted(names)}"
        )
    count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in expected
    )
    return sorted(names), count


def set_temporal_head_only(model: Any, *, include_actionness: bool = False) -> dict[str, Any]:
    names, count = exact_temporal_head_parameters(
        model, include_actionness=include_actionness
    )
    allowed = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    actual_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if sorted(actual_trainable) != names:
        raise RuntimeError(
            "strict temporal-head allowlist failed: "
            f"actual={sorted(actual_trainable)}, expected={names}"
        )
    return {
        "allowed_parameter_names": names,
        "trainable_scalar_count": count,
        "include_actionness": include_actionness,
        "upstream_decoder_allowed": False,
        "all_other_parameters_frozen": True,
    }


def snapshot_full_model_state(model: Any) -> dict[str, Any]:
    """Clone all model parameters and buffers to CPU for episodic reset."""

    snapshot: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        if hasattr(value, "detach"):
            snapshot[name] = value.detach().cpu().clone()
        else:
            snapshot[name] = copy.deepcopy(value)
    return snapshot


def snapshot_module_training_modes(model: Any) -> dict[str, bool]:
    """Capture every module's train/eval flag for an episodic reset."""

    return {
        name: bool(module.training)
        for name, module in model.named_modules()
    }


def snapshot_requires_grad(model: Any) -> dict[str, bool]:
    """Capture parameter gradient-enable flags before temporary head freezing."""

    return {
        name: bool(parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }


def restore_module_training_modes(
    model: Any, modes: Mapping[str, bool]
) -> None:
    """Restore module train/eval flags without changing state tensors."""

    by_name = dict(model.named_modules())
    if set(by_name) != set(modes):
        raise RuntimeError(
            "module topology changed during episodic adaptation; "
            f"missing={sorted(set(modes) - set(by_name))[:10]}, "
            f"unexpected={sorted(set(by_name) - set(modes))[:10]}"
        )
    for name, module in by_name.items():
        module.training = bool(modes[name])


def restore_requires_grad(model: Any, flags: Mapping[str, bool]) -> None:
    """Restore all parameter ``requires_grad`` flags after a reset."""

    by_name = dict(model.named_parameters())
    if set(by_name) != set(flags):
        raise RuntimeError(
            "parameter topology changed during episodic adaptation; "
            f"missing={sorted(set(flags) - set(by_name))[:10]}, "
            f"unexpected={sorted(set(by_name) - set(flags))[:10]}"
        )
    for name, parameter in by_name.items():
        parameter.requires_grad_(bool(flags[name]))


def module_training_modes_equal(model: Any, modes: Mapping[str, bool]) -> bool:
    return snapshot_module_training_modes(model) == dict(modes)


def requires_grad_equal(model: Any, flags: Mapping[str, bool]) -> bool:
    return snapshot_requires_grad(model) == dict(flags)


def restore_full_model_state(model: Any, snapshot: Mapping[str, Any]) -> None:
    """Restore a complete state dict and fail on architecture drift."""

    result = model.load_state_dict(dict(snapshot), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "full-state reset was not strict: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def full_state_equal(model: Any, snapshot: Mapping[str, Any]) -> bool:
    current = model.state_dict()
    if set(current) != set(snapshot):
        return False
    for name, value in current.items():
        expected = snapshot[name]
        if hasattr(value, "detach"):
            if not bool(value.detach().cpu().equal(expected)):
                return False
        elif value != expected:
            return False
    return True


def _valid_mask_from_batch(batch: Mapping[str, Any], time_length: int, device: Any) -> Any:
    import torch

    durations = [int(value) for value in batch["durations"]]
    mask = torch.zeros((len(durations), time_length), dtype=torch.bool, device=device)
    for index, duration in enumerate(durations):
        mask[index, : min(duration, time_length)] = True
    return mask


def _forward_with_temporal_trace(
    model: Any,
    batch: Mapping[str, Any],
    *,
    iteration_rate: int = 0,
    device: Any | None = None,
    autocast_enabled: bool | None = None,
    autocast_dtype: str | None = "float16",
) -> tuple[MutableMapping[str, Any], Any]:
    """Run TA forward and capture query-conditioned per-frame features.

    The collated batch is copied onto the complete model's device before the
    call.  ``iteration_rate=0`` keeps the trace to one decoder invocation;
    callers that need the released model's refinement behavior can pass ``-1``
    explicitly.  Autocast is opt-in for CPU and defaults to enabled only on an
    explicit CUDA device.
    """

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.prepare_tastvg import capture_query_conditioned_temporal_features

    if device is None:
        device = _model_device(model)
    elif not hasattr(device, "type"):
        device = _resolve_device(device)
    batch = _batch_to_device(batch, device)
    videos = batch["videos"]
    texts = batch["texts"]
    targets = batch["targets"]
    if not hasattr(videos, "tensors") or videos.tensors.device != device:
        raise RuntimeError(
            "TA-STVG batch/video device mismatch: "
            f"videos={getattr(getattr(videos, 'tensors', None), 'device', None)}, "
            f"model={device}"
        )
    autocast_requested = bool(
        device.type == "cuda" if autocast_enabled is None else autocast_enabled
    )
    resolved_dtype = _autocast_dtype(autocast_dtype, device)
    autocast_effective = bool(autocast_requested and resolved_dtype is not None)
    with _autocast_context(
        device, enabled=autocast_enabled, dtype=autocast_dtype
    ):
        with capture_query_conditioned_temporal_features(model) as captured:
            outputs = model(videos, texts, targets, iteration_rate=iteration_rate)
    if not captured:
        raise RuntimeError("TA-STVG temporal feature hook captured no decoder call")
    hidden = captured[-1]
    per_frame = hidden[-1]
    if outputs["pred_sted"].shape[:2] != per_frame.shape[:2]:
        raise RuntimeError(
            "query-conditioned feature/logit time dimensions disagree: "
            f"features={tuple(per_frame.shape)}, logits={tuple(outputs['pred_sted'].shape)}"
        )
    trace = {
        "decoder_invocation_count": len(captured),
        "all_layer_features_shape": list(hidden.shape),
        "final_query_conditioned_per_frame_shape": list(per_frame.shape),
        "final_per_frame_features": per_frame,
        "device": str(device),
        "autocast_enabled": autocast_effective,
        "autocast_dtype": None if resolved_dtype is None else str(resolved_dtype),
    }
    return outputs, trace


def _temporal_entropy_loss(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> tuple[Any, dict[str, float]]:
    """Span entropy retained for query diagnostics and the Span-TTA objective."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.prepare_tastvg import valid_span_entropy

    logits = outputs["pred_sted"]
    valid_mask = _valid_mask_from_batch(batch, logits.shape[1], logits.device)
    return valid_span_entropy(logits, valid_mask)


def _boundary_entropy_loss(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> tuple[Any, dict[str, float]]:
    """Use the same normalized independent endpoint baseline as TubeDETR."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from vg_tta.tta import temporal_entropy_loss

    logits = outputs["pred_sted"]
    valid_mask = _valid_mask_from_batch(batch, logits.shape[1], logits.device)
    return temporal_entropy_loss(dict(outputs), time_mask=valid_mask)


def _trajectory_calibration_loss(
    outputs: Mapping[str, Any],
    trace: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    trajectory_weight: float,
    trajectory_temperature: float,
    trajectory_top_k: int | None,
) -> tuple[Any, dict[str, float]]:
    """Return the detached trajectory-target KL objective for one clip.

    TA-STVG's temporal head is the only student path here.  The trajectory
    identity evidence is computed from the detached final decoder features
    captured immediately before ``temp_embed``; it therefore cannot create a
    gradient path into the visual, language, encoder, or decoder modules.
    """

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from vg_tta.phase3 import temporal_probabilities, trajectory_temporal_target
    from vg_tta.tta import temporal_target_kl_loss

    logits = outputs["pred_sted"]
    features = trace["final_per_frame_features"]
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("TA-STVG TrajCal currently requires a single support clip (B=1)")
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError("captured TA-STVG temporal features must have shape 1xTxH")
    probabilities = temporal_probabilities(logits)
    target_probability, diagnostics = trajectory_temporal_target(
        probabilities,
        features[0],
        trajectory_weight=float(trajectory_weight),
        temperature=float(trajectory_temperature),
        top_k=trajectory_top_k,
    )
    valid_mask = _valid_mask_from_batch(batch, logits.shape[1], logits.device)
    loss, loss_info = temporal_target_kl_loss(
        dict(outputs), target_probability, time_mask=valid_mask
    )
    return loss, {
        **diagnostics,
        **loss_info,
        "objective": "trajectory_target_kl",
        "target_shape": list(target_probability.shape),
        "target_device": str(target_probability.device),
        "features_shape": list(features.shape),
        "features_device": str(features.device),
    }


def _ttaug_consistency_loss(
    model: Any,
    batch: Mapping[str, Any],
    *,
    device: Any | None = None,
    autocast_enabled: bool | None = None,
    autocast_dtype: str | None = "float16",
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    original, original_trace = _forward_with_temporal_trace(
        model,
        batch,
        device=device,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
    )
    batch = _batch_to_device(batch, device or _model_device(model))
    videos = batch["videos"]
    # Horizontal flip is a true frame-space augmentation.  Temporal logits
    # should remain invariant, while spatial target boxes are deliberately not
    # consumed by this consistency objective.
    flipped_videos = type(videos)(
        torch.flip(videos.tensors, dims=[3]), videos.mask, videos.durations
    )
    flipped_batch = dict(batch)
    flipped_batch["videos"] = flipped_videos
    flipped, flipped_trace = _forward_with_temporal_trace(
        model,
        flipped_batch,
        device=device,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
    )
    loss = F.mse_loss(original["pred_sted"], flipped["pred_sted"])
    return (
        loss,
        {
            "ttaug_consistency_mse": float(loss.detach().item()),
        },
        {
            "original": original_trace,
            "flipped": flipped_trace,
        },
    )


def _slice_temporal_metadata(value: Any, offset: int) -> Any:
    """Slice a target field in the same way as ``NestedTensor.subsample``."""

    if value is None:
        return None
    if hasattr(value, "ndim") and hasattr(value, "__getitem__"):
        try:
            return value[offset::2]
        except (IndexError, TypeError):
            return value
    if isinstance(value, (list, tuple)):
        return value[offset::2]
    return value


def _make_temporal_view_batch(
    batch: Mapping[str, Any], *, offset: int, flip: bool
) -> dict[str, Any]:
    """Create one official evaluation view without mutating the source batch.

    TA-STVG's evaluator evaluates the two temporal half-rate views separately
    and merges them after ``PostProcess``.  The four-view TTAug baseline adds a
    horizontal flip to each of those two views.  Only the model input and the
    frame-aligned target metadata are changed; the original ``boxs`` object is
    retained because the official forward path does not consume it.
    """

    import torch

    if offset not in {0, 1}:
        raise ValueError("TA-STVG temporal view offset must be 0 or 1")
    videos = batch["videos"]
    if not hasattr(videos, "subsample"):
        raise TypeError("TA-STVG batch videos must provide NestedTensor.subsample")
    sampled = videos.subsample(2, start_idx=offset)
    if not sampled.durations or int(sampled.durations[0]) < 1:
        raise ValueError(
            "cannot create a non-empty offset view for a clip shorter than two frames"
        )
    if flip:
        sampled = type(sampled)(
            torch.flip(sampled.tensors, dims=[3]), sampled.mask, sampled.durations
        )

    targets = batch.get("targets")
    if not isinstance(targets, list) or len(targets) != 1:
        raise ValueError("TA-STVG view ensemble currently requires one target (B=1)")
    target = dict(targets[0])
    original_frame_ids = list(target.get("frame_ids", []))
    frame_ids = original_frame_ids[offset::2]
    duration = int(sampled.durations[0])
    if len(frame_ids) != duration:
        raise RuntimeError(
            "temporal view frame ids and video duration disagree: "
            f"offset={offset}, frame_ids={len(frame_ids)}, duration={duration}"
        )
    target["frame_ids"] = frame_ids
    for key in ("actioness", "start_heatmap", "end_heatmap", "orignal_frame"):
        if key in target:
            target[key] = _slice_temporal_metadata(target[key], offset)
    return {
        "durations": list(sampled.durations),
        "videos": sampled,
        "texts": list(batch["texts"]),
        "targets": [target],
    }


def _target_original_size(target: Mapping[str, Any]) -> tuple[int, int]:
    """Return TA target ``ori_size`` as ``(height, width)``."""

    value = target.get("ori_size")
    if value is None or len(value) != 2:
        raise ValueError(f"TA-STVG target has no two-dimensional ori_size: {value!r}")
    return int(value[0]), int(value[1])


def _postprocess_temporal_view(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    postprocessor: Any,
    *,
    offset: int,
    flip: bool,
) -> dict[str, Any]:
    """Apply the pinned official ``models.post_processor.PostProcess``."""

    import torch

    targets = batch["targets"]
    target = targets[0]
    frame_ids = [int(value) for value in target["frame_ids"]]
    durations = [int(value) for value in batch["durations"]]
    if len(durations) != 1 or durations[0] != len(frame_ids):
        raise ValueError("official TA-STVG postprocessing requires B=1 aligned frames")
    height, width = _target_original_size(target)
    # The official model flattens its final spatial boxes to [B*T,4], so the
    # postprocessor expects one original image size per sampled frame.
    target_sizes = torch.tensor(
        [[height, width] for _ in range(durations[0])],
        dtype=torch.float32,
        device=outputs["pred_boxes"].device,
    )
    pred_boxes, pred_att, pred_steds, pred_kf = postprocessor(
        outputs, target_sizes, [frame_ids], durations
    )
    boxes_abs = pred_boxes.reshape(1, durations[0], 4)[0].detach()
    if flip:
        unflipped = boxes_abs.clone()
        unflipped[:, 0] = float(width) - boxes_abs[:, 2]
        unflipped[:, 2] = float(width) - boxes_abs[:, 0]
        boxes_abs = unflipped
    raw_boxes = outputs["pred_boxes"].reshape(1, durations[0], 4)[0].detach()
    if flip:
        raw_boxes = raw_boxes.clone()
        raw_boxes[:, 0] = 1.0 - raw_boxes[:, 0]
    temporal_logits = outputs["pred_sted"][0].detach()
    sted = [int(pred_steds[0][0]), int(pred_steds[0][1])]
    try:
        start_idx = frame_ids.index(sted[0])
        end_idx = frame_ids.index(sted[1] - 1)
    except ValueError as exc:
        raise RuntimeError(
            "official postprocessor returned a span outside the view frame ids: "
            f"sted={sted}, frame_ids={frame_ids}"
        ) from exc
    return {
        "offset": int(offset),
        "flip": bool(flip),
        "frame_ids": frame_ids,
        "boxes_abs": boxes_abs,
        "raw_boxes": raw_boxes,
        "temporal_logits": temporal_logits,
        "pred_att": pred_att,
        "pred_kf": pred_kf,
        "pred_sted": sted,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "outputs": outputs,
    }


def _merge_postprocessed_views(
    view_records: Sequence[Mapping[str, Any]],
    full_frame_ids: Sequence[int],
) -> dict[str, Any]:
    """Merge official postprocessed views frame-by-frame and temporally."""

    import torch

    if not view_records:
        raise ValueError("at least one TA-STVG postprocessed view is required")
    ordered_frame_ids = [int(value) for value in full_frame_ids]
    global_frame_index = {frame_id: index for index, frame_id in enumerate(ordered_frame_ids)}
    expected = set(ordered_frame_ids)
    box_by_frame: dict[int, list[Any]] = {}
    raw_by_frame: dict[int, list[Any]] = {}
    for record in view_records:
        frame_ids = [int(value) for value in record["frame_ids"]]
        boxes_abs = record["boxes_abs"]
        raw_boxes = record["raw_boxes"]
        if len(frame_ids) != len(boxes_abs) or len(frame_ids) != len(raw_boxes):
            raise RuntimeError("TA-STVG view boxes and frame ids are not aligned")
        for local_index, frame_id in enumerate(frame_ids):
            box_by_frame.setdefault(frame_id, []).append(boxes_abs[local_index])
            raw_by_frame.setdefault(frame_id, []).append(raw_boxes[local_index])
    missing = sorted(expected - set(box_by_frame))
    if missing:
        raise RuntimeError(
            "TA-STVG temporal views did not cover every full-resolution frame: "
            f"missing={missing[:10]}"
        )
    merged_boxes = torch.stack(
        [torch.stack(box_by_frame[frame_id]).float().mean(dim=0) for frame_id in ordered_frame_ids]
    )
    merged_raw_boxes = torch.stack(
        [torch.stack(raw_by_frame[frame_id]).float().mean(dim=0) for frame_id in ordered_frame_ids]
    )
    predicted_start = min(
        global_frame_index[int(record["pred_sted"][0])] for record in view_records
    )
    predicted_end = max(
        global_frame_index[int(record["pred_sted"][1]) - 1] for record in view_records
    )
    if predicted_start >= predicted_end:
        raise RuntimeError(
            "merged TA-STVG postprocessed span is not strict start<end: "
            f"({predicted_start}, {predicted_end})"
        )
    return {
        "predicted_indices": (predicted_start, predicted_end),
        "predicted_frame_span": [
            int(ordered_frame_ids[predicted_start]),
            int(ordered_frame_ids[predicted_end]) + 1,
        ],
        "boxes_abs": merged_boxes,
        "raw_boxes": merged_raw_boxes,
        "temporal_logits": torch.stack(
            [
                torch.stack(
                    [
                        record["temporal_logits"][local_index]
                        for record in view_records
                        for local_index, frame_id in enumerate(record["frame_ids"])
                        if int(frame_id) == int(ordered_frame_id)
                    ]
                ).float().mean(dim=0)
                for ordered_frame_id in ordered_frame_ids
            ]
        ),
        "view_metadata": [
            {
                "offset": int(record["offset"]),
                "flip": bool(record["flip"]),
                "frame_count": len(record["frame_ids"]),
                "predicted_sted": list(record["pred_sted"]),
            }
            for record in view_records
        ],
        "postprocessor": "external/TA-STVG/models/post_processor.py:PostProcess",
        "temporal_merge": "minimum start and maximum end over official views",
        "spatial_merge": "mean of unflipped official absolute xyxy boxes by frame id",
    }


def _metric_target_sequence(
    batch: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[int, int], list[int], list[int]]:
    """Convert an official target to ``vg_tta.metrics`` frame targets."""

    import torch

    target = batch["targets"][0]
    frame_ids = [int(value) for value in target["frame_ids"]]
    actioness = target.get("actioness")
    if isinstance(actioness, torch.Tensor):
        active = torch.nonzero(actioness.detach().cpu().flatten().bool(), as_tuple=False).flatten()
    else:
        active = torch.nonzero(torch.as_tensor(actioness).flatten().bool(), as_tuple=False).flatten()
    if active.numel() == 0:
        raise ValueError("TA-STVG metric target has no active actioness frames")
    if int(active[-1]) >= len(frame_ids):
        raise ValueError("TA-STVG actioness is longer than its frame id sequence")
    box_list = target.get("boxs")
    if box_list is None:
        raise ValueError("TA-STVG metric target has no boxs field")
    if hasattr(box_list, "bbox"):
        boxes = box_list.bbox.detach().float().cpu()
        box_mode = getattr(box_list, "mode", "xywh")
        if box_mode == "xyxy":
            # The official Normalize transform normally leaves ``boxs`` in
            # normalized cxcywh (the BoxList calls this mode ``xywh``).  This
            # fallback keeps the metric adapter explicit if a custom loader
            # supplies pixel xyxy boxes.
            size = getattr(box_list, "size", None)
            if size is None:
                raise ValueError("xyxy BoxList metric target has no image size")
            width, height = float(size[0]), float(size[1])
            x1, y1, x2, y2 = boxes.unbind(-1)
            boxes = torch.stack(
                ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                 (x2 - x1) / width, (y2 - y1) / height),
                dim=-1,
            )
    else:
        boxes = torch.as_tensor(box_list, dtype=torch.float32).cpu()
    if len(boxes) != int(active.numel()):
        raise ValueError(
            "TA-STVG target box/actionness counts disagree: "
            f"boxes={len(boxes)}, active_frames={int(active.numel())}"
        )
    frame_targets: list[dict[str, Any]] = [{} for _ in frame_ids]
    for box_index, frame_index in enumerate(active.tolist()):
        frame_targets[int(frame_index)] = {"boxes": boxes[box_index : box_index + 1]}
    gt_indices = (int(active[0]), int(active[-1]))
    gt_frame_interval = [frame_ids[gt_indices[0]], frame_ids[gt_indices[1]] + 1]
    return frame_targets, gt_indices, frame_ids, gt_frame_interval


def _absolute_boxes_to_normalized_cxcywh(
    boxes_abs: Any, target: Mapping[str, Any]
) -> Any:
    import torch

    height, width = _target_original_size(target)
    boxes = boxes_abs.detach().float()
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        ((x1 + x2) / (2.0 * float(width)), (y1 + y2) / (2.0 * float(height)),
         (x2 - x1) / float(width), (y2 - y1) / float(height)),
        dim=-1,
    )


def _dense_official_metric_result(
    batch: Mapping[str, Any], ensemble: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Build metric inputs on the official dense source-frame timeline.

    ``TA-STVG`` evaluates the two half-rate views after ``PostProcess``,
    merges their predictions by frame id, and then calls
    ``engine.evaluate.linear_interp`` before handing the dense boxes to its
    dataset evaluator.  The model input in this runner is intentionally
    sampled (``sample_frames`` is 64 in the formal protocol and 8 in smoke),
    so computing vIoU directly on those sparse samples would make the union
    denominator depend on the sampling rate.  ``_batch_for_dataset_item``
    preserves the original ``all_gt_data`` record under a private key; this
    adapter uses that record only for evaluation metadata and never feeds it
    to the model.

    Return ``None`` when a caller supplied a synthetic/manual batch without a
    full record.  Real official-loader batches have the record and therefore
    take this dense path; the sampled path below remains useful for the
    standalone synthetic postprocessing tests.
    """

    import torch

    full_record = batch.get("_tastvg_full_record")
    if not isinstance(full_record, Mapping):
        return None

    target = batch.get("targets", [{}])[0]
    if not isinstance(target, Mapping):
        raise TypeError("TA-STVG batch target must be a mapping")
    full_frame_ids_raw = full_record.get("frame_ids")
    full_actioness_raw = full_record.get("actioness")
    full_bboxs_raw = full_record.get("bboxs")
    if full_frame_ids_raw is None or full_actioness_raw is None or full_bboxs_raw is None:
        raise ValueError(
            "TA-STVG full metric record must contain frame_ids, actioness, and bboxs"
        )
    full_frame_ids = [int(value) for value in full_frame_ids_raw]
    if not full_frame_ids:
        raise ValueError("TA-STVG full metric record has an empty frame timeline")
    if full_frame_ids != sorted(set(full_frame_ids)):
        raise ValueError("TA-STVG full metric frame ids must be sorted and unique")
    # The official interpolation helper assumes a dense integer source-frame
    # timeline.  The prepared VidSTG/HC-STVG2 input caches satisfy this by
    # construction; fail loudly if a future conversion violates it.
    expected_frame_ids = list(range(full_frame_ids[0], full_frame_ids[-1] + 1))
    if full_frame_ids != expected_frame_ids:
        raise ValueError(
            "TA-STVG full metric frame ids are not a dense integer timeline: "
            f"first={full_frame_ids[0]}, last={full_frame_ids[-1]}, "
            f"count={len(full_frame_ids)}"
        )

    full_actioness = torch.as_tensor(full_actioness_raw, dtype=torch.bool).flatten().cpu()
    if full_actioness.numel() != len(full_frame_ids):
        raise ValueError(
            "TA-STVG full metric actioness/frame-id lengths disagree: "
            f"actioness={full_actioness.numel()}, frame_ids={len(full_frame_ids)}"
        )
    active = torch.nonzero(full_actioness, as_tuple=False).flatten()
    if active.numel() == 0:
        raise ValueError("TA-STVG full metric record has no active actioness frames")

    if isinstance(full_bboxs_raw, Mapping):
        # This is the shape used by the official anno cache.  The input cache
        # normally stores the equivalent dense active-frame ndarray, but
        # accepting the mapping keeps this adapter useful for direct evaluator
        # records as well.
        box_values: list[Any] = []
        for frame_index in active.tolist():
            frame_id = full_frame_ids[int(frame_index)]
            value = full_bboxs_raw.get(frame_id, full_bboxs_raw.get(str(frame_id)))
            if value is None:
                raise ValueError(
                    f"TA-STVG full metric bbox mapping has no frame {frame_id}"
                )
            box_values.append(value)
        full_boxes_abs = torch.as_tensor(box_values, dtype=torch.float32).cpu()
    else:
        full_boxes_abs = torch.as_tensor(full_bboxs_raw, dtype=torch.float32).cpu()
    if full_boxes_abs.ndim == 1 and full_boxes_abs.numel() == 4:
        full_boxes_abs = full_boxes_abs.reshape(1, 4)
    if tuple(full_boxes_abs.shape) != (int(active.numel()), 4):
        raise ValueError(
            "TA-STVG full metric boxes must be one pixel xyxy box per active "
            f"frame: boxes={tuple(full_boxes_abs.shape)}, active={int(active.numel())}"
        )
    if not bool(torch.isfinite(full_boxes_abs).all()):
        raise ValueError("TA-STVG full metric boxes contain non-finite values")

    sampled_frame_ids = [int(value) for value in target.get("frame_ids", [])]
    sampled_boxes_abs = ensemble["boxes_abs"].detach().float().cpu()
    if len(sampled_frame_ids) != len(sampled_boxes_abs):
        raise ValueError(
            "TA-STVG sampled prediction boxes/frame ids disagree: "
            f"boxes={len(sampled_boxes_abs)}, frame_ids={len(sampled_frame_ids)}"
        )
    full_frame_set = set(full_frame_ids)
    if len(set(sampled_frame_ids)) != len(sampled_frame_ids) or not set(
        sampled_frame_ids
    ).issubset(full_frame_set):
        raise ValueError("TA-STVG sampled prediction frame ids are not in the full timeline")
    if len(sampled_frame_ids) < 2:
        raise ValueError("TA-STVG dense metric interpolation needs at least two samples")

    # Keep the exact upstream interpolation implementation in the metric
    # provenance.  It mutates its input mapping, so construct a private plain
    # Python copy from CPU postprocessed absolute xyxy boxes.
    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))
    from engine.evaluate import linear_interp  # type: ignore[import-not-found]

    sampled_bbox_dict = {
        frame_id: [box.tolist()]
        for frame_id, box in zip(sampled_frame_ids, sampled_boxes_abs)
    }
    interpolated = linear_interp(sampled_bbox_dict)
    if set(interpolated) != full_frame_set:
        raise RuntimeError(
            "official TA-STVG linear_interp did not cover the full source "
            f"timeline: interpolated={len(interpolated)}, full={len(full_frame_ids)}, "
            f"sampled_range=({min(sampled_frame_ids)}, {max(sampled_frame_ids)}), "
            f"full_range=({full_frame_ids[0]}, {full_frame_ids[-1]})"
        )
    dense_boxes_abs = torch.as_tensor(
        [interpolated[frame_id][0] for frame_id in full_frame_ids],
        dtype=torch.float32,
    )
    height = int(full_record.get("height"))
    width = int(full_record.get("width"))
    if height <= 0 or width <= 0:
        raise ValueError(f"TA-STVG full metric record has invalid size {width}x{height}")
    size_target = {"ori_size": (height, width)}
    dense_pred_boxes = _absolute_boxes_to_normalized_cxcywh(
        dense_boxes_abs, size_target
    )
    dense_target_boxes = _absolute_boxes_to_normalized_cxcywh(
        full_boxes_abs, size_target
    )
    frame_targets: list[dict[str, Any]] = [{} for _ in full_frame_ids]
    for box_index, frame_index in enumerate(active.tolist()):
        frame_targets[int(frame_index)] = {
            "boxes": dense_target_boxes[box_index : box_index + 1]
        }

    full_frame_index = {frame_id: index for index, frame_id in enumerate(full_frame_ids)}
    predicted_frame_span = ensemble.get("predicted_frame_span")
    if not isinstance(predicted_frame_span, (list, tuple)) or len(predicted_frame_span) != 2:
        raise ValueError(
            "TA-STVG ensemble is missing a two-element predicted_frame_span"
        )
    predicted_start_frame = int(predicted_frame_span[0])
    predicted_end_exclusive = int(predicted_frame_span[1])
    if predicted_start_frame not in full_frame_index:
        raise ValueError(
            f"predicted span starts outside the full timeline: {predicted_start_frame}"
        )
    predicted_end_frame = predicted_end_exclusive - 1
    if predicted_end_frame not in full_frame_index:
        raise ValueError(
            "predicted span end is not an inclusive source-frame id: "
            f"end_exclusive={predicted_end_exclusive}, full_range="
            f"({full_frame_ids[0]}, {full_frame_ids[-1]})"
        )
    pred_indices = (
        full_frame_index[predicted_start_frame],
        full_frame_index[predicted_end_frame],
    )
    gt_indices = (int(active[0]), int(active[-1]))
    gt_frame_interval = [
        full_frame_ids[gt_indices[0]],
        full_frame_ids[gt_indices[1]] + 1,
    ]

    from vg_tta.metrics import compute_stvg_metrics

    metrics = compute_stvg_metrics(
        dense_pred_boxes,
        frame_targets,
        pred_indices,
        gt_indices,
        frame_ids=full_frame_ids,
        gt_frame_interval=gt_frame_interval,
    )
    return (
        {str(key): value for key, value in metrics.items()},
        {
            "metric_timeline": "full dense source frame ids from official all_gt_data",
            "spatial_interpolation": "external/TA-STVG/engine/evaluate.py:linear_interp",
            "target_source": "batch['_tastvg_full_record']",
            "full_frame_count": len(full_frame_ids),
            "sampled_frame_count": len(sampled_frame_ids),
            "full_frame_range": [full_frame_ids[0], full_frame_ids[-1]],
            "dense_prediction_box_format": "pixel xyxy after official PostProcess, linearly interpolated",
            "dense_target_box_format": "pixel xyxy from official all_gt_data bboxs, normalized to cxcywh for vg_tta.metrics",
        },
    )


def _compute_corrected_metrics(
    batch: Mapping[str, Any], ensemble: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute corrected vIoU and all paired-protocol metrics.

    Spatial/temporal predictions come from the official postprocessor and its
    two-view temporal merge.  The metric implementation supplies the corrected
    union denominator while retaining the legacy value for auditability.
    """

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from vg_tta.metrics import compute_stvg_metrics

    dense_result = _dense_official_metric_result(batch, ensemble)
    if dense_result is None:
        frame_targets, gt_indices, frame_ids, gt_frame_interval = _metric_target_sequence(batch)
        pred_indices = tuple(int(value) for value in ensemble["predicted_indices"])
        pred_boxes = _absolute_boxes_to_normalized_cxcywh(
            ensemble["boxes_abs"], batch["targets"][0]
        )
        metrics = compute_stvg_metrics(
            pred_boxes,
            frame_targets,
            pred_indices,
            gt_indices,
            frame_ids=frame_ids,
            gt_frame_interval=gt_frame_interval,
        )
        metric_provenance = {
            "metric_timeline": "sampled frame ids (synthetic/manual batch without full record)",
        }
    else:
        metrics, metric_provenance = dense_result
    metrics = {str(key): value for key, value in metrics.items()}
    metrics["vIoU@0.3"] = float(metrics["vIoU_at_0.3"])
    metrics["vIoU@0.5"] = float(metrics["vIoU_at_0.5"])
    metrics["vIoU@.3"] = float(metrics["vIoU_at_0.3"])
    metrics["vIoU@.5"] = float(metrics["vIoU_at_0.5"])
    metrics["start_error"] = float(metrics["start_error_frames"])
    metrics["end_error"] = float(metrics["end_error_frames"])
    metrics["start_abs_error"] = float(metrics["start_abs_error_frames"])
    metrics["end_abs_error"] = float(metrics["end_abs_error_frames"])
    return {
        "metrics": metrics,
        "metric_provenance": {
            "postprocessor": "external/TA-STVG/models/post_processor.py:PostProcess",
            "metric_function": "vg_tta.metrics.compute_stvg_metrics",
            "primary_viou": "vIoU_corrected",
            "corrected_union_end": "max(pred_end, gt_end)",
            "target_boxes": "official normalized BoxList cxcywh aligned to active actioness frames",
            **metric_provenance,
        },
    }


def _query_view_ensemble(
    model: Any,
    batch: Mapping[str, Any],
    *,
    mode: str,
    device: Any,
    postprocessor: Any,
    autocast_enabled: bool | None,
    autocast_dtype: str | None,
) -> dict[str, Any]:
    """Run official two-view inference, or four-view no-param TTAug."""

    import torch

    if mode not in {"frozen", "ttaug", "ttaug-consistency", "boundary-entropy", "trajcal", "span-tta"}:
        raise ValueError(f"unsupported query ensemble mode: {mode}")
    offsets = (0, 1)
    flips = (False, True) if mode == "ttaug" else (False,)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in offsets:
            for flip in flips:
                view_batch = _make_temporal_view_batch(batch, offset=offset, flip=flip)
                outputs, trace = _forward_with_temporal_trace(
                    model,
                    view_batch,
                    device=device,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                )
                record = _postprocess_temporal_view(
                    outputs,
                    view_batch,
                    postprocessor,
                    offset=offset,
                    flip=flip,
                )
                record["trace"] = trace
                records.append(record)
    merged = _merge_postprocessed_views(records, batch["targets"][0]["frame_ids"])
    merged["primary_outputs"] = records[0]["outputs"]
    merged["primary_trace"] = records[0]["trace"]
    merged["view_count"] = len(records)
    merged["ensemble_name"] = "ttaug_4view" if mode == "ttaug" else "official_temporal_2view"
    return merged


def _summarize_outputs(
    outputs: Mapping[str, Any],
    trace: Mapping[str, Any],
    batch: Mapping[str, Any] | None = None,
    ensemble: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pred_sted_shape": list(outputs["pred_sted"].shape),
        "pred_actioness_shape": list(outputs["pred_actioness"].shape),
        "pred_sted_min": float(outputs["pred_sted"].detach().min().item()),
        "pred_sted_max": float(outputs["pred_sted"].detach().max().item()),
        "decoder_invocation_count": int(trace["decoder_invocation_count"]),
        "all_layer_features_shape": list(trace["all_layer_features_shape"]),
        "final_query_conditioned_per_frame_shape": list(
            trace["final_query_conditioned_per_frame_shape"]
        ),
        "device": str(outputs["pred_sted"].device),
    }
    if batch is not None:
        valid_mask = _valid_mask_from_batch(
            batch, outputs["pred_sted"].shape[1], outputs["pred_sted"].device
        )
        predicted_indices = _predicted_legal_span(outputs["pred_sted"], valid_mask)
        if ensemble is not None:
            predicted_indices = tuple(int(value) for value in ensemble["predicted_indices"])
        target_indices = _target_temporal_span(batch)
        result["predicted_span_indices"] = list(predicted_indices)
        result["target_span_indices"] = (
            None if target_indices is None else list(target_indices)
        )
        result["temporal_iou"] = _span_iou(predicted_indices, target_indices)
        frame_ids = batch.get("targets", [{}])[0].get("frame_ids", [])
        frame_ids = [int(value) for value in frame_ids]
        if frame_ids and predicted_indices[1] < len(frame_ids):
            result["predicted_frame_span"] = [
                frame_ids[predicted_indices[0]],
                frame_ids[predicted_indices[1]] + 1,
            ]
        if target_indices is not None and target_indices[1] < len(frame_ids):
            result["target_frame_span"] = [
                frame_ids[target_indices[0]],
                frame_ids[target_indices[1]] + 1,
            ]
        result["temporal_iou_convention"] = (
            "inclusive sampled-index spans; frame spans are half-open"
        )
        if ensemble is not None:
            metric_result = _compute_corrected_metrics(batch, ensemble)
            result.update(metric_result)
            # Keep the formal fields available both in the structured
            # ``metrics`` object and as query-record columns.  The latter is
            # convenient for tabular/cluster-bootstrap consumers and mirrors
            # the existing vg_tta report schema.
            for metric_name in (
                "tIoU",
                "sIoU",
                "vIoU",
                "vIoU_corrected",
                "vIoU_at_0.3",
                "vIoU_at_0.5",
                "vIoU@.3",
                "vIoU@.5",
                "start_error_frames",
                "end_error_frames",
                "start_abs_error_frames",
                "end_abs_error_frames",
            ):
                if metric_name in result["metrics"]:
                    result[metric_name] = result["metrics"][metric_name]
            result["temporal_iou"] = float(result["metrics"]["tIoU"])
            result["predicted_frame_span"] = list(ensemble["predicted_frame_span"])
            result["view_ensemble"] = {
                "name": str(ensemble["ensemble_name"]),
                "view_count": int(ensemble["view_count"]),
                "views": list(ensemble["view_metadata"]),
                "postprocessor": str(ensemble["postprocessor"]),
                "temporal_merge": str(ensemble["temporal_merge"]),
                "spatial_merge": str(ensemble["spatial_merge"]),
            }
            entropy_logits = ensemble["temporal_logits"].unsqueeze(0)
            entropy_mask = _valid_mask_from_batch(
                batch, entropy_logits.shape[1], entropy_logits.device
            )
            entropy, entropy_info = _temporal_entropy_loss(
                {"pred_sted": entropy_logits}, batch
            )
            result["valid_span_entropy"] = float(entropy.detach().item())
            result["valid_span_entropy_info"] = entropy_info
    return result


def _predicted_legal_span(logits: Any, valid_mask: Any) -> tuple[int, int]:
    """Decode the official strict ``start < end`` temporal span for B=1."""

    import torch

    scores = logits.detach().float()
    if scores.ndim == 3:
        if scores.shape[0] != 1:
            raise ValueError("span decoding expects B=1")
        scores = scores[0]
    mask = valid_mask.detach().to(device=scores.device, dtype=torch.bool)
    if mask.ndim == 2:
        if mask.shape[0] != 1:
            raise ValueError("span decoding expects one valid-mask row")
        mask = mask[0]
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if positions.numel() < 2:
        raise ValueError("span decoding needs at least two valid frames")
    selected = scores.index_select(0, positions)
    joint = (
        selected[:, 0].log_softmax(dim=0)[:, None]
        + selected[:, 1].log_softmax(dim=0)[None, :]
    )
    legal = torch.triu(
        torch.ones_like(joint, dtype=torch.bool), diagonal=1
    )
    best = joint.masked_fill(~legal, -torch.inf).reshape(-1).argmax()
    start = int(best.item() // joint.shape[1])
    end = int(best.item() % joint.shape[1])
    return int(positions[start].item()), int(positions[end].item())


def _target_temporal_span(batch: Mapping[str, Any]) -> tuple[int, int] | None:
    """Read a metadata-only inclusive actionness span from a collated batch."""

    import torch

    targets = batch.get("targets", [])
    if not targets:
        return None
    actioness = targets[0].get("actioness")
    if actioness is None:
        return None
    if isinstance(actioness, torch.Tensor):
        values = actioness.detach().cpu().flatten().bool()
    else:
        values = torch.as_tensor(actioness).flatten().bool()
    positions = torch.nonzero(values, as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    return int(positions[0].item()), int(positions[-1].item())


def _span_iou(first: tuple[int, int] | None, second: tuple[int, int] | None) -> float | None:
    """Compute inclusive-index temporal IoU for two ``(start,end)`` spans."""

    if first is None or second is None:
        return None
    start = max(first[0], second[0])
    end = min(first[1], second[1])
    intersection = max(0, end - start + 1)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return float(intersection / union) if union > 0 else 0.0


def _trace_metadata(trace: Any) -> Any:
    """Keep hook shape metadata while removing live tensors from JSON."""

    if isinstance(trace, Mapping):
        return {
            str(key): _trace_metadata(value)
            for key, value in trace.items()
            if key != "final_per_frame_features"
        }
    if hasattr(trace, "shape") and hasattr(trace, "dtype"):
        return {
            "shape": list(trace.shape),
            "dtype": str(trace.dtype),
            "device": str(trace.device),
        }
    if isinstance(trace, (list, tuple)):
        return [_trace_metadata(value) for value in trace]
    return trace


def run_support_query_episode(
    model: Any,
    support_batch: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    query_batch: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    mode: str,
    adaptation_steps: int = 1,
    learning_rate: float = 2e-4,
    optimizer_eps: float = 1e-4,
    weight_decay: float = 0.0,
    include_actionness: bool = False,
    reset_after: bool = True,
    device: Any | None = None,
    autocast_enabled: bool | None = None,
    autocast_dtype: str | None = "float16",
    trajectory_weight: float = 8.0,
    trajectory_temperature: float = 1.0,
    trajectory_top_k: int | None = 20,
) -> dict[str, Any]:
    """Run one frozen/TTAug/entropy/TrajCal/Span-TTA support-to-query episode.

    ``support_batch`` and ``query_batch`` are regular batches returned by the
    official TA-STVG loader.  A sequence of support batches is accepted for a
    K-shot protocol and is adapted sequentially with one fresh optimizer per
    episode.  A sequence of query batches is evaluated under the same adapted
    state; it is never re-adapted per query.  This is the invariant required by
    paired K-shot evaluation.  No ground-truth labels enter an adaptation
    objective; ``actioness`` remains metadata used only to size the legal-frame
    mask.  Query inference uses the official two temporal offset views, while
    ``mode='ttaug'`` adds horizontal flips for a no-parameter four-view
    ensemble.  CUDA is used only when ``device`` is explicitly set to an
    available CUDA device; the default follows the model's current device.
    """

    import torch

    if mode not in {
        "frozen",
        "ttaug",
        "ttaug-consistency",
        "boundary-entropy",
        "trajcal",
        "span-tta",
    }:
        raise ValueError(f"unsupported episode mode: {mode}")
    if adaptation_steps < 0:
        raise ValueError("adaptation_steps must be non-negative")
    if trajectory_top_k is not None and trajectory_top_k < 1:
        raise ValueError("trajectory_top_k must be positive or None")
    if isinstance(support_batch, Mapping):
        support_batches = [support_batch]
    else:
        support_batches = support_batch
    if not support_batches:
        raise ValueError("support_batch sequence must be non-empty")
    if isinstance(query_batch, Mapping):
        query_batches = [query_batch]
    else:
        query_batches = query_batch
    if not query_batches:
        raise ValueError("query_batch sequence must be non-empty")
    if device is None:
        device = _model_device(model)
    elif not hasattr(device, "type"):
        device = _resolve_device(device)
    autocast_requested = bool(
        device.type == "cuda" if autocast_enabled is None else autocast_enabled
    )
    resolved_autocast_dtype = _autocast_dtype(autocast_dtype, device)
    autocast_active = bool(autocast_requested and resolved_autocast_dtype is not None)
    scaler = _make_grad_scaler(
        device, enabled=autocast_active, dtype=resolved_autocast_dtype
    )
    # Keep the episode collection on host memory and transfer one clip at a
    # time.  A formal K=64/Q=64 TA-STVG episode is roughly 100 MiB per decoded
    # clip; eagerly materializing every clip on CUDA would consume most of a
    # 32 GiB device before the first forward pass and can turn an otherwise
    # valid protocol into an avoidable OOM.
    initial_training_modes = snapshot_module_training_modes(model)
    initial_requires_grad = snapshot_requires_grad(model)
    head_audit = set_temporal_head_only(model, include_actionness=include_actionness)
    snapshot = snapshot_full_model_state(model)
    # The official target includes full unresized frames. Retain only IDs,
    # never all target dictionaries for an entire support/query episode.
    support_vids = (
        support_batches.source_vids()
        if hasattr(support_batches, "source_vids")
        else [str(batch["targets"][0].get("vid")) for batch in support_batches]
    )
    query_vids = (
        query_batches.source_vids()
        if hasattr(query_batches, "source_vids")
        else [str(batch["targets"][0].get("vid")) for batch in query_batches]
    )
    query_vid = query_vids[0]
    optimizer = None
    adaptation_records: list[dict[str, Any]] = []
    query_summaries: list[dict[str, Any]] = []
    reset_verified = False
    training_mode_reset_verified = False
    requires_grad_reset_verified = False
    allowed_names = set(
        exact_temporal_head_parameters(
            model, include_actionness=include_actionness
        )[0]
    )
    try:
        model.eval()
        # ``ttaug`` is intentionally a no-parameter query-time baseline.  A
        # caller may leave the common adaptation_steps flag at its default,
        # but it must not create an optimizer or alter the model.
        effective_adaptation_steps = (
            0 if mode in {"frozen", "ttaug"} else adaptation_steps
        )
        if effective_adaptation_steps:
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable,
                lr=learning_rate,
                eps=optimizer_eps,
                weight_decay=weight_decay,
            )
            for step in range(effective_adaptation_steps):
                for support_index, current_support in enumerate(support_batches):
                    current_support_device = _batch_to_device(current_support, device)
                    optimizer.zero_grad(set_to_none=True)
                    if mode == "ttaug-consistency":
                        loss, info, trace = _ttaug_consistency_loss(
                            model,
                            current_support_device,
                            device=device,
                            autocast_enabled=autocast_enabled,
                            autocast_dtype=autocast_dtype,
                        )
                    else:
                        outputs, trace = _forward_with_temporal_trace(
                            model,
                            current_support_device,
                            device=device,
                            autocast_enabled=autocast_enabled,
                            autocast_dtype=autocast_dtype,
                        )
                        if mode == "trajcal":
                            loss, info = _trajectory_calibration_loss(
                                outputs,
                                trace,
                                current_support_device,
                                trajectory_weight=trajectory_weight,
                                trajectory_temperature=trajectory_temperature,
                                trajectory_top_k=trajectory_top_k,
                            )
                        elif mode == "boundary-entropy":
                            loss, info = _boundary_entropy_loss(
                                outputs, current_support_device
                            )
                        else:
                            loss, info = _temporal_entropy_loss(
                                outputs, current_support_device
                            )
                    if not bool(torch.isfinite(loss).all()):
                        raise FloatingPointError(
                            f"non-finite adaptation loss at step={step}, "
                            f"support_index={support_index}"
                        )
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    gradient_names = []
                    nonfinite_gradient_names = []
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None:
                            continue
                        if bool(torch.isfinite(parameter.grad).all()):
                            gradient_names.append(name)
                        else:
                            nonfinite_gradient_names.append(name)
                    if nonfinite_gradient_names:
                        raise FloatingPointError(
                            "non-finite temporal-head gradients: "
                            f"{nonfinite_gradient_names[:5]}"
                        )
                    forbidden_gradients = [
                        name
                        for name, parameter in model.named_parameters()
                        if name not in allowed_names and parameter.grad is not None
                    ]
                    if forbidden_gradients:
                        raise RuntimeError(
                            "strict head-only adaptation received forbidden gradients: "
                            f"{forbidden_gradients[:5]}"
                        )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    adaptation_records.append(
                        {
                            "step": step,
                            "support_index": support_index,
                            "support_vid": support_vids[support_index],
                            "loss": float(loss.detach().item()),
                            "objective": mode,
                            "info": info,
                            "finite_gradient_names": gradient_names,
                            "trace": _trace_metadata(trace),
                        }
                    )
                    if (support_index + 1) % 16 == 0 or support_index + 1 == len(support_batches):
                        print(f"[{mode} support] {support_index + 1}/{len(support_batches)}", flush=True)
                    del current_support_device
        postprocessor = official_imports()["postprocessor"]()
        for query_position, current_query in enumerate(query_batches):
            current_query_device = _batch_to_device(current_query, device)
            ensemble = _query_view_ensemble(
                model,
                current_query_device,
                mode=mode,
                device=device,
                postprocessor=postprocessor,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )
            query_summary = _summarize_outputs(
                ensemble["primary_outputs"],
                ensemble["primary_trace"],
                current_query_device,
                ensemble=ensemble,
            )
            query_target = current_query["targets"][0]
            query_summary["query_vid"] = str(query_target.get("vid"))
            query_summary["query_item_id"] = str(query_target.get("item_id"))
            # A source video is the natural cluster for VidSTG's multiple
            # query records; HC-STVG has one query per video in this loader.
            query_summary["query_cluster"] = str(
                query_target.get("vid", query_target.get("item_id"))
            )
            query_summaries.append(query_summary)
            if (query_position + 1) % 16 == 0 or query_position + 1 == len(query_batches):
                print(f"[{mode} query] {query_position + 1}/{len(query_batches)}", flush=True)
            del current_query_device
    finally:
        if reset_after:
            restore_full_model_state(model, snapshot)
            restore_module_training_modes(model, initial_training_modes)
            restore_requires_grad(model, initial_requires_grad)
            reset_verified = full_state_equal(model, snapshot)
            training_mode_reset_verified = module_training_modes_equal(
                model, initial_training_modes
            )
            requires_grad_reset_verified = requires_grad_equal(
                model, initial_requires_grad
            )
            if not reset_verified:
                raise RuntimeError("full-state reset verification failed")
            if not training_mode_reset_verified:
                raise RuntimeError("module training-mode reset verification failed")
            if not requires_grad_reset_verified:
                raise RuntimeError("requires_grad reset verification failed")
        optimizer = None

    return {
        "status": "passed",
        "device": str(device),
        "mode": mode,
        "support_vid": support_vids[0],
        "support_vids": support_vids,
        "support_size": len(support_batches),
        "query_vid": query_vid,
        "query_vids": query_vids,
        "query_count": len(query_batches),
        "support_query_source_disjoint": all(
            support_vid != query_source
            for support_vid in support_vids
            for query_source in query_vids
        ),
        "adaptation_steps": adaptation_steps,
        "effective_adaptation_steps": (
            0 if mode in {"frozen", "ttaug"} else adaptation_steps
        ),
        "learning_rate": learning_rate,
        "optimizer": {
            "name": "AdamW",
            "eps": float(optimizer_eps),
            "weight_decay": float(weight_decay),
        },
        "include_actionness": include_actionness,
        "autocast": {
            "enabled": autocast_active,
            "dtype": None
            if resolved_autocast_dtype is None
            else str(resolved_autocast_dtype),
            "grad_scaler": scaler is not None,
        },
        "trajectory": {
            "weight": float(trajectory_weight),
            "temperature": float(trajectory_temperature),
            "top_k": None if trajectory_top_k is None else int(trajectory_top_k),
        },
        "head_audit": head_audit,
        "adaptation_records": adaptation_records,
        "query": query_summaries[0] if query_summaries else None,
        "queries": query_summaries,
        "reset_after": reset_after,
        "reset_verified": reset_verified if reset_after else None,
        "training_mode_reset_verified": (
            training_mode_reset_verified if reset_after else None
        ),
        "requires_grad_reset_verified": (
            requires_grad_reset_verified if reset_after else None
        ),
        "full_state_reset_scope": (
            "all model state_dict parameters and buffers, module training flags, "
            "requires_grad flags, and optimizer state"
        ),
    }


def load_model_on_device(
    dataset: str,
    runtime_dir: Path,
    *,
    checkpoint: Path | None = None,
    resolution: int = 224,
    sample_frames: int = 64,
    device: str | Any = "cpu",
    source_dataset: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Construct/load TA-STVG on CPU, then explicitly move to ``device``.

    The checkpoint is always deserialized with ``map_location=cpu`` and the
    complete model is kept on CPU until strict compatibility checks finish.
    For a cross-dataset run, the model is constructed from the source
    dataset's YAML/runtime so its ``APP_NUM``/``MOT_NUM`` classifiers have the
    exact checkpoint shapes.  The target dataset is used later only to build
    the loader/evaluator and to install a small metadata-key compatibility
    map.  No target-shaped classifier is randomly initialized.
    """

    requested_device = _resolve_device(device)
    checkpoint_dataset = source_dataset or dataset
    checkpoint_path = _checkpoint_for(checkpoint_dataset, checkpoint)
    inferred_source = _infer_checkpoint_source(checkpoint_path)
    source_dataset = source_dataset or (
        inferred_source if inferred_source in DATASET_DEFAULTS else dataset
    )
    runtime_dir = runtime_dir.resolve()
    target_manifest = _load_manifest(runtime_dir)
    source_runtime_dir = (
        runtime_dir.parent / str(_dataset_spec(source_dataset)["runtime_dirname"])
    ).resolve()
    if not (source_runtime_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            "source-dataset runtime manifest missing; stage the source runtime "
            f"before strict cross-dataset load: {source_runtime_dir / 'manifest.json'}"
        )
    source_manifest = _load_manifest(source_runtime_dir)
    model_root = runtime_dir.parent
    model_zoo = prepare_model_zoo(model_root)
    os.environ["TORCH_HOME"] = str(TORCH_HOME.resolve())
    os.environ["HF_HOME"] = str(HF_HOME.resolve())
    _prepend_path(PROJECT_ROOT / ".conda" / "tastvg" / "bin")
    imports = official_imports()
    cfg = build_runtime_config(
        source_dataset,
        source_runtime_dir,
        resolution=resolution,
        sample_frames=sample_frames,
        use_model_defaults=False,
    )
    cfg_dict = {
        "model_architecture_dataset": str(cfg.DATASET.NAME),
        "loader_dataset": dataset,
        "data_dir": str(cfg.DATA_DIR),
        "source_runtime_dir": str(source_runtime_dir),
        "target_runtime_dir": str(runtime_dir),
        "resolution": int(cfg.INPUT.RESOLUTION),
        "sample_frames": int(sample_frames),
        "app_num": int(cfg.DATASET.APP_NUM),
        "mot_num": int(cfg.DATASET.MOT_NUM),
        "theta": 0.45 if str(cfg.DATASET.NAME) == "VidSTG" else 0.7,
        "device": str(cfg.MODEL.DEVICE),
        "use_lstm": bool(cfg.MODEL.USE_LSTM),
        "decoder_layers": int(cfg.MODEL.TASTVG.DEC_LAYERS),
    }
    logger = logging.getLogger("tastvg.device.runner")
    logger.setLevel(logging.INFO)
    model = None
    loss_model = None
    with _temporary_cwd(model_root), _cpu_only_torch_guard():
        model, loss_model, _weight_dict = imports["build_model"](cfg)
        model.to("cpu")
        model.eval()
        # Keep the official model constructor, but use a strict compatibility
        # loader here.  The released checkpoint has one deterministic
        # position_ids buffer that recent Transformers marks non-persistent;
        # VSTGCheckpointer's unconditional strict load rejects that harmless
        # version skew before it can report the actual model state.
        checkpoint_load = _load_checkpoint_cpu_compat(
            model,
            checkpoint_path,
            source_dataset=source_dataset,
            target_dataset=dataset,
        )
    cpu_parameter_report = _model_parameter_device_report(
        model, expected_device="cpu"
    )
    if not cpu_parameter_report["all_cpu"]:
        raise RuntimeError(
            "CPU model load placed parameters on non-CPU devices: "
            f"{cpu_parameter_report}"
        )
    # This is the only transfer to a non-CPU device in the loader.  It is
    # intentionally after constructor + checkpoint load to keep map_location
    # and failure behavior deterministic.
    if requested_device.type != "cpu":
        model.to(requested_device)
    model.eval()
    parameter_report = _model_parameter_device_report(
        model, expected_device=requested_device
    )
    if not parameter_report["all_on_expected_device"]:
        raise RuntimeError(
            "TA-STVG model transfer landed on unexpected devices: "
            f"{parameter_report}"
        )
    head_names, head_count = exact_temporal_head_parameters(model)
    report = {
        "status": "passed",
        "device": str(requested_device),
        "requested_device": str(requested_device),
        "load_device": "cpu",
        "target_dataset": dataset,
        "model_architecture_dataset": source_dataset,
        "source_dataset": source_dataset,
        "inferred_checkpoint_source": inferred_source,
        "forward_executed": False,
        "runtime_manifest": target_manifest,
        "target_runtime_manifest": target_manifest,
        "source_runtime_manifest": source_manifest,
        "model_zoo": model_zoo,
        "config": cfg_dict,
        "checkpoint": {
            "path": _relative(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "checkpoint_load": checkpoint_load,
        "parameter_device": parameter_report,
        "cpu_parameter_device_before_transfer": cpu_parameter_report,
        "strict_boundary_head": {
            "parameter_names": head_names,
            "trainable_scalar_count": head_count,
        },
        "lazy_import": install_optional_torchtext_shim(),
    }
    return model, cfg, report


def load_cpu_model(
    dataset: str,
    runtime_dir: Path,
    *,
    checkpoint: Path | None = None,
    resolution: int = 224,
    sample_frames: int = 64,
    source_dataset: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Backward-compatible CPU-only wrapper around :func:`load_model_on_device`."""

    return load_model_on_device(
        dataset,
        runtime_dir,
        checkpoint=checkpoint,
        resolution=resolution,
        sample_frames=sample_frames,
        device="cpu",
        source_dataset=source_dataset,
    )


def _sample_batches(
    dataset: str,
    runtime_dir: Path,
    *,
    resolution: int,
    sample_frames: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _set_random_seed(seed)
    imports = official_imports()
    cfg = build_runtime_config(
        dataset,
        runtime_dir,
        resolution=resolution,
        sample_frames=sample_frames,
    )
    with _trusted_runtime_torch_load():
        dataset_object = imports["build_dataset"](
            cfg, "test", imports["build_transforms"](cfg, is_train=False)
        )
    indices = _shortest_distinct_indices(dataset_object, count=2)
    support = _batch_for_dataset_item(dataset_object, indices[0], imports["collate_fn"])
    query = _batch_for_dataset_item(dataset_object, indices[1], imports["collate_fn"] if len(indices) > 1 else imports["collate_fn"])
    return support, query, {
        "support_index": indices[0],
        "query_index": indices[1] if len(indices) > 1 else indices[0],
        "support_item_id": str(dataset_object.all_gt_data[indices[0]]["item_id"]),
        "query_item_id": str(dataset_object.all_gt_data[indices[1]]["item_id"] if len(indices) > 1 else dataset_object.all_gt_data[indices[0]]["item_id"]),
    }


def _build_runtime_dataset(
    dataset: str,
    runtime_dir: Path,
    *,
    resolution: int,
    sample_frames: int,
    seed: int,
) -> tuple[Any, dict[str, Any], Any]:
    """Build one real official test dataset and return it with its collator."""

    _set_random_seed(seed)
    imports = official_imports()
    cfg = build_runtime_config(
        dataset,
        runtime_dir,
        resolution=resolution,
        sample_frames=sample_frames,
    )
    with _trusted_runtime_torch_load():
        dataset_object = imports["build_dataset"](
            cfg, "test", imports["build_transforms"](cfg, is_train=False)
        )
    return dataset_object, imports, cfg


def configure_target_loader_compatibility(
    model: Any,
    dataset_object: Any,
    target_dataset: str,
) -> dict[str, Any]:
    """Separate source-model architecture from target-loader metadata.

    ``TASTVGNet`` stores two dataset-dependent pieces of behavior in its
    constructor/forward: ``APP_NUM``/``MOT_NUM`` determine the spatial
    activation classifier shapes, and ``cfg.DATASET.NAME`` selects both the
    temporal attention threshold and the key used to look up the subject and
    attribute labels.  The model must keep the source architecture and
    checkpoint intact for natural-shift evaluation.  Only the latter lookup
    metadata is adapted here, after the target loader has been built.

    The pinned official checkout has no ``query_frame_info`` reference; its
    target frame metadata is carried by ``targets[0]['frame_ids']``.  The
    compatibility map therefore keys target records by ``item_id`` when the
    source architecture is VidSTG and by ``vid`` when it is HC-STVG, matching
    the exact branch in ``models/pipeline.py:TASTVGNet.forward``.  It supplies
    explicit target-safe subject/attribute labels without changing any model
    parameter or buffer.
    """

    model_cfg = getattr(model, "cfg", None)
    source_cfg_dataset = getattr(model_cfg, "DATASET", None)
    source_name = str(getattr(source_cfg_dataset, "NAME", ""))
    source_dataset = {
        "VidSTG": "vidstg",
        "HC-STVG": "hcstvg2",
    }.get(source_name)
    if source_dataset is None:
        raise ValueError(
            "unsupported source model DATASET.NAME for target compatibility: "
            f"{source_name!r}"
        )
    if target_dataset not in DATASET_DEFAULTS:
        raise ValueError(f"unknown target dataset for compatibility: {target_dataset}")

    # SpatialActivation keeps the constructor vocabulary size explicitly;
    # unlike the temporal MLP heads, its classification projection is nested
    # inside BertLMPredictionHead and has no public ``layers`` ModuleList.
    app_num = int(model.s_spatial_clas.vocab_size)
    mot_num = int(model.t_spatial_clas.vocab_size)
    theta = float(model.theta)
    source_key_field = "item_id" if source_name == "VidSTG" else "vid"
    target_records = list(getattr(dataset_object, "all_gt_data", []))
    if not target_records:
        raise ValueError("target loader has no all_gt_data records")

    result: dict[str, Any] = {
        "status": "passed",
        "applied": False,
        "source_dataset": source_dataset,
        "source_cfg_dataset_name": source_name,
        "target_loader_dataset": target_dataset,
        "source_key_field": source_key_field,
        "source_app_num": app_num,
        "source_mot_num": mot_num,
        "source_theta": theta,
        "target_record_count": len(target_records),
        "target_loader_frame_field": "targets[0]['frame_ids']",
        "query_frame_info_reference": "not present in pinned official TA-STVG checkout",
        "pipeline_audit": {
            "cfg_dataset_name": "models/pipeline.py:TASTVGNet uses source cfg for APP_NUM/MOT_NUM, theta, and info-key branch",
            "classifier_construction": "models/pipeline.py:TASTVGNet.__init__: build_SpatialActivation(hidden_dim, cfg.DATASET.APP_NUM/MOT_NUM)",
            "theta": "models/pipeline.py:TASTVGNet.__init__: 0.45 for VidSTG, 0.7 otherwise",
            "label_lookup": "models/pipeline.py:TASTVGNet.forward: item_id for VidSTG, vid otherwise",
            "query_frame_info": "no reference found; target frame ids remain in targets[0]['frame_ids']",
            "item_id": "target item_id is stringified only for the source-VidSTG label map",
        },
        "spatial_classifier_weight_policy": (
            "source checkpoint tensors are loaded strictly; no target-shaped "
            "classifier is initialized or skipped"
        ),
    }
    if source_dataset == target_dataset:
        result["reason"] = "native source and target dataset; source runtime labels already use the same key convention"
        return result

    labels: dict[str, dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    conflicting_keys: list[str] = []
    for item in target_records:
        if not isinstance(item, Mapping):
            raise TypeError("target loader all_gt_data contains a non-mapping record")
        raw_key = item.get(source_key_field)
        if raw_key is None:
            raise ValueError(
                f"target loader record has no source-key field {source_key_field!r}: "
                f"{sorted(item.keys())}"
            )
        key = str(raw_key)
        subject = item.get("sub") or item.get("object") or "person"
        verb_labels = item.get("verb_index_list", [])
        attr_labels = item.get("adj_index_list", [])
        label_entry = {
            "sub": str(subject),
            "verb_index_list": list(verb_labels) if verb_labels is not None else [],
            "adj_index_list": list(attr_labels) if attr_labels is not None else [],
        }
        if key in labels:
            duplicate_keys.append(key)
            if labels[key] != label_entry:
                conflicting_keys.append(key)
            continue
        labels[key] = label_entry
    if conflicting_keys:
        raise ValueError(
            "target loader records have conflicting labels under the source "
            f"model's {source_key_field} lookup: {conflicting_keys[:5]}"
        )
    if len(labels) + len(duplicate_keys) != len(target_records):
        raise RuntimeError(
            "target compatibility label map does not cover every loader record: "
            f"unique_labels={len(labels)}, duplicates={len(duplicate_keys)}, "
            f"records={len(target_records)}"
        )

    # These are Python metadata dictionaries, not state-dict tensors.  Set
    # both training/eval maps because the official forward picks one by
    # ``self.training`` even though this runner normally evaluates in eval.
    model.verb_label = copy.deepcopy(labels)
    model.verb_label2 = copy.deepcopy(labels)
    result.update(
        {
            "applied": True,
            "reason": "cross-dataset target metadata lookup adapted after source-architecture construction",
            "label_map_count": len(labels),
            "label_map_key_sha256": hashlib.sha256(
                json.dumps(sorted(labels), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "label_map_subject": "target record object/sub fallback, default person",
            "verb_attribute_labels": "target-safe explicit lists; no model weights changed",
            "duplicate_source_key_records": len(duplicate_keys),
            "duplicate_source_keys": sorted(set(duplicate_keys))[:20],
            "duplicate_key_policy": (
                "allowed when all duplicate target records share identical labels; "
                "conflicting labels are rejected because the source model branch "
                "cannot distinguish them by this key"
            ),
        }
    )
    return result


def _protocol_indices(
    dataset_object: Any,
    *,
    support_size: int,
    support_seeds: Sequence[int],
    query_count: int,
) -> dict[str, Any]:
    """Select deterministic disjoint K-shot support/query records.

    Query records are the shortest distinct videos; each seed permutes the
    remaining distinct-video pool.  This makes all seed rows paired on the
    same query records while avoiding support/query leakage at the video level.
    """

    if support_size < 1:
        raise ValueError("support_size/K must be positive")
    if query_count < 1:
        raise ValueError("query_count must be positive")
    if not support_seeds:
        raise ValueError("at least one support seed is required")
    rows: list[tuple[int, int, str]] = []
    for index, item in enumerate(dataset_object.all_gt_data):
        rows.append(
            (
                len(item.get("frame_ids", [])),
                index,
                str(item.get("vid", "")),
            )
        )
    rows.sort()
    distinct_rows: list[tuple[int, int, str]] = []
    seen_vids: set[str] = set()
    for row in rows:
        if row[2] in seen_vids:
            continue
        distinct_rows.append(row)
        seen_vids.add(row[2])
    if len(distinct_rows) < query_count + support_size:
        raise ValueError(
            "not enough distinct videos for disjoint protocol: "
            f"need={query_count + support_size}, available={len(distinct_rows)}"
        )
    query_rows = distinct_rows[:query_count]
    query_indices = [row[1] for row in query_rows]
    query_vids = [row[2] for row in query_rows]
    support_pool = [row for row in distinct_rows if row[2] not in set(query_vids)]
    per_seed: dict[str, list[int]] = {}
    for raw_seed in support_seeds:
        seed = int(raw_seed)
        generator = random.Random(seed)
        shuffled = list(support_pool)
        generator.shuffle(shuffled)
        selected = shuffled[:support_size]
        per_seed[str(seed)] = [row[1] for row in selected]
    dataset_records = [
        {
            "index": int(index),
            "item_id": str(item.get("item_id", "")),
            "vid": str(item.get("vid", "")),
            "frame_count": len(item.get("frame_ids", [])),
        }
        for index, item in enumerate(dataset_object.all_gt_data)
    ]
    split_payload = {
        "dataset_records": dataset_records,
        "query_indices": query_indices,
        "query_videos": query_vids,
        "support_indices_by_seed": per_seed,
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    query_records = [
        {
            "index": int(row[1]),
            "item_id": str(dataset_object.all_gt_data[row[1]].get("item_id", "")),
            "vid": str(row[2]),
            "cluster": str(row[2]),
        }
        for row in query_rows
    ]
    return {
        "protocol_id": "tastvg-fixed-shortest-distinct-v1",
        "query_indices": query_indices,
        "query_videos": query_vids,
        "query_records": query_records,
        "support_indices_by_seed": per_seed,
        "support_size": int(support_size),
        "support_seeds": [int(seed) for seed in support_seeds],
        "query_count": int(query_count),
        "selection": "shortest distinct query videos; seeded shuffled disjoint support pool",
        "source_disjoint_by_construction": True,
        "dataset_record_count": len(dataset_records),
        "dataset_selection_sha256": split_hash,
        "dataset_records_sha256": hashlib.sha256(
            json.dumps(dataset_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "adaptation_once_per_seed": True,
        "query_evaluated_under_single_adapted_state": True,
        "expected_episode_call_count": 2 + len(support_seeds),
        "expected_adaptation_episode_call_count": len(support_seeds),
    }


def _paired_numeric_delta(
    baseline: Mapping[str, Any], adapted: Mapping[str, Any]
) -> dict[str, float]:
    fields = ("temporal_iou", "valid_span_entropy", "pred_sted_min", "pred_sted_max")
    delta: dict[str, float] = {}
    for field in fields:
        first = baseline.get(field)
        second = adapted.get(field)
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            delta[field] = float(second) - float(first)
    baseline_metrics = baseline.get("metrics", {})
    adapted_metrics = adapted.get("metrics", {})
    if isinstance(baseline_metrics, Mapping) and isinstance(adapted_metrics, Mapping):
        metric_fields = (
            "tIoU",
            "sIoU",
            "vIoU",
            "vIoU_corrected",
            "vIoU_at_0.3",
            "vIoU_at_0.5",
            "start_error_frames",
            "end_error_frames",
            "start_abs_error_frames",
            "end_abs_error_frames",
        )
        for field in metric_fields:
            first = baseline_metrics.get(field)
            second = adapted_metrics.get(field)
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                delta[field] = float(second) - float(first)
    return delta


def _metric_value(summary: Mapping[str, Any], metric: str) -> float | None:
    metrics = summary.get("metrics", {})
    value = metrics.get(metric) if isinstance(metrics, Mapping) else summary.get(metric)
    if value is None:
        value = summary.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _mean_metric(summaries: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [
        value
        for summary in summaries
        if (value := _metric_value(summary, metric)) is not None
    ]
    return None if not values else float(sum(values) / len(values))


def _cluster_bootstrap_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    adapted_key: str,
    baseline_key: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Return cluster-aware paired CIs for every formal query metric."""

    if n_bootstrap < 1:
        raise ValueError("bootstrap sample count must be positive")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from vg_tta.metrics import (
        cluster_macro_paired_bootstrap_ci,
        cluster_paired_bootstrap_ci,
    )

    metric_names = (
        "tIoU",
        "sIoU",
        "vIoU_corrected",
        "vIoU_at_0.3",
        "vIoU_at_0.5",
        "start_error_frames",
        "end_error_frames",
        "start_abs_error_frames",
        "end_abs_error_frames",
    )
    result: dict[str, Any] = {}
    for metric_index, metric in enumerate(metric_names):
        adapted_values: list[float] = []
        baseline_values: list[float] = []
        clusters: list[str] = []
        for row in rows:
            adapted = row.get(adapted_key)
            baseline = row.get(baseline_key)
            if not isinstance(adapted, Mapping) or not isinstance(baseline, Mapping):
                continue
            adapted_value = _metric_value(adapted, metric)
            baseline_value = _metric_value(baseline, metric)
            if (
                adapted_value is None
                or baseline_value is None
                or not math.isfinite(adapted_value)
                or not math.isfinite(baseline_value)
            ):
                continue
            adapted_values.append(adapted_value)
            baseline_values.append(baseline_value)
            clusters.append(str(row.get("query_cluster", row.get("query_vid", ""))))
        if not adapted_values:
            result[metric] = {
                "status": "unavailable",
                "reason": "no finite paired metric rows",
            }
            continue
        metric_seed = int(seed) + metric_index * 1009
        result[metric] = {
            "status": "passed",
            "cluster_micro": cluster_paired_bootstrap_ci(
                adapted_values,
                baseline_values,
                clusters,
                n_bootstrap=n_bootstrap,
                seed=metric_seed,
            ),
            "cluster_macro": cluster_macro_paired_bootstrap_ci(
                adapted_values,
                baseline_values,
                clusters,
                n_bootstrap=n_bootstrap,
                seed=metric_seed,
            ),
        }
    return result


def run_paired_seed_protocol(
    model: Any,
    dataset_object: Any,
    collate_fn: Any,
    *,
    mode: str,
    support_size: int = 64,
    support_seeds: Sequence[int] = (41001, 41002, 41003),
    query_count: int = 1,
    adaptation_steps: int = 1,
    learning_rate: float = 2e-4,
    optimizer_eps: float = 1e-4,
    weight_decay: float = 0.0,
    include_actionness: bool = False,
    reset_after: bool = True,
    device: Any | None = None,
    autocast_enabled: bool | None = None,
    autocast_dtype: str | None = "float16",
    trajectory_weight: float = 8.0,
    trajectory_temperature: float = 1.0,
    trajectory_top_k: int | None = 20,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260904,
) -> dict[str, Any]:
    """Run paired Frozen-vs-adapted rows for fixed queries.

    The critical K-shot invariant is enforced here: each support seed calls
    :func:`run_support_query_episode` exactly once with all query batches.  The
    optimizer therefore sees the K support clips once per adaptation step, and
    every query for that seed is evaluated under the same adapted state before
    the complete model reset.  Frozen and four-view TTAug baselines are also
    each evaluated once over the complete query set.
    """

    if device is None:
        device = _model_device(model)
    elif not hasattr(device, "type"):
        device = _resolve_device(device)
    protocol = _protocol_indices(
        dataset_object,
        support_size=support_size,
        support_seeds=support_seeds,
        query_count=query_count,
    )
    # Bound both host and device memory. VidSTG clips can be substantially
    # larger than HC clips; an unbounded decoded cache exceeded 29 GiB RSS.
    from functools import lru_cache

    import torch
    cache_identity = {
        "version": "tastvg-deterministic-eval-input-v1",
        "cfg": str(dataset_object.cfg),
        "dataset_records_sha256": protocol["dataset_records_sha256"],
        "official_repository_commit": _git_commit(OFFICIAL_REPO),
    }
    cache_fingerprint = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode()
    ).hexdigest()
    disk_cache = PROJECT_ROOT / "artifacts" / "tastvg_decoded_cache" / cache_fingerprint
    disk_cache.mkdir(parents=True, exist_ok=True)
    identity_path = disk_cache / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != cache_identity:
            raise RuntimeError("decoded cache identity mismatch")
    else:
        identity_path.write_text(json.dumps(cache_identity, indent=2) + "\n")

    @lru_cache(maxsize=2)
    def get_batch(index: int) -> dict[str, Any]:
        cache_path = disk_cache / f"{index}.pt"
        digest_path = cache_path.with_suffix(".sha256")
        if cache_path.exists() and digest_path.exists():
            if sha256_file(cache_path) != digest_path.read_text().strip():
                raise RuntimeError(f"decoded cache content checksum mismatch: {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        batch = _batch_for_dataset_item(dataset_object, index, collate_fn)
        temporary_path = cache_path.with_suffix(".tmp")
        torch.save(batch, temporary_path)
        temporary_path.replace(cache_path)
        digest_path.write_text(sha256_file(cache_path) + "\n")
        return batch

    class LazyBatches(Sequence):
        def __init__(self, indices: Sequence[int]) -> None:
            self.indices = tuple(int(index) for index in indices)

        def __len__(self) -> int:
            return len(self.indices)

        def source_vids(self) -> list[str]:
            return [str(dataset_object.all_gt_data[index]["vid"]) for index in self.indices]

        def __getitem__(self, position: Any) -> Any:
            if isinstance(position, slice):
                return LazyBatches(self.indices[position])
            return get_batch(self.indices[position])

    query_batches = LazyBatches(protocol["query_indices"])
    baseline_support_index = protocol["support_indices_by_seed"][
        str(protocol["support_seeds"][0])
    ][0]
    frozen_baseline = run_support_query_episode(
        model,
        get_batch(int(baseline_support_index)),
        query_batches,
        mode="frozen",
        adaptation_steps=0,
        learning_rate=learning_rate,
        optimizer_eps=optimizer_eps,
        weight_decay=weight_decay,
        include_actionness=include_actionness,
        reset_after=reset_after,
        device=device,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
        trajectory_weight=trajectory_weight,
        trajectory_temperature=trajectory_temperature,
        trajectory_top_k=trajectory_top_k,
    )
    ttaug_baseline = run_support_query_episode(
        model,
        get_batch(int(baseline_support_index)),
        query_batches,
        mode="ttaug",
        adaptation_steps=0,
        learning_rate=learning_rate,
        optimizer_eps=optimizer_eps,
        weight_decay=weight_decay,
        include_actionness=include_actionness,
        reset_after=reset_after,
        device=device,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
        trajectory_weight=trajectory_weight,
        trajectory_temperature=trajectory_temperature,
        trajectory_top_k=trajectory_top_k,
    )
    frozen_queries = list(frozen_baseline.get("queries", []))
    ttaug_queries = list(ttaug_baseline.get("queries", []))
    if len(frozen_queries) != len(query_batches) or len(ttaug_queries) != len(query_batches):
        raise RuntimeError("baseline episode did not return one summary per fixed query")

    rows: list[dict[str, Any]] = []
    for raw_seed in protocol["support_seeds"]:
        seed = int(raw_seed)
        _set_random_seed(seed, device)
        support_indices = protocol["support_indices_by_seed"][str(seed)]
        support_batches = LazyBatches(support_indices)
        # One call per seed is intentional: K support clips are accumulated in
        # one adapted model state, then all fixed queries are evaluated before
        # the finally-block restores the checkpoint.
        adapted = run_support_query_episode(
            model,
            support_batches,
            query_batches,
            mode=mode,
            adaptation_steps=adaptation_steps,
            learning_rate=learning_rate,
            optimizer_eps=optimizer_eps,
            weight_decay=weight_decay,
            include_actionness=include_actionness,
            reset_after=reset_after,
            device=device,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            trajectory_weight=trajectory_weight,
            trajectory_temperature=trajectory_temperature,
            trajectory_top_k=trajectory_top_k,
        )
        adapted_queries = list(adapted.get("queries", []))
        if len(adapted_queries) != len(query_batches):
            raise RuntimeError(
                "one adapted K-shot episode did not return one summary per fixed query"
            )
        for query_position, query_index in enumerate(protocol["query_indices"]):
            baseline_query = frozen_queries[query_position] or {}
            ttaug_query = ttaug_queries[query_position] or {}
            adapted_query = adapted_queries[query_position] or {}
            rows.append(
                {
                    "seed": seed,
                    "query_index": int(query_index),
                    "query_vid": str(dataset_object.all_gt_data[query_index]["vid"]),
                    "support_indices": [int(index) for index in support_indices],
                    "support_vids": [
                        str(dataset_object.all_gt_data[index]["vid"])
                        for index in support_indices
                    ],
                    "support_query_source_disjoint": bool(
                        adapted["support_query_source_disjoint"]
                    ),
                    "query_cluster": str(
                        adapted_query.get(
                            "query_cluster",
                            dataset_object.all_gt_data[query_index].get("vid", query_index),
                        )
                    ),
                    "mode": mode,
                    "baseline": baseline_query,
                    "ttaug_baseline": ttaug_query,
                    "adapted": adapted_query,
                    "delta_adapted_minus_baseline": _paired_numeric_delta(
                        baseline_query, adapted_query
                    ),
                    "delta_adapted_minus_ttaug": _paired_numeric_delta(
                        ttaug_query, adapted_query
                    ),
                    "reset_verified": adapted.get("reset_verified"),
                }
            )
    formal_metrics = (
        "tIoU",
        "sIoU",
        "vIoU_corrected",
        "vIoU_at_0.3",
        "vIoU_at_0.5",
        "start_error_frames",
        "end_error_frames",
        "start_abs_error_frames",
        "end_abs_error_frames",
    )
    summary: dict[str, Any] = {
        "frozen": {
            metric: _mean_metric(frozen_queries, metric) for metric in formal_metrics
        },
        "ttaug_4view": {
            metric: _mean_metric(ttaug_queries, metric) for metric in formal_metrics
        },
        "adapted": {
            metric: _mean_metric([row["adapted"] for row in rows], metric)
            for metric in formal_metrics
        },
        "adapted_minus_frozen": {
            metric: _mean_metric(
                [
                    {"metrics": {metric: row["delta_adapted_minus_baseline"].get(metric)}}
                    for row in rows
                    if metric in row["delta_adapted_minus_baseline"]
                ],
                metric,
            )
            for metric in formal_metrics
        },
        "adapted_minus_ttaug_4view": {
            metric: _mean_metric(
                [
                    {"metrics": {metric: row["delta_adapted_minus_ttaug"].get(metric)}}
                    for row in rows
                    if metric in row["delta_adapted_minus_ttaug"]
                ],
                metric,
            )
            for metric in formal_metrics
        },
        "legacy_temporal_iou_alias": {
            "frozen_mean": _mean_metric(frozen_queries, "tIoU"),
            "ttaug_4view_mean": _mean_metric(ttaug_queries, "tIoU"),
            "adapted_mean": _mean_metric([row["adapted"] for row in rows], "tIoU"),
        },
    }
    return {
        "status": "passed",
        "device": str(device),
        "mode": mode,
        "protocol": protocol,
        "decoded_input_cache": {"identity": cache_identity, "path": str(disk_cache),
                                "sha256_verified_each_load": True, "host_lru_size": 2},
        "episode_call_count": 2 + len(protocol["support_seeds"]),
        "adaptation_episode_call_count": len(protocol["support_seeds"]),
        "paired_rows": rows,
        "paired_row_count": len(rows),
        "all_rows_reset_verified": all(row["reset_verified"] is True for row in rows),
        "baseline_reset_verified": bool(
            frozen_baseline.get("reset_verified") is True
            and ttaug_baseline.get("reset_verified") is True
        ),
        "summary": summary,
        "cluster_bootstrap": {
            "adapted_minus_frozen": _cluster_bootstrap_metrics(
                rows,
                adapted_key="adapted",
                baseline_key="baseline",
                n_bootstrap=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "adapted_minus_ttaug_4view": _cluster_bootstrap_metrics(
                rows,
                adapted_key="adapted",
                baseline_key="ttaug_baseline",
                n_bootstrap=bootstrap_samples,
                seed=bootstrap_seed + 500_000,
            ),
        },
        "cluster_bootstrap_provenance": {
            "cluster_field": "query_cluster",
            "cluster_definition": "target vid/source video; all records from a source video share one cluster",
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "ci": "95 percent percentile paired cluster bootstrap",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="hcstvg2")
    parser.add_argument(
        "--source-dataset",
        choices=sorted(DATASET_DEFAULTS),
        default=None,
        help="dataset identity of the released source checkpoint (for natural shift)",
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="fail-safe CPU by default; CUDA requires this explicit opt-in",
    )
    parser.add_argument(
        "--autocast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable CUDA autocast (default: enabled for explicit CUDA)",
    )
    parser.add_argument(
        "--autocast-dtype",
        choices=["float16", "bfloat16", "none"],
        default="float16",
    )
    parser.add_argument("--prepare-runtime", action="store_true")
    parser.add_argument("--loader-smoke", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument(
        "--run-mode",
        choices=[
            "frozen",
            "ttaug",
            "ttaug-consistency",
            "boundary-entropy",
            "trajcal",
            "span-tta",
        ],
        default=None,
    )
    parser.add_argument("--adaptation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--optimizer-eps",
        type=float,
        default=1e-4,
        help="AdamW epsilon for temporal-head adaptation",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay for temporal-head adaptation",
    )
    parser.add_argument("--include-actionness", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="formal TA-STVG input resolution (CPU smoke should pass 64)",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=64,
        help="formal sampled frame count (CPU smoke should pass 8)",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--support-size",
        "--k",
        dest="support_size",
        type=int,
        default=64,
        help="K support clips for --paired-metrics (default: 64)",
    )
    parser.add_argument(
        "--support-seeds",
        "--seeds",
        dest="support_seeds",
        type=int,
        nargs="+",
        default=[41001, 41002, 41003],
        help="paired support-selection seeds (default: 41001 41002 41003)",
    )
    parser.add_argument("--query-count", type=int, default=1)
    parser.add_argument(
        "--paired-metrics",
        action="store_true",
        help="run fixed-query Frozen-vs-adapted rows for each support seed",
    )
    parser.add_argument("--trajectory-weight", type=float, default=8.0)
    parser.add_argument("--trajectory-temperature", type=float, default=1.0)
    parser.add_argument(
        "--trajectory-top-k",
        type=int,
        default=20,
        help="restrict TrajCal to base top-K spans; use 0 for all legal spans",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="cluster-bootstrap replicates for paired reports (smoke may use 100)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260904,
        help="base seed for deterministic cluster bootstrap reports",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.trajectory_top_k < 0:
        parser.error("--trajectory-top-k must be non-negative; use 0 for all spans")
    if args.optimizer_eps <= 0:
        parser.error("--optimizer-eps must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    trajectory_top_k = None if args.trajectory_top_k == 0 else args.trajectory_top_k
    requested_device = _resolve_device(args.device)
    if args.paired_metrics and args.run_mode is None:
        parser.error("--paired-metrics requires --run-mode")
    _set_random_seed(args.seed, requested_device)
    runtime_root = args.runtime_root.resolve()
    runtime_dir = runtime_root / str(_dataset_spec(args.dataset)["runtime_dirname"])
    source_dataset = args.source_dataset
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": args.dataset,
        "target_dataset": args.dataset,
        "source_dataset": source_dataset,
        "device": str(requested_device),
        "gpu_execution": (
            "not requested; fail-safe CPU default"
            if requested_device.type == "cpu"
            else "explicit CUDA opt-in requested"
        ),
        "forward_executed": False,
        "provenance": {
            "runner": _relative(Path(__file__)),
            "runner_sha256": sha256_file(Path(__file__)),
            "command": [str(value) for value in sys.argv],
            "project_root": str(PROJECT_ROOT.resolve()),
            "runtime_root": str(runtime_root),
            "runtime_dir": str(runtime_dir),
            "seed": int(args.seed),
            "device": str(requested_device),
            "source_dataset": source_dataset,
            "resolution": int(args.resolution),
            "sample_frames": int(args.sample_frames),
            "support_size": int(args.support_size),
            "support_seeds": [int(seed) for seed in args.support_seeds],
            "query_count": int(args.query_count),
            "paired_metrics": bool(args.paired_metrics),
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "autocast_enabled": args.autocast,
            "autocast_dtype": args.autocast_dtype,
            "trajectory_top_k": trajectory_top_k,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": float(args.learning_rate),
                "eps": float(args.optimizer_eps),
                "weight_decay": float(args.weight_decay),
            },
            "official_repository_url": "https://github.com/HengLan/TA-STVG",
            "official_repository_commit": _git_commit(OFFICIAL_REPO),
        },
        "official_repository": str(OFFICIAL_REPO.resolve()),
        "official_repository_commit": _git_commit(OFFICIAL_REPO),
        "requested": {
            "prepare_runtime": bool(args.prepare_runtime),
            "loader_smoke": bool(args.loader_smoke),
            "load_model": bool(args.load_model),
            "run_mode": args.run_mode,
            "paired_metrics": bool(args.paired_metrics),
        },
    }
    try:
        if args.prepare_runtime or not (runtime_dir / "manifest.json").is_file():
            report["runtime"] = prepare_runtime(
                args.dataset,
                source_root=args.source_root,
                runtime_root=runtime_root,
            )
        if args.loader_smoke:
            report["loader_smoke"] = loader_smoke(
                args.dataset,
                runtime_dir,
                resolution=args.resolution,
                sample_frames=args.sample_frames,
                seed=args.seed,
            )
        model = None
        if args.load_model or args.run_mode is not None:
            model, _cfg, model_report = load_model_on_device(
                args.dataset,
                runtime_dir,
                checkpoint=args.checkpoint,
                source_dataset=source_dataset,
                # Model input resolution changes tensor shapes but not the
                # checkpoint parameter shapes. Keep the CLI smoke small.
                resolution=args.resolution,
                sample_frames=args.sample_frames,
                device=requested_device,
            )
            report["model_load"] = model_report
            source_dataset = model_report.get("source_dataset", source_dataset)
            report["source_dataset"] = source_dataset
            report["provenance"]["source_dataset"] = source_dataset
        if args.run_mode is not None:
            if model is None:
                raise RuntimeError("internal error: --run-mode requires a loaded model")
            if args.paired_metrics:
                dataset_object, imports, _dataset_cfg = _build_runtime_dataset(
                    args.dataset,
                    runtime_dir,
                    resolution=args.resolution,
                    sample_frames=args.sample_frames,
                    seed=args.seed,
                )
                report["target_loader_compatibility"] = configure_target_loader_compatibility(
                    model, dataset_object, args.dataset
                )
                report["paired_metrics"] = run_paired_seed_protocol(
                    model,
                    dataset_object,
                    imports["collate_fn"],
                    mode=args.run_mode,
                    support_size=args.support_size,
                    support_seeds=args.support_seeds,
                    query_count=args.query_count,
                    adaptation_steps=args.adaptation_steps,
                    learning_rate=args.learning_rate,
                    optimizer_eps=args.optimizer_eps,
                    weight_decay=args.weight_decay,
                    include_actionness=args.include_actionness,
                    reset_after=not args.no_reset,
                    device=requested_device,
                    autocast_enabled=args.autocast,
                    autocast_dtype=args.autocast_dtype,
                    trajectory_weight=args.trajectory_weight,
                    trajectory_temperature=args.trajectory_temperature,
                    trajectory_top_k=trajectory_top_k,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=args.bootstrap_seed,
                )
                report["episode_selection"] = report["paired_metrics"]["protocol"]
                report["provenance"]["protocol_id"] = report["paired_metrics"][
                    "protocol"
                ]["protocol_id"]
                report["provenance"]["dataset_selection_sha256"] = report[
                    "paired_metrics"
                ]["protocol"]["dataset_selection_sha256"]
            else:
                dataset_object, imports, _dataset_cfg = _build_runtime_dataset(
                    args.dataset,
                    runtime_dir,
                    resolution=args.resolution,
                    sample_frames=args.sample_frames,
                    seed=args.seed,
                )
                report["target_loader_compatibility"] = configure_target_loader_compatibility(
                    model, dataset_object, args.dataset
                )
                indices = _shortest_distinct_indices(dataset_object, count=2)
                support = _batch_for_dataset_item(
                    dataset_object, indices[0], imports["collate_fn"]
                )
                query = _batch_for_dataset_item(
                    dataset_object, indices[1], imports["collate_fn"]
                )
                selection = {
                    "support_index": indices[0],
                    "query_index": indices[1],
                    "support_item_id": str(
                        dataset_object.all_gt_data[indices[0]]["item_id"]
                    ),
                    "query_item_id": str(
                        dataset_object.all_gt_data[indices[1]]["item_id"]
                    ),
                }
                report["episode_selection"] = selection
                report["episode"] = run_support_query_episode(
                    model,
                    support,
                    query,
                    mode=args.run_mode,
                    adaptation_steps=args.adaptation_steps,
                    learning_rate=args.learning_rate,
                    optimizer_eps=args.optimizer_eps,
                    weight_decay=args.weight_decay,
                    include_actionness=args.include_actionness,
                    reset_after=not args.no_reset,
                    device=requested_device,
                    autocast_enabled=args.autocast,
                    autocast_dtype=args.autocast_dtype,
                    trajectory_weight=args.trajectory_weight,
                    trajectory_temperature=args.trajectory_temperature,
                    trajectory_top_k=trajectory_top_k,
                )
            report["forward_executed"] = True
        if not any([args.prepare_runtime, args.loader_smoke, args.load_model, args.run_mode]):
            report["hint"] = "pass --prepare-runtime, --loader-smoke, --load-model, or --run-mode"
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    output = args.output or (RUNNER_ARTIFACT_ROOT / f"{args.dataset}_runtime_report.json")
    output = output if output.is_absolute() else PROJECT_ROOT / output
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
