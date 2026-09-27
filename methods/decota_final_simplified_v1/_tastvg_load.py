"""Strict TA-STVG loader compatibility, extracted from the sealed local loader.

Only the load_model_on_device call closure is retained (20 functions). No
adaptation baselines, GT metrics, dataset preparation or experiment runners.
Function bodies are identical; only this module's PROJECT_ROOT depth changes.
Upstream loading uses model_ema when provided, with strict tensor-shape checks.
See SOURCE_MAP.json. No remote downloads or experiment calls are added here.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

