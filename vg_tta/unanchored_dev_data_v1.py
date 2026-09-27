"""Read-only bridge for the existing TubeDETR development caches.

This module is deliberately narrower than an experiment runner.  It exposes
the already materialised, GT-free temporal-head inputs used by the unanchored
development groups and performs scoring only after a caller supplies every
prediction in a group.  In particular, it does not open the ``labels_only``
files from :func:`prepare_group` or :func:`load_features`.

Mouse and Football use the released AnyGroundBench sparse Vidi evaluator.  The
old ``scripts/run_ours_validation_tuning_v1.py`` dense ``prediction_record``
path is retained only for HC/Vid, where those datasets are native dense STVG
data.  MECCANO's fourteen train rows already have the same sparse Vidi cache
contract and are exposed as ``meccano14train``.

No model forward, optimiser step, GPU allocation, or file mutation is done by
this module.  ``load_source`` reconstructs the two-layer TubeDETR temporal MLP
from the locked serialized head and verifies its state fingerprint against the
locked checkpoint's ``model_ema.sted_embed`` state.  This makes the helper
usable by a later tuner without trusting an arbitrary pickled module class.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OURS_ROOT = ROOT / "artifacts" / "ours_validation_tuning_v1"
OURS_LOCK = OURS_ROOT / "lock.json"
OURS_RUNTIME_LOCK = OURS_ROOT / "runtime_lock.json"
OURS_PROTOCOL = ROOT / "protocols" / "ours_validation_tuning_v1.json"
MEC_ROOT = ROOT / "artifacts" / "meccano_available_evaluation_v1"
MEC_LOCK = MEC_ROOT / "lock.json"
MEC_PROTOCOL = ROOT / "protocols" / "meccano_available_evaluation_v1.json"
MEC_TEMPORAL_LOCK = ROOT / "artifacts" / "meccano_temporal_coordinate_v1" / "lock.json"
MEC_SOURCE_HEAD = ROOT / "artifacts" / "meccano_temporal_coordinate_v1" / "source_head_state.pt"

MOUSE_OFFICIAL_METADATA = (
    ROOT / "data" / "decoder_mouse_v1" / "source_release" / "data" / "animal" / "meta-data" / "st_train.json"
)
MOUSE_OFFICIAL_METADATA_SHA256 = "7b4896cf22f40f992c28c4264c99d1eb439ae82aea1614e8f319cac18a388226"
FOOTBALL_OFFICIAL_METADATA = (
    ROOT / "artifacts" / "anygroundbench_floor_check" / "hf_release" / "data" / "sports" / "meta-data" / "st_train.json"
)
FOOTBALL_OFFICIAL_METADATA_SHA256 = "2a5a5d91a10a9635e016ea383bcff482efd9044b65c154f03c7b1d5aeed0a5f6"
MEC_OFFICIAL_METADATA = ROOT / "data" / "stvg_domain_expansion_v2" / "industry" / "meta-data" / "st_train.json"
MEC_OFFICIAL_METADATA_SHA256 = "e422293d814fcca2f0210d636dd5eda4ba228ff71a3c3b80144957c4e57f1658"

CHECKPOINTS = {
    "hc": ROOT / "checkpoints" / "tubedetr_hcstvg2_res224_stride2.pth",
    "vid": ROOT / "checkpoints" / "tubedetr_vidstg_res224_stride2.pth",
}
CHECKPOINT_SHA256 = {
    "hc": "c646a8b7131bb28806c350e16479cd8bb31d4425ccb15129b26a6e4d38dd4222",
    "vid": "2802c66049b2e7986a826b4bbd291eb8521446cfba6fee03b56729438bf44213",
}

OURS_GROUPS = {
    "hc_to_vid": {"source": "hc", "target": "vid"},
    "vid_to_hc": {"source": "vid", "target": "hc"},
    "hc_to_mouse": {"source": "hc", "target": "mouse"},
    "hc_to_football": {"source": "hc", "target": "football"},
}
GROUPS = tuple((*OURS_GROUPS, "meccano14train"))
OFFICIAL_TARGETS = {"mouse", "football", "meccano14train"}
FORBIDDEN_CACHE_KEYS = {
    "targets",
    "annotation",
    "gt",
    "gt_boxes",
    "video_target",
    "labels",
    "spatio_temporal_label",
    "temporal_range",
    "sparse_bbox_frame_keys",
}


class DataContractError(RuntimeError):
    """Raised when a locked cache or scoring contract is not satisfied."""


def _absolute(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=None)
def _sha256_cached(path_string: str) -> str:
    path = Path(path_string)
    if not path.is_file():
        raise DataContractError(f"missing pinned file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: str | Path) -> str:
    # Keep the literal path in the cache key.  This avoids silently changing a
    # symlink-keyed provenance path into a different spelling while hashing.
    return _sha256_cached(str(_absolute(path)))


def _read_json(path: str | Path) -> Any:
    actual = _absolute(path)
    if not actual.is_file():
        raise DataContractError(f"missing JSON: {actual}")
    try:
        return json.loads(actual.read_text())
    except Exception as exc:  # pragma: no cover - gives a useful contract error
        raise DataContractError(f"invalid JSON: {actual}: {exc}") from exc


def _state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        if not torch.is_tensor(value):
            raise DataContractError(f"non-tensor source-head state at {name}")
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _normalize_head_state(value: Mapping[str, Any] | nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(value, nn.Module):
        raw = value.state_dict()
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise DataContractError(f"source head must be a module or state mapping, got {type(value).__name__}")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        if not torch.is_tensor(tensor):
            raise DataContractError(f"source head field {name} is not a tensor")
        state[str(name)] = tensor.detach().cpu().clone()
    expected = {"layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias"}
    if set(state) != expected:
        raise DataContractError(f"unexpected temporal-head keys: {sorted(state)}")
    expected_shapes = {
        "layers.0.weight": (256, 256),
        "layers.0.bias": (256,),
        "layers.1.weight": (2, 256),
        "layers.1.bias": (2,),
    }
    for name, shape in expected_shapes.items():
        if tuple(state[name].shape) != shape or not torch.isfinite(state[name]).all().item():
            raise DataContractError(f"invalid temporal-head state {name}: {tuple(state[name].shape)}")
    return state


class _TemporalMLP(nn.Module):
    """Numerically matching ``models.tubedetr.MLP(256,256,2,2,dropout=.5)``."""

    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 2
        self.layers = nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 2)])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
            if self.dropout and index < self.num_layers:
                x = self.dropout(x)
        return x


class _MlpUnpickleAliases:
    """Temporarily map the trusted cached MLP pickle to our local class."""

    def __enter__(self) -> "_MlpUnpickleAliases":
        self._old = {name: sys.modules.get(name) for name in ("models", "models.tubedetr")}
        package = types.ModuleType("models")
        package.__path__ = []  # type: ignore[attr-defined]
        module = types.ModuleType("models.tubedetr")
        module.MLP = _TemporalMLP
        package.tubedetr = module  # type: ignore[attr-defined]
        sys.modules["models"] = package
        sys.modules["models.tubedetr"] = module
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for name, old in self._old.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_head_artifact(path: Path) -> dict[str, torch.Tensor]:
    with _MlpUnpickleAliases():
        value = torch.load(path, map_location="cpu", weights_only=False)
    return _normalize_head_state(value)


@lru_cache(maxsize=None)
def _checkpoint_head_fingerprint(path_string: str, expected_sha256: str) -> str:
    path = Path(path_string)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise DataContractError(f"checkpoint SHA mismatch: {path}")
    # This is intentionally the one place that opens the large checkpoint.
    # The result is memoized so HC is not read three times for the three HC
    # source groups and neither source is read once per query.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    key = "model_ema" if "model_ema" in checkpoint else "model"
    if key not in checkpoint:
        raise DataContractError(f"checkpoint has no model_ema/model state: {path}")
    state = {
        name[len("sted_embed.") :]: tensor.detach().cpu()
        for name, tensor in checkpoint[key].items()
        if name.startswith("sted_embed.")
    }
    return _state_fingerprint(_normalize_head_state(state))


def _expected_official(target: str) -> tuple[Path, str]:
    if target == "mouse":
        return MOUSE_OFFICIAL_METADATA, MOUSE_OFFICIAL_METADATA_SHA256
    if target == "football":
        return FOOTBALL_OFFICIAL_METADATA, FOOTBALL_OFFICIAL_METADATA_SHA256
    if target == "meccano14train":
        return MEC_OFFICIAL_METADATA, MEC_OFFICIAL_METADATA_SHA256
    raise DataContractError(f"no sparse official metadata for {target}")


def _query_id_for(target: str, index: int, metadata: Mapping[str, Any]) -> str:
    if target == "mouse":
        return f"mouse_train_{index}"
    if target == "football":
        value = metadata.get("official_annotation_key")
        if not value:
            raise DataContractError(f"Football query {index} has no official_annotation_key")
        return str(value)
    return f"{target}_{index}"


def _ours_group(group: str) -> dict[str, Any]:
    if group not in OURS_GROUPS:
        raise DataContractError(f"unknown Ours development group: {group}")
    lock = _read_json(OURS_LOCK)
    protocol = _read_json(OURS_PROTOCOL)
    relation = lock.get("groups", {}).get(group)
    expected_relation = OURS_GROUPS[group]
    if relation != expected_relation:
        raise DataContractError(f"locked relation for {group} changed: {relation!r}")
    source, target = expected_relation["source"], expected_relation["target"]
    dataset = lock.get("datasets", {}).get(target)
    if not isinstance(dataset, Mapping):
        raise DataContractError(f"missing locked target dataset {target}")
    indices = [int(value) for value in dataset.get("indices", [])]
    if not indices or len(indices) != int(dataset.get("query_count", -1)):
        raise DataContractError(f"invalid locked query count for {group}")
    folder = OURS_ROOT / group
    complete_path = folder / "cache_complete.json"
    complete = _read_json(complete_path)
    if complete.get("indices") != indices or complete.get("gt_used") is not False:
        raise DataContractError(f"{group} cache_complete is incomplete or not GT-free")
    runtime_lock_sha = str(complete.get("runtime_lock_sha256", ""))
    if not runtime_lock_sha or not OURS_RUNTIME_LOCK.is_file() or _sha256(OURS_RUNTIME_LOCK) != runtime_lock_sha:
        raise DataContractError(f"runtime lock mismatch for {group}")
    cache_shas = complete.get("cache_sha256")
    label_shas = complete.get("label_sha256")
    if not isinstance(cache_shas, Mapping) or not isinstance(label_shas, Mapping):
        raise DataContractError(f"cache_complete lacks cache/label SHA maps for {group}")
    source_head_path = folder / "source_head.pt"
    checkpoint_path = _absolute(lock["checkpoints"][source])
    expected_checkpoint_sha = str(lock["checkpoint_sha256"][source])
    if _sha256(checkpoint_path) != expected_checkpoint_sha:
        raise DataContractError(f"locked {source} checkpoint changed")
    if _sha256(source_head_path) != complete.get("source_head_sha256"):
        raise DataContractError(f"locked source head changed for {group}")

    metadata_path, metadata_sha = _expected_official(target) if target in OFFICIAL_TARGETS else (None, None)
    queries: list[dict[str, Any]] = []
    for index in indices:
        key = str(index)
        identity = complete.get("query_identities", {}).get(key)
        qmeta = dataset.get("query_metadata", {}).get(key)
        if not isinstance(identity, Mapping) or not isinstance(qmeta, Mapping):
            raise DataContractError(f"missing query identity/metadata for {group}:{index}")
        cache_path = folder / "cache" / f"{index:06d}.pt"
        cache_sha = cache_shas.get(key)
        if not cache_sha or _sha256(cache_path) != cache_sha:
            raise DataContractError(f"cache SHA mismatch for {group}:{index}")
        label_path = folder / "validation_labels_only" / f"{index:06d}.pt"
        label_sha = label_shas.get(key)
        if not label_sha or not label_path.is_file():
            raise DataContractError(f"missing label provenance for {group}:{index}")
        if str(identity.get("source")) != str(dataset.get("sources", {}).get(key)):
            raise DataContractError(f"source identity mismatch for {group}:{index}")
        query_id = _query_id_for(target, index, qmeta)
        query: dict[str, Any] = {
            "index": index,
            "query_id": query_id,
            "source": str(identity["source"]),
            "source_cluster": str(qmeta.get("source_cluster", identity["source"])),
            "caption": str(qmeta.get("caption", "")),
            "identity": copy.deepcopy(dict(identity)),
            "cache_path": str(cache_path.absolute()),
            "cache_sha256": str(cache_sha),
            "runtime_lock_path": str(OURS_RUNTIME_LOCK.absolute()),
            "runtime_lock_sha256": runtime_lock_sha,
            "head_input_source": "cache.head_input",
            "labels_path": str(label_path.absolute()),
            "label_sha256": str(label_sha),
            "labels_loaded_stage": "score_predictions_after_all_group_predictions",
            "official_annotation_key": query_id if target in OFFICIAL_TARGETS else None,
            "official_metadata_path": str(metadata_path.absolute()) if metadata_path else None,
            "official_metadata_sha256": metadata_sha,
            "official_sample_ids_source": (
                "validation_labels_only.video_target.frames_id; local TubeDETR grid, read after prediction barrier"
                if target in {"mouse", "football"}
                else "validation_labels_only.video_target.frames_id"
            ),
            "official_sample_ids_are_model_mandate": False if target in {"mouse", "football"} else None,
        }
        for field in ("video_path", "video_filename", "fps", "frame_count", "width", "height", "original_video_id"):
            if field in qmeta:
                query[field] = copy.deepcopy(qmeta[field])
        queries.append(query)

    query_shape_summary = _attach_cache_shapes(queries)
    pins = {
        str(OURS_LOCK.absolute()): _sha256(OURS_LOCK),
        str(OURS_PROTOCOL.absolute()): _sha256(OURS_PROTOCOL),
        str(complete_path.absolute()): _sha256(complete_path),
        str(source_head_path.absolute()): str(complete["source_head_sha256"]),
        str(checkpoint_path.absolute()): expected_checkpoint_sha,
        str(OURS_RUNTIME_LOCK.absolute()): runtime_lock_sha,
        str(_absolute(dataset["annotation"]).absolute()): str(dataset["annotation_sha256"]),
    }
    pins.update({q["cache_path"]: q["cache_sha256"] for q in queries})
    pins.update({q["labels_path"]: q["label_sha256"] for q in queries})
    if metadata_path is not None:
        # This is a deferred GT pin: do not open the official metadata before
        # score_predictions establishes the complete prediction barrier.
        pins[str(metadata_path.absolute())] = str(metadata_sha)
    return {
        "schema_version": "unanchored_dev_data_v1_group",
        "group": group,
        "kind": "ours_validation_tuning_cache",
        "source_dataset": source,
        "target_dataset": target,
        "source_checkpoint": str(checkpoint_path.absolute()),
        "source_checkpoint_sha256": expected_checkpoint_sha,
        "checkpoint": str(checkpoint_path.absolute()),
        "checkpoint_sha256": expected_checkpoint_sha,
        "source_head_path": str(source_head_path.absolute()),
        "source_head_sha256": str(complete["source_head_sha256"]),
        "source_head_format": "serialized_tubedetr_mlp; state is checked against checkpoint model_ema by load_source",
        "source_head_parameter_count": 66306,
        "metric_name": "official_vidi_volume_iou" if target in OFFICIAL_TARGETS else "vIoU_corrected",
        "metric_contract": (
            "AnyGroundBench released sparse Vidi; prediction boxes are interpolated only to released GT timestamps; GT is never filled"
            if target in OFFICIAL_TARGETS
            else "TubeDETR corrected dense STVG vIoU on the native dataset's validation labels"
        ),
        "query_count": len(queries),
        "source_count": len({q["source_cluster"] for q in queries}),
        "source_multiplicity": dict(Counter(q["source_cluster"] for q in queries)),
        "queries": queries,
        "cache_head_input_shape_summary": query_shape_summary,
        "labels": {
            "fit_input_contains_gt": False,
            "labels_loaded_in_prepare": False,
            "labels_loaded_in_load_features": False,
            "labels_loaded_in": "score_predictions only, after exact all-query prediction barrier",
            "path_pattern": str((folder / "validation_labels_only").absolute()) if target not in {"meccano14train"} else None,
        },
        "pins": pins,
        "runtime": {
            "lock": str(OURS_RUNTIME_LOCK.absolute()),
            "lock_sha256": runtime_lock_sha,
            "cache_items_bind_runtime_lock": True,
        },
        "protocol_path": str(OURS_PROTOCOL.absolute()),
        "protocol_sha256": _sha256(OURS_PROTOCOL),
        "no_gpu_or_model_forward": True,
    }


def _meccano_group() -> dict[str, Any]:
    lock = _read_json(MEC_LOCK)
    protocol = _read_json(MEC_PROTOCOL)
    temporal_lock = _read_json(MEC_TEMPORAL_LOCK)
    train_rows = [row for row in lock.get("queries", []) if row.get("role") == "train"]
    if len(train_rows) != 14 or lock.get("split_counts", {}).get("train") != 14:
        raise DataContractError(f"MECCANO train roster is not exactly 14 rows: {len(train_rows)}")
    source_head_path = _absolute(temporal_lock["source_head_state"])
    checkpoint_path = _absolute(lock["checkpoint"])
    checkpoint_sha = str(lock["checkpoint_sha256"])
    if _sha256(checkpoint_path) != checkpoint_sha:
        raise DataContractError("MECCANO HC checkpoint changed")
    source_head_sha = str(temporal_lock["source_head_state_sha256"])
    if _sha256(source_head_path) != source_head_sha:
        raise DataContractError("MECCANO source head state changed")
    mec_lock_sha = _sha256(MEC_LOCK)
    queries: list[dict[str, Any]] = []
    for row in train_rows:
        query_id = str(row["query_id"])
        cache_path = MEC_ROOT / "cache" / f"{query_id}.pt"
        if not cache_path.is_file():
            raise DataContractError(f"missing MECCANO train cache {query_id}")
        cache_sha = _sha256(cache_path)
        sample_ids = [int(value) for value in row["sample_ids"]]
        queries.append({
            "index": query_id,
            "query_id": query_id,
            "source": str(row["source_cluster"]),
            "source_cluster": str(row["source_cluster"]),
            "caption": str(row["caption"]),
            "identity": {
                "query_id": query_id,
                "source_cluster": str(row["source_cluster"]),
                "split": "train",
                "raw_source_sha256": str(row["raw_source_sha256"]),
            },
            "cache_path": str(cache_path.absolute()),
            "cache_sha256": cache_sha,
            "cache_lock_sha256": mec_lock_sha,
            "head_input_source": "cache.head_input",
            "sample_ids": sample_ids,
            "sample_ids_sha256": str(row.get("sample_ids_sha256")),
            "official_annotation_key": query_id,
            "official_metadata_path": str(MEC_OFFICIAL_METADATA.absolute()),
            "official_metadata_sha256": MEC_OFFICIAL_METADATA_SHA256,
            "official_sample_ids_source": "meccano_available_evaluation_v1/lock.json query.sample_ids; GT-free physical grid",
            "official_sample_ids_are_model_mandate": True,
            "metadata": str(_absolute(row["metadata"]).absolute()),
            "metadata_sha256": str(row["metadata_sha256"]),
            "raw_source": str(_absolute(row["raw_source"]).absolute()),
            "raw_source_sha256": str(row["raw_source_sha256"]),
            "media": copy.deepcopy(row["media"]),
            "fps": float(row["media"]["fps"]),
            "frame_count": int(row["media"]["frame_count"]),
            "width": int(row["media"]["width"]),
            "height": int(row["media"]["height"]),
            "split": "train",
        })
    query_shape_summary = _attach_cache_shapes(queries)
    pins = {
        str(MEC_LOCK.absolute()): _sha256(MEC_LOCK),
        str(MEC_PROTOCOL.absolute()): _sha256(MEC_PROTOCOL),
        str(MEC_TEMPORAL_LOCK.absolute()): _sha256(MEC_TEMPORAL_LOCK),
        str(source_head_path.absolute()): source_head_sha,
        str(checkpoint_path.absolute()): checkpoint_sha,
        str(MEC_OFFICIAL_METADATA.absolute()): MEC_OFFICIAL_METADATA_SHA256,
    }
    pins.update({q["cache_path"]: q["cache_sha256"] for q in queries})
    return {
        "schema_version": "unanchored_dev_data_v1_group",
        "group": "meccano14train",
        "kind": "meccano_available_train_cache",
        "source_dataset": "hc",
        "target_dataset": "meccano14train",
        "source_checkpoint": str(checkpoint_path.absolute()),
        "source_checkpoint_sha256": checkpoint_sha,
        "checkpoint": str(checkpoint_path.absolute()),
        "checkpoint_sha256": checkpoint_sha,
        "source_head_path": str(source_head_path.absolute()),
        "source_head_sha256": source_head_sha,
        "source_head_format": "state_dict_tubedetr_mlp; extracted model_ema.sted_embed",
        "source_head_parameter_count": 66306,
        "metric_name": "official_vidi_volume_iou",
        "metric_contract": "AnyGroundBench released sparse Vidi; original sparse GT and physical sample grid remain unchanged",
        "query_count": len(queries),
        "source_count": len({q["source_cluster"] for q in queries}),
        "source_multiplicity": dict(Counter(q["source_cluster"] for q in queries)),
        "queries": queries,
        "cache_head_input_shape_summary": query_shape_summary,
        "labels": {
            "fit_input_contains_gt": False,
            "labels_loaded_in_prepare": False,
            "labels_loaded_in_load_features": False,
            "labels_loaded_in": "official metadata only inside score_predictions after exact all-query prediction barrier",
            "path_pattern": None,
        },
        "pins": pins,
        "runtime": {
            "checkpoint": str(checkpoint_path.absolute()),
            "checkpoint_sha256": checkpoint_sha,
            "cache_lock_sha256": mec_lock_sha,
        },
        "protocol_path": str(MEC_PROTOCOL.absolute()),
        "protocol_sha256": _sha256(MEC_PROTOCOL),
        "no_gpu_or_model_forward": True,
        "source_cluster_note": "All 14 development queries are one MECCANO source cluster (meccano:0005); source-macro is therefore one-cluster evidence.",
    }


def _attach_cache_shapes(queries: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    frozen_box_shapes: Counter[str] = Counter()
    frozen_sted_shapes: Counter[str] = Counter()
    for query in queries:
        loaded = load_features(query)
        head_input = loaded["head_input"]
        frozen = loaded["frozen"]
        query["head_input_shape"] = list(head_input.shape)
        query["head_input_dtype"] = str(head_input.dtype)
        query["head_input_finite"] = bool(torch.isfinite(head_input).all().item())
        query["frozen_pred_boxes_shape"] = list(frozen["pred_boxes"].shape)
        query["frozen_pred_sted_shape"] = list(frozen["pred_sted"].shape)
        counts["x".join(map(str, head_input.shape))] += 1
        dtypes[str(head_input.dtype)] += 1
        frozen_box_shapes["x".join(map(str, frozen["pred_boxes"].shape))] += 1
        frozen_sted_shapes["x".join(map(str, frozen["pred_sted"].shape))] += 1
    return {
        "head_input": dict(counts),
        "dtype": dict(dtypes),
        "frozen_pred_boxes": dict(frozen_box_shapes),
        "frozen_pred_sted": dict(frozen_sted_shapes),
        "all_gt_free_finite": True,
    }


def prepare_group(group: str) -> dict[str, Any]:
    """Return an immutable, GT-free description of one development group."""

    if group in OURS_GROUPS:
        return _ours_group(group)
    if group == "meccano14train":
        return _meccano_group()
    raise DataContractError(f"unknown development group {group!r}; expected {GROUPS}")


def _assert_no_forbidden_keys(value: Any, path: str = "cache") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_CACHE_KEYS:
                raise DataContractError(f"GT-like cache field {path}.{key}")
            _assert_no_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, f"{path}[{index}]")


def load_features(query: Mapping[str, Any]) -> dict[str, Any]:
    """Load one cached head input and frozen temporal output, never labels."""

    if not isinstance(query, Mapping) or "cache_path" not in query:
        raise DataContractError("query must be a prepared query mapping")
    path = _absolute(str(query["cache_path"]))
    expected_sha = query.get("cache_sha256")
    if expected_sha and _sha256(path) != str(expected_sha):
        raise DataContractError(f"cache changed: {path}")
    item = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(item, Mapping):
        raise DataContractError(f"cache is not a mapping: {path}")
    _assert_no_forbidden_keys(item)
    if item.get("gt_used") not in (None, False):
        raise DataContractError(f"cache declares GT use: {path}")
    expected_runtime_sha = query.get("runtime_lock_sha256")
    if expected_runtime_sha is not None and item.get("runtime_lock_sha256") != str(expected_runtime_sha):
        raise DataContractError(f"cache runtime lock mismatch: {path}")
    expected_cache_lock_sha = query.get("cache_lock_sha256")
    if expected_cache_lock_sha is not None and item.get("lock_sha256") != str(expected_cache_lock_sha):
        raise DataContractError(f"cache lock mismatch: {path}")
    head_input = item.get("head_input")
    if not torch.is_tensor(head_input) or head_input.ndim != 4 or tuple(head_input.shape[:2]) != (6, 1) or int(head_input.shape[3]) != 256:
        raise DataContractError(f"invalid head_input shape in {path}: {getattr(head_input, 'shape', None)}")
    if head_input.requires_grad or not torch.isfinite(head_input).all().item():
        raise DataContractError(f"head_input is not detached finite data: {path}")
    frozen = item.get("frozen")
    if frozen is None and isinstance(item.get("predictions"), Mapping):
        frozen = item["predictions"].get("frozen")
    if not isinstance(frozen, Mapping) or not torch.is_tensor(frozen.get("pred_boxes")) or not torch.is_tensor(frozen.get("pred_sted")):
        raise DataContractError(f"cache has no frozen pred_boxes/pred_sted: {path}")
    boxes = frozen["pred_boxes"]
    sted = frozen["pred_sted"]
    if boxes.ndim != 2 or tuple(boxes.shape[1:]) != (4,) or int(boxes.shape[0]) != int(head_input.shape[2]):
        raise DataContractError(f"frozen boxes/head-input length mismatch: {path}")
    if sted.ndim not in (2, 3) or int(sted.shape[-1]) != 2 or int(sted.shape[-2]) != int(head_input.shape[2]):
        raise DataContractError(f"frozen temporal logits/head-input length mismatch: {path}")
    if not torch.isfinite(boxes).all().item() or not torch.isfinite(sted).all().item():
        raise DataContractError(f"frozen outputs are non-finite: {path}")
    cache_identity = item.get("identity")
    if cache_identity is not None and dict(cache_identity) != dict(query.get("identity", {})):
        raise DataContractError(f"cache identity mismatch: {path}")
    if query.get("query_id", "").startswith("meccano_"):
        if item.get("query_id") != query.get("query_id") or item.get("source_cluster") != query.get("source_cluster"):
            raise DataContractError(f"MECCANO cache identity mismatch: {path}")
    return {
        "head_input": head_input.detach().cpu().clone(),
        "frozen": {
            "pred_boxes": boxes.detach().cpu().clone(),
            "pred_sted": sted.detach().cpu().clone(),
        },
        "identity": copy.deepcopy(dict(query.get("identity", cache_identity or {}))),
    }


def load_source(group_spec: Mapping[str, Any]) -> nn.Module:
    """Load the locked temporal head and verify it against checkpoint ``model_ema``."""

    if not isinstance(group_spec, Mapping):
        raise DataContractError("group_spec must be a prepared mapping")
    checkpoint_path = _absolute(str(group_spec.get("source_checkpoint", group_spec.get("checkpoint"))))
    checkpoint_sha = str(group_spec.get("source_checkpoint_sha256", group_spec.get("checkpoint_sha256")))
    head_path = _absolute(str(group_spec["source_head_path"]))
    if _sha256(head_path) != str(group_spec["source_head_sha256"]):
        raise DataContractError(f"source head changed: {head_path}")
    artifact_state = _load_head_artifact(head_path)
    artifact_fp = _state_fingerprint(artifact_state)
    checkpoint_fp = _checkpoint_head_fingerprint(str(checkpoint_path), checkpoint_sha)
    if artifact_fp != checkpoint_fp:
        raise DataContractError(
            f"source head does not match locked checkpoint model_ema: {head_path} vs {checkpoint_path}"
        )
    head = _TemporalMLP()
    head.load_state_dict(artifact_state, strict=True)
    head.eval().requires_grad_(False)
    return head


def _normalize_prediction_mapping(
    group_spec: Mapping[str, Any], predictions: Mapping[Any, Mapping[str, Any]]
) -> dict[int | str, Mapping[str, Any]]:
    if not isinstance(predictions, Mapping):
        raise DataContractError("predictions must be a mapping keyed by every prepared query index")
    queries = list(group_spec.get("queries", []))
    expected_raw = [q["index"] for q in queries]
    expected = {str(value) for value in expected_raw}
    canonical_keys = [str(value) for value in predictions]
    if len(canonical_keys) != len(set(canonical_keys)):
        raise DataContractError("prediction barrier failed: duplicate int/string query keys")
    actual = set(canonical_keys)
    if actual != expected:
        raise DataContractError(
            "prediction barrier failed: expected exactly "
            f"{len(expected)} query predictions, missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    normalized: dict[int | str, Mapping[str, Any]] = {}
    for query in queries:
        key = query["index"]
        value = predictions.get(key, predictions.get(str(key)))
        if not isinstance(value, Mapping):
            raise DataContractError(f"prediction {key!r} is not a mapping")
        if any(str(field).lower() in FORBIDDEN_CACHE_KEYS for field in value):
            raise DataContractError(f"prediction {key!r} contains GT-like fields")
        boxes, sted = value.get("pred_boxes"), value.get("pred_sted")
        if not torch.is_tensor(boxes) or not torch.is_tensor(sted):
            raise DataContractError(f"prediction {key!r} needs tensor pred_boxes and pred_sted")
        if boxes.ndim != 2 or tuple(boxes.shape[1:]) != (4,) or sted.ndim not in (2, 3) or int(sted.shape[-1]) != 2:
            raise DataContractError(f"prediction {key!r} has invalid output shapes")
        if int(boxes.shape[0]) != int(sted.shape[-2]) or not torch.isfinite(boxes).all().item() or not torch.isfinite(sted).all().item():
            raise DataContractError(f"prediction {key!r} is non-finite or temporally misaligned")
        normalized[key] = value
    return normalized


def _load_label_file(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise DataContractError(f"labels file is not a mapping: {path}")
    return value


def _official_media(metadata: Mapping[str, Any], query: Mapping[str, Any]) -> dict[str, Any]:
    info = metadata.get("meta_info")
    if not isinstance(info, Mapping):
        raise DataContractError(f"official metadata lacks meta_info for {query['query_id']}")
    # ``media`` is the locked prediction timeline.  In particular, MECCANO's
    # released GT ``meta_info`` has no ``img_num``; its physical clip geometry
    # lives in the evaluation lock.  GT geometry/fps is still taken by the
    # official evaluator from ``metadata['meta_info']`` and is not silently
    # replaced with prediction-media values.
    query_media = query.get("media")
    if isinstance(query_media, Mapping):
        media = {
            "fps": float(query_media["fps"]),
            "frame_count": int(query_media["frame_count"]),
            "width": int(query_media["width"]),
            "height": int(query_media["height"]),
        }
    elif all(field in query for field in ("fps", "frame_count", "width", "height")):
        media = {
            "fps": float(query["fps"]),
            "frame_count": int(query["frame_count"]),
            "width": int(query["width"]),
            "height": int(query["height"]),
        }
    else:
        try:
            media = {
                "fps": float(info["fps"]),
                "frame_count": int(info["img_num"]),
                "width": int(info["width"]),
                "height": int(info["height"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise DataContractError(f"prediction media geometry missing for {query['query_id']}") from exc
    if media["fps"] <= 0 or media["frame_count"] < 2 or media["width"] <= 0 or media["height"] <= 0:
        raise DataContractError(f"invalid prediction media geometry for {query['query_id']}")
    return media


def _normal_caption(value: Any) -> str:
    return " ".join(str(value).split())


def _verify_official_identity(
    target: str, query: Mapping[str, Any], official_key: str, metadata: Mapping[str, Any], official: Mapping[str, Any]
) -> None:
    """Bind the locked local query to a released row, beyond a row count."""

    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("meta_info"), Mapping):
        raise DataContractError(f"official row has no meta_info: {official_key}")
    if _normal_caption(metadata.get("text")) != _normal_caption(query.get("caption")):
        raise DataContractError(f"official/local caption mismatch for {query['query_id']}")
    info = metadata["meta_info"]
    if target == "mouse":
        # Verify the index-derived key by the released media filename and
        # source clip, rather than trusting that both files happen to have
        # 108 entries in the same order.
        filename = str(query.get("video_filename", ""))
        matches = []
        for key, row in official.items():
            if not isinstance(row, Mapping) or not isinstance(row.get("meta_info"), Mapping):
                continue
            if (
                str(row["meta_info"].get("source_video_name", "")) == filename
                and _normal_caption(row.get("text")) == _normal_caption(query.get("caption"))
            ):
                matches.append(str(key))
        if matches != [official_key]:
            raise DataContractError(
                f"Mouse released media does not uniquely bind {query['query_id']}: {matches!r}"
            )
        if str(info.get("clip_id", "")) != str(query.get("source_cluster", "")):
            raise DataContractError(f"Mouse source-cluster mismatch for {query['query_id']}")
    elif target == "football":
        source_name = str(info.get("source_video_name", ""))
        expected_source = f"sports:{Path(source_name).stem}" if source_name else ""
        if expected_source != str(query.get("source_cluster", "")):
            raise DataContractError(f"Football source-cluster mismatch for {query['query_id']}")
    elif target == "meccano14train":
        source_name = str(info.get("source_video_name", ""))
        expected_source = f"meccano:{Path(source_name).stem}" if source_name else ""
        if expected_source != str(query.get("source_cluster", "")):
            raise DataContractError(f"MECCANO source-cluster mismatch for {query['query_id']}")


def _score_official(
    query: Mapping[str, Any], prediction: Mapping[str, Any], metadata: Mapping[str, Any], sample_ids: Sequence[int],
    *, cached_grid: bool,
) -> dict[str, Any]:
    boxes = prediction["pred_boxes"]
    sted = prediction["pred_sted"]
    if len(sample_ids) != int(boxes.shape[0]):
        raise DataContractError(f"official sample grid/prediction mismatch for {query['query_id']}")
    media = _official_media(metadata, query)
    if cached_grid:
        # Mouse/Football's locked TubeDETR grids are often endpoint-short.
        # The endpoint-requiring sparse helper would reject these historical
        # grids; cached_grid evaluates them without changing the grid or GT.
        from vg_tta.anyground_cached_grid_eval_v1 import evaluate_cached_grid

        metrics = evaluate_cached_grid(
            {"pred_boxes": boxes, "pred_sted": sted}, list(map(int, sample_ids)), metadata, media
        )
    else:
        from vg_tta.anyground_sparse_eval import evaluate_sparse

        metrics = evaluate_sparse(
            {"pred_boxes": boxes, "pred_sted": sted}, list(map(int, sample_ids)), metadata, media
        )
    metric = float(metrics["official_vidi_volume_iou"])
    return {
        "index": query["index"],
        "query_id": query["query_id"],
        "source": query["source"],
        "source_cluster": query["source_cluster"],
        "metric_name": "official_vidi_volume_iou",
        "metric": metric,
        "metrics": metrics,
        "diagnostics": {
            "labels_loaded_after_prediction_barrier": True,
            "sample_ids_source": query["official_sample_ids_source"],
            "official_metadata_path": query["official_metadata_path"],
        },
    }


def _score_dense(query: Mapping[str, Any], prediction: Mapping[str, Any], label: Mapping[str, Any]) -> dict[str, Any]:
    from vg_tta.metrics import compute_stvg_metrics, interval_from_logits

    try:
        targets = label["targets"]
        video_target = label["video_target"]
        annotation = label["annotation"]
        frame_ids = [int(value) for value in video_target["frames_id"]]
        pred_indices = interval_from_logits(prediction["pred_sted"])
        gt_indices = tuple(video_target["inter_idx"])
        metrics = compute_stvg_metrics(
            prediction["pred_boxes"],
            targets,
            pred_indices,
            gt_indices,
            frame_ids=frame_ids,
            gt_frame_interval=(annotation["tube_start_frame"], annotation["tube_end_frame"]),
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise DataContractError(f"dense label/cache schema failed for {query['query_id']}: {exc}") from exc
    return {
        "index": query["index"],
        "query_id": query["query_id"],
        "source": query["source"],
        "source_cluster": query["source_cluster"],
        "metric_name": "vIoU_corrected",
        "metric": float(metrics["vIoU_corrected"]),
        "metrics": metrics,
        "diagnostics": {
            "labels_loaded_after_prediction_barrier": True,
            "sample_ids_source": "validation_labels_only.video_target.frames_id",
            "legacy_dense_proxy": True,
        },
    }


def score_predictions(
    group_spec: Mapping[str, Any], predictions: Mapping[Any, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Score a complete group, loading labels only after the prediction barrier.

    The exact-key check is intentionally before *any* label or official
    metadata read.  A partial mapping therefore fails closed and cannot turn a
    failed GT-free prediction stage into a partially scored development row.
    """

    normalized = _normalize_prediction_mapping(group_spec, predictions)
    target = str(group_spec.get("target_dataset"))
    queries = list(group_spec.get("queries", []))
    if len(queries) != int(group_spec.get("query_count", -1)):
        raise DataContractError("prepared group query count changed")

    # The barrier is complete here.  Everything below is label/official-GT
    # access and must remain below this line.
    if target in {"mouse", "football"}:
        metadata_path = _absolute(str(group_spec["queries"][0]["official_metadata_path"]))
        expected = str(group_spec["queries"][0]["official_metadata_sha256"])
        if _sha256(metadata_path) != expected:
            raise DataContractError(f"official metadata changed: {metadata_path}")
        official = _read_json(metadata_path)
        if not isinstance(official, Mapping):
            raise DataContractError(f"official metadata is not a mapping: {metadata_path}")
        labels: dict[int, Mapping[str, Any]] = {}
        sample_ids: dict[int, list[int]] = {}
        for query in queries:
            label_path = _absolute(str(query["labels_path"]))
            expected_label_sha = query.get("label_sha256")
            if expected_label_sha and _sha256(label_path) != str(expected_label_sha):
                raise DataContractError(f"label changed: {label_path}")
            label = _load_label_file(label_path)
            if label.get("identity") != query.get("identity"):
                raise DataContractError(f"label identity mismatch for {query['query_id']}")
            video_target = label.get("video_target")
            if not isinstance(video_target, Mapping) or video_target.get("caption") != query.get("caption"):
                raise DataContractError(f"label caption mismatch for {query['query_id']}")
            frames = video_target.get("frames_id")
            if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
                raise DataContractError(f"missing local sample frame IDs for {query['query_id']}")
            labels[int(query["index"])] = label
            sample_ids[int(query["index"])] = [int(value) for value in frames]
        rows: list[dict[str, Any]] = []
        for query in queries:
            key = int(query["index"])
            official_key = str(query["official_annotation_key"])
            if official_key not in official:
                raise DataContractError(f"official query ID missing: {official_key}")
            _verify_official_identity(target, query, official_key, official[official_key], official)
            rows.append(_score_official(
                query, normalized[query["index"]], official[official_key], sample_ids[key],
                cached_grid=target in {"mouse", "football"},
            ))
        return rows

    if target == "meccano14train":
        metadata_path = _absolute(str(queries[0]["official_metadata_path"]))
        expected = str(queries[0]["official_metadata_sha256"])
        if _sha256(metadata_path) != expected:
            raise DataContractError(f"MECCANO official metadata changed: {metadata_path}")
        official = _read_json(metadata_path)
        if not isinstance(official, Mapping):
            raise DataContractError(f"MECCANO metadata is not a mapping: {metadata_path}")
        rows = []
        for query in queries:
            key = str(query["official_annotation_key"])
            if key not in official:
                raise DataContractError(f"MECCANO official query ID missing: {key}")
            _verify_official_identity(target, query, key, official[key], official)
            rows.append(_score_official(
                query, normalized[query["index"]], official[key], query["sample_ids"], cached_grid=False
            ))
        return rows

    # HC/Vid labels are dense native STVG labels.  They are opened only after
    # the complete mapping check above, and are never part of fit inputs.
    rows = []
    for query in queries:
        label_path = _absolute(str(query["labels_path"]))
        expected_label_sha = query.get("label_sha256")
        if expected_label_sha and _sha256(label_path) != str(expected_label_sha):
            raise DataContractError(f"label changed: {label_path}")
        label = _load_label_file(label_path)
        if label.get("identity") != query.get("identity"):
            raise DataContractError(f"label identity mismatch for {query['query_id']}")
        rows.append(_score_dense(query, normalized[query["index"]], label))
    return rows


__all__ = [
    "GROUPS",
    "DataContractError",
    "load_features",
    "load_source",
    "prepare_group",
    "score_predictions",
]
