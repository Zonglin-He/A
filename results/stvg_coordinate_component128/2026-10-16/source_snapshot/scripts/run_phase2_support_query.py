#!/usr/bin/env python3
"""Source-cluster-disjoint support-to-query Phase-2 experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_feasibility import load_full_video, validate_output
from scripts.run_phase1 import base_record
from vg_tta.augmentations import apply_condition
from vg_tta.metrics import (
    TEMPORAL_DECODER_VERSION,
    interval_from_logits,
    cluster_macro_paired_bootstrap_ci,
    cluster_paired_bootstrap_ci,
    paired_bootstrap_ci,
)
from vg_tta.phase2 import (
    aggregate_predictions,
    cluster_macro_mean,
    deterministic_views,
    source_cluster,
    subset_prediction_frames,
    temporally_subsample_sample,
)
from vg_tta.phase3 import temporal_probabilities, trajectory_temporal_target
from vg_tta.tta import (
    EpisodicHeadAdapter,
    teacher_interval_mask,
    temporal_span_entropy_loss,
)
from vg_tta.tubedetr_runtime import (
    add_repo_to_path,
    build_model,
    dataset_args,
    forward_video,
    forward_video_with_features,
    forward_video_with_temporal_head_input,
    load_official_checkpoint,
    vidstg_dataset_args,
)


METHODS = (
    "tta_ensemble",
    "gt_support",
    "clean_teacher_support",
    "entropy_support",
    "span_entropy_support",
    "multiview_temporal_support",
    "trajcal_support",
    "multiview_support",
)
CONDITIONS = ("clean", "low_light", "blur", "temporal_subsample")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(PROJECT_ROOT / "external" / "TubeDETR"))
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "checkpoints" / "tubedetr_hcstvg2_res224_stride2.pth"),
    )
    parser.add_argument("--checkpoint-domain", default="hcstvg2")
    parser.add_argument(
        "--dataset",
        choices=("hcstvg", "vidstg"),
        default="hcstvg",
        help="Target dataset consumed by the source-cluster-disjoint protocol.",
    )
    parser.add_argument(
        "--dataset-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For VidSTG, load test.json instead of val.json.",
    )
    parser.add_argument(
        "--video-root", default=str(PROJECT_ROOT / "data" / "hcstvg2_confirm512")
    )
    parser.add_argument(
        "--annotation-root",
        default=str(PROJECT_ROOT / "data" / "hcstvg2_confirm512" / "annotations"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional immutable run identifier; a UUID is generated when omitted.",
    )
    parser.add_argument(
        "--fixed-split-manifest",
        default=None,
        help=(
            "Optional prepared-data manifest containing either a canonical split "
            "object or selection/entries metadata. No split is regenerated when set."
        ),
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["blur"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--support-sizes", nargs="+", type=int, default=[4, 16, 64])
    parser.add_argument("--support-seeds", nargs="+", type=int, default=[41001, 41002, 41003])
    parser.add_argument("--query-clusters", type=int, default=64)
    parser.add_argument("--max-query-clips-per-cluster", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=20260906)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--temporal-factor", type=int, default=2)
    parser.add_argument("--scope", default="heads_ln_projection")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps-per-support", type=int, default=1)
    parser.add_argument("--view-count", type=int, default=4)
    parser.add_argument(
        "--trajectory-top-k",
        type=int,
        default=20,
        help="restrict TrajCal to the base top-K spans; use 0 for all legal spans",
    )
    parser.add_argument("--trajectory-weight", type=float, default=8.0)
    parser.add_argument("--trajectory-temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--cache-temporal-query-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Memoize frozen query decoder features for temporal-only adapted "
            "query evaluation. Every frozen temporal logit is exactly rechecked."
        ),
    )
    return parser.parse_args()


def build_split(
    annotations: list[dict[str, Any]] | dict[str, Any],
    *,
    query_cluster_count: int,
    max_query_clips_per_cluster: int,
    split_seed: int,
    support_sizes: list[int],
    support_seeds: list[int],
) -> dict[str, Any]:
    annotations = dataset_annotation_rows(annotations)
    if not support_sizes or any(int(size) <= 0 for size in support_sizes):
        raise ValueError("support sizes must be a non-empty list of positive integers")
    if not support_seeds:
        raise ValueError("support seeds must be non-empty")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, annotation in enumerate(annotations):
        grouped[source_cluster(annotation["video_path"])].append(index)
    clusters = np.asarray(sorted(grouped), dtype=object)
    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(clusters)
    if query_cluster_count <= 0 or query_cluster_count >= len(shuffled):
        raise ValueError("query cluster count must leave a non-empty support pool")
    query_clusters = [str(value) for value in shuffled[:query_cluster_count]]
    support_pool = [str(value) for value in shuffled[query_cluster_count:]]
    if max(support_sizes) > len(support_pool):
        raise ValueError("largest support size exceeds source-disjoint support pool")

    query_indices: list[int] = []
    query_selection: dict[str, list[int]] = {}
    for cluster in query_clusters:
        candidates = np.asarray(sorted(grouped[cluster]), dtype=np.int64)
        count = min(max_query_clips_per_cluster, len(candidates))
        selected = sorted(int(value) for value in rng.choice(candidates, size=count, replace=False))
        query_selection[cluster] = selected
        query_indices.extend(selected)

    support_plans: dict[str, dict[str, Any]] = {}
    for seed in support_seeds:
        seed_rng = np.random.default_rng(seed)
        ordered_clusters = [str(value) for value in seed_rng.permutation(support_pool)]
        ordered_indices = [
            int(seed_rng.choice(np.asarray(grouped[cluster], dtype=np.int64)))
            for cluster in ordered_clusters
        ]
        support_plans[str(seed)] = {
            "ordered_clusters": ordered_clusters,
            "ordered_indices": ordered_indices,
        }
    return {
        "query_clusters": query_clusters,
        "query_indices": sorted(query_indices),
        "query_selection": query_selection,
        "support_pool_clusters": support_pool,
        "support_plans": support_plans,
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    """Serialize provenance values deterministically and reject NaN metadata."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def dataset_annotation_rows(
    annotations_or_dataset: Any,
) -> list[dict[str, Any]]:
    """Return annotation rows in dataset-index order for HC-STVG and VidSTG.

    TubeDETR stores HC-STVG annotations as a list but VidSTG annotations as a
    dictionary containing a ``videos`` list.  The support/query protocol is
    index based, so every identity and split check must use the same normalized
    row list rather than indexing the raw container directly.
    """

    raw = getattr(annotations_or_dataset, "annotations", annotations_or_dataset)
    if isinstance(raw, list) and all(isinstance(value, dict) for value in raw):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("videos"), list):
        rows = raw["videos"]
        if not all(isinstance(value, dict) for value in rows):
            raise TypeError("dataset annotations['videos'] must contain dictionaries")
        return rows
    raise TypeError(
        "expected annotations as a list or a VidSTG dictionary containing 'videos'"
    )


def _canonical_video_path(
    value: Any,
    *,
    video_root: str | Path | None = None,
) -> str:
    """Canonicalize an annotation video path to a dataset-relative token."""

    text = str(value).replace("\\", "/")
    path = Path(text).expanduser()
    if video_root is not None and path.is_absolute():
        root = Path(video_root).expanduser().resolve()
        # ``args.video_root`` points to the data root while prepared manifests
        # commonly point to data_root/video/file.  Prefer the latter.
        for candidate in (root / "video", root):
            try:
                return path.resolve().relative_to(candidate.resolve()).as_posix()
            except ValueError:
                continue
    return text.lstrip("./")


def _annotation_query(annotation: dict[str, Any]) -> str:
    value = annotation.get("caption", annotation.get("query"))
    if value is None:
        value = annotation.get("description")
    if value is None:
        raise ValueError("annotation row has no caption/query/description field")
    return str(value)


def _sample_identity(
    annotation: dict[str, Any],
    index: int,
    *,
    video_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a stable, human-auditable identity for one dataset row."""

    if "video_path" not in annotation:
        raise ValueError(f"annotation row {index} has no video_path")
    raw_video_path = str(annotation["video_path"])
    identity: dict[str, Any] = {
        "dataset_index": int(index),
        "video_path": _canonical_video_path(raw_video_path, video_root=video_root),
        "query": _annotation_query(annotation),
        "source_cluster": source_cluster(raw_video_path),
        # The row hash includes the full row (including the tube/trajectory),
        # while the explicit fields above make split audits readable.
        "annotation_sha256": _json_sha256(annotation),
    }
    for key in (
        "video_id",
        "original_video_id",
        "target_id",
        "tube_start_frame",
        "tube_end_frame",
        "start_frame",
        "end_frame",
        "frame_count",
        "width",
        "height",
        "fps",
    ):
        if key in annotation:
            identity[key] = annotation[key]
    return identity


def resolve_annotation_file(
    dataset: str,
    annotation_root: str | Path,
    *,
    dataset_test: bool = True,
) -> Path:
    """Resolve the exact annotation file consumed by TubeDETR's dataset builder."""

    root = Path(annotation_root).expanduser().resolve()
    if root.is_file():
        return root
    if dataset == "hcstvg":
        return root / "valv2_proc.json"
    if dataset == "vidstg":
        return root / ("test.json" if dataset_test else "val.json")
    raise ValueError(f"unsupported dataset {dataset!r}")


def build_dataset_provenance(
    annotations: list[dict[str, Any]] | dict[str, Any],
    *,
    dataset: str,
    annotation_file: str | Path,
    video_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint the exact dataset rows and annotation file used by a run."""

    rows = dataset_annotation_rows(annotations)
    annotation_path = Path(annotation_file).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"annotation file not found: {annotation_path}")
    samples = [
        _sample_identity(annotation, index, video_root=video_root)
        for index, annotation in enumerate(rows)
    ]
    sample_identity_sha256 = _json_sha256(samples)
    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "sample_count": len(samples),
        "annotation_file": str(annotation_path),
        "annotation_file_sha256": _sha256(annotation_path),
        "sample_identity_sha256": sample_identity_sha256,
        "samples": samples,
    }


def _manifest_sample_identities(
    prepared: dict[str, Any],
    *,
    video_root: str | Path | None,
) -> dict[int, dict[str, Any]]:
    """Collect identity records from current and legacy prepared manifests."""

    candidates: dict[int, dict[str, Any]] = {}
    provenance = prepared.get("dataset_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("samples"), list):
        sources = provenance["samples"]
    else:
        sources = []
    entries = prepared.get("entries") or prepared.get("records")
    if isinstance(entries, list):
        sources = [*sources, *entries]
    for value in sources:
        if not isinstance(value, dict) or value.get("dataset_index") is None:
            continue
        index = int(value["dataset_index"])
        raw_path = value.get("video_relpath", value.get("video_path"))
        if raw_path is None:
            raise ValueError(f"fixed split identity for index {index} has no video path")
        query = value.get("query", value.get("caption", value.get("description")))
        if query is None:
            raise ValueError(f"fixed split identity for index {index} has no query")
        identity: dict[str, Any] = {
            "dataset_index": index,
            "video_path": _canonical_video_path(raw_path, video_root=video_root),
            "query": str(query),
            "source_cluster": str(
                value.get("source_cluster", source_cluster(str(raw_path)))
            ),
        }
        for key in (
            "annotation_sha256",
            "video_id",
            "original_video_id",
            "target_id",
            "tube_start_frame",
            "tube_end_frame",
            "start_frame",
            "end_frame",
            "frame_count",
            "width",
            "height",
            "fps",
        ):
            if key in value:
                identity[key] = value[key]
        previous = candidates.get(index)
        if previous is not None and previous != identity:
            raise ValueError(f"fixed split contains conflicting identities for index {index}")
        candidates[index] = identity
    return candidates


def _declared_annotation_sha256(
    prepared: dict[str, Any],
    *,
    current_annotation_file: Path,
) -> str | None:
    """Find a target annotation SHA without confusing source archive SHAs."""

    provenance = prepared.get("dataset_provenance")
    if isinstance(provenance, dict):
        value = provenance.get("annotation_file_sha256")
        if value:
            return str(value)
    for key in ("annotation_file_sha256", "data_annotation_sha256"):
        if prepared.get(key):
            return str(prepared[key])
    annotation_meta = prepared.get("annotation")
    if isinstance(annotation_meta, dict):
        value = annotation_meta.get("annotation_file_sha256")
        if value:
            return str(value)
    # Existing VidSTG preparation manifests record the exact TubeDETR target
    # annotation path, while their ``vidstg_annotation_sha256`` refers to the
    # upstream source annotation and must not be compared to test.json.
    tube_detr = prepared.get("tube_detr")
    if isinstance(tube_detr, dict):
        value = tube_detr.get("annotation_file_sha256")
        if value:
            return str(value)
        path = tube_detr.get("annotation_file")
        if path:
            candidate = Path(str(path)).expanduser()
            if candidate.is_file():
                return _sha256(candidate)
    return None


def load_fixed_split(
    path: str | Path,
    annotations: list[dict[str, Any]] | dict[str, Any],
    *,
    support_sizes: list[int],
    support_seeds: list[int],
    dataset: str | None = None,
    annotation_file: str | Path | None = None,
    video_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a predeclared split without re-randomizing confirmation data."""

    if annotation_file is None:
        raise ValueError(
            "annotation_file is required for fixed split provenance validation"
        )
    if not support_sizes or any(int(size) <= 0 for size in support_sizes):
        raise ValueError("support sizes must be a non-empty list of positive integers")
    if not support_seeds:
        raise ValueError("support seeds must be non-empty")
    rows = dataset_annotation_rows(annotations)
    current_dataset_provenance = None
    current_annotation_path = Path(annotation_file).expanduser().resolve()
    current_dataset_provenance = build_dataset_provenance(
        rows,
        dataset=dataset or "unknown",
        annotation_file=current_annotation_path,
        video_root=video_root,
    )

    manifest_path = Path(path).expanduser().resolve()
    prepared = json.loads(manifest_path.read_text())
    if "split" in prepared and isinstance(prepared["split"], dict):
        split = prepared["split"]
    else:
        selection = prepared.get("selection")
        entries = prepared.get("entries") or prepared.get("records")
        if not isinstance(selection, dict) or not isinstance(entries, list):
            raise ValueError(
                "fixed split manifest needs split or selection plus entries/records"
            )
        query_indices = sorted(
            int(value["dataset_index"])
            for value in entries
            if str(value.get("role")) == "query"
        )
        support_plans: dict[str, dict[str, Any]] = {}
        for seed in support_seeds:
            source = selection.get("support_plans", {}).get(str(seed))
            if not isinstance(source, dict):
                raise KeyError(f"fixed split lacks support plan for seed {seed}")
            ordered_indices = source.get("ordered_indices")
            if ordered_indices is None:
                ordered_indices = source.get("ordered_dataset_indices")
            if ordered_indices is None:
                raise KeyError(f"support plan {seed} lacks ordered dataset indices")
            support_plans[str(seed)] = {
                "ordered_clusters": [str(value) for value in source["ordered_clusters"]],
                "ordered_indices": [int(value) for value in ordered_indices],
            }
        split = {
            "query_clusters": [str(value) for value in selection["query_clusters"]],
            "query_indices": query_indices,
            "query_selection": selection.get("query_selection", {}),
            "support_pool_clusters": [
                str(value) for value in selection["support_pool_clusters"]
            ],
            "support_plans": support_plans,
        }

    if not isinstance(split, dict):
        raise ValueError("fixed split must be a JSON object")
    query_indices = [int(value) for value in split["query_indices"]]
    if not query_indices or min(query_indices) < 0 or max(query_indices) >= len(rows):
        raise ValueError("fixed query indices are empty or outside the prepared dataset")
    if len(set(query_indices)) != len(query_indices):
        raise ValueError("fixed query indices contain duplicates")
    query_cluster_values = [str(value) for value in split["query_clusters"]]
    support_pool_values = [str(value) for value in split["support_pool_clusters"]]
    if len(set(query_cluster_values)) != len(query_cluster_values):
        raise ValueError("fixed query clusters contain duplicates")
    if len(set(support_pool_values)) != len(support_pool_values):
        raise ValueError("fixed support pool clusters contain duplicates")
    query_clusters = set(query_cluster_values)
    support_pool = set(support_pool_values)
    if query_clusters & support_pool:
        raise ValueError("fixed query/support source clusters overlap")
    observed_query_clusters = {
        source_cluster(rows[index]["video_path"]) for index in query_indices
    }
    if observed_query_clusters != query_clusters:
        raise ValueError("fixed query indices do not realize the declared query clusters")
    for seed in support_seeds:
        support_plans = split.get("support_plans")
        if not isinstance(support_plans, dict) or str(seed) not in support_plans:
            raise KeyError(f"fixed split lacks support plan for seed {seed}")
        plan = support_plans[str(seed)]
        indices = [int(value) for value in plan["ordered_indices"]]
        clusters = [str(value) for value in plan["ordered_clusters"]]
        if len(indices) < max(support_sizes) or len(clusters) < max(support_sizes):
            raise ValueError(f"fixed support plan {seed} is too short")
        if len(indices) != len(clusters) or len(set(indices)) != len(indices):
            raise ValueError(f"fixed support plan {seed} is duplicated or unaligned")
        if len(set(clusters)) != len(clusters):
            raise ValueError(f"fixed support plan {seed} repeats a source cluster")
        if set(clusters) != support_pool:
            raise ValueError(f"fixed support plan {seed} does not cover its support pool")
        for index, cluster in zip(indices, clusters):
            if not 0 <= index < len(rows):
                raise ValueError(f"fixed support index {index} is outside the dataset")
            if source_cluster(rows[index]["video_path"]) != cluster:
                raise ValueError(f"fixed support plan {seed} has an index/cluster mismatch")
        if query_clusters & set(clusters):
            raise ValueError(f"fixed support plan {seed} overlaps query clusters")

    declared_identities = _manifest_sample_identities(
        prepared,
        video_root=video_root,
    )
    referenced_indices = set(query_indices)
    for seed in support_seeds:
        referenced_indices.update(
            int(value)
            for value in split["support_plans"][str(seed)]["ordered_indices"]
        )
    if not declared_identities:
        raise ValueError(
            "fixed split manifest has no dataset sample identities; regenerate it "
            "with dataset_provenance or entries/records"
        )
    for index in sorted(referenced_indices):
        expected = declared_identities.get(index)
        if expected is None:
            raise ValueError(
                f"fixed split manifest has no identity for referenced dataset index {index}"
            )
        actual = _sample_identity(rows[index], index, video_root=video_root)
        for key in ("video_path", "query", "source_cluster"):
            if expected.get(key) != actual.get(key):
                raise ValueError(
                    "fixed split dataset identity mismatch for index "
                    f"{index} field {key}: expected={expected.get(key)!r} "
                    f"actual={actual.get(key)!r}"
                )
        for key in (
            "annotation_sha256",
            "video_id",
            "original_video_id",
            "target_id",
            "tube_start_frame",
            "tube_end_frame",
            "start_frame",
            "end_frame",
            "frame_count",
            "width",
            "height",
            "fps",
        ):
            if key in expected and expected[key] != actual.get(key):
                raise ValueError(
                    "fixed split dataset identity mismatch for index "
                    f"{index} field {key}: expected={expected[key]!r} "
                    f"actual={actual.get(key)!r}"
                )
    declared_annotation_sha256 = _declared_annotation_sha256(
        prepared,
        current_annotation_file=current_annotation_path,
    )
    if declared_annotation_sha256 is None:
        raise ValueError(
            "fixed split manifest has no target annotation file SHA; regenerate it "
            "with dataset_provenance"
        )
    if declared_annotation_sha256 != current_dataset_provenance["annotation_file_sha256"]:
        raise ValueError(
            "fixed split annotation file SHA mismatch: "
            f"expected={declared_annotation_sha256} "
            f"actual={current_dataset_provenance['annotation_file_sha256']}"
        )
    declared_provenance = prepared.get("dataset_provenance")
    if isinstance(declared_provenance, dict):
        declared_dataset = declared_provenance.get("dataset")
        if dataset is not None and declared_dataset not in (None, dataset):
            raise ValueError(
                f"fixed split dataset mismatch: expected={declared_dataset!r} "
                f"actual={dataset!r}"
            )
        declared_fingerprint = declared_provenance.get("sample_identity_sha256")
        if (
            declared_fingerprint
            and int(declared_provenance.get("sample_count", -1)) == len(rows)
            and str(declared_fingerprint)
            != current_dataset_provenance["sample_identity_sha256"]
        ):
            raise ValueError(
                "fixed split dataset sample fingerprint mismatch: "
                f"expected={declared_fingerprint} "
                f"actual={current_dataset_provenance['sample_identity_sha256']}"
            )
    identity_audit = {
        "status": "verified",
        "referenced_sample_count": len(referenced_indices),
        "annotation_file_sha256": current_dataset_provenance[
            "annotation_file_sha256"
        ],
        "sample_identity_sha256": current_dataset_provenance[
            "sample_identity_sha256"
        ],
    }
    return split, {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "identity_audit": identity_audit,
    }


def prepare_condition(
    video: torch.Tensor,
    targets: list[dict[str, Any]],
    video_target: dict[str, Any],
    condition: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    if condition == "temporal_subsample":
        return temporally_subsample_sample(
            video,
            targets,
            video_target,
            factor=args.temporal_factor,
        )
    return (
        apply_condition(video, condition, severity=args.severity),
        targets,
        video_target,
    )


def forward_timed(
    model: torch.nn.Module,
    video: torch.Tensor,
    caption: str,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        output = forward_video(
            model,
            video,
            caption,
            repo=args.repo,
            stride=args.stride,
            device="cuda",
        )
    torch.cuda.synchronize()
    return output, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 1024**3


def build_temporal_query_cache(
    model: torch.nn.Module,
    dataset: Any,
    query_indices: list[int],
    conditions: list[str],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Cache frozen outputs and final decoder features for temporal-only runs."""

    if args.scope != "temporal":
        raise ValueError("query feature caching is valid only for temporal scope")
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    max_abs_reapplied_difference = 0.0
    decoded_interval_checks = 0
    for condition in conditions:
        for position, sample_index in enumerate(query_indices):
            full_video, targets, video_target = load_full_video(dataset[sample_index])
            shifted, shifted_targets, shifted_video_target = prepare_condition(
                full_video, targets, video_target, condition, args
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                output, head_input = forward_video_with_temporal_head_input(
                    model,
                    shifted,
                    shifted_video_target["caption"],
                    repo=args.repo,
                    stride=args.stride,
                    device="cuda",
                )
            torch.cuda.synchronize()
            runtime = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 1024**3
            validate_output(output, shifted.shape[1])
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                reapplied = model.sted_embed(head_input)[-1]
            difference = float(
                (reapplied.float() - output["pred_sted"].float()).abs().max().item()
            )
            max_abs_reapplied_difference = max(
                max_abs_reapplied_difference, difference
            )
            if difference != 0.0:
                raise RuntimeError(
                    "cached temporal head did not exactly reproduce frozen logits: "
                    f"condition={condition} sample={sample_index} max_abs={difference}"
                )
            if interval_from_logits(output["pred_sted"]) != interval_from_logits(
                reapplied.detach().cpu()
            ):
                raise RuntimeError("cached/full temporal interval decoding differs")
            decoded_interval_checks += 1
            cache[(condition, sample_index)] = {
                "pred_boxes": output["pred_boxes"].detach().cpu(),
                "pred_sted": output["pred_sted"].detach().cpu(),
                "head_input": head_input.detach().cpu(),
                "full_forward_runtime": runtime,
                "peak_vram_gb": peak,
                # Small evaluation metadata only: never retain decoded pixels
                # or original-resolution visualization frames in the host cache.
                "targets": [{"boxes": target.get("boxes", torch.empty(0, 4)).detach().cpu().clone()}
                            for target in shifted_targets],
                "video_target": {key: shifted_video_target[key]
                                 for key in ("caption", "video_id", "inter_idx", "frames_id")},
                "num_frames": int(shifted.shape[1]),
            }
            if (position + 1) % 25 == 0 or position + 1 == len(query_indices):
                print(
                    f"[cache temporal query {condition}] {position + 1}/{len(query_indices)}",
                    flush=True,
                )
    return cache, {
        "enabled": True,
        "entries": len(cache),
        "max_abs_frozen_temporal_logit_reapplication_difference": (
            max_abs_reapplied_difference
        ),
        "exact_frozen_reapplication": max_abs_reapplied_difference == 0.0,
        "exact_decoded_interval_checks": decoded_interval_checks,
        "temporal_decoder_version": TEMPORAL_DECODER_VERSION,
        "runtime_accounting": "full frozen forward plus adapted temporal-head reapplication",
    }


def cached_temporal_query_forward(
    model: torch.nn.Module,
    cached: dict[str, Any],
    *,
    frozen: bool,
) -> tuple[dict[str, torch.Tensor], float, float]:
    """Apply the current temporal head to a cached, head-independent feature."""

    if frozen:
        return (
            {
                "pred_boxes": cached["pred_boxes"],
                "pred_sted": cached["pred_sted"],
            },
            float(cached["full_forward_runtime"]),
            float(cached["peak_vram_gb"]),
        )
    head_input = cached["head_input"].to("cuda", non_blocking=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        pred_sted = model.sted_embed(head_input)[-1]
    torch.cuda.synchronize()
    head_runtime = time.perf_counter() - started
    return (
        {
            "pred_boxes": cached["pred_boxes"],
            "pred_sted": pred_sted.detach().cpu(),
        },
        float(cached["full_forward_runtime"]) + head_runtime,
        max(
            float(cached["peak_vram_gb"]),
            torch.cuda.max_memory_allocated() / 1024**3,
        ),
    )


def ensemble_timed(
    model: torch.nn.Module,
    video: torch.Tensor,
    caption: str,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    predictions = []
    with torch.no_grad():
        for view in deterministic_views(video, seed=seed, count=args.view_count):
            predictions.append(
                forward_video(
                    model,
                    view,
                    caption,
                    repo=args.repo,
                    stride=args.stride,
                    device="cuda",
                )
            )
    output = aggregate_predictions(predictions)
    torch.cuda.synchronize()
    return output, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 1024**3


def new_adapter(model: torch.nn.Module, args: argparse.Namespace) -> EpisodicHeadAdapter:
    return EpisodicHeadAdapter(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scope=args.scope,
        minimum_loss_for_update=None,
        optimizer_name=args.optimizer,
        optimizer_eps=args.optimizer_eps,
        momentum=args.momentum,
    )


def cpu_prediction(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "pred_boxes": output["pred_boxes"].detach().float().cpu(),
        "pred_sted": output["pred_sted"].detach().float().cpu(),
    }


def to_device_prediction(
    output: dict[str, torch.Tensor], device: str = "cuda"
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in output.items()}


def adapt_on_support(
    model: torch.nn.Module,
    dataset: Any,
    support_indices: list[int],
    condition: str,
    method: str,
    args: argparse.Namespace,
    *,
    seed: int,
    support_size: int,
    run_id: str,
    provenance: dict[str, Any],
    support_log_handle: Any,
) -> tuple[EpisodicHeadAdapter, dict[str, float]]:
    annotation_rows = dataset_annotation_rows(dataset)
    if int(support_size) != len(support_indices):
        raise ValueError(
            "support_size must equal the number of support_indices in an episode"
        )
    clean_teacher_cache: dict[int, dict[str, torch.Tensor]] = {}
    if method == "clean_teacher_support":
        for sample_index in support_indices:
            full_video, _, video_target = load_full_video(dataset[sample_index])
            output, _, _ = forward_timed(model, full_video, video_target["caption"], args)
            clean_teacher_cache[sample_index] = cpu_prediction(output)

    adapter = new_adapter(model, args)
    adapter.reset()
    last_loss: dict[str, float] = {}
    try:
        for support_position, sample_index in enumerate(support_indices):
            full_video, targets, video_target = load_full_video(dataset[sample_index])
            shifted, shifted_targets, shifted_video_target = prepare_condition(
                full_video, targets, video_target, condition, args
            )
            caption = shifted_video_target["caption"]
            for local_step in range(args.steps_per_support):
                if method == "gt_support":
                    output = forward_video(
                        model,
                        shifted,
                        caption,
                        repo=args.repo,
                        stride=args.stride,
                        device="cuda",
                    )
                    mask = torch.ones_like(output["pred_sted"][..., 0], dtype=torch.bool)
                    last_loss = adapter.step_ground_truth(
                        output,
                        shifted_targets,
                        tuple(shifted_video_target["inter_idx"]),
                        time_mask=mask,
                    )
                elif method == "clean_teacher_support":
                    teacher = to_device_prediction(clean_teacher_cache[sample_index])
                    positions = shifted_video_target.get("temporal_subsample_positions")
                    if positions is not None:
                        teacher = subset_prediction_frames(teacher, positions)
                    student = forward_video(
                        model,
                        shifted,
                        caption,
                        repo=args.repo,
                        stride=args.stride,
                        device="cuda",
                    )
                    time_mask = torch.ones_like(teacher["pred_sted"][..., 0], dtype=torch.bool)
                    last_loss = adapter.step(
                        student,
                        teacher,
                        time_mask=time_mask,
                        box_mask=teacher_interval_mask(teacher["pred_sted"], time_mask=time_mask),
                    )
                elif method == "entropy_support":
                    output = forward_video(
                        model,
                        shifted,
                        caption,
                        repo=args.repo,
                        stride=args.stride,
                        device="cuda",
                    )
                    mask = torch.ones_like(output["pred_sted"][..., 0], dtype=torch.bool)
                    last_loss = adapter.step_temporal_entropy(output, time_mask=mask)
                elif method == "span_entropy_support":
                    output = forward_video(
                        model,
                        shifted,
                        caption,
                        repo=args.repo,
                        stride=args.stride,
                        device="cuda",
                    )
                    mask = torch.ones_like(output["pred_sted"][..., 0], dtype=torch.bool)
                    last_loss = adapter.step_temporal_span_entropy(output, time_mask=mask)
                    # The optimized signal is reported before and immediately
                    # after every support update. This extra read-only forward
                    # is intentionally retained for mechanism auditing.
                    with torch.no_grad():
                        post_update = forward_video(
                            model,
                            shifted,
                            caption,
                            repo=args.repo,
                            stride=args.stride,
                            device="cuda",
                        )
                        _, post_components = temporal_span_entropy_loss(
                            post_update,
                            time_mask=torch.ones_like(
                                post_update["pred_sted"][..., 0], dtype=torch.bool
                            ),
                        )
                    last_loss["post_update_temporal_span_entropy"] = post_components[
                        "loss_temporal_span_entropy"
                    ]
                    last_loss["immediate_temporal_span_entropy_delta"] = (
                        last_loss["post_update_temporal_span_entropy"]
                        - last_loss["loss_temporal_span_entropy"]
                    )
                elif method == "multiview_temporal_support":
                    with torch.no_grad():
                        predictions = [
                            forward_video(
                                model,
                                view,
                                caption,
                                repo=args.repo,
                                stride=args.stride,
                                device="cuda",
                            )
                            for view in deterministic_views(
                                shifted,
                                seed=seed + sample_index * 10 + local_step,
                                count=args.view_count,
                            )
                        ]
                        teacher = aggregate_predictions(predictions)
                        temporal_target = temporal_probabilities(teacher["pred_sted"])
                    student = forward_video(
                        model,
                        shifted,
                        caption,
                        repo=args.repo,
                        stride=args.stride,
                        device="cuda",
                    )
                    time_mask = torch.ones_like(
                        student["pred_sted"][..., 0], dtype=torch.bool
                    )
                    last_loss = adapter.step_temporal_target(
                        student,
                        temporal_target,
                        time_mask=time_mask,
                    )
                elif method == "trajcal_support":
                    predictions = []
                    decoder_features = []
                    with torch.no_grad():
                        for view in deterministic_views(
                            shifted,
                            seed=seed + sample_index * 10 + local_step,
                            count=args.view_count,
                        ):
                            prediction, features = forward_video_with_features(
                                model,
                                view,
                                caption,
                                repo=args.repo,
                                stride=args.stride,
                                device="cuda",
                            )
                            predictions.append(prediction)
                            decoder_features.append(
                                F.normalize(features.detach().float(), dim=-1)
                            )
                        teacher = aggregate_predictions(predictions)
                        consensus_features = torch.stack(decoder_features).mean(dim=0)
                        temporal_target, trajectory_diagnostics = trajectory_temporal_target(
                            temporal_probabilities(teacher["pred_sted"]),
                            consensus_features,
                            trajectory_weight=args.trajectory_weight,
                            temperature=args.trajectory_temperature,
                            top_k=(
                                None
                                if args.trajectory_top_k == 0
                                else args.trajectory_top_k
                            ),
                        )
                    student = forward_video(
                        model,
                        shifted,
                        caption,
                        repo=args.repo,
                        stride=args.stride,
                        device="cuda",
                    )
                    time_mask = torch.ones_like(
                        student["pred_sted"][..., 0], dtype=torch.bool
                    )
                    last_loss = adapter.step_temporal_target(
                        student,
                        temporal_target,
                        time_mask=time_mask,
                    )
                    last_loss.update(
                        {
                            f"trajectory_{key}": value
                            for key, value in trajectory_diagnostics.items()
                        }
                    )
                elif method == "multiview_support":
                    with torch.no_grad():
                        predictions = [
                            forward_video(
                                model,
                                view,
                                caption,
                                repo=args.repo,
                                stride=args.stride,
                                device="cuda",
                            )
                            for view in deterministic_views(
                                shifted,
                                seed=seed + sample_index * 10 + local_step,
                                count=args.view_count,
                            )
                        ]
                        teacher = aggregate_predictions(predictions)
                    student = forward_video(
                        model,
                        shifted,
                        caption,
                        repo=args.repo,
                        stride=args.stride,
                        device="cuda",
                    )
                    time_mask = torch.ones_like(teacher["pred_sted"][..., 0], dtype=torch.bool)
                    last_loss = adapter.step(
                        student,
                        teacher,
                        time_mask=time_mask,
                        box_mask=teacher_interval_mask(teacher["pred_sted"], time_mask=time_mask),
                    )
                else:
                    raise ValueError(f"unsupported support method: {method}")
            support_log_handle.write(
                json.dumps(
                    {
                        "method": method,
                        "condition": condition,
                        "run_id": run_id,
                        "support_size": int(support_size),
                        "support_seed": seed,
                        "support_position": support_position,
                        "support_index": int(sample_index),
                        "support_cluster": source_cluster(
                            annotation_rows[sample_index]["video_path"]
                        ),
                        # Retain the historical key for downstream readers.
                        "sample_index": int(sample_index),
                        "source_cluster": source_cluster(
                            annotation_rows[sample_index]["video_path"]
                        ),
                        "last_loss": last_loss,
                        **provenance,
                        **adapter.delta_from_initial(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            support_log_handle.flush()
        return adapter, adapter.delta_from_initial()
    except Exception:
        adapter.reset()
        raise


def evaluate_query(
    model: torch.nn.Module,
    dataset: Any,
    query_indices: list[int],
    condition: str,
    method: str,
    args: argparse.Namespace,
    *,
    support_size: int,
    support_seed: int,
    run_id: str,
    provenance: dict[str, Any],
    delta: dict[str, float] | None,
    handle: Any,
    temporal_query_cache: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    annotation_rows = dataset_annotation_rows(dataset)
    records: list[dict[str, Any]] = []
    for query_position, sample_index in enumerate(query_indices):
        if temporal_query_cache is not None and method != "tta_ensemble":
            cached = temporal_query_cache[(condition, sample_index)]
            shifted_targets = cached["targets"]
            shifted_video_target = cached["video_target"]
            num_frames = cached["num_frames"]
        else:
            full_video, targets, video_target = load_full_video(dataset[sample_index])
            shifted, shifted_targets, shifted_video_target = prepare_condition(
                full_video, targets, video_target, condition, args
            )
            num_frames = int(shifted.shape[1])
        annotation = annotation_rows[sample_index]
        caption = shifted_video_target["caption"]
        if method == "tta_ensemble":
            output, runtime, peak = ensemble_timed(
                model,
                shifted,
                caption,
                args,
                seed=args.split_seed + sample_index * 10,
            )
        elif temporal_query_cache is not None:
            output, runtime, peak = cached_temporal_query_forward(
                model,
                temporal_query_cache[(condition, sample_index)],
                frozen=method == "frozen",
            )
        else:
            output, runtime, peak = forward_timed(model, shifted, caption, args)
        validate_output(output, num_frames)
        record = base_record(
            output,
            shifted_targets,
            shifted_video_target,
            annotation,
            sample_index=sample_index,
            condition=condition,
            method=method,
            runtime=runtime,
            peak=peak,
        )
        record.update(
            run_id=run_id,
            support_size=support_size,
            support_seed=support_seed,
            checkpoint_domain=args.checkpoint_domain,
            **provenance,
            **(delta or {}),
        )
        _, span_components = temporal_span_entropy_loss(output)
        record["temporal_span_entropy"] = span_components[
            "loss_temporal_span_entropy"
        ]
        records.append(record)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        if (query_position + 1) % 25 == 0 or query_position + 1 == len(query_indices):
            print(
                f"[query {method} {condition} K={support_size} seed={support_seed}] "
                f"{query_position + 1}/{len(query_indices)}",
                flush=True,
            )
    return records


def summarize(
    records: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace
) -> None:
    frozen = {
        (record["sample_index"], record["condition"]): record
        for record in records
        if record["method"] == "frozen"
    }
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["method"] != "frozen":
            grouped[
                (record["condition"], record["method"], int(record["support_size"]))
            ].append(record)

    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for (condition, method, support_size), values in sorted(grouped.items()):
        per_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            per_sample[value["sample_index"]].append(value)
        sample_indices = sorted(per_sample)
        adapted = np.asarray(
            [np.mean([item["vIoU_corrected"] for item in per_sample[index]]) for index in sample_indices]
        )
        baseline = np.asarray(
            [frozen[(index, condition)]["vIoU_corrected"] for index in sample_indices]
        )
        clusters = [per_sample[index][0]["source_cluster"] for index in sample_indices]
        seed_deltas: dict[str, float] = {}
        for seed in sorted({value["support_seed"] for value in values}):
            seed_values = [value for value in values if value["support_seed"] == seed]
            seed_deltas[str(seed)] = float(
                np.mean(
                    [
                        value["vIoU_corrected"]
                        - frozen[(value["sample_index"], condition)]["vIoU_corrected"]
                        for value in seed_values
                    ]
                )
            )
        clip_micro = paired_bootstrap_ci(
            adapted,
            baseline,
            n_bootstrap=args.bootstrap_samples,
            seed=args.split_seed,
        )
        cluster_micro = cluster_paired_bootstrap_ci(
            adapted,
            baseline,
            clusters,
            n_bootstrap=args.bootstrap_samples,
            seed=args.split_seed,
        )
        cluster_macro = cluster_macro_paired_bootstrap_ci(
            adapted,
            baseline,
            clusters,
            n_bootstrap=args.bootstrap_samples,
            seed=args.split_seed,
        )
        key = f"{condition}|{method}|K={support_size}"
        detail[key] = {
            "n_query_clips": len(sample_indices),
            "n_query_clusters": len(set(clusters)),
            "n_support_seeds": len(seed_deltas),
            "support_seed_clip_micro_deltas": seed_deltas,
            "frozen_clip_micro": float(baseline.mean()),
            "adapted_clip_micro": float(adapted.mean()),
            "frozen_cluster_macro": cluster_macro_mean(baseline.tolist(), clusters),
            "adapted_cluster_macro": cluster_macro_mean(adapted.tolist(), clusters),
            "clip_micro_paired_bootstrap": clip_micro,
            "cluster_resampled_clip_micro_bootstrap": cluster_micro,
            "cluster_macro_paired_bootstrap": cluster_macro,
            "paired_median_delta": float(np.median(adapted - baseline)),
        }
        rows.append(
            {
                "condition": condition,
                "method": method,
                "support_size": support_size,
                "n_query_clips": len(sample_indices),
                "n_query_clusters": len(set(clusters)),
                "support_seeds": len(seed_deltas),
                "frozen_vIoU_micro": float(baseline.mean()),
                "adapted_vIoU_micro": float(adapted.mean()),
                "clip_micro_delta_pp": 100.0 * clip_micro["mean_difference"],
                "cluster_macro_delta_pp": 100.0 * cluster_macro["mean_difference"],
                "cluster_macro_ci_low_pp": 100.0 * cluster_macro["ci_low"],
                "cluster_macro_ci_high_pp": 100.0 * cluster_macro["ci_high"],
                "median_delta_pp": 100.0 * float(np.median(adapted - baseline)),
                "seed_delta_min_pp": 100.0 * min(seed_deltas.values()),
                "seed_delta_max_pp": 100.0 * max(seed_deltas.values()),
            }
        )
    (output_dir / "support_query_statistics.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False) + "\n"
    )
    if rows:
        with (output_dir / "support_query_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if any(size <= 0 for size in args.support_sizes):
        raise ValueError("support sizes must be positive")
    if args.trajectory_top_k < 0:
        raise ValueError("trajectory top-K must be non-negative; use 0 for all spans")
    if not args.support_seeds:
        raise ValueError("support seeds must be non-empty")
    run_id = str(args.run_id or uuid.uuid4().hex)
    if not run_id.strip():
        raise ValueError("run id must be non-empty")
    args.run_id = run_id
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    add_repo_to_path(args.repo)
    from datasets import build_dataset

    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)
    model, model_args = build_model(
        args.repo,
        device="cuda",
        resolution=args.resolution,
        stride=args.stride,
    )
    checkpoint_report = load_official_checkpoint(model, args.checkpoint)
    if checkpoint_report["missing_keys"] or checkpoint_report["unexpected_keys"]:
        raise RuntimeError(f"checkpoint restore failed: {checkpoint_report}")
    model.eval()
    if args.dataset == "hcstvg":
        target_dataset_args = dataset_args(
            model_args, args.video_root, args.annotation_root
        )
    else:
        target_dataset_args = vidstg_dataset_args(
            model_args,
            args.video_root,
            args.annotation_root,
            test=args.dataset_test,
        )
    dataset = build_dataset(
        args.dataset,
        image_set="val",
        args=target_dataset_args,
    )
    annotation_rows = dataset_annotation_rows(dataset)
    annotation_file = resolve_annotation_file(
        args.dataset,
        args.annotation_root,
        dataset_test=args.dataset_test,
    )
    dataset_provenance = build_dataset_provenance(
        annotation_rows,
        dataset=args.dataset,
        annotation_file=annotation_file,
        video_root=args.video_root,
    )
    if args.fixed_split_manifest:
        split, fixed_split_source = load_fixed_split(
            args.fixed_split_manifest,
            annotation_rows,
            support_sizes=args.support_sizes,
            support_seeds=args.support_seeds,
            dataset=args.dataset,
            annotation_file=annotation_file,
            video_root=args.video_root,
        )
    else:
        split = build_split(
            annotation_rows,
            query_cluster_count=args.query_clusters,
            max_query_clips_per_cluster=args.max_query_clips_per_cluster,
            split_seed=args.split_seed,
            support_sizes=args.support_sizes,
            support_seeds=args.support_seeds,
        )
        fixed_split_source = None
    split_sha256 = _json_sha256(split)
    checkpoint_sha256 = _sha256(args.checkpoint)
    config_args = vars(args).copy()
    config_sha256 = _json_sha256(config_args)
    record_provenance = {
        "temporal_decoder_version": TEMPORAL_DECODER_VERSION,
        "evaluator_sha256": _sha256(PROJECT_ROOT / "vg_tta/metrics.py"),
        "runner_sha256": _sha256(__file__),
        "checkpoint_sha256": checkpoint_sha256,
        "data_fingerprint": dataset_provenance["sample_identity_sha256"],
        "annotation_file_sha256": dataset_provenance["annotation_file_sha256"],
        "split_sha256": split_sha256,
        "config_sha256": config_sha256,
    }
    manifest = {
        "temporal_decoder_version": TEMPORAL_DECODER_VERSION,
        "evaluator_sha256": record_provenance["evaluator_sha256"],
        "runner_sha256": record_provenance["runner_sha256"],
        "run_id": run_id,
        "phase": "phase2_support_to_source_cluster_disjoint_query",
        "evidence_role": (
            "mechanism/development on previously viewed HC-STVG2; natural checkpoint shift "
            "is external-training-domain evidence but HC query clips are not newly untouched"
        ),
        "split": split,
        "split_sha256": split_sha256,
        "fixed_split_source": fixed_split_source,
        "args": vars(args),
        "config_provenance": {
            "sha256": config_sha256,
            "args": config_args,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_report": checkpoint_report,
        "target_dataset": args.dataset,
        "dataset_provenance": dataset_provenance,
        "data_fingerprint": dataset_provenance["sample_identity_sha256"],
        "annotation_file_sha256": dataset_provenance["annotation_file_sha256"],
        "model_training_mode": model.training,
        "protocol": (
            "one clip per support source cluster; query source clusters are disjoint; "
            "adaptation state persists across support and is reset before every run"
        ),
    }
    if args.cache_temporal_query_features:
        temporal_query_cache, cache_audit = build_temporal_query_cache(
            model,
            dataset,
            split["query_indices"],
            args.conditions,
            args,
        )
    else:
        temporal_query_cache = None
        cache_audit = {"enabled": False}
    manifest["temporal_query_cache_audit"] = cache_audit
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    records: list[dict[str, Any]] = []
    with (output_dir / "query_records.jsonl").open("w") as query_handle, (
        output_dir / "support_updates.jsonl"
    ).open("w") as support_handle:
        # Frozen query predictions are shared by every support run.
        for condition in args.conditions:
            frozen_records = evaluate_query(
                model,
                dataset,
                split["query_indices"],
                condition,
                "frozen",
                args,
                support_size=0,
                support_seed=0,
                run_id=run_id,
                provenance=record_provenance,
                delta=None,
                handle=query_handle,
                temporal_query_cache=temporal_query_cache,
            )
            records.extend(frozen_records)

        if "tta_ensemble" in args.methods:
            for condition in args.conditions:
                ensemble_records = evaluate_query(
                    model,
                    dataset,
                    split["query_indices"],
                    condition,
                    "tta_ensemble",
                    args,
                    support_size=0,
                    support_seed=0,
                    run_id=run_id,
                    provenance=record_provenance,
                    delta=None,
                    handle=query_handle,
                    temporal_query_cache=None,
                )
                records.extend(ensemble_records)

        support_methods = [method for method in args.methods if method != "tta_ensemble"]
        for method in support_methods:
            for condition in args.conditions:
                for support_size in sorted(args.support_sizes):
                    for seed in args.support_seeds:
                        plan = split["support_plans"][str(seed)]
                        support_indices = plan["ordered_indices"][:support_size]
                        adapter, delta = adapt_on_support(
                            model,
                            dataset,
                            support_indices,
                            condition,
                            method,
                            args,
                            seed=seed,
                            support_size=support_size,
                            run_id=run_id,
                            provenance=record_provenance,
                            support_log_handle=support_handle,
                        )
                        try:
                            adapted_records = evaluate_query(
                                model,
                                dataset,
                                split["query_indices"],
                                condition,
                                method,
                                args,
                                support_size=support_size,
                                support_seed=seed,
                                run_id=run_id,
                                provenance=record_provenance,
                                delta=delta,
                                handle=query_handle,
                                temporal_query_cache=temporal_query_cache,
                            )
                            records.extend(adapted_records)
                        finally:
                            adapter.reset()
                            if adapter.delta_from_initial()["parameter_delta_max_abs"] != 0.0:
                                raise RuntimeError("support-run reset was not exact")
    summarize(records, output_dir, args)


if __name__ == "__main__":
    main()
