"""Source-supervised, frozen-PTD fit for the three Desta3D reader designs.

``propose`` and ``cpu-check`` are CPU-only. ``register``, ``capture-queries``,
``fit`` and ``eval-corrupt`` are explicit actions; this file does not start a
GPU job merely by being imported. The model's official PTD decoder and output
grammar stay intact. Validation labels are opened only after a prediction
manifest has been written and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from scripts.decota_matrix_common_v1 import sha

PARENT = ROOT / "artifacts/desta3d_v1"
OUT = PARENT / "source_fit"
INPUTS = PARENT / "SOURCE_INPUTS.json"
LABELS = PARENT / "SOURCE_LABELS_TRAINING_ONLY.json"
PROVENANCE = PARENT / "SOURCE_LABEL_PROVENANCE.json"
FULL_BARRIER = PARENT / "SOURCE_FULL_FEATURE_BARRIER.json"
CAP_SECONDS = 28_800
QUERY_STAGE_CAP = 3_600
FIT_STAGE_CAP = 7_200
CORRUPT_STAGE_CAP = 1_800
DISK_FREE_MIN_BYTES = 8 * 1024**3 + 200 * 1024**2
SEED = 20260927
ARCHITECTURES = {
    "early_factorized": {"hidden_dim": 275, "expected_active_parameters": 3_679_587},
    "shared3d": {"hidden_dim": 265, "expected_active_parameters": 3_685_267},
    "dual3d": {"hidden_dim": 256, "expected_active_parameters": 3_686_146},
}
LOSS_WEIGHTS = {"ptd_source_ce_ntp_mtp": 1.0, "referent_soft_occupancy_bce": 0.1, "event_soft_occupancy_bce": 0.1}


def jread(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def jwrite_once(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"refusing to replace immutable artifact: {path}"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def save_pt_once(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f"refusing to replace immutable tensor artifact: {path}"
    torch.save(obj, path)


def _assert_disk_reserve(operation: str) -> dict[str, int]:
    usage = shutil.disk_usage(ROOT)
    assert usage.total > 0 and usage.free >= 0
    if usage.free < DISK_FREE_MIN_BYTES:
        raise RuntimeError(
            f"disk reserve reached before {operation}: {usage.free} bytes free; "
            f"requires at least {DISK_FREE_MIN_BYTES} bytes (8 GiB plus 200 MiB staging reserve)"
        )
    return {"free_bytes": usage.free, "required_free_bytes": DISK_FREE_MIN_BYTES}


def load_official_builder():
    path = ROOT / "external/ParallelTubeDecoding/data/prepare_vidstg.py"
    spec = importlib.util.spec_from_file_location("desta3d_source_fit_prepare_vidstg", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_rows() -> list[dict[str, Any]]:
    rows = jread(INPUTS)
    assert len(rows) == 816
    assert len({r["key"] for r in rows}) == 816
    return rows


def _raw_referent_observation(prov: dict[str, Any], target_id: int, frame_id: int) -> dict[str, Any]:
    observations = [
        obs
        for frame in prov["frames"]
        if int(frame["frame_id"]) == int(frame_id)
        for obs in frame["observations"]
        if int(obs["tid"]) == target_id and bool(obs["is_referent"])
    ]
    assert len(observations) == 1, f"raw target trajectory must be unique: tid={target_id}, frame={frame_id}"
    obs = observations[0]
    return {"generated": int(obs["generated"]), "tracker": str(obs["tracker"])}


def _make_label_record(row: dict[str, Any], lab: dict[str, Any], prov: dict[str, Any], builder) -> dict[str, Any]:
    q = row["input"]
    assert lab["frame_ids"] == q["frame_ids"], f"label/input frame mismatch: {row['key']}"
    assert len(lab["boxes_xyxy"]) == len(q["frame_ids"])
    iv = lab["event_interval"]
    begin, end = int(iv["begin_fid"]), int(iv["end_fid"])
    assert begin < end
    for fid, active in zip(q["frame_ids"], lab["event_active"]):
        assert bool(active) == (begin <= int(fid) < end), f"event mask/physical interval mismatch: {row['key']}"

    build_boxes: list[tuple[int, list[int]]] = []
    tokenized_positions: list[int] = []
    raw_flags: dict[str, dict[str, Any]] = {}
    for pos, (fid, xyxy, valid, active) in enumerate(
        zip(q["frame_ids"], lab["boxes_xyxy"], lab["box_valid"], lab["event_active"]), start=1
    ):
        if not bool(valid):
            continue
        x0, y0, x1, y1 = (float(v) for v in xyxy)
        assert all(math.isfinite(v) for v in (x0, y0, x1, y1))
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0, f"invalid normalized source box: {row['key']}:{pos}"
        raw_flags[str(pos)] = _raw_referent_observation(prov, int(lab["target_id"]), int(fid))
        if not bool(active):
            continue
        w, h = float(q["width"]), float(q["height"])
        xy_pixel = [float(xyxy[0]) * w, float(xyxy[1]) * h, float(xyxy[2]) * w, float(xyxy[3]) * h]
        normalized_tokens = builder.normalize_box(xy_pixel, w, h)
        if normalized_tokens is not None:
            build_boxes.append((pos, normalized_tokens))
            tokenized_positions.append(pos)

    response_record = builder.build_record(q["video_path"], q["caption"].strip(), build_boxes)
    return {
        "key": row["key"],
        "source": str(row["source"]),
        "split": row["split"],
        "video_sha256": q["video_sha256"],
        "caption": q["caption"],
        "frame_ids": [int(x) for x in q["frame_ids"]],
        "fps": float(q["fps"]),
        "frame_count": len(q["frame_ids"]),
        "width": int(q["width"]),
        "height": int(q["height"]),
        "target_id": int(lab["target_id"]),
        "category": lab["category"],
        "event_interval": {"begin_fid": begin, "end_fid": end},
        "boxes_xyxy": [[float(v) for v in box] for box in lab["boxes_xyxy"]],
        "box_valid": [bool(x) for x in lab["box_valid"]],
        "event_active": [bool(x) for x in lab["event_active"]],
        "raw_referent_flags_by_sample_position": raw_flags,
        "response_eligible": response_record is not None,
        "response": response_record["conversations"][1]["value"] if response_record is not None else None,
        "response_box_positions_1based": tokenized_positions,
        "response_box_count": len(tokenized_positions),
        "source_label_semantics": lab["label_semantics"],
        "training_semantics": "box_valid frames are referent support positives; event_active && box_valid frames support event field; known inactive event frames are event-field zero; active missing boxes are masked unknown",
    }


def prepare() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _all_rows()
    labels = jread(LABELS)
    provenance = jread(PROVENANCE)["records"]
    builder = load_official_builder()
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    assert (len(train), len({r["source"] for r in train})) == (618, 95)
    assert (len(val), len({r["source"] for r in val})) == (198, 31)
    train_parents = {str(r["source"]) for r in train}
    val_parents = {str(r["source"]) for r in val}
    train_videos = {str(r["input"]["video_sha256"]) for r in train}
    val_videos = {str(r["input"]["video_sha256"]) for r in val}
    assert train_parents.isdisjoint(val_parents), "source parents overlap between train and validation"
    assert train_videos.isdisjoint(val_videos), "video media hashes overlap between train and validation"
    train_records = []
    counts: dict[str, dict[str, int]] = {}
    for split_rows in (train, val):
        ok = support = all_valid = 0
        for row in split_rows:
            key = row["key"]
            lab, prov = labels[key], provenance[key]
            assert str(prov["source"]) == str(row["source"]) and prov["split"] == row["split"]
            rec = _make_label_record(row, lab, prov, builder)
            ok += int(rec["response_eligible"])
            support += int(any(v and a for v, a in zip(rec["box_valid"], rec["event_active"])))
            all_valid += int(any(rec["box_valid"]))
            if row["split"] == "train":
                train_records.append(rec)
        counts[split_rows[0]["split"]] = {
            "queries": len(split_rows),
            "parents": len({r["source"] for r in split_rows}),
            "unique_media_hashes": len({str(r["input"]["video_sha256"]) for r in split_rows}),
            "official_response_eligible": ok,
            "response_ineligible": len(split_rows) - ok,
            "queries_with_any_box_valid": all_valid,
            "queries_with_event_active_box_valid_support": support,
            "queries_without_event_active_box_support": len(split_rows) - support,
        }
    assert counts["train"]["official_response_eligible"] == 612
    assert counts["validation"]["official_response_eligible"] == 193
    return train_records, counts


def _write_or_check(path: Path, payload: Any) -> None:
    if path.exists():
        assert jread(path) == payload, f"existing immutable artifact differs: {path}"
    else:
        jwrite_once(path, payload)


def propose() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_records, split_counts = prepare()
    train_path = OUT / "SOURCE_TRAIN_RECORDS.json"
    proposal_path = OUT / "PROPOSAL.json"
    _write_or_check(train_path, train_records)
    from vg_tta.desta3d_v1 import Desta3DAdapter

    counts = {}
    for arch, c in ARCHITECTURES.items():
        model = Desta3DAdapter(in_channels=2560, query_dim=2560, hidden_dim=c["hidden_dim"], architecture=arch)
        actual = model.active_parameter_count()
        assert actual == c["expected_active_parameters"], f"capacity mismatch for {arch}: {actual}"
        counts[arch] = {**c, "active_parameters_measured": actual}
        del model
    proposal = {
        "version": "desta3d_source_fit_v1",
        "status": "proposal_cpu_ready_not_registered_not_run",
        "task": "fit three capacity-matched Desta3D adapters through the frozen official PTD-4B visual.merger while preserving the stock PTD decoder and response grammar",
        "data": {
            "source_split_counts": split_counts,
            "train_validation_parent_and_media_hash_intersection": {"parents": 0, "video_sha256": 0},
            "training_records_file": str(train_path),
            "training_records_sha256": sha(train_path),
            "train_parent_validation_parent_disjoint": True,
            "train_validation_media_hashes_disjoint": True,
            "target_data_used": False,
            "source32_sample_frames": "use each SOURCE_INPUTS row's locked exact frame_ids; source videos decoded from their registered path and hash",
            "query_vectors": "exact frozen 2560-D mean of original PTD prompt-prefill language hidden states after the final video token; capture once per clean query; no target/response tokens",
            "source_response_rule": "official prepare_vidstg.build_record plus official _append_ptd_targets; contiguous, event-active valid referent boxes only; builder-ineligible samples remain in auxiliary fit",
            "raw_trajectory_flags": "generated/tracker preserved for each valid raw referent frame",
        },
        "model": {
            "checkpoint": str(ROOT / "checkpoints/ParallelTubeDecoding-Qwen3-VL-4B"),
            "decoder": "official PTD 4B frozen; unchanged tokenizer, decoder, generation, and PTD response grammar",
            "hook": "model.model.visual.merger; inject zero-initialized Desta3DAdapter updated_tokens in the native merged-token stream",
            "first_source_step_noop_check": "for each arm, compare stock and hooked PTD source CE and merger tokens on the first legal source response before its first optimizer update",
            "3d_semantics": "THW video feature grid (time, merged height, merged width), never XYZ; temporal convolutions currently use sampled-frame index adjacency, not physical time-gap scaling",
            "backbone_trainable": False,
            "adapters": counts,
            "capacity_spread_percent": (max(x["expected_active_parameters"] for x in ARCHITECTURES.values()) / min(x["expected_active_parameters"] for x in ARCHITECTURES.values()) - 1) * 100,
        },
        "objective": {
            "loss_weights": LOSS_WEIGHTS,
            "official_ce": "official PTD joint_loss (NTP+MTP sequence CE) on legal source responses; no CE term on response-ineligible rows",
            "referent_field": "soft cell-area occupancy of normalized xyxy target box on every box_valid sample frame, including valid boxes outside the event; all grid cells supervised there",
            "event_field": "same soft occupancy on event_active && box_valid frames; all-zero occupancy on known inactive frames; event_active && !box_valid frames masked unknown",
            "field_semantics": "box-support occupancy only, not object segmentation; no missing-box or event-external object-absence assumption",
            "response_ineligible_rows": "included in source training; their auxiliary fields remain trainable whenever valid known source masks exist",
        },
        "optimization": {
            "epochs_max": 2,
            "order": "same fixed-seed epoch permutation of the 618 train queries for every arm",
            "arm_schedule": "for each source query, execute one update for each of the three arms before advancing; save/resume only at equal-progress triplet boundaries",
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "grad_norm_clip": 1.0,
            "selection": "best of epochs 1-2 by clean source-validation full-tube vIoU, average queries within each of 31 validation parents then equally average parents; ties choose earliest epoch",
            "validation_gt_access": "source labels were already read for CPU eligibility accounting and the authorized native Frozen source reference; for each new epoch, read the validation annotations for scoring/selection only after all three arms' 198 prediction files are saved and sealed",
            "reader_diagnostics": "sealed predictions store CPU float16 referent/event logits and merged THW; after all arms are sealed, report masked soft-occupancy BCE and parent-macro event-frame AUROC; referent AUROC uses occupancy>0 box-support cells only and is not segmentation",
        },
        "evaluation": {
            "primary": "all 198 clean source-validation queries, official PTD decode through the adapter merger hook, GT support is box_valid && event_active, physical begin_fid/end_fid retained, corrected sampled temporal-union denominator",
            "no_support_policy": "retain query; empty spatial numerator yields vIoU=0 with original sample-union denominator; report spatial-support query/frame counts",
            "frozen_baseline": "native clean 4B outputs from hash-verified SOURCE_FULL_FEATURE_BARRIER caches, scored with the same labels/metric",
            "secondary": "31 validation parents under noise_medium and defocus_extreme, using each corruption's own frozen prompt-prefill query; diagnostics only, never checkpoint selection",
            "metric": "vg_tta.metrics.compute_stvg_metrics; prediction coordinate tokens are xyxy/1000 converted to normalized cxcywh; GT source xyxy converted to cxcywh",
        },
        "resource_caps": {"cumulative_gpu_seconds": CAP_SECONDS, "query_capture_stage_seconds": QUERY_STAGE_CAP, "three_arm_fit_and_validation_stage_seconds": FIT_STAGE_CAP, "corruption_readout_stage_seconds": CORRUPT_STAGE_CAP, "includes_prior_failures_and_nested_receipts": True},
        "status_gates": {"code_and_proposal_cpu_only": True, "no_gpu_started": True, "register_required_before_capture_or_fit": True, "parent_review_required_before_any_gpu_action": True, "this_is_not_target_tta_or_deployable_method_evidence": True},
    }
    _write_or_check(proposal_path, proposal)
    print(json.dumps({"proposal": str(proposal_path), "train_records": str(train_path), "split_counts": split_counts, "adapter_counts": counts}, indent=2))


def cell_occupancy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Fraction of each normalized [H,W] grid cell covered by xyxy box."""
    if boxes.shape[-1] != 4 or height < 1 or width < 1:
        raise ValueError("expected normalized xyxy boxes and positive grid dimensions")
    boxes = boxes.float()
    xs0 = torch.arange(width, device=boxes.device, dtype=boxes.dtype) / width
    xs1 = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 1) / width
    ys0 = torch.arange(height, device=boxes.device, dtype=boxes.dtype) / height
    ys1 = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 1) / height
    iw = (torch.minimum(boxes[..., 2, None], xs1) - torch.maximum(boxes[..., 0, None], xs0)).clamp_min(0)
    ih = (torch.minimum(boxes[..., 3, None], ys1) - torch.maximum(boxes[..., 1, None], ys0)).clamp_min(0)
    return (ih[..., :, None] * iw[..., None, :] * (height * width)).clamp_(0.0, 1.0)


def aux_targets(record: dict[str, Any], height: int, width: int, device: torch.device | str) -> dict[str, torch.Tensor]:
    t = len(record["frame_ids"])
    ref_y = torch.zeros((1, t, height, width), device=device, dtype=torch.float32)
    ref_mask = torch.zeros_like(ref_y, dtype=torch.bool)
    event_y = torch.zeros_like(ref_y)
    event_mask = torch.zeros_like(ref_mask)
    for i, (box, valid, active) in enumerate(zip(record["boxes_xyxy"], record["box_valid"], record["event_active"])):
        if valid:
            occupancy = cell_occupancy(torch.tensor(box, device=device).view(1, 4), height, width)[0]
            ref_y[0, i] = occupancy
            ref_mask[0, i] = True
        if not active:
            event_mask[0, i] = True
        elif valid:
            event_y[0, i] = cell_occupancy(torch.tensor(box, device=device).view(1, 4), height, width)[0]
            event_mask[0, i] = True
        # active and missing/invalid: retain mask=False (unknown, not object absent)
    return {"referent_target": ref_y, "referent_mask": ref_mask, "event_target": event_y, "event_mask": event_mask}


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or mask.shape != target.shape:
        raise ValueError(f"field BCE shape mismatch: {logits.shape}, {target.shape}, {mask.shape}")
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits.float()[mask], target.float()[mask])


def cpu_check() -> None:
    from vg_tta.desta3d_v1 import Desta3DAdapter

    # The four cell occupancies of this quarter-cell box sum to its area share.
    occ = cell_occupancy(torch.tensor([[0.0, 0.0, 0.25, 0.25]]), 2, 2)
    assert torch.allclose(occ, torch.tensor([[[0.25, 0.0], [0.0, 0.0]]]))
    synthetic = {
        "frame_ids": [10, 20, 30],
        "boxes_xyxy": [[0.0, 0.0, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.5, 0.5]],
        "box_valid": [True, False, True],
        "event_active": [False, True, True],
    }
    masks = aux_targets(synthetic, 2, 2, "cpu")
    assert masks["referent_mask"][0, 0].all() and masks["referent_mask"][0, 2].all()
    assert not masks["referent_mask"][0, 1].any()
    assert masks["event_mask"][0, 0].all() and not masks["event_target"][0, 0].any()
    assert not masks["event_mask"][0, 1].any(), "active missing box must be unknown, not negative"
    assert masks["event_mask"][0, 2].all() and masks["event_target"][0, 2].sum() > 0

    measured = {}
    for arch, conf in ARCHITECTURES.items():
        adapter = Desta3DAdapter(2560, 2560, conf["hidden_dim"], arch)
        nparams = adapter.active_parameter_count()
        assert nparams == conf["expected_active_parameters"]
        x = torch.randn(1, 3, 2, 2, 2560)
        q = torch.randn(1, 2560)
        with torch.no_grad():
            out = adapter(x, q)
        assert torch.equal(out["updated_tokens"], x), f"{arch} is not exact-zero initialized"
        measured[arch] = nparams
        del adapter

    gradient_groups = {}
    for arch in ARCHITECTURES:
        torch.manual_seed(SEED)
        adapter = Desta3DAdapter(16, 16, 8, arch)
        x, q = torch.randn(1, 3, 2, 2, 16), torch.randn(1, 16)
        out = adapter(x, q)
        field_target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]).expand(1, 3, 2, 2).clone()
        aux_loss = F.binary_cross_entropy_with_logits(out["referent_logits"], field_target)
        aux_loss += F.binary_cross_entropy_with_logits(out["event_logits"], field_target)
        visual_loss = (out["updated_tokens"] - torch.full_like(x, 0.01)).square().mean()
        (visual_loss + 0.1 * aux_loss).backward()
        group_rows = _grad_groups(adapter)
        required = ["input_proj", "query_proj", "out_proj", "referent_head", "event_head"]
        required += ["factorized_reader"] if arch == "early_factorized" else ["stem", "shared_reader"] if arch == "shared3d" else ["stem", "short_reader", "long_reader"]
        for group in required:
            assert group_rows.get(group, {}).get("grad_norm", 0.0) > 0.0, f"synthetic objective missed {arch}.{group}"
        gradient_groups[arch] = {group: group_rows[group]["grad_norm"] for group in required}
        del adapter

    # Check eligibility counts and build only source-train target records.
    records, split_counts = prepare()
    assert len(records) == 618 and sum(r["response_eligible"] for r in records) == 612
    assert split_counts["validation"]["queries"] == 198
    record_map = {r["key"]: r for r in records}
    assert all(record_map[key]["response_eligible"] for order in _training_order(record_map) for key in order[:2]), "first two source steps must carry legal PTD source responses for gradient gates"
    metric_diffs = _frozen_metric_agreement()
    resume_check = _cpu_resume_failure_check()
    query_cache_check = _cpu_query_cache_roundtrip_check()
    aux_metric_check = _cpu_aux_field_metrics_check()
    invocation_check = _cpu_invocation_signature_check()
    payload = {
        "status": "cpu_check_passed_no_gpu_no_backbone_weights",
        "cell_occupancy_geometry": True,
        "active_missing_event_box_masked_unknown": True,
        "known_inactive_event_frames_zero_supervised": True,
        "outside_event_valid_boxes_remain_referent_positives": True,
        "adapter_exact_zero_noop_all_arms": True,
        "adapter_active_parameter_counts": measured,
        "synthetic_reader_and_visual_losses_have_nonzero_gradients": gradient_groups,
        "frozen_project_metric_matches_independent_source_readback_max_abs_diff": metric_diffs,
        "second_arm_failure_resume_preserves_equal_progress": resume_check,
        "query_cache_sidecar_manifest_roundtrip": query_cache_check,
        "source_reader_field_metric_semantics": aux_metric_check,
        "cpu_callsite_signature_binding": invocation_check,
        "split_counts": split_counts,
        "target_GT_used": False,
        "GPU_started": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "CPU_CHECK.json"
    _write_or_check(path, payload)
    print(json.dumps(payload, indent=2))


def gpu_receipt_files() -> list[Path]:
    return sorted(
        p for p in PARENT.rglob("*.json")
        if any(part in {"receipts", "worker_receipts"} for part in p.parts)
    )


def spent_gpu_seconds() -> tuple[float, list[dict[str, Any]]]:
    total = 0.0
    rows = []
    for path in gpu_receipt_files():
        try:
            payload = jread(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot audit GPU receipt {path}") from exc
        seconds = payload.get("seconds")
        assert isinstance(seconds, (int, float)), f"GPU receipt has no numeric seconds: {path}"
        seconds = float(seconds)
        assert math.isfinite(seconds) and seconds >= 0.0, f"GPU receipt has invalid seconds: {path}"
        total += seconds
        rows.append({"path": str(path), "seconds": seconds, "stage": payload.get("stage")})
    return total, rows


def source_rows_and_records() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = {r["key"]: r for r in _all_rows()}
    records = {r["key"]: r for r in jread(OUT / "SOURCE_TRAIN_RECORDS.json")}
    return rows, records


def _registration_pins() -> dict[str, str]:
    checkpoint = ROOT / "checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/OFFICIAL_RECEIPT.json"
    files = [
        Path(__file__),
        ROOT / "scripts/desta3d_adapter_train_v1.py",  # exact frozen query capture helper
        ROOT / "scripts/ptd_spatial_adapter_ab_v1.py",
        ROOT / "scripts/ptd_8b_teacher_feasibility_v1.py",  # official append_targets/joint_loss only
        ROOT / "external/ParallelTubeDecoding/data/prepare_vidstg.py",
        ROOT / "external/ParallelTubeDecoding/src/dataset/sft_dataset.py",
        ROOT / "external/ParallelTubeDecoding/src/model/ptd_generation.py",
        ROOT / "vg_tta/desta3d_v1.py",
        ROOT / "vg_tta/ptd_spatial_adapter_ab_v1.py",
        ROOT / "vg_tta/metrics.py",
        INPUTS,
        LABELS,
        PROVENANCE,
        PARENT / "SOURCE_PREPARATION.json",
        FULL_BARRIER,
        PARENT / "SOURCE_FULL_INDEPENDENT_READBACK.json",
        PARENT / "SOURCE_FROZEN_INDEPENDENT_METRICS.json",
        ROOT / "scripts/desta3d_source_readback_v1.py",
        ROOT / "scripts/desta3d_source_supervisor_v1.py",
        PARENT / "BUDGET_AUTHORIZATION.json",
        ROOT / "methods/CURRENT_METHOD.json",
        checkpoint,
        OUT / "SOURCE_TRAIN_RECORDS.json",
        OUT / "PROPOSAL.json",
        OUT / "CPU_CHECK.json",
    ]
    return {str(path): sha(path) for path in files}


def register() -> None:
    assert (OUT / "PROPOSAL.json").is_file() and (OUT / "CPU_CHECK.json").is_file()
    assert jread(OUT / "CPU_CHECK.json")["GPU_started"] is False
    auth = jread(PARENT / "BUDGET_AUTHORIZATION.json")
    assert int(auth["GPU_seconds_cap"]) == CAP_SECONDS, "budget authorization does not match the cumulative cap"
    barrier = jread(FULL_BARRIER)
    assert barrier["GT_read"] is False
    assert len(barrier["files"]) >= 800, "full source frozen feature barrier is incomplete"
    before, receipts = spent_gpu_seconds()
    assert before < CAP_SECONDS, f"prior GPU receipts already reach/exceed cap: {before:.3f}s"
    cfg = {
        "version": "desta3d_source_fit_v1",
        "status": "registered_not_run_parent_review_required",
        "proposal_sha256": sha(OUT / "PROPOSAL.json"),
        "train_records_sha256": sha(OUT / "SOURCE_TRAIN_RECORDS.json"),
        "source_split": {"train_queries": 618, "train_parents": 95, "train_media_hashes": 95, "validation_queries": 198, "validation_parents": 31, "validation_media_hashes": 31, "parent_intersection": 0, "media_hash_intersection": 0},
        "architectures": ARCHITECTURES,
        "loss_weights": LOSS_WEIGHTS,
        "epochs_max": 2,
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.0, "grad_clip": 1.0},
        "arm_schedule": "round-robin per train query; all three architectures update before cursor advances; resume only from safe equal-progress triplets",
        "selection": "best clean source-validation full-tube vIoU; query means within parent then equal-weight over 31 parents; earliest epoch on tie",
        "reader_field_audit": "sealed GT-free predictions retain CPU float16 THW referent/event logits; after all three arms seal, report masked soft-occupancy BCE, parent-macro event-frame max-logit AUROC and secondary referent box-support-cell AUROC (occupancy>0 only, not segmentation); these metrics do not select epochs",
        "real_ptd_zero_noop_gate": "each arm compares stock and zero-init hooked merger tokens and source NTP+MTP CE on first legal train source response before its first update",
        "resource_caps": {"cumulative_gpu_seconds": CAP_SECONDS, "query_capture_stage_seconds": QUERY_STAGE_CAP, "fit_plus_validation_stage_seconds": FIT_STAGE_CAP, "corruption_readout_stage_seconds": CORRUPT_STAGE_CAP, "spent_before_registration_seconds": before, "prior_receipts": receipts},
        "validation_prediction_gt_firewall": "source validation labels have been read for CPU eligibility counts and native Frozen reference; for each new epoch, score/select only after all three arms' 198 prediction files are sealed; score full sampled T with box_valid && event_active GT boxes only",
        "target_GT_read": False,
        "GPU_started": False,
        "parent_gpu_invocation_review": "required before capture-queries or fit; register itself is CPU-only",
    }
    cfg_path, lock_path = OUT / "TRAIN_CONFIG.json", OUT / "TRAIN_LOCK.json"
    _write_or_check(cfg_path, cfg)
    lock = {"created_unix": time.time(), "pins": _registration_pins(), "config_sha256": sha(cfg_path), "script_sha256": sha(Path(__file__)), "status": cfg["status"]}
    if lock_path.exists():
        assert jread(lock_path) == lock, "existing source-fit registration differs; do not overwrite"
    else:
        jwrite_once(lock_path, lock)
    print(json.dumps({"lock": str(lock_path), "config": str(cfg_path), "GPU_started": False, "spent_gpu_seconds": before}, indent=2))


def verify_lock() -> dict[str, Any]:
    lock_path = OUT / "TRAIN_LOCK.json"
    assert lock_path.is_file(), "run the CPU-only register action before a GPU action"
    lock = jread(lock_path)
    assert sha(Path(__file__)) == lock["script_sha256"], "source-fit script changed after registration"
    assert sha(OUT / "TRAIN_CONFIG.json") == lock["config_sha256"]
    for path, digest in lock["pins"].items():
        assert sha(path) == digest, f"registered source/dependency changed: {path}"
    cfg = jread(OUT / "TRAIN_CONFIG.json")
    assert cfg["resource_caps"]["cumulative_gpu_seconds"] == CAP_SECONDS
    assert cfg["resource_caps"]["fit_plus_validation_stage_seconds"] == FIT_STAGE_CAP
    return cfg


def _query_path(key: str) -> Path:
    return OUT / "query_clean" / f"{hashlib.sha256(key.encode()).hexdigest()}.pt"


def _query_manifest() -> Path:
    return OUT / "QUERY_CLEAN_SEAL.json"


def _verify_query_manifest_payload(manifest: dict[str, Any], expected_count: int) -> dict[str, Any]:
    assert manifest["count"] == expected_count and manifest["GT_read"] is False
    assert len(manifest["files"]) == expected_count
    assert len({row["key"] for row in manifest["files"]}) == expected_count
    for row in manifest["files"]:
        path = Path(row["path"])
        assert sha(path) == row["sha256"], f"exact-query cache changed: {path}"
        assert sha(row["metadata_path"]) == row["metadata_sha256"]
        metadata = jread(row["metadata_path"])
        assert metadata["query_sha256"] == row["sha256"] and metadata["key"] == row["key"]
        q = torch.load(path, map_location="cpu", weights_only=True)
        assert tuple(q.shape) == (2560,) and torch.isfinite(q).all()
    return manifest


def verify_query_manifest() -> dict[str, Any]:
    return _verify_query_manifest_payload(jread(_query_manifest()), 816)


def _cpu_query_cache_roundtrip_check() -> dict[str, Any]:
    import tempfile

    key = "query-cache-cpu-fixture"
    with tempfile.TemporaryDirectory(prefix="desta3d-query-fixture-") as temp:
        qpath = Path(temp) / "query.pt"
        meta = Path(temp) / "query.json"
        torch.save(torch.linspace(-1.0, 1.0, 2560), qpath)
        qhash = sha(qpath)
        jwrite_once(meta, {"key": key, "query_sha256": qhash, "GT_read": False})
        manifest = {"count": 1, "GT_read": False, "files": [{"key": key, "path": str(qpath), "sha256": qhash, "metadata_path": str(meta), "metadata_sha256": sha(meta)}]}
        _verify_query_manifest_payload(manifest, 1)
    return {"status": "passed", "fixture_count": 1, "query_dim": 2560, "metadata_and_tensor_hashes_match": True}


def capture_queries() -> None:
    cfg = verify_lock()
    assert not _query_manifest().exists(), "exact clean query cache was already sealed"
    assert torch.cuda.is_available(), "query capture requires GPU"
    before, _ = spent_gpu_seconds()
    stage_before = _stage_receipt_seconds("exact_clean_query_capture")
    started = time.monotonic()
    assert before < CAP_SECONDS and stage_before < QUERY_STAGE_CAP
    from scripts.run_final_simplification_v1 import lease
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load, frames_for, inputs_for
    from scripts.desta3d_adapter_train_v1 import capture_exact_query

    rows = _all_rows()
    row_manifest = []
    guard = lease()
    peak = 0
    try:
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        processor, model = processor_load(), model_load()
        model.eval().requires_grad_(False)
        model.config.use_cache = False
        assert all(not p.requires_grad for p in model.parameters())
        for index, row in enumerate(rows, start=1):
            _assert_disk_reserve(f"clean query capture {index}/816")
            assert before + time.monotonic() - started < CAP_SECONDS
            assert stage_before + time.monotonic() - started < QUERY_STAGE_CAP
            path = _query_path(row["key"])
            if path.exists():
                cached = torch.load(path, map_location="cpu", weights_only=True)
                assert tuple(cached.shape) == (2560,)
                meta_path = path.with_suffix(".json")
                assert meta_path.exists(), "partial clean query cache lacks metadata; inspect before reuse"
                metadata = jread(meta_path)
                assert metadata["key"] == row["key"] and metadata["query_sha256"] == sha(path)
            else:
                frames, frame_ids = frames_for(row, "clean")
                assert frame_ids == row["input"]["frame_ids"]
                inputs, preprocess = inputs_for(row, processor, frames)
                query = capture_exact_query(model, inputs).squeeze(0).detach().float().cpu().contiguous()
                assert tuple(query.shape) == (2560,) and torch.isfinite(query).all()
                save_pt_once(path, query)
                meta_path = path.with_suffix(".json")
                jwrite_once(meta_path, {"key": row["key"], "source": str(row["source"]), "split": row["split"], "query_sha256": sha(path), "pixel_sha256": preprocess["pixel_sha"], "frame_ids": frame_ids, "query_definition": "frozen PTD prompt-prefill hidden mean after final video token", "GT_read": False})
                row_manifest.append({"key": row["key"], "source": str(row["source"]), "split": row["split"], "path": str(path), "sha256": sha(path), "pixel_sha256": preprocess["pixel_sha"], "frame_count": len(frame_ids)})
                del frames, inputs, query
                torch.cuda.empty_cache()
            peak = max(peak, torch.cuda.max_memory_allocated())
            if index % 25 == 0:
                print("QUERY_CAPTURE", index, row["key"], flush=True)
        # Include previously written files only if a prior process stopped before the seal.
        files = []
        for row in rows:
            path = _query_path(row["key"])
            meta_path = path.with_suffix(".json")
            assert path.is_file()
            assert meta_path.is_file()
            files.append({"key": row["key"], "source": str(row["source"]), "split": row["split"], "path": str(path), "sha256": sha(path), "metadata_path": str(meta_path), "metadata_sha256": sha(meta_path)})
        seal = {"status": "complete_gt_free_exact_query_capture", "count": len(files), "files": files, "query_definition": "exact frozen PTD prompt-prefill mean after final video token", "GT_read": False, "seconds": time.monotonic() - started, "peak_gpu_bytes": peak, "train_lock_sha256": sha(OUT / "TRAIN_LOCK.json")}
        jwrite_once(_query_manifest(), seal)
        verify_query_manifest()
    except BaseException as exc:
        jwrite_once(OUT / f"QUERY_CAPTURE_FAILURE_{time.time_ns()}.json", {"error": repr(exc), "traceback": traceback.format_exc(), "GPU_started": True})
        raise
    finally:
        elapsed = time.monotonic() - started
        OUT.joinpath("receipts").mkdir(parents=True, exist_ok=True)
        jwrite_once(OUT / "receipts" / f"query_capture_{time.time_ns()}.json", {"stage": "exact_clean_query_capture", "seconds": elapsed, "prior_seconds": before, "prior_query_capture_stage_seconds": stage_before, "peak_bytes": peak, "count": len(row_manifest), "GT_read": False, "cumulative_cap_seconds": CAP_SECONDS, "stage_cap_seconds": QUERY_STAGE_CAP})
        guard.close()


class BudgetPause(RuntimeError):
    pass


def _fit_receipt_seconds() -> float:
    _, rows = spent_gpu_seconds()
    return sum(row["seconds"] for row in rows if row["stage"] == "source_three_arm_fit_and_clean_validation")


def _stage_receipt_seconds(stage_name: str) -> float:
    _, rows = spent_gpu_seconds()
    return sum(row["seconds"] for row in rows if row["stage"] == stage_name)


def _budget_guard(started: float, spent_before: float, *, margin: float = 30.0, stage_cap: int = FIT_STAGE_CAP,
                  stage_name: str = "source_three_arm_fit_and_clean_validation") -> None:
    elapsed = time.monotonic() - started
    total = spent_before + elapsed
    stage = _stage_receipt_seconds(stage_name) + elapsed
    if total + margin >= CAP_SECONDS:
        raise BudgetPause(f"cumulative GPU cap reserve reached: {total:.1f}s + {margin:.1f}s >= {CAP_SECONDS}s")
    if stage + margin >= stage_cap:
        raise BudgetPause(f"GPU stage reserve reached for {stage_name}: {stage:.1f}s + {margin:.1f}s >= {stage_cap}s")


def _clean_query_tensor(key: str) -> torch.Tensor:
    value = torch.load(_query_path(key), map_location="cpu", weights_only=True).float()
    assert tuple(value.shape) == (2560,) and torch.isfinite(value).all()
    return value


def _query_context_for_inputs(model, row, processor, condition: str, query_cache: bool = True):
    from scripts.ptd_spatial_adapter_ab_v1 import frames_for, inputs_for

    frames, frame_ids = frames_for(row, condition)
    assert frame_ids == row["input"]["frame_ids"]
    inputs, preprocess = inputs_for(row, processor, frames)
    if query_cache:
        q = _clean_query_tensor(row["key"])
    else:
        from scripts.desta3d_adapter_train_v1 import capture_exact_query
        q = capture_exact_query(model, inputs).squeeze(0).detach().float().cpu()
    q = q.to("cuda", dtype=torch.float32).reshape(1, 2560)
    return frames, frame_ids, inputs, q, preprocess


def _training_inputs(processor, model, row: dict[str, Any], record: dict[str, Any]):
    from scripts.ptd_spatial_adapter_ab_v1 import frames_for, inputs_for
    from scripts.ptd_8b_teacher_feasibility_v1 import append_targets

    frames, frame_ids = frames_for(row, "clean")
    assert frame_ids == row["input"]["frame_ids"] == record["frame_ids"]
    prompt, preprocess = inputs_for(row, processor, frames)
    if record["response_eligible"]:
        prompt_ids = prompt["input_ids"][0].detach().cpu()
        response_ids = processor.tokenizer.encode(record["response"] + "<|im_end|>\n", add_special_tokens=False)
        start = int(prompt_ids.numel())
        response = torch.tensor(response_ids, dtype=torch.long)
        labels = torch.cat((torch.full((start,), -100, dtype=torch.long), response))
        mm = prompt.get("mm_token_type_ids", torch.zeros_like(prompt["input_ids"]))[0].detach().cpu()
        mm = torch.cat((mm, torch.zeros(len(response_ids), dtype=mm.dtype)))
        data = {
            "input_ids": torch.cat((prompt_ids, response)),
            "labels": labels,
            "mm_token_type_ids": mm,
            "video_grid_thw": prompt["video_grid_thw"].detach().cpu(),
            "pixel_values_videos": prompt["pixel_values_videos"].detach().cpu(),
        }
        data = append_targets(processor, data, start)
        data["ptd_prefix_lengths"] = data.pop("ptd_prefix_length").unsqueeze(0)
        for name in ("input_ids", "labels", "mm_token_type_ids", "attention_mask", "ptd_position_ids", "ptd_context_limits"):
            data[name] = data[name].unsqueeze(0).to("cuda")
        for name in ("video_grid_thw", "pixel_values_videos"):
            data[name] = data[name].to("cuda")
        if "second_per_grid_ts" in prompt:
            data["second_per_grid_ts"] = prompt["second_per_grid_ts"].to("cuda")
        data["ptd_prefix_lengths"] = data["ptd_prefix_lengths"].to("cuda")
        prompt_inputs = None
    else:
        data = None
        prompt_inputs = prompt
    del frames
    return data, prompt_inputs, preprocess


def _processor_prompt_kwargs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keep = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"}
    return {k: v for k, v in inputs.items() if k in keep}


def _attach_adapter(adapter, data: dict[str, torch.Tensor] | None, prompt: dict[str, torch.Tensor] | None,
                    query: torch.Tensor, record: dict[str, Any], model, ce_fn, *, exact_snapshot: bool = False):
    from vg_tta.desta3d_v1 import reshape_merged_video_tokens

    source_data = data if data is not None else prompt
    assert source_data is not None
    grid_thw = source_data["video_grid_thw"][0]
    frame_ids = record["frame_ids"]
    fps = float(record["fps"])
    capture: dict[str, Any] = {"calls": 0}

    def hook(module, args, output):
        capture["calls"] += 1
        grid = reshape_merged_video_tokens(output.float(), grid_thw)
        assert grid.shape[1] == len(frame_ids), f"merged THW T={grid.shape[1]} does not match source sampled frames={len(frame_ids)}"
        times = torch.tensor(frame_ids, dtype=torch.float32, device=output.device)[None] / fps
        result = adapter(grid, query, frame_times=times)
        capture["fields"] = result
        capture["zero_noop"] = bool(torch.equal(output, result["updated_tokens"].reshape_as(output).to(output.dtype)))
        updated = result["updated_tokens"].reshape_as(output).to(output.dtype)
        if exact_snapshot:
            capture["input_tokens_cpu"] = output.detach().cpu().clone()
            capture["updated_tokens_cpu"] = updated.detach().cpu().clone()
        return updated

    handle = model.model.visual.merger.register_forward_hook(hook)
    try:
        if data is not None:
            ce, ce_stats = ce_fn(model, data)
        else:
            with torch.enable_grad():
                model.model(**_processor_prompt_kwargs(prompt), use_cache=False)
            fields = capture.get("fields")
            assert fields is not None
            ce = fields["event_logits"].sum() * 0.0
            ce_stats = {"ntp_loss": None, "mtp_loss": None, "ntp_count": 0, "mtp_count": 0}
        assert capture["calls"] == 1, f"merger hook expected exactly one invocation, got {capture['calls']}"
        fields = capture["fields"]
        ref_logits, event_logits = fields["referent_logits"], fields["event_logits"]
        b, t, h, w = ref_logits.shape
        assert b == 1 and t == len(frame_ids) == event_logits.shape[1]
        targets = aux_targets(record, h, w, ref_logits.device)
        ref_loss = masked_bce(ref_logits, targets["referent_target"], targets["referent_mask"])
        event_loss = masked_bce(event_logits, targets["event_target"], targets["event_mask"])
        loss = LOSS_WEIGHTS["ptd_source_ce_ntp_mtp"] * ce
        loss = loss + LOSS_WEIGHTS["referent_soft_occupancy_bce"] * ref_loss
        loss = loss + LOSS_WEIGHTS["event_soft_occupancy_bce"] * event_loss
        aux = {
            "source_ce": float(ce.detach()),
            "referent_bce": float(ref_loss.detach()),
            "event_bce": float(event_loss.detach()),
            "referent_supervised_frames": int(targets["referent_mask"].any(dim=(-1, -2)).sum()),
            "event_supervised_frames": int(targets["event_mask"].any(dim=(-1, -2)).sum()),
            "event_active_unknown_frames_masked": int((torch.tensor(record["event_active"], device=event_logits.device) & ~torch.tensor(record["box_valid"], device=event_logits.device)).sum()),
            "zero_initialized_visual_noop": capture["zero_noop"],
        }
        return loss, {**ce_stats, **aux}, ref_loss, event_loss, capture
    finally:
        handle.remove()


def _grad_groups(adapter) -> dict[str, Any]:
    groups: dict[str, dict[str, float | int]] = {}
    for name, p in adapter.named_parameters():
        group = name.split(".", 1)[0]
        row = groups.setdefault(group, {"parameters": 0, "grad_none_parameters": 0, "nonzero_grad_parameters": 0, "norm_sq": 0.0})
        row["parameters"] += p.numel()
        if p.grad is None:
            row["grad_none_parameters"] += p.numel()
        else:
            norm = float(p.grad.detach().float().norm())
            row["norm_sq"] += norm * norm
            if norm > 0:
                row["nonzero_grad_parameters"] += p.numel()
    for row in groups.values():
        row["grad_norm"] = math.sqrt(float(row.pop("norm_sq")))
    return groups


def _training_order(records: dict[str, dict[str, Any]]) -> list[list[str]]:
    keys = sorted(records)
    orders = []
    for epoch in range(1, 3):
        order = np.random.default_rng(SEED + epoch).permutation(keys).tolist()
        if not records[order[0]]["response_eligible"]:
            eligible = next((k for k in order if records[k]["response_eligible"]), None)
            assert eligible
            order.remove(eligible)
            order.insert(0, eligible)
        assert len(order) == 618 and len(set(order)) == 618
        orders.append(order)
    return orders


def _state_to_cpu(adapter) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}


def _save_progress(path: Path, payload: dict[str, Any]) -> None:
    _assert_disk_reserve(f"checkpoint write {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _load_progress(path: Path, adapters: dict[str, Any], optimizers: dict[str, Any]) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["train_lock_sha256"] == sha(OUT / "TRAIN_LOCK.json")
    assert state["query_manifest_sha256"] == sha(_query_manifest())
    assert state["train_records_sha256"] == sha(OUT / "SOURCE_TRAIN_RECORDS.json")
    for arch in ARCHITECTURES:
        adapters[arch].load_state_dict(state["adapter_states"][arch])
        if arch in state["optimizer_states"]:
            optimizers[arch].load_state_dict(state["optimizer_states"][arch])
            for slot in optimizers[arch].state.values():
                for k, v in slot.items():
                    if torch.is_tensor(v):
                        slot[k] = v.to("cuda")
    return state


def _checkpoint_payload(phase: str, epoch: int, next_index: int, adapters, optimizers, history, best):
    return {
        "train_lock_sha256": sha(OUT / "TRAIN_LOCK.json"),
        "query_manifest_sha256": sha(_query_manifest()),
        "train_records_sha256": sha(OUT / "SOURCE_TRAIN_RECORDS.json"),
        "phase": phase,
        "epoch": epoch,
        "next_source_index": next_index,
        "adapter_states": {arch: _state_to_cpu(adapters[arch]) for arch in ARCHITECTURES},
        "optimizer_states": {arch: optimizers[arch].state_dict() for arch in ARCHITECTURES},
        "history": history,
        "best": best,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def _prediction_dir(arch: str, epoch: int, condition: str = "clean") -> Path:
    return OUT / "predictions" / f"epoch_{epoch}_{arch}" / condition


def _prediction_path(arch: str, epoch: int, key: str, condition: str = "clean") -> Path:
    return _prediction_dir(arch, epoch, condition) / f"{hashlib.sha256(key.encode()).hexdigest()}.pt"


def _prediction_seal_path(arch: str, epoch: int, condition: str = "clean") -> Path:
    return OUT / f"SEALED_{condition.upper()}_VAL_{arch}_E{epoch}.json"


def _state_dict_digest(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _adapter_digest(adapter) -> str:
    return _state_dict_digest(adapter.state_dict())


def _validate_prediction_seal(arch: str, epoch: int, rows: list[dict[str, Any]], condition: str,
                             adapter_sha256: str) -> dict[str, Any]:
    manifest = jread(_prediction_seal_path(arch, epoch, condition))
    assert manifest["architecture"] == arch and manifest["epoch"] == epoch and manifest["condition"] == condition
    assert manifest["count"] == len(rows) and manifest["GT_read"] is False
    assert len(rows) == (198 if condition == "clean" else 31)
    assert adapter_sha256 and manifest["adapter_sha256"] == adapter_sha256, "sealed predictions came from a different or unspecified adapter state"
    assert {r["key"] for r in manifest["files"]} == {r["key"] for r in rows}
    for item in manifest["files"]:
        assert sha(item["path"]) == item["sha256"], f"sealed validation prediction changed: {item['path']}"
    return manifest


def _condition_query_path(condition: str, key: str) -> Path:
    return OUT / "query_conditions" / condition / f"{hashlib.sha256(key.encode()).hexdigest()}.pt"


def _adapted_generation(model, processor, adapter, row, condition: str, adapter_sha256: str) -> dict[str, Any]:
    from scripts.ptd_spatial_adapter_ab_v1 import infer
    from vg_tta.desta3d_v1 import reshape_merged_video_tokens
    from scripts.ptd_spatial_adapter_ab_v1 import frames_for, inputs_for

    frames, ids = frames_for(row, condition)
    assert ids == row["input"]["frame_ids"]
    inputs, preprocess = inputs_for(row, processor, frames)
    if condition == "clean":
        q = _clean_query_tensor(row["key"])
    else:
        qpath = _condition_query_path(condition, row["key"])
        meta_path = qpath.with_suffix(".json")
        if qpath.exists():
            q = torch.load(qpath, map_location="cpu", weights_only=True).float()
            assert meta_path.exists(), "partial corruption query cache lacks metadata; inspect before reuse"
            metadata = jread(meta_path)
            assert metadata["key"] == row["key"] and metadata["condition"] == condition
            assert metadata["pixel_sha256"] == preprocess["pixel_sha"] and metadata["query_sha256"] == sha(qpath)
        else:
            from scripts.desta3d_adapter_train_v1 import capture_exact_query
            q = capture_exact_query(model, inputs).squeeze(0).detach().float().cpu().contiguous()
            save_pt_once(qpath, q)
            jwrite_once(meta_path, {"key": row["key"], "source": str(row["source"]), "split": row["split"], "condition": condition, "pixel_sha256": preprocess["pixel_sha"], "query_sha256": sha(qpath), "frame_ids": ids, "query_definition": "frozen base PTD prompt-prefill hidden mean after final video token for this corrupted input", "GT_read": False})
    assert tuple(q.shape) == (2560,) and torch.isfinite(q).all()
    q = q.to("cuda", dtype=torch.float32).reshape(1, 2560)
    times = torch.tensor(ids, device="cuda", dtype=torch.float32)[None] / float(row["input"]["fps"])
    grid_thw = inputs["video_grid_thw"][0]
    hook_state = {"calls": 0}

    def merger_hook(module, args, output):
        hook_state["calls"] += 1
        visual = reshape_merged_video_tokens(output.float(), grid_thw)
        assert visual.shape[1] == len(ids)
        value = adapter(visual, q, frame_times=times)
        hook_state["merged_thw"] = [int(x) for x in visual.shape[1:4]]
        hook_state["referent_shape"] = list(value["referent_logits"].shape)
        hook_state["referent_logits_cpu_half"] = value["referent_logits"][0].detach().to(device="cpu", dtype=torch.float16).contiguous()
        hook_state["event_logits_cpu_half"] = value["event_logits"][0].detach().to(device="cpu", dtype=torch.float16).contiguous()
        return value["updated_tokens"].reshape_as(output).to(output.dtype)

    handle = model.model.visual.merger.register_forward_hook(merger_hook)
    try:
        result = infer(model, processor, inputs)
    finally:
        handle.remove()
    assert hook_state["calls"] == 1
    assert result.get("GT_used") is False
    return {
        "key": row["key"],
        "source": str(row["source"]),
        "condition": condition,
        "frame_ids": [int(x) for x in ids],
        "positions": [int(x) for x in result.get("positions", [])] if result.get("positions") is not None else None,
        "boxes_cxcywh": result.get("boxes", torch.empty((0, 4))).detach().float().cpu(),
        "geometry_valid": result.get("geometry_valid", torch.empty((0,), dtype=torch.bool)).detach().bool().cpu(),
        "interval": result.get("interval"),
        "format_ok": bool(result.get("format_ok", False)),
        "completion": result.get("completion", ""),
        "hook_calls": hook_state["calls"],
        "referent_field_shape": hook_state.get("referent_shape"),
        "merged_thw": hook_state.get("merged_thw"),
        "aux_field_logits": {
            "referent": hook_state["referent_logits_cpu_half"],
            "event": hook_state["event_logits_cpu_half"],
            "storage_dtype": "float16_cpu",
            "meaning": "query-conditioned box-support and event-field logits; source diagnostic only",
        },
        "preprocess": preprocess,
        "adapter_sha256": adapter_sha256,
        "GT_used": False,
    }


def generate_validation_predictions(arch: str, epoch: int, model, processor, adapter, rows: list[dict[str, Any]], started: float, spent_before: float,
                                    condition: str, adapter_sha256: str,
                                    stage_cap: int = FIT_STAGE_CAP,
                                    stage_name: str = "source_three_arm_fit_and_clean_validation") -> dict[str, Any]:
    seal_path = _prediction_seal_path(arch, epoch, condition)
    if seal_path.exists():
        return _validate_prediction_seal(arch, epoch, rows, condition, adapter_sha256)
    for index, row in enumerate(rows, start=1):
        _assert_disk_reserve(f"validation prediction {arch} epoch {epoch} {index}/{len(rows)}")
        path = _prediction_path(arch, epoch, row["key"], condition)
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=False)
            assert existing["key"] == row["key"] and existing["frame_ids"] == row["input"]["frame_ids"]
            assert existing["adapter_sha256"] == adapter_sha256, "partial validation prediction belongs to another model state"
            continue
        _budget_guard(started, spent_before, margin=15.0, stage_cap=stage_cap, stage_name=stage_name)
        result = _adapted_generation(model, processor, adapter, row, condition, adapter_sha256)
        save_pt_once(path, result)
        if index % 25 == 0:
            print("SOURCE_VAL_GENERATION", arch, epoch, index, row["key"], result["format_ok"], flush=True)
    files = [
        {"key": row["key"], "source": str(row["source"]), "path": str(_prediction_path(arch, epoch, row["key"], condition)), "sha256": sha(_prediction_path(arch, epoch, row["key"], condition))}
        for row in rows
    ]
    query_context_hash = sha(_query_manifest()) if condition == "clean" else hashlib.sha256(
        "\n".join(sha(_condition_query_path(condition, row["key"]).with_suffix(".json")) for row in rows).encode()
    ).hexdigest()
    manifest = {
        "status": "sealed_clean_source_validation_predictions",
        "architecture": arch,
        "epoch": epoch,
        "condition": condition,
        "count": len(files),
        "files": files,
        "native_decoder_and_grammar": True,
        "adapter_hook": "frozen PTD visual.merger output",
        "adapter_sha256": adapter_sha256,
        "GT_read": False,
        "query_context_sha256": query_context_hash,
        "train_lock_sha256": sha(OUT / "TRAIN_LOCK.json"),
    }
    jwrite_once(seal_path, manifest)
    return _validate_prediction_seal(arch, epoch, rows, condition, adapter_sha256)


def _dense_positions_to_boxes(positions: list[int] | None, boxes_cxcywh: torch.Tensor, frame_count: int) -> torch.Tensor:
    dense = torch.zeros((frame_count, 4), dtype=torch.float32)
    if positions is None:
        return dense
    positions = [int(x) for x in positions]
    if len(positions) != len(boxes_cxcywh) or len(set(positions)) != len(positions):
        raise ValueError("PTD positions and sparse box predictions must be unique and aligned")
    for pos, box in zip(positions, boxes_cxcywh):
        if not 0 <= pos < frame_count:
            raise ValueError(f"PTD output sample position is outside input T: {pos}/{frame_count}")
        dense[pos] = box.detach().float().cpu()
    return dense


def _tube_metric(prediction: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    from vg_tta.metrics import compute_stvg_metrics

    frame_ids = [int(x) for x in prediction["frame_ids"]]
    assert frame_ids == [int(x) for x in label["frame_ids"]]
    support = [bool(v) and bool(a) for v, a in zip(label["box_valid"], label["event_active"])]
    support_count = sum(support)
    interval = prediction.get("interval")
    t = len(frame_ids)
    valid_interval = isinstance(interval, (list, tuple)) and len(interval) == 2 and 0 <= int(interval[0]) <= int(interval[1]) < t
    if not prediction.get("format_ok", False) or not valid_interval:
        return {"vIoU": 0.0, "sIoU": 0.0, "tIoU": 0.0, "format_ok": False, "spatial_support_frames": support_count, "temporal_union_sample_count": 0}

    boxes = prediction["boxes_cxcywh"].detach().float().cpu()
    positions = prediction.get("positions")
    if len(boxes) == t and positions == list(range(t)):
        # Full-frame outputs are already in canonical sampled index order.
        pred_boxes = boxes
    else:
        pred_boxes = _dense_positions_to_boxes(positions, boxes, t)
    geometry = prediction.get("geometry_valid")
    if geometry is not None and len(geometry) == len(boxes):
        # Invalid xyxy token rectangles are zero-boxes in PTD's parser.
        for pos, valid in zip(positions or [], geometry.tolist()):
            if not valid and 0 <= int(pos) < t:
                pred_boxes[int(pos)] = 0

    targets = []
    for box, valid in zip(label["boxes_xyxy"], support):
        if not valid:
            targets.append({"boxes": None})
        else:
            x1, y1, x2, y2 = [float(v) for v in box]
            targets.append({"boxes": torch.tensor([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]], dtype=torch.float32)})
    begin, end = int(label["event_interval"]["begin_fid"]), int(label["event_interval"]["end_fid"])
    metrics = compute_stvg_metrics(
        pred_boxes,
        targets,
        (int(interval[0]), int(interval[1])),
        (0, t - 1),
        frame_ids=frame_ids,
        gt_frame_interval=(begin, end),
    )
    return {
        "vIoU": float(metrics["vIoU"]),
        "sIoU": float(metrics["sIoU"]),
        "tIoU": float(metrics["tIoU"]),
        "format_ok": True,
        "spatial_support_frames": support_count,
        "temporal_union_sample_count": int(metrics["temporal_union_sample_count"]),
    }


def _frozen_prediction(row: dict[str, Any], barrier: dict[str, Any]) -> dict[str, Any]:
    from vg_tta.ptd_spatial_adapter_ab_v1 import boxes_from_tokens

    key = row["key"]
    path = PARENT / "source_features" / "clean" / f"{hashlib.sha256(key.encode()).hexdigest()}.pt"
    assert sha(path) == barrier["files"][str(path)]
    cache = torch.load(path, map_location="cpu", weights_only=False)
    assert cache["key"] == key and cache["frame_ids"] == row["input"]["frame_ids"]
    token_ids = cache["box_logits"].argmax(-1).long()
    boxes, valid = boxes_from_tokens(token_ids)
    return {
        "key": key,
        "source": str(row["source"]),
        "frame_ids": [int(x) for x in cache["frame_ids"]],
        "positions": [int(x) for x in cache["spatial_positions"]],
        "boxes_cxcywh": boxes.float().cpu(),
        "geometry_valid": valid.bool().cpu(),
        "interval": cache["interval"],
        "format_ok": bool(cache["format_ok"]),
    }


def _score_predictions(predictions: dict[str, dict[str, Any]], rows: list[dict[str, Any]], labels: dict[str, Any]) -> dict[str, Any]:
    result = []
    for row in rows:
        key = row["key"]
        metric = _tube_metric(predictions[key], labels[key])
        result.append({"key": key, "source": str(row["source"]), "metrics": metric})
    parents = sorted({x["source"] for x in result})
    parent_rows = []
    for parent in parents:
        subset = [x for x in result if x["source"] == parent]
        parent_rows.append({
            "source": parent,
            "queries": len(subset),
            **{m: sum(x["metrics"][m] for x in subset) / len(subset) for m in ("vIoU", "sIoU", "tIoU")},
        })
    parent_macro = {m: sum(x[m] for x in parent_rows) / len(parent_rows) for m in ("vIoU", "sIoU", "tIoU")}
    return {
        "queries": len(result),
        "parents": len(parents),
        "query_macro": {m: sum(x["metrics"][m] for x in result) / len(result) for m in ("vIoU", "sIoU", "tIoU")},
        "parent_macro": parent_macro,
        "parent_rows": parent_rows,
        "spatial_support_queries": sum(any(labels[x["key"]]["box_valid"][i] and labels[x["key"]]["event_active"][i] for i in range(len(labels[x["key"]]["frame_ids"]))) for x in result),
        "spatial_support_frames": sum(x["metrics"]["spatial_support_frames"] for x in result),
        "no_spatial_support_queries_included_with_zero_vIoU": sum(x["metrics"]["spatial_support_frames"] == 0 for x in result),
        "rows": result,
    }


def _binary_auc(labels: list[bool] | np.ndarray, scores: list[float] | np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.bool_).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    assert len(y) == len(s) and np.isfinite(s).all()
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y].sum())
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _score_aux_fields(predictions: dict[str, dict[str, Any]], rows: list[dict[str, Any]],
                      labels: dict[str, Any]) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    parent_event: dict[str, dict[str, list[Any]]] = {}
    parent_referent: dict[str, dict[str, list[Any]]] = {}
    for row in rows:
        key, parent = row["key"], str(row["source"])
        prediction, label = predictions[key], labels[key]
        fields = prediction.get("aux_field_logits")
        assert fields and fields.get("storage_dtype") == "float16_cpu", f"missing sealed reader fields for {key}"
        referent_logits = torch.as_tensor(fields["referent"], dtype=torch.float32)
        event_logits = torch.as_tensor(fields["event"], dtype=torch.float32)
        assert referent_logits.ndim == event_logits.ndim == 3
        assert tuple(referent_logits.shape) == tuple(event_logits.shape)
        t, h, w = referent_logits.shape
        assert t == len(label["frame_ids"]) == len(prediction["frame_ids"])
        targets = aux_targets(label, h, w, "cpu")
        ref_mask, event_mask = targets["referent_mask"][0], targets["event_mask"][0]
        ref_target, event_target = targets["referent_target"][0], targets["event_target"][0]
        ref_bce = None
        event_bce = None
        if bool(ref_mask.any()):
            ref_bce = float(F.binary_cross_entropy_with_logits(referent_logits[ref_mask], ref_target[ref_mask]))
        if bool(event_mask.any()):
            event_bce = float(F.binary_cross_entropy_with_logits(event_logits[event_mask], event_target[event_mask]))
        active = np.asarray(label["event_active"], dtype=np.bool_)
        event_frame_scores = event_logits.amax(dim=(-1, -2)).numpy().astype(np.float64).tolist()
        event_bucket = parent_event.setdefault(parent, {"labels": [], "scores": [], "queries": set()})
        event_bucket["labels"].extend(active.tolist())
        event_bucket["scores"].extend(event_frame_scores)
        event_bucket["queries"].add(key)
        box_valid = torch.tensor(label["box_valid"], dtype=torch.bool)
        if bool(box_valid.any()):
            support_labels = (ref_target[box_valid] > 0).reshape(-1).numpy().astype(np.bool_)
            support_scores = referent_logits[box_valid].reshape(-1).numpy().astype(np.float64).tolist()
            ref_bucket = parent_referent.setdefault(parent, {"labels": [], "scores": [], "queries": set()})
            ref_bucket["labels"].extend(support_labels.tolist())
            ref_bucket["scores"].extend(support_scores)
            ref_bucket["queries"].add(key)
        per_query.append({
            "key": key,
            "source": parent,
            "referent_masked_bce": ref_bce,
            "referent_supervised_box_valid_frames": int(ref_mask.any(dim=(-1, -2)).sum()),
            "event_masked_bce": event_bce,
            "event_supervised_frames": int(event_mask.any(dim=(-1, -2)).sum()),
            "event_active_missing_box_frames_masked": int((active & ~box_valid.numpy()).sum()),
        })

    def bce_summary(name: str) -> dict[str, Any]:
        parent_values = []
        parent_counts = []
        for parent in sorted({str(r["source"]) for r in rows}):
            values = [r[name] for r in per_query if r["source"] == parent and r[name] is not None]
            if values:
                parent_values.append(sum(values) / len(values))
                parent_counts.append({"source": parent, "queries": len(values), "mean_bce": sum(values) / len(values)})
        return {"parent_macro_query_mean": sum(parent_values) / len(parent_values) if parent_values else None,
                "eligible_parents": len(parent_values), "total_parents": len({str(r["source"]) for r in rows}),
                "parent_rows": parent_counts}

    event_rows = []
    for parent, values in sorted(parent_event.items()):
        auc = _binary_auc(values["labels"], values["scores"])
        event_rows.append({"source": parent, "queries": len(values["queries"]), "frames": len(values["labels"]),
                           "positive_frames": int(sum(values["labels"])), "negative_frames": len(values["labels"]) - int(sum(values["labels"])), "auc": auc})
    support_rows = []
    for parent, values in sorted(parent_referent.items()):
        auc = _binary_auc(values["labels"], values["scores"])
        support_rows.append({"source": parent, "queries": len(values["queries"]), "grid_cells": len(values["labels"]),
                             "positive_cells": int(sum(values["labels"])), "negative_cells": len(values["labels"]) - int(sum(values["labels"])), "auc": auc})
    event_valid = [r["auc"] for r in event_rows if r["auc"] is not None]
    support_valid = [r["auc"] for r in support_rows if r["auc"] is not None]
    return {
        "selection_use": "diagnostic_only; does not affect clean full-tube vIoU epoch selection",
        "referent_field": {"masked_soft_occupancy_bce": bce_summary("referent_masked_bce"),
                           "box_support_cell_auroc": {"parent_macro": sum(support_valid) / len(support_valid) if support_valid else None,
                                                       "eligible_parents": len(support_valid), "total_parents": len({str(r['source']) for r in rows}),
                                                       "target": "occupancy > 0 as spatial box-support; only box_valid frames; not object segmentation", "parent_rows": support_rows}},
        "event_field": {"masked_soft_occupancy_bce": bce_summary("event_masked_bce"),
                        "frame_max_logit_event_auroc": {"parent_macro": sum(event_valid) / len(event_valid) if event_valid else None,
                                                        "eligible_parents": len(event_valid), "total_parents": len({str(r['source']) for r in rows}),
                                                        "target": "event_active labels for every sampled frame; active missing-box does not affect event annotation", "parent_rows": event_rows}},
        "per_query": per_query,
    }


def _cpu_aux_field_metrics_check() -> dict[str, Any]:
    label = {
        "frame_ids": [10, 20, 30],
        "boxes_xyxy": [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        "box_valid": [True, True, False],
        "event_active": [False, True, True],
    }
    targets = aux_targets(label, 2, 2, "cpu")
    ref_target, event_target = targets["referent_target"][0], targets["event_target"][0]
    ref_logits = torch.where(ref_target > 0, torch.full_like(ref_target, 2.0), torch.full_like(ref_target, -2.0))
    event_logits = torch.where(event_target > 0, torch.full_like(event_target, 2.0), torch.full_like(event_target, -2.0))
    event_logits[2] = 4.0  # active with no box is unknown for BCE but remains an annotated active frame for AUROC.
    rows = [{"key": "q", "source": "p"}]
    predictions = {"q": {"frame_ids": label["frame_ids"], "aux_field_logits": {
        "referent": ref_logits.half(), "event": event_logits.half(), "storage_dtype": "float16_cpu"}}}
    summary = _score_aux_fields(predictions, rows, {"q": label})
    event_auc = summary["event_field"]["frame_max_logit_event_auroc"]["parent_macro"]
    support_auc = summary["referent_field"]["box_support_cell_auroc"]["parent_macro"]
    assert event_auc == 1.0 and support_auc == 1.0
    assert summary["per_query"][0]["event_supervised_frames"] == 2
    assert summary["per_query"][0]["event_active_missing_box_frames_masked"] == 1
    assert summary["event_field"]["masked_soft_occupancy_bce"]["eligible_parents"] == 1
    return {"status": "passed", "event_frame_max_logit_auc": event_auc,
            "referent_box_support_auc": support_auc,
            "active_missing_box_masked_from_bce_but_kept_for_event_auc": True,
            "referent_auc_is_box_support_not_segmentation": True}


def _cpu_invocation_signature_check() -> dict[str, Any]:
    import inspect

    generation = inspect.signature(generate_validation_predictions)
    fit_args = ("shared3d", 1, None, None, None, [], 0.0, 0.0)
    generation.bind(*fit_args, condition="clean", adapter_sha256="0" * 64)
    generation.bind(*fit_args, condition="noise_medium", adapter_sha256="1" * 64,
                    stage_cap=CORRUPT_STAGE_CAP, stage_name="source_validation_corruptions")
    inspect.signature(_score_epoch_predictions).bind(1, [], {a: "0" * 64 for a in ARCHITECTURES})
    inspect.signature(_validate_prediction_seal).bind("shared3d", 1, [], "clean", "0" * 64)
    return {"status": "passed", "fit_clean_generation_call_binds": True,
            "corruption_generation_call_binds": True, "scorer_requires_adapter_digests": True}


def _frozen_metric_agreement() -> dict[str, float]:
    reference_path = PARENT / "SOURCE_FROZEN_INDEPENDENT_METRICS.json"
    assert reference_path.is_file(), "independent native Frozen source readback is required before source-fit lock"
    reference = jread(reference_path)
    reference_rows = {r["key"]: r["metrics"] for r in reference["rows"] if r["condition"] == "clean"}
    rows = [r for r in _all_rows() if r["split"] == "validation"]
    labels = jread(LABELS)
    barrier = jread(FULL_BARRIER)
    assert len(reference_rows) == len(rows) == 198
    diffs = {"vIoU": 0.0, "sIoU": 0.0, "tIoU": 0.0}
    for row in rows:
        metric = _tube_metric(_frozen_prediction(row, barrier), labels[row["key"]])
        expected = reference_rows[row["key"]]
        for name in diffs:
            diffs[name] = max(diffs[name], abs(float(metric[name]) - float(expected[name])))
    assert all(value <= 1e-6 for value in diffs.values()), f"project metric differs from independent readback: {diffs}"
    return diffs


def _cpu_resume_failure_check() -> dict[str, Any]:
    """Simulate an arm-2 exception and verify resume from the prior safe group."""
    import copy

    arms = ("early_factorized", "shared3d", "dual3d")
    torch.manual_seed(SEED + 80)
    initial = {a: torch.nn.Linear(1, 1, bias=False) for a in arms}
    initial_opt = {a: torch.optim.Adam(initial[a].parameters(), lr=1e-2) for a in arms}
    safe_model = {a: copy.deepcopy(initial[a].state_dict()) for a in arms}
    safe_opt = {a: copy.deepcopy(initial_opt[a].state_dict()) for a in arms}

    def one_update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(torch.tensor([[1.0]])) - 1.0).square().sum()
        loss.backward()
        optimizer.step()

    # A is mutated in memory, then arm B raises. The implementation keeps the
    # disk snapshot unchanged and rehydrates all arms from it before replay.
    one_update(initial[arms[0]], initial_opt[arms[0]])
    simulated_failure = True
    assert simulated_failure
    resumed = {a: torch.nn.Linear(1, 1, bias=False) for a in arms}
    resumed_opt = {a: torch.optim.Adam(resumed[a].parameters(), lr=1e-2) for a in arms}
    for a in arms:
        resumed[a].load_state_dict(safe_model[a])
        resumed_opt[a].load_state_dict(safe_opt[a])
        one_update(resumed[a], resumed_opt[a])
    reference = {a: torch.nn.Linear(1, 1, bias=False) for a in arms}
    reference_opt = {a: torch.optim.Adam(reference[a].parameters(), lr=1e-2) for a in arms}
    for a in arms:
        reference[a].load_state_dict(safe_model[a])
        reference_opt[a].load_state_dict(safe_opt[a])
        one_update(reference[a], reference_opt[a])
        assert all(torch.equal(resumed[a].state_dict()[k], reference[a].state_dict()[k]) for k in safe_model[a])
    failure_rows = _failed_history_rows([{"architecture": arms[0], "source_index": 10}], 123, "train", 1, 10, [10, 11])
    assert failure_rows[0]["run_id"] == 123 and failure_rows[0]["committed_to_safe_checkpoint"] is False
    assert failure_rows[0]["replay_source_index_range_inclusive_exclusive"] == [10, 11]
    classified, committed_count = _classify_history_against_checkpoint(
        [{"architecture": arms[0], "epoch": 1, "source_index": 10, "key": "a"},
         {"architecture": arms[1], "epoch": 1, "source_index": 11, "key": "b"}],
        125,
        {"phase": "train", "epoch": 1, "next_source_index": 11,
         "history": [{"architecture": arms[0], "epoch": 1, "source_index": 10, "key": "a", "committed_to_safe_checkpoint": True}]},
        [11, 12],
    )
    assert committed_count == 1 and classified[0]["committed_to_safe_checkpoint"] is True
    assert classified[1]["committed_to_safe_checkpoint"] is False and classified[1]["replay_source_index_range_inclusive_exclusive"] == [11, 12]
    committed_rows = _committed_history_rows([{"architecture": a, "source_index": 10} for a in arms], 124, 11)
    assert all(r["committed_to_safe_checkpoint"] is True for r in committed_rows)
    return {"simulated_failure_after_first_arm": True, "safe_checkpoint_preserved": True, "resumed_triplet_matches_reference": True, "resumed_optimizer_steps_per_arm": 1, "failed_attempts_marked_uncommitted_and_replay_range": True, "checkpointed_pending_rows_reconciled_as_committed": True}


def _score_epoch_predictions(epoch: int, val_rows: list[dict[str, Any]], adapter_sha256_by_arch: dict[str, str]) -> dict[str, Any]:
    # Every arm's complete prediction seal is required before reading validation labels.
    assert set(adapter_sha256_by_arch) == set(ARCHITECTURES)
    seals = {arch: _validate_prediction_seal(arch, epoch, val_rows, "clean", adapter_sha256_by_arch[arch]) for arch in ARCHITECTURES}
    assert all(seal["GT_read"] is False for seal in seals.values())
    labels = jread(LABELS)
    predictions: dict[str, dict[str, dict[str, Any]]] = {arch: {} for arch in ARCHITECTURES}
    for arch in ARCHITECTURES:
        for row in val_rows:
            value = torch.load(_prediction_path(arch, epoch, row["key"]), map_location="cpu", weights_only=False)
            predictions[arch][row["key"]] = value
    scores = {arch: _score_predictions(predictions[arch], val_rows, labels) for arch in ARCHITECTURES}
    aux_scores = {arch: _score_aux_fields(predictions[arch], val_rows, labels) for arch in ARCHITECTURES}

    baseline_path = OUT / "FROZEN_SOURCE_VAL_BASELINE.json"
    if baseline_path.exists():
        frozen_summary = jread(baseline_path)["summary"]
    else:
        barrier = jread(FULL_BARRIER)
        frozen = {row["key"]: _frozen_prediction(row, barrier) for row in val_rows}
        frozen_summary = _score_predictions(frozen, val_rows, labels)
        jwrite_once(baseline_path, {
            "status": "native_frozen_source_validation_baseline",
            "summary": frozen_summary,
            "cache_barrier_sha256": sha(FULL_BARRIER),
            "labels_sha256": sha(LABELS),
            "GT_opened_after_all_three_arm_prediction_seals": True,
        })
    score_path = OUT / f"SOURCE_VAL_METRICS_E{epoch}.json"
    payload = {
        "status": "completed_source_validation_epoch_selection_readout",
        "epoch": epoch,
        "metric": "vg_tta.metrics.compute_stvg_metrics corrected sampled physical-time union",
        "full_tubes": "all T samples scored; targets only box_valid && event_active; half-open physical event interval",
        "val_label_sha256": sha(LABELS),
        "sealed_prediction_manifests": {arch: sha(_prediction_seal_path(arch, epoch)) for arch in ARCHITECTURES},
        "prediction_adapter_sha256": adapter_sha256_by_arch,
        "source_frozen_baseline": frozen_summary,
        "adapters": scores,
        "adapter_reader_fields": aux_scores,
        "selection": "highest parent_macro.vIoU separately for each architecture; ties choose earliest epoch",
    }
    _write_or_check(score_path, payload)
    return payload


def _init_fit_objects():
    from vg_tta.desta3d_v1 import Desta3DAdapter

    torch.manual_seed(SEED)
    adapters = {}
    optimizers = {}
    for arch, conf in ARCHITECTURES.items():
        adapter = Desta3DAdapter(2560, 2560, conf["hidden_dim"], arch).cuda().train()
        assert adapter.active_parameter_count() == conf["expected_active_parameters"]
        adapters[arch] = adapter
        optimizers[arch] = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.0)
    return adapters, optimizers


def _write_arm_epoch_checkpoint(arch: str, epoch: int, adapter, optimizer, directory: Path) -> Path:
    _assert_disk_reserve(f"epoch optimizer checkpoint {arch} E{epoch}")
    path = directory / f"{arch}_E{epoch}.pt"
    adapter_state = _state_to_cpu(adapter)
    adapter_sha = _state_dict_digest(adapter_state)
    identity = {"train_lock_sha256": sha(OUT / "TRAIN_LOCK.json"), "query_manifest_sha256": sha(_query_manifest()), "train_records_sha256": sha(OUT / "SOURCE_TRAIN_RECORDS.json")}
    if not path.exists():
        save_pt_once(path, {"architecture": arch, "epoch": epoch, "adapter_sha256": adapter_sha, "source_fit_identity": identity, "adapter": adapter_state, "optimizer": optimizer.state_dict()})
    else:
        prior = torch.load(path, map_location="cpu", weights_only=False)
        assert prior["architecture"] == arch and prior["epoch"] == epoch
        assert prior["adapter_sha256"] == adapter_sha and _state_dict_digest(prior["adapter"]) == adapter_sha
        assert prior["source_fit_identity"] == identity
    return path


def _write_best_checkpoint(arch: str, best: dict[str, Any]) -> None:
    path = OUT / "checkpoints" / f"{arch}_BEST.pt"
    adapter_state = best[arch]["adapter_state"]
    identity = {"train_lock_sha256": sha(OUT / "TRAIN_LOCK.json"), "query_manifest_sha256": sha(_query_manifest()), "train_records_sha256": sha(OUT / "SOURCE_TRAIN_RECORDS.json")}
    _save_progress(path, {"architecture": arch, "best_epoch": best[arch]["epoch"], "source_parent_macro_vIoU": best[arch]["vIoU"], "adapter_sha256": _state_dict_digest(adapter_state), "source_fit_identity": identity, "adapter": adapter_state})


def _check_real_ptd_zero_noop(arch: str, key: str, adapter, data, prompt, query, record, model, joint_loss, pixel_sha: str) -> None:
    """Compare stock PTD against the registered zero-init hook before step one."""
    path = OUT / "NOOP_CHECKS" / f"{arch}.json"
    if path.exists():
        report = jread(path)
        assert report["architecture"] == arch and report["source_key"] == key
        assert report["exact_merger_tokens_equal"] and report["source_ce_abs_diff"] <= 1e-6
        return

    assert bool(torch.count_nonzero(adapter.out_proj.weight).item() == 0)
    assert adapter.out_proj.bias is None or bool(torch.count_nonzero(adapter.out_proj.bias).item() == 0)
    stock = {"calls": 0, "tokens": None}

    def stock_hook(module, args, output):
        stock["calls"] += 1
        stock["tokens"] = output.detach().cpu().clone()

    handle = model.model.visual.merger.register_forward_hook(stock_hook)
    try:
        with torch.no_grad():
            stock_ce, stock_stats = joint_loss(model, data)
    finally:
        handle.remove()
    assert stock["calls"] == 1 and stock["tokens"] is not None

    with torch.no_grad():
        hooked_loss, hooked_stats, _, _, capture = _attach_adapter(
            adapter, data, prompt, query, record, model, joint_loss, exact_snapshot=True
        )
    assert capture["calls"] == 1 and capture["zero_noop"]
    exact_tokens = torch.equal(stock["tokens"], capture["input_tokens_cpu"]) and torch.equal(
        capture["input_tokens_cpu"], capture["updated_tokens_cpu"]
    )
    assert exact_tokens, f"{arch}: zero-init hook changed stock merger tokens"
    ce_abs_diff = abs(float(stock_ce.detach().float()) - float(hooked_stats["source_ce"]))
    assert ce_abs_diff <= 1e-6, f"{arch}: zero-init source CE differs from stock PTD by {ce_abs_diff}"
    jwrite_once(path, {
        "architecture": arch,
        "source_key": key,
        "input_pixels_sha256": pixel_sha,
        "source_ce_stock": float(stock_ce.detach().float()),
        "source_ce_zero_hook": float(hooked_stats["source_ce"]),
        "source_ce_abs_diff": ce_abs_diff,
        "ntp_count_stock": int(stock_stats["ntp_count"]),
        "mtp_count_stock": int(stock_stats["mtp_count"]),
        "exact_merger_tokens_equal": exact_tokens,
        "adapter_output_projection_zero_before_step": True,
        "target_GT_read": False,
    })
    assert float(hooked_loss.detach()) >= 0.0


def _train_one_arm_example(arch: str, epoch: int, source_index: int, key: str, adapter, optimizer,
                           model, processor, source_rows, train_records, history: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss

    row = source_rows[key]
    record = train_records[key]
    data, prompt, preprocess = _training_inputs(processor, model, row, record)
    q = _clean_query_tensor(key).to("cuda", dtype=torch.float32).reshape(1, 2560)
    if epoch == 1 and source_index == 0:
        _check_real_ptd_zero_noop(arch, key, adapter, data, prompt, q, record, model, joint_loss, preprocess["pixel_sha"])
    optimizer.zero_grad(set_to_none=True)
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    loss, fields, _, _, capture = _attach_adapter(adapter, data, prompt, q, record, model, joint_loss)
    assert torch.isfinite(loss), f"nonfinite source-fit loss for {arch}, {key}"
    loss.backward()
    assert all(p.grad is None for p in model.parameters()), "frozen 4B acquired gradients"
    group_grad = _grad_groups(adapter)
    if source_index in (0, 1):
        required = ["input_proj", "query_proj", "out_proj", "referent_head", "event_head"]
        required += ["factorized_reader"] if arch == "early_factorized" else ["stem", "shared_reader"] if arch == "shared3d" else ["stem", "short_reader", "long_reader"]
        if source_index == 1:
            required.append("film")
        for group in required:
            norm = float(group_grad.get(group, {}).get("grad_norm", 0.0))
            assert math.isfinite(norm) and norm > 0.0, f"{arch}.{group} lacks a finite nonzero source-fit gradient at source step {source_index}"
    grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
    assert math.isfinite(grad_norm), f"nonfinite adapter gradient norm for {arch}, {key}"
    optimizer.step()
    result = {
        "architecture": arch,
        "epoch": epoch,
        "source_index": source_index,
        "key": key,
        "source": str(row["source"]),
        "response_eligible": bool(record["response_eligible"]),
        "loss": float(loss.detach()),
        "gradient_norm_preclip": grad_norm,
        "adapter_gradient_groups": group_grad,
        "gradient_groups_required_nonzero_this_step": required if source_index in (0, 1) else [],
        "backbone_all_grad_none": True,
        "input_pixels_sha256": preprocess["pixel_sha"],
        "source_label_flags": record["raw_referent_flags_by_sample_position"],
        "zero_initialized_visual_noop": bool(capture["zero_noop"]),
        **fields,
    }
    history.append(result)
    del data, prompt, q, loss
    torch.cuda.empty_cache()
    return result


def _load_progress_or_new(progress_path: Path, adapters, optimizers):
    if progress_path.exists():
        state = _load_progress(progress_path, adapters, optimizers)
        torch.set_rng_state(state["torch_rng_state"])
        if state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        return state, True
    state = {
        "train_lock_sha256": sha(OUT / "TRAIN_LOCK.json"),
        "query_manifest_sha256": sha(_query_manifest()),
        "train_records_sha256": sha(OUT / "SOURCE_TRAIN_RECORDS.json"),
        "phase": "train",
        "epoch": 1,
        "next_source_index": 0,
        "adapter_states": {arch: _state_to_cpu(adapters[arch]) for arch in ARCHITECTURES},
        "optimizer_states": {arch: optimizers[arch].state_dict() for arch in ARCHITECTURES},
        "history": [],
        "best": {},
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    return state, False


def _load_adapters_from_state(adapters, state):
    for arch in ARCHITECTURES:
        adapters[arch].load_state_dict(state["adapter_states"][arch])


def _history_path() -> Path:
    return OUT / "TRAIN_STEP_HISTORY.jsonl"


def _append_history(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _committed_history_rows(rows: list[dict[str, Any]], run_id: int, safe_cursor: int) -> list[dict[str, Any]]:
    return [
        {**row, "run_id": run_id, "committed_to_safe_checkpoint": True,
         "safe_checkpoint_next_source_index": safe_cursor, "replayed_after_failure": False}
        for row in rows
    ]


def _failed_history_rows(rows: list[dict[str, Any]], run_id: int, safe_phase: str | None,
                         safe_epoch: int | None, safe_cursor: int | None,
                         replay_range: list[int] | None) -> list[dict[str, Any]]:
    return [
        {**row, "run_id": run_id, "committed_to_safe_checkpoint": False,
         "safe_checkpoint_phase": safe_phase, "safe_checkpoint_epoch": safe_epoch,
         "safe_checkpoint_next_source_index": safe_cursor,
         "replay_source_index_range_inclusive_exclusive": replay_range,
         "replayed_after_failure": True}
        for row in rows
    ]


def _classify_history_against_checkpoint(rows: list[dict[str, Any]], run_id: int,
                                         safe_state: dict[str, Any] | None,
                                         replay_range: list[int] | None) -> tuple[list[dict[str, Any]], int]:
    safe_state = safe_state or {}
    safe_history = {
        (r.get("architecture"), r.get("epoch"), r.get("source_index"), r.get("key"))
        for r in safe_state.get("history", [])
        if isinstance(r, dict) and r.get("committed_to_safe_checkpoint") is True
    }
    safe_cursor = safe_state.get("next_source_index")
    safe_phase, safe_epoch = safe_state.get("phase"), safe_state.get("epoch")
    classified = []
    committed_count = 0
    for row in rows:
        identity = (row.get("architecture"), row.get("epoch"), row.get("source_index"), row.get("key"))
        if identity in safe_history:
            classified.append({**row, "run_id": run_id, "committed_to_safe_checkpoint": True,
                               "safe_checkpoint_phase": safe_phase, "safe_checkpoint_epoch": safe_epoch,
                               "safe_checkpoint_next_source_index": safe_cursor,
                               "replay_source_index_range_inclusive_exclusive": None,
                               "replayed_after_failure": False})
            committed_count += 1
        else:
            classified.append({**row, "run_id": run_id, "committed_to_safe_checkpoint": False,
                               "safe_checkpoint_phase": safe_phase, "safe_checkpoint_epoch": safe_epoch,
                               "safe_checkpoint_next_source_index": safe_cursor,
                               "replay_source_index_range_inclusive_exclusive": replay_range,
                               "replayed_after_failure": True})
    return classified, committed_count


def fit() -> None:
    cfg = verify_lock()
    assert _query_manifest().is_file(), "capture exact clean prompt queries before fit"
    query_manifest = verify_query_manifest()
    assert not (OUT / "FIT_COMPLETE.json").exists(), "source-fit already completed"
    assert torch.cuda.is_available(), "source fit requires GPU"
    before, _ = spent_gpu_seconds()
    stage_before = _fit_receipt_seconds()
    assert before < CAP_SECONDS and stage_before < FIT_STAGE_CAP

    from scripts.run_final_simplification_v1 import lease
    from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load
    from vg_tta.desta3d_v1 import Desta3DAdapter

    run_id = time.time_ns()
    run_started = time.monotonic()
    OUT.joinpath("receipts").mkdir(parents=True, exist_ok=True)
    progress_path = OUT / "FIT_PROGRESS.pt"
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    guard = lease()
    peak = 0
    steps_this_run = 0
    committed_steps_this_run = 0
    status = "running"
    error = None
    current_key = None
    completed_arms_in_current_group: list[str] = []
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    epoch, next_idx, phase = 1, 0, "initialize"
    local_history: list[dict[str, Any]] = []
    try:
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        source_rows, train_records = source_rows_and_records()
        train_keys = sorted(train_records)
        val_rows = [r for r in _all_rows() if r["split"] == "validation"]
        orders = _training_order(train_records)
        order_path = OUT / "TRAIN_ORDER.json"
        order_payload = {"seed": SEED, "epoch_orders": orders, "architecture_order_per_source": list(ARCHITECTURES), "equal_arm_progress": True}
        _write_or_check(order_path, order_payload)
        processor = processor_load()
        model = model_load()
        model.requires_grad_(False)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        assert all(not p.requires_grad for p in model.parameters())
        frozen_versions = {name: p._version for name, p in model.named_parameters()}
        adapters, optimizers = _init_fit_objects()
        state, resumed = _load_progress_or_new(progress_path, adapters, optimizers)
        if resumed:
            _load_adapters_from_state(adapters, state)
        else:
            # This is the first safe checkpoint. An exception in any arm of the
            # first three-way group must never serialize only a subset of steps.
            _save_progress(progress_path, state)
        history = list(state["history"])
        best = dict(state["best"])
        epoch = int(state["epoch"])
        next_idx = int(state["next_source_index"])
        phase = state["phase"]
        while epoch <= 2:
            if phase == "train":
                order = orders[epoch - 1]
                assert len(order) == len(train_keys) == 618
                while next_idx < len(order):
                    # A complete source query is applied to all three arms before
                    # the cursor advances, so a safe pause cannot favor one arm.
                    _assert_disk_reserve(f"fit triplet epoch {epoch} source {next_idx + 1}/618")
                    _budget_guard(run_started, before, margin=60.0)
                    key = order[next_idx]
                    current_key = key
                    completed_arms_in_current_group = []
                    for arch, adapter in adapters.items():
                        adapter.train()
                        optimizers[arch].zero_grad(set_to_none=True)
                        example = _train_one_arm_example(arch, epoch, next_idx, key, adapter, optimizers[arch], model, processor, source_rows, train_records, local_history)
                        steps_this_run += 1
                        peak = max(peak, torch.cuda.max_memory_allocated())
                        if not example["backbone_all_grad_none"]:
                            raise AssertionError("backbone gradient gate failed")
                        completed_arms_in_current_group.append(arch)
                    next_idx += 1
                    current_key = None
                    completed_arms_in_current_group = []
                    if next_idx % 25 == 0 or next_idx == len(order):
                        committed_rows = _committed_history_rows(local_history, run_id, next_idx)
                        history.extend(committed_rows)
                        _save_progress(progress_path, _checkpoint_payload("train", epoch, next_idx, adapters, optimizers, history, best))
                        _append_history(committed_rows, _history_path())
                        committed_steps_this_run += len(committed_rows)
                        local_history.clear()
                    if next_idx % 50 == 0:
                        print("SOURCE_FIT_PROGRESS", epoch, next_idx, len(order), flush=True)
                phase, next_idx = "validate", 0
                _save_progress(progress_path, _checkpoint_payload(phase, epoch, next_idx, adapters, optimizers, history, best))

            if phase == "validate":
                model.eval()
                for arch, adapter in adapters.items():
                    _budget_guard(run_started, before, margin=60.0)
                    adapter.eval()
                    adapter_hash = _adapter_digest(adapter)
                    generate_validation_predictions(arch, epoch, model, processor, adapter, val_rows, run_started, before, condition="clean", adapter_sha256=adapter_hash)
                    _budget_guard(run_started, before, margin=30.0)
                # Source validation labels are opened only after all three full
                # prediction filesets have been sealed.
                score_payload = _score_epoch_predictions(epoch, val_rows, {arch: _adapter_digest(adapter) for arch, adapter in adapters.items()})
                for arch in ARCHITECTURES:
                    score = float(score_payload["adapters"][arch]["parent_macro"]["vIoU"])
                    if arch not in best or score > float(best[arch]["vIoU"]):
                        best[arch] = {"epoch": epoch, "vIoU": score, "adapter_state": _state_to_cpu(adapters[arch])}
                    _write_best_checkpoint(arch, best)
                    _write_arm_epoch_checkpoint(arch, epoch, adapters[arch], optimizers[arch], checkpoint_dir)
                history.append({"kind": "source_val_selection", "epoch": epoch, "scores": {a: score_payload["adapters"][a]["parent_macro"]["vIoU"] for a in ARCHITECTURES}, "frozen_vIoU": score_payload["source_frozen_baseline"]["parent_macro"]["vIoU"]})
                if epoch == 2:
                    phase, epoch, next_idx = "complete", 3, 0
                    _save_progress(progress_path, _checkpoint_payload(phase, epoch, next_idx, adapters, optimizers, history, best))
                    break
                epoch, phase, next_idx = epoch + 1, "train", 0
                model.train()
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.eval()
                _save_progress(progress_path, _checkpoint_payload(phase, epoch, next_idx, adapters, optimizers, history, best))
                continue

        if phase == "complete":
            summary = {arch: {"best_epoch": best[arch]["epoch"], "source_parent_macro_vIoU": best[arch]["vIoU"]} for arch in ARCHITECTURES}
            jwrite_once(OUT / "FIT_COMPLETE.json", {
                "status": "completed_source_only_fit_and_clean_validation_selection",
                "architectures": summary,
                "frozen_backbone": True,
                "stock_ptd_decoder_and_grammar": True,
                "target_GT_read": False,
                "TTA_or_deployment_claim": False,
                "best_checkpoint_hashes": {arch: sha(OUT / "checkpoints" / f"{arch}_BEST.pt") for arch in ARCHITECTURES},
                "best_adapter_sha256": {arch: _state_dict_digest(best[arch]["adapter_state"]) for arch in ARCHITECTURES},
                "query_cache_sha256": sha(_query_manifest()),
                "train_lock_sha256": sha(OUT / "TRAIN_LOCK.json"),
                "train_records_sha256": sha(OUT / "SOURCE_TRAIN_RECORDS.json"),
                "step_history_contract": "attempt logs include run_id and committed_to_safe_checkpoint; only true rows count as committed science steps; failed uncommitted rows mark the source-index replay range",
            })
            status = "completed"
        else:
            status = "paused_at_safe_equal_arm_boundary"
        assert all(p._version == frozen_versions[name] for name, p in model.named_parameters()), "frozen PTD weights changed"
        peak = max(peak, torch.cuda.max_memory_allocated())
    except BudgetPause as exc:
        error = str(exc)
        status = "paused_budget_equal_arm_resume_state_saved"
        if "adapters" in locals() and "optimizers" in locals():
            committed_rows = _committed_history_rows(local_history, run_id, next_idx)
            history.extend(committed_rows)
            _save_progress(progress_path, _checkpoint_payload(phase, epoch, next_idx, adapters, optimizers, history, best))
            _append_history(committed_rows, _history_path())
            committed_steps_this_run += len(committed_rows)
            local_history.clear()
        jwrite_once(OUT / f"FIT_PAUSE_{run_id}.json", {"status": status, "reason": error, "phase": phase, "epoch": epoch, "next_source_index": next_idx, "all_arms_have_same_progress": True, "GT_read_in_this_partial_epoch": False})
    except BaseException as exc:
        error = repr(exc)
        status = "failed_partial_state_preserved"
        # Keep the last complete three-arm group checkpoint untouched. Any
        # incomplete group is explicitly uncommitted and is replayed from that
        # checkpoint on resume; elapsed GPU time remains in this receipt.
        safe_cursor = None
        safe_phase = None
        safe_epoch = None
        safe_state = None
        if progress_path.exists():
            safe_state = torch.load(progress_path, map_location="cpu", weights_only=False)
            safe_cursor = int(safe_state.get("next_source_index", 0))
            safe_phase = safe_state.get("phase")
            safe_epoch = int(safe_state.get("epoch", 0))
        replay_end = (int(next_idx) + int(current_key is not None)) if "next_idx" in locals() else safe_cursor
        replay_range = [safe_cursor, replay_end] if safe_cursor is not None and replay_end is not None and replay_end > safe_cursor else None
        failure_rows, checkpointed_pending_steps = _classify_history_against_checkpoint(local_history, run_id, safe_state, replay_range)
        committed_steps_this_run += checkpointed_pending_steps
        _append_history(failure_rows, _history_path())
        local_history.clear()
        has_safe_checkpoint = safe_state is not None
        jwrite_once(OUT / f"FIT_FAILURE_{run_id}.json", {"run_id": run_id, "status": status, "error": error, "traceback": traceback.format_exc(), "phase": phase if "phase" in locals() else None, "epoch": epoch if "epoch" in locals() else None, "in_memory_next_source_index": next_idx if "next_idx" in locals() else None, "safe_checkpoint_phase": safe_phase, "safe_checkpoint_epoch": safe_epoch, "safe_checkpoint_next_source_index": safe_cursor, "replay_source_index_range_inclusive_exclusive": replay_range, "uncommitted_source_key": current_key, "arms_updated_in_uncommitted_group": completed_arms_in_current_group, "uncommitted_history_rows": sum(not row["committed_to_safe_checkpoint"] for row in failure_rows), "safe_checkpoint_preserved": has_safe_checkpoint, "resume_replays_uncommitted_group": has_safe_checkpoint and replay_range is not None, "elapsed_GPU_seconds_counted_in_receipt": True})
        raise
    finally:
        elapsed = time.monotonic() - run_started
        assert not local_history, "unclassified source-fit history escaped success/pause/failure handler"
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_allocated())
        jwrite_once(OUT / "receipts" / f"fit_{run_id}.json", {"stage": "source_three_arm_fit_and_clean_validation", "seconds": elapsed, "prior_seconds": before, "prior_fit_stage_seconds": stage_before, "adapter_updates_completed_in_memory_this_run": steps_this_run, "adapter_updates_committed_to_safe_checkpoint_this_run": committed_steps_this_run, "partial_or_replayed_updates_still_counted_in_GPU_seconds": True, "run_id": run_id, "peak_bytes": peak, "status": status, "error": error, "cumulative_cap_seconds": CAP_SECONDS, "stage_cap_seconds": FIT_STAGE_CAP})
        guard.close()
    print(json.dumps({"status": status, "steps_completed_this_run": steps_this_run, "seconds": time.monotonic() - run_started, "fit_stage_seconds_before": stage_before}, indent=2))


def eval_corrupt() -> None:
    """Secondary source-val readout on the 31 registered noise/blur pairs."""
    verify_lock()
    assert (OUT / "FIT_COMPLETE.json").is_file(), "complete clean source-fit selection before corruption readout"
    assert _query_manifest().is_file()
    assert torch.cuda.is_available(), "corruption readout requires GPU"
    before, _ = spent_gpu_seconds()
    assert before < CAP_SECONDS
    stage_before = _stage_receipt_seconds("source_validation_corruptions")
    assert stage_before < CORRUPT_STAGE_CAP
    run_started = time.monotonic()
    run_id = time.time_ns()
    status, error, peak = "running", None, 0
    guard = None
    try:
        from scripts.run_final_simplification_v1 import lease
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load, model_load
        from vg_tta.desta3d_v1 import Desta3DAdapter

        initial_rows = jread(PARENT / "SOURCE_INITIAL_INPUTS.json")
        rows = [r for r in initial_rows if r["split"] == "validation"]
        assert len(rows) == 31 and len({r["source"] for r in rows}) == 31
        rows.sort(key=lambda r: r["key"])
        guard = lease()
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        processor, model = processor_load(), model_load()
        model.eval().requires_grad_(False)
        model.config.use_cache = False
        fit_complete = jread(OUT / "FIT_COMPLETE.json")
        result_arch_epochs = {}
        best_states = {}
        for arch, conf in ARCHITECTURES.items():
            best = torch.load(OUT / "checkpoints" / f"{arch}_BEST.pt", map_location="cpu", weights_only=False)
            result_arch_epochs[arch] = int(best["best_epoch"])
            assert best["architecture"] == arch
            assert fit_complete["architectures"][arch]["best_epoch"] == result_arch_epochs[arch]
            assert fit_complete["best_checkpoint_hashes"][arch] == sha(OUT / "checkpoints" / f"{arch}_BEST.pt")
            assert best["source_fit_identity"] == {"train_lock_sha256": sha(OUT / "TRAIN_LOCK.json"), "query_manifest_sha256": sha(_query_manifest()), "train_records_sha256": sha(OUT / "SOURCE_TRAIN_RECORDS.json")}
            adapter = Desta3DAdapter(2560, 2560, conf["hidden_dim"], arch).cuda().eval()
            adapter.load_state_dict(best["adapter"])
            digest = _adapter_digest(adapter)
            assert best["adapter_sha256"] == digest, f"best adapter digest mismatch: {arch}"
            best_states[arch] = (adapter, digest, best["best_epoch"])

        for arch in ARCHITECTURES:
            adapter, digest, epoch = best_states[arch]
            for condition in ("noise_medium", "defocus_extreme"):
                _budget_guard(run_started, before, margin=60.0, stage_cap=CORRUPT_STAGE_CAP, stage_name="source_validation_corruptions")
                generate_validation_predictions(arch, epoch, model, processor, adapter, rows, run_started, before,
                                                condition=condition, adapter_sha256=digest, stage_cap=CORRUPT_STAGE_CAP,
                                                stage_name="source_validation_corruptions")
                peak = max(peak, torch.cuda.max_memory_allocated())

        # The readback occurs only after each condition x arm has its own seal.
        for arch in ARCHITECTURES:
            adapter, digest, epoch = best_states[arch]
            for condition in ("noise_medium", "defocus_extreme"):
                _validate_prediction_seal(arch, epoch, rows, condition, digest)
        labels = jread(LABELS)
        independent = jread(PARENT / "SOURCE_FROZEN_INDEPENDENT_METRICS.json")
        independent_rows = {(r["key"], r["condition"]): r["metrics"] for r in independent["rows"]}
        same31_keys = {r["key"] for r in rows}
        summaries = {}
        for arch in ARCHITECTURES:
            epoch = result_arch_epochs[arch]
            summaries[arch] = {}
            clean_metrics = jread(OUT / f"SOURCE_VAL_METRICS_E{epoch}.json")["adapters"][arch]["rows"]
            clean_rows = [r for r in clean_metrics if r["key"] in same31_keys]
            assert len(clean_rows) == len(rows)
            clean_parent_rows = []
            for parent in sorted({str(r["source"]) for r in clean_rows}):
                subset = [r["metrics"] for r in clean_rows if str(r["source"]) == parent]
                clean_parent_rows.append({"source": parent, **{m: sum(x[m] for x in subset) / len(subset) for m in ("vIoU", "sIoU", "tIoU")}})
            clean_same31 = {"queries": len(clean_rows), "parents": len(clean_parent_rows),
                            "parent_macro": {m: sum(x[m] for x in clean_parent_rows) / len(clean_parent_rows) for m in ("vIoU", "sIoU", "tIoU")},
                            "parent_rows": clean_parent_rows}
            for condition in ("noise_medium", "defocus_extreme"):
                preds = {r["key"]: torch.load(_prediction_path(arch, epoch, r["key"], condition), map_location="cpu", weights_only=False) for r in rows}
                summary = _score_predictions(preds, rows, labels)
                aux_summary = _score_aux_fields(preds, rows, labels)
                frozen_refs = [independent_rows[(r["key"], condition)] for r in rows]
                frozen_summary = {
                    "queries": len(rows),
                    "parents": 31,
                    "parent_macro": {m: sum(x[m] for x in frozen_refs) / len(frozen_refs) for m in ("vIoU", "sIoU", "tIoU")},
                }
                summaries[arch][condition] = {
                    "adapted": summary,
                    "adapter_reader_fields_diagnostic_only": aux_summary,
                    "frozen_reference_same31": frozen_summary,
                    "clean_adapter_same31": clean_same31,
                    "delta_parent_macro_vIoU_vs_frozen": summary["parent_macro"]["vIoU"] - frozen_summary["parent_macro"]["vIoU"],
                    "delta_parent_macro_vIoU_vs_clean_adapter": summary["parent_macro"]["vIoU"] - clean_same31["parent_macro"]["vIoU"],
                }
        payload = {
            "status": "completed_secondary_source_validation_corruption_readout",
            "scope": "31 source validation parents only; not independent target evidence or checkpoint selection",
            "source_labels_opened_after_all_six_arm_condition_prediction_seals": True,
            "conditions": ["noise_medium", "defocus_extreme"],
            "query_context": "separate frozen exact prompt-prefill hidden mean from the corresponding corrupted input per query and condition",
            "adapter_best_epochs": result_arch_epochs,
            "metric": "same corrected sampled physical-time vIoU and parent macro as clean validation; compare only common 31 parents",
            "summaries": summaries,
            "target_test_GT_read": False,
            "train_lock_sha256": sha(OUT / "TRAIN_LOCK.json"),
        }
        _write_or_check(OUT / "SOURCE_VAL_CORRUPTION_METRICS.json", payload)
        status = "completed"
    except BudgetPause as exc:
        status, error = "paused_at_equal_condition_arm_boundary", str(exc)
        jwrite_once(OUT / f"CORRUPT_PAUSE_{run_id}.json", {"status": status, "reason": error, "target_test_GT_read": False})
    except BaseException as exc:
        status, error = "failed_partial_predictions_preserved", repr(exc)
        jwrite_once(OUT / f"CORRUPT_FAILURE_{run_id}.json", {"status": status, "error": error, "traceback": traceback.format_exc(), "target_test_GT_read": False})
        raise
    finally:
        elapsed = time.monotonic() - run_started
        OUT.joinpath("receipts").mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_allocated())
        jwrite_once(OUT / "receipts" / f"corrupt_{run_id}.json", {"stage": "source_validation_corruptions", "seconds": elapsed, "prior_seconds": before, "prior_corruption_stage_seconds": stage_before, "peak_bytes": peak, "status": status, "error": error, "cumulative_cap_seconds": CAP_SECONDS, "stage_cap_seconds": CORRUPT_STAGE_CAP})
        if guard is not None:
            guard.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["propose", "cpu-check", "register", "capture-queries", "fit", "eval-corrupt"])
    args = parser.parse_args()
    actions = {
        "propose": propose,
        "cpu-check": cpu_check,
        "register": register,
        "capture-queries": capture_queries,
        "fit": fit,
        "eval-corrupt": eval_corrupt,
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
