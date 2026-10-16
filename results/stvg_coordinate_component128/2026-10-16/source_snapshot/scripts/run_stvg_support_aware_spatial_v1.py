"""Finite eight-case source-preserving Direct vs conservative spatial OPD worker.

Importing this module is inert.  The root-controlled single-GPU launcher must
provide a locked runtime with hashes for every input/source dependency.  This
worker reads no ground-truth labels and never changes the frozen source model.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_support_aware_temporal_spatial_update_v1"
OLD_BASE = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stvg_temporal_caption_query_v1 import (  # noqa: E402
    catalog, inference_guard, read as old_read, roster as old_roster,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _require_pin(runtime, relpath: str, path: Path):
    pins = runtime.get("pins")
    if not isinstance(pins, dict) or pins.get(relpath) != _sha(path):
        raise RuntimeError(f"locked runtime does not bind required input: {relpath}")


def _file_index(barrier):
    rows = barrier.get("files")
    if not isinstance(rows, list):
        raise ValueError("sealed barrier has no files list")
    index = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("malformed sealed file row")
        if row["path"] in index:
            raise ValueError("duplicate path in sealed file inventory")
        index[row["path"]] = row
    return index


def _verify_file_inventory(barrier, *, expected_count: int | None = None):
    index = _file_index(barrier)
    if expected_count is not None and len(index) != expected_count:
        raise ValueError("sealed file inventory count differs")
    for relpath, row in index.items():
        path = ROOT / relpath
        if not path.is_file() or _sha(path) != row.get("sha256"):
            raise RuntimeError(f"sealed input file hash mismatch: {relpath}")
        if "bytes" in row and path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"sealed input file byte count mismatch: {relpath}")
    return index


def _xyxy(boxes):
    import numpy as np

    values = boxes.detach().cpu().double().numpy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("expected normalized cxcywh box matrix")
    return np.concatenate((values[:, :2] - values[:, 2:] / 2,
                           values[:, :2] + values[:, 2:] / 2), axis=1)


def _tensor_sha(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    payload = (str(value.dtype) + "|" + repr(tuple(value.shape)) + "|").encode("ascii")
    return hashlib.sha256(payload + value.numpy().tobytes()).hexdigest()


def _same_logits(left, right):
    import torch

    return len(left) == len(right) and all(torch.equal(a.detach().cpu(), b.detach().cpu())
                                            for a, b in zip(left, right))


def _direct_step_readback(core, base, initial, frame_ids, key, fit, torch):
    """Materialize exact cached boxes for every saved Direct optimizer step."""
    replay, _ = core._trainable_replay(base, frame_ids, key, "joint")
    trace = []
    try:
        if len(fit.get("path", [])) != int(fit.get("declared_steps", -1)):
            raise RuntimeError("Direct fit did not retain every configured step")
        for entry in fit["path"]:
            if "state" not in entry or "boxes" not in entry:
                raise RuntimeError("Direct step omitted its pre-update state or full decoder boxes")
            before_state = {name: entry["state"][name].to(initial[name]) for name in initial}
            replay.restore(before_state)
            with torch.no_grad():
                before_boxes = replay.values()["boxes"].detach().cpu().clone()
            if not torch.equal(before_boxes, entry["boxes"].detach().cpu()):
                raise RuntimeError("Direct pre-step decoder boxes do not reproduce from saved state")
            after_saved = entry.get("update_state", entry["state"])
            after_state = {name: after_saved[name].to(initial[name]) for name in initial}
            replay.restore(after_state)
            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().cpu().clone()
            trace.append({
                "step": int(entry["step"]),
                "state_before": {k: v.detach().cpu().clone() for k, v in before_state.items()},
                "state_after": {k: v.detach().cpu().clone() for k, v in after_state.items()},
                "optimizer_state_before": core._copy_cpu(entry.get("optimizer_before", entry.get("optimizer_state"))),
                "optimizer_state_after": core._copy_cpu(entry.get("optimizer_state")),
                "boxes_before": before_boxes,
                "boxes_after": after_boxes,
                "no_update": bool(entry.get("skipped", False)),
            })
        replay.restore({name: value.to(initial[name]) for name, value in fit["final_state"].items()})
        with torch.no_grad():
            final_boxes = replay.values()["boxes"].detach().cpu()
        if not torch.equal(final_boxes, fit["final_cached_decoder_boxes"].detach().cpu()):
            raise RuntimeError("Direct final replay differs from its saved cached decoder")
        if not trace or not torch.equal(trace[-1]["boxes_after"], final_boxes):
            raise RuntimeError("Direct final step boxes differ from the saved fit result")
        return trace
    finally:
        replay.restore({name: value.to(initial[name]) for name, value in initial.items()})
        base.restore(initial)


def _anchors_for_case(record, case, reference, frame_ids, torch):
    if reference.get("case_id") != record["case_id"] or reference.get("source_case_sha256") != record["source_case_sha256"]:
        raise ValueError("spatial reference is not bound to this locked source case")
    response_rel = reference.get("actual_response_path")
    response_sha = reference.get("actual_response_sha256")
    if not isinstance(response_rel, str) or not isinstance(response_sha, str):
        raise ValueError("spatial reference omitted its sealed response receipt")
    response_path = ROOT / response_rel
    if not response_path.is_file() or _sha(response_path) != response_sha:
        raise RuntimeError("spatial reference receipt changed after lock")
    candidate = reference.get("candidate")
    positions = list(reference.get("observed_positions", []))
    if candidate is None:
        if positions:
            raise ValueError("reference without a candidate cannot carry observed positions")
        return []
    boxes = candidate.get("boxes_xyxy_normalized")
    if not isinstance(boxes, list) or len(boxes) != len(frame_ids):
        raise ValueError("selected spatial candidate does not cover the locked source frame grid")
    if not positions or len(positions) > 4 or len(set(positions)) != len(positions):
        raise ValueError("spatial reference must provide one to four distinct verified positions")
    anchors = []
    for position in positions:
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(frame_ids):
            raise ValueError("invalid sparse observation position")
        row = boxes[position]
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError("selected candidate box has the wrong shape")
        x1, y1, x2, y2 = map(float, row)
        if not all(map(lambda x: __import__("math").isfinite(x), (x1, y1, x2, y2))):
            raise ValueError("selected candidate box is not finite")
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("selected candidate box is outside normalized xyxy bounds")
        anchors.append({
            "position": position,
            "physical_frame_id": int(frame_ids[position]),
            "box": torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2,
                                 x2 - x1, y2 - y1], dtype=torch.float32),
            "weight": 1.0,
        })
    return anchors


def _validate_runtime():
    runtime_path = BASE / "STUDENT_RUNTIME.json"
    if not runtime_path.is_file():
        raise RuntimeError("root-locked STUDENT_RUNTIME.json is required")
    runtime = _json(runtime_path)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("student runtime is not locked before GT with GT_read=false")
    if runtime.get("scope") != "joint" or runtime.get("steps") != 5:
        raise RuntimeError("runtime must lock the 1,792-parameter joint scope and five steps")
    if runtime.get("source_frame_rms_limit") != 0.01 or runtime.get("backtracking_halvings") != 20:
        raise RuntimeError("runtime does not match the fixed conservative OPD constraint")
    if not isinstance(runtime.get("pins"), dict) or not runtime["pins"]:
        raise RuntimeError("runtime must pin code and sealed input files")
    required = {
        "scripts/run_stvg_support_aware_spatial_v1.py": ROOT / "scripts/run_stvg_support_aware_spatial_v1.py",
        "scripts/stvg_source_anchored_spatial_v1.py": ROOT / "scripts/stvg_source_anchored_spatial_v1.py",
        "scripts/stvg_temporal_caption_query_v1.py": ROOT / "scripts/stvg_temporal_caption_query_v1.py",
        "scripts/run_tastvg_evidence_vulnerability_v2.py": ROOT / "scripts/run_tastvg_evidence_vulnerability_v2.py",
        "scripts/run_decota_paper_main_v1.py": ROOT / "scripts/run_decota_paper_main_v1.py",
        "scripts/stvg_mllm_feedback_common_v1.py": ROOT / "scripts/stvg_mllm_feedback_common_v1.py",
        "scripts/stvg_target_teacher_student_v1.py": ROOT / "scripts/stvg_target_teacher_student_v1.py",
        "vg_tta/decota_spatial_opd_tunable_v1.py": ROOT / "vg_tta/decota_spatial_opd_tunable_v1.py",
        "vg_tta/stvg_opd_frozen_mask_cache012.py": ROOT / "vg_tta/stvg_opd_frozen_mask_cache012.py",
        "vg_tta/stvg_opd_frozen_visual_host011.py": ROOT / "vg_tta/stvg_opd_frozen_visual_host011.py",
        "vg_tta/stvg_opd_frozen_attention_lifetime010.py": ROOT / "vg_tta/stvg_opd_frozen_attention_lifetime010.py",
        "vg_tta/stvg_opd_frozen_module_cpu008.py": ROOT / "vg_tta/stvg_opd_frozen_module_cpu008.py",
        "methods/decota_final_simplified_v1/objectives.py": ROOT / "methods/decota_final_simplified_v1/objectives.py",
        "methods/decota_final_simplified_v1/tensors.py": ROOT / "methods/decota_final_simplified_v1/tensors.py",
        "methods/decota_final_simplified_v1/backbone.py": ROOT / "methods/decota_final_simplified_v1/backbone.py",
        "methods/decota_spatial_opd_v1/configs.json": ROOT / "methods/decota_spatial_opd_v1/configs.json",
        "artifacts/stvg_temporal_caption_query_spatial_opd_v1/ROSTER_LOCK.json": OLD_BASE / "ROSTER_LOCK.json",
        "artifacts/stvg_temporal_caption_query_spatial_opd_v1/RUNTIME.json": OLD_BASE / "RUNTIME.json",
        "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_RUNTIME.json": OLD_BASE / "STUDENT_RUNTIME.json",
        "artifacts/stvg_temporal_caption_query_spatial_opd_v1/TEACHER_GLOBAL_BARRIER.json": OLD_BASE / "TEACHER_GLOBAL_BARRIER.json",
        "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json": OLD_BASE / "STUDENT_GLOBAL_BARRIER.json",
        "artifacts/stvg_temporal_caption_query_spatial_opd_v1/SPATIAL_REFERENCE_LOCK.json": OLD_BASE / "SPATIAL_REFERENCE_LOCK.json",
    }
    for relpath, path in required.items():
        _require_pin(runtime, relpath, path)
    return runtime_path, runtime


def main():
    import collections
    import gc
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_decota_paper_main_v1 import model_for, frames_for
    from scripts.stvg_mllm_feedback_common_v1 import raw_row
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward, transient_native_masks
    from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
    from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
    from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts.stvg_source_anchored_spatial_v1 import fit_source_anchored_opd

    runtime_path, runtime = _validate_runtime()
    install_clean_loader()
    sys.addaudithook(inference_guard)
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    old_roster_path = OLD_BASE / "ROSTER_LOCK.json"
    teacher_barrier_path = OLD_BASE / "TEACHER_GLOBAL_BARRIER.json"
    old_student_barrier_path = OLD_BASE / "STUDENT_GLOBAL_BARRIER.json"
    reference_lock_path = OLD_BASE / "SPATIAL_REFERENCE_LOCK.json"
    old_roster_lock = _json(old_roster_path)
    teacher_barrier = _json(teacher_barrier_path)
    old_student_barrier = _json(old_student_barrier_path)
    reference_lock = _json(reference_lock_path)
    if old_roster_lock.get("status") != "locked" or old_roster_lock.get("count") != 8 or old_roster_lock.get("GT_read", False) is not False:
        raise RuntimeError("previous eight-case roster is not a valid pre-GT lock")
    if teacher_barrier.get("status") != "complete" or teacher_barrier.get("count") != 64 or teacher_barrier.get("GT_read") is not False:
        raise RuntimeError("previous teacher request barrier is incomplete or invalid")
    if old_student_barrier.get("status") != "complete" or old_student_barrier.get("count") != 8 or old_student_barrier.get("GT_read") is not False:
        raise RuntimeError("previous student outputs are not sealed as an eight-case pre-GT stage")
    if reference_lock.get("status") != "locked" or reference_lock.get("count") != 8 or reference_lock.get("GT_read") is not False:
        raise RuntimeError("the prior locked spatial reference catalog is incomplete")
    if teacher_barrier.get("runtime_sha256") != _sha(OLD_BASE / "RUNTIME.json"):
        raise RuntimeError("previous teacher barrier does not bind its runtime")
    if old_student_barrier.get("student_runtime_sha256") != _sha(OLD_BASE / "STUDENT_RUNTIME.json"):
        raise RuntimeError("previous student barrier does not bind its runtime")
    if old_student_barrier.get("teacher_barrier_sha256") != _sha(teacher_barrier_path):
        raise RuntimeError("previous student barrier does not bind the sealed teacher stage")
    teacher_files = _verify_file_inventory(teacher_barrier, expected_count=64)
    old_student_files = _verify_file_inventory(old_student_barrier, expected_count=8)

    locked_roster = old_roster_lock.get("records")
    records = old_roster()
    if not isinstance(locked_roster, list) or len(records) != 8 or len(locked_roster) != 8:
        raise RuntimeError("the existing roster reader did not return exactly eight cases")
    roster_index = {row["case_id"]: row for row in locked_roster}
    if len(roster_index) != 8:
        raise RuntimeError("duplicate case IDs in the existing locked roster")
    for row in records:
        locked = roster_index.get(row["case_id"])
        if locked is None or locked.get("source_case_sha256") != row.get("source_case_sha256") or locked.get("dataset") != row.get("dataset"):
            raise RuntimeError("runtime roster differs from the prior locked source cohort")

    references = reference_lock.get("records")
    reference_index = {row["case_id"]: row for row in references or []}
    if len(reference_index) != 8 or set(reference_index) != set(roster_index):
        raise RuntimeError("spatial reference catalog does not cover the exact locked cases")
    for ref in references:
        response = ref.get("actual_response_path")
        receipt_sha = ref.get("actual_response_sha256")
        if not isinstance(response, str) or not isinstance(receipt_sha, str):
            raise RuntimeError("spatial reference omitted its response receipt")
        path = ROOT / response
        if not path.is_file() or _sha(path) != receipt_sha:
            raise RuntimeError("locked spatial reference receipt bytes changed")

    output_barrier = BASE / "STUDENT_GLOBAL_BARRIER.json"
    if output_barrier.exists():
        raise FileExistsError("new spatial output barrier already exists")
    for row in records:
        if (BASE / "student" / f"{row['case_id']}.pt").exists():
            raise FileExistsError("new spatial case output already exists")

    references_by_case = reference_index
    model = None
    current_source = None
    output_files = []
    full_model_capture_calls = 0
    full_model_final_readout_calls = 0
    total_direct_steps = 0
    total_conservative_steps = 0
    started_all = time.perf_counter()
    BASE.mkdir(parents=True, exist_ok=True)
    for number, record in enumerate(records, start=1):
        case, _catalog = catalog(record)
        if case.get("_source_case_sha256") != record["source_case_sha256"]:
            raise RuntimeError("source catalog digest differs from the locked roster")
        ref = references_by_case[record["case_id"]]
        old_rel = f"{OLD_BASE.relative_to(ROOT)}/student/{record['case_id']}.pt"
        old_receipt = old_student_files.get(old_rel)
        if old_receipt is None:
            raise RuntimeError("previous S2 result is absent from its sealed output barrier")
        old_path = ROOT / old_rel
        old_case_sha = _sha(old_path)
        old_case = torch.load(old_path, map_location="cpu", weights_only=False)
        if old_case.get("GT_read") is not False or old_case.get("source_case_sha256") != record["source_case_sha256"]:
            raise RuntimeError("previous student result is not bound to the same pre-GT source")
        if old_case.get("case_id") != record["case_id"] or list(old_case.get("frame_ids", [])) != list(case["frame_ids"]):
            raise RuntimeError("previous S2 result changed case identity or physical frame grid")
        if old_case_sha != old_receipt.get("sha256"):
            raise RuntimeError("previous S2 result digest differs from its seal")
        cold_receipt = old_case.get("source_cold_readback", {})
        if (cold_receipt.get("status") != "pass"
                or cold_receipt.get("pixel_sha256") != case.get("pixel_sha256")
                or cold_receipt.get("full_model_state_sha256") != case.get("source_model_state_sha256")
                or cold_receipt.get("source_boxes_logits_bitwise") is not True
                or cold_receipt.get("hidden_head_bitwise") is not True):
            raise RuntimeError("previous S2 result lacks exact source-capture provenance")

        if current_source != case["source"]:
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            model = model_for(case["source"])
            current_source = case["source"]
        expected_model_hash = case.get("source_model_state_sha256")
        if model._fixed_full_state_hash != expected_model_hash or model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("loaded full model is not the frozen source checkpoint")

        row = raw_row(case)
        frames, frame_ids, _ = frames_for(record["dataset"], row, collections.OrderedDict())
        if list(frame_ids) != list(case["frame_ids"]):
            raise RuntimeError("canonical source reader changed physical frame IDs")
        pixel_sha = hashlib.sha256(frames.tobytes()).hexdigest()
        if pixel_sha != case["pixel_sha256"]:
            raise RuntimeError("canonical source pixels differ from the source lock")
        full_model_capture_calls += 1
        batch, source_records, base = frozen_forward(model, frames, row)
        native = prediction(base.zero["logits"], base.zero["boxes"], source_records, list(frame_ids))
        if source_records != case["records"]:
            raise RuntimeError("fresh source replay records differ from the locked source case")
        initial = {name: value.detach().clone() for name, value in base.initial.items()}
        if any(not torch.equal(value.detach().cpu(), case["source_light_initial_state"][name])
               for name, value in base.initial.items()):
            raise RuntimeError("fresh source initial student state differs from the locked source state")
        if state_hash(model.state_dict()) != expected_model_hash:
            raise RuntimeError("source full-model state hash changed during capture")
        if state_hash(initial) != state_hash(old_case["spatial_fit"]["initial_state"]):
            raise RuntimeError("fresh source light initial state differs from prior S2 initialization")
        source_boxes_xyxy = _xyxy(native["boxes"])
        prior_source = old_case["arms"]["F"]
        if prior_source["interval"] != case["Native_interval"] or not __import__("numpy").array_equal(source_boxes_xyxy, prior_source["boxes"]):
            raise RuntimeError("source arm cannot be reused exactly from the prior sealed student output")
        old_fit = old_case["spatial_fit"]
        old_s2_readout = old_case["spatial_native_full_readout"]
        if not torch.equal(old_s2_readout["boxes"].detach().cpu(), old_fit["final_cached_decoder_boxes"].detach().cpu()):
            raise RuntimeError("previous S2 full-model readout differs from its cached decoder output")
        prior_s2 = old_case["arms"]["S"]
        if prior_s2["interval"] != case["Native_interval"] or not __import__("numpy").array_equal(_xyxy(old_s2_readout["boxes"]), prior_s2["boxes"]):
            raise RuntimeError("old S2 result is not an exact, reusable arm")
        if not _same_logits(old_s2_readout["logits"], native["logits"]):
            raise RuntimeError("old S2 full-model readout changed source temporal logits")

        anchors = _anchors_for_case(record, case, ref, frame_ids, torch)
        config = core.load_original_opd_config(record["dataset"])
        config["steps"] = 5
        if config["samples"] != 32:
            raise RuntimeError("original OPD sampling count changed")
        key = "caption_query_fixed_space|" + record["case_id"]
        if old_case.get("spatial_reference_sha256") != ref["actual_response_sha256"]:
            raise RuntimeError("previous S2 fit used a different spatial-reference receipt")
        if list(old_case.get("supervised_positions", [])) != [a["position"] for a in anchors]:
            raise RuntimeError("previous S2 fit used a different sparse observation set")
        if old_fit.get("config") != config or old_fit.get("active_parameters") != 1792 or old_fit.get("declared_steps") != 5:
            raise RuntimeError("previous S2 fit did not use the pinned original joint 5-step configuration")
        for step_row in old_fit.get("path", []):
            rollout_record = step_row.get("rollout") if isinstance(step_row, dict) else None
            if rollout_record is not None:
                step_index = step_row.get("step")
                if isinstance(step_index, int) and "seed" in rollout_record:
                    expected_seed = int(hashlib.sha256(
                        f"spatial-opd-v1|{key}|{step_index}".encode()).hexdigest()[:15], 16)
                    if rollout_record["seed"] != expected_seed:
                        raise RuntimeError("previous S2 Gaussian rollout did not use the locked case key")

        def checkpoint_writer(arm_name):
            def save_progress(progress):
                destination = (BASE / "student_partial" / record["case_id"] /
                              arm_name / f"step{progress['next_step']}.pt")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(destination)
                temp = destination.with_suffix(destination.suffix + ".tmp")
                torch.save(progress, temp)
                os.replace(temp, destination)
            return save_progress

        direct_fit = core._fit_direct(
            base, initial, anchors, frame_ids, key, config, scope="joint",
            checkpoint_callback=checkpoint_writer("S1_direct"))
        if direct_fit.get("config") != config or direct_fit.get("active_parameters") != 1792 or direct_fit.get("declared_steps") != 5:
            raise RuntimeError("S1 Direct fit did not use the locked joint five-step configuration")
        total_direct_steps += int(direct_fit["optimizer_steps"])
        direct_trace = _direct_step_readback(core, base, initial, frame_ids, key, direct_fit, torch)
        if state_hash(model.state_dict()) != expected_model_hash:
            raise RuntimeError("Direct fit changed the frozen full-model weights")

        conservative_fit = fit_source_anchored_opd(
            base, initial, anchors, frame_ids, key, config, scope="joint",
            source_frame_rms_limit=0.01, backtracking_halvings=20,
            checkpoint_callback=checkpoint_writer("S3_conservative_opd"))
        total_conservative_steps += int(conservative_fit["optimizer_steps"])
        if state_hash(model.state_dict()) != expected_model_hash:
            raise RuntimeError("conservative OPD changed the frozen full-model weights")

        with torch.no_grad(), transient_native_masks(model), frozen_visual_host(model), \
                frozen_attention_storage(model), inactive_modules_on_host(model), \
                query_subject(model, batch, row["parses"]["subject"]):
            direct_readout = core._finalize_student_readout(
                model, batch, frame_ids, source_records, direct_fit["final_state"])
        full_model_final_readout_calls += 1
        if state_hash(model.state_dict()) != expected_model_hash:
            raise RuntimeError("S1 full-model readout failed to restore the frozen source model")
        with torch.no_grad(), transient_native_masks(model), frozen_visual_host(model), \
                frozen_attention_storage(model), inactive_modules_on_host(model), \
                query_subject(model, batch, row["parses"]["subject"]):
            conservative_readout = core._finalize_student_readout(
                model, batch, frame_ids, source_records, conservative_fit["final_state"])
        full_model_final_readout_calls += 1
        if state_hash(model.state_dict()) != expected_model_hash:
            raise RuntimeError("S3 full-model readout failed to restore the frozen source model")

        for name, readout, fit in (
            ("S1_direct", direct_readout, direct_fit),
            ("S3_conservative_opd", conservative_readout, conservative_fit),
        ):
            if not torch.equal(readout["boxes"].detach().cpu(), fit["final_cached_decoder_boxes"].detach().cpu()):
                raise AssertionError(f"{name} full-model boxes differ from the final cached decoder")
            if not _same_logits(readout["logits"], native["logits"]):
                raise AssertionError(f"{name} changed the frozen source temporal logits")
            if state_hash(model.state_dict()) != expected_model_hash:
                raise AssertionError(f"{name} readout did not restore the frozen source model")
        if not conservative_fit["final_constraint"]["all_source_frames_within_limit"]:
            raise AssertionError("conservative OPD final output exceeds the source RMS cap")

        arms = {
            "S0_source": {"boxes": prior_source["boxes"].copy(), "interval": list(prior_source["interval"])},
            "S1_direct": {"boxes": _xyxy(direct_readout["boxes"]), "interval": list(direct_readout["physical_interval"])},
            "S2_old_opd": {"boxes": prior_s2["boxes"].copy(), "interval": list(prior_s2["interval"])},
            "S3_conservative_opd": {"boxes": _xyxy(conservative_readout["boxes"]),
                                     "interval": list(conservative_readout["physical_interval"])},
        }
        result = {
            "schema": "stvg-support-aware-spatial-case-v1",
            "case_id": record["case_id"], "dataset": record["dataset"],
            "source_case_sha256": record["source_case_sha256"],
            "source_model_state_sha256": expected_model_hash,
            "pixel_sha256": pixel_sha, "frame_ids": list(frame_ids),
            "native_interval": list(case["Native_interval"]),
            "arms": arms,
            "fits": {"S1_direct": direct_fit, "S3_conservative_opd": conservative_fit},
            "direct_step_readback": direct_trace,
            "readouts": {"S0_source": native,
                         "S1_direct": direct_readout,
                         "S2_old_opd": old_s2_readout,
                         "S3_conservative_opd": conservative_readout},
            "reference_binding": {"spatial_reference_lock_sha256": _sha(reference_lock_path),
                                  "response_sha256": ref["actual_response_sha256"],
                                  "candidate_id": (ref["candidate"].get("candidate_id")
                                                   if ref.get("candidate") is not None else None),
                                  "observed_positions": [a["position"] for a in anchors],
                                  "reference_is_student_pseudoreference": True},
            "supervision_anchors": [{"position": a["position"],
                                     "physical_frame_id": a["physical_frame_id"],
                                     "box_cxcywh_normalized": a["box"].detach().cpu(),
                                     "weight": a["weight"]} for a in anchors],
            "reuse_binding": {"previous_student_path": old_rel,
                              "previous_student_sha256": old_case_sha,
                              "previous_source_arm_exactly_reused": True,
                              "previous_S2_arm_exactly_reused": True,
                              "source_zero_boxes_sha256": _tensor_sha(base.zero["boxes"]),
                              "source_logits_bitwise_preserved": True,
                              "full_model_state_restored": True},
            "configuration": {"spatial_opd": config,
                               "spatial_opd_config_sha256": core.OPD_CONFIG_SHA256,
                               "key": key, "scope": "joint",
                               "source_frame_rms_limit": 0.01,
                               "backtracking_halvings": 20},
            "compute": {"source_capture_calls": 1,
                        "new_full_model_readouts": 2,
                        "old_S2_readout_reused": True,
                        "direct_optimizer_steps": int(direct_fit["optimizer_steps"]),
                        "conservative_optimizer_steps": int(conservative_fit["optimizer_steps"])},
            "GT_read": False, "teacher_weight_update": False,
            "source_model_state_restored": True,
        }
        destination = BASE / "student" / f"{record['case_id']}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(result, temp)
        os.replace(temp, destination)
        output_files.append({"path": str(destination.relative_to(ROOT)),
                             "sha256": _sha(destination), "bytes": destination.stat().st_size})
        _write_json(BASE / "STUDENT_GPU_STATUS.json", {
            "status": "running", "completed_cases": number, "total_cases": 8,
            "actual_direct_fit_cases": number,
            "actual_conservative_opd_fit_cases": number,
            "GT_read": False, "time": time.time(),
        }, replace=True)
        del base, batch, frames, direct_fit, conservative_fit, direct_readout, conservative_readout, old_case, result
        gc.collect()
        torch.cuda.empty_cache()

    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    barrier = {
        "schema": "stvg-support-aware-spatial-barrier-v1",
        "status": "complete", "count": 8, "files": output_files,
        "old_roster_sha256": _sha(old_roster_path),
        "old_teacher_barrier_sha256": _sha(teacher_barrier_path),
        "old_student_barrier_sha256": _sha(old_student_barrier_path),
        "spatial_reference_lock_sha256": _sha(reference_lock_path),
        "runtime_sha256": _sha(runtime_path),
        "GT_read": False, "teacher_weight_update": False,
        "actual_direct_fit_cases": 8,
        "actual_conservative_opd_fit_cases": 8,
        "total_direct_optimizer_steps": total_direct_steps,
        "total_conservative_optimizer_steps": total_conservative_steps,
        "source_capture_calls": full_model_capture_calls,
        "new_full_model_final_readout_calls": full_model_final_readout_calls,
        "previous_source_and_S2_outputs_reused": True,
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    _write_json(output_barrier, barrier)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Keep the failure receipt small and free of raw input text.
        failure = BASE / "STUDENT_FAILURE_CAPTURE.json"
        if not failure.exists():
            _write_json(failure, {"status": "actual_failure", "error_type": type(exc).__name__,
                                  "GT_read": False, "time": time.time()})
        raise
