#!/usr/bin/env python3
"""Read back the eight sealed native-instance saved fit paths using CPU only.

This is a scalar diagnostic against the frozen teacher anchors, not a GT
evaluation. It does not run a model or open video, annotations, RGB, or query
text. Per-step cached decoder boxes are checked through the saved transition
deltas and summarized without exporting coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/stvg_instance_identity_update_v1"
OLD = ROOT / "artifacts/stvg_native_instance_spatial_opd_v1"
RUNTIME = BASE / "STEP_DIAGNOSTIC_RUNTIME.json"
DESIGN_LOCK = BASE / "DESIGN_LOCK.json"
PROTOCOL = ROOT / "protocols/stvg_instance_identity_update_v1.md"
OUT = BASE / "STEP_DIAGNOSTIC_READBACK.json"
CUSTODY = BASE / "STEP_DIAGNOSTIC_CUSTODY.json"

CASE_ALIASES = {
    "hc2_000": "HC01", "hc2_001": "HC02", "hc2_002": "HC03", "hc2_003": "HC04",
    "vidstg_000": "V01", "vidstg_001": "V02", "vidstg_002": "V03", "vidstg_003": "V04",
}
ARMS = ("direct", "iou", "reliability_weighted_iou")
EXPECTED_PINS = {
    "scripts/run_stvg_native_instance_student_v1.py":
        "d0299e21b555cd180f8282e628960a8e9e73a7b0e2cca8a21ba59627b0e2dc13",
    "scripts/stvg_native_instance_spatial_fit_v1.py":
        "316078ad420d6ad6fedbdb21cee119523d8b81e4807d42b42afc5942986dafeb",
    "scripts/stvg_target_teacher_student_v1.py":
        "e792d73414ce0639517faef351330cda8ea97298f9b7854b0ab7055de2bbbc69",
    "vg_tta/decota_spatial_opd_tunable_v1.py":
        "3f49e8c95fddc3e67f1b929c9114749815e16bb1d0029161bf1edae1570ad171",
    "methods/decota_final_simplified_v1/objectives.py":
        "c2dc28e23e4213ff2559482f7362e32bec81e767786b5b53c4501dd6017aaeaf",
}
NOFIT_REASON = {
    "native_r1_not_valid:out_of_clock_interval": "native_interval_out_of_clock",
    "no_selected_mask_at_fixed_center": "no_selected_mask_at_fixed_center",
}


class ReadbackError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ReadbackError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def tensor_array(value: Any, torch, name: str) -> np.ndarray:
    require(torch.is_tensor(value) and value.device.type == "cpu", f"{name} must be a CPU tensor")
    array = value.detach().numpy()
    require(bool(np.isfinite(array).all()), f"{name} contains a non-finite value")
    return np.asarray(array, dtype=np.float64)


def iou_cxcywh(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pxy = np.concatenate((pred[..., :2] - pred[..., 2:] / 2,
                          pred[..., :2] + pred[..., 2:] / 2), axis=-1)
    txy = np.concatenate((target[..., :2] - target[..., 2:] / 2,
                          target[..., :2] + target[..., 2:] / 2), axis=-1)
    left_top = np.maximum(pxy[..., :2], txy[..., :2])
    right_bottom = np.minimum(pxy[..., 2:], txy[..., 2:])
    intersection = np.maximum(right_bottom - left_top, 0).prod(axis=-1)
    parea = np.maximum(pxy[..., 2:] - pxy[..., :2], 0).prod(axis=-1)
    tarea = np.maximum(txy[..., 2:] - txy[..., :2], 0).prod(axis=-1)
    return intersection / np.maximum(parea + tarea - intersection, 1e-7)


def partition_displacement(left: np.ndarray, right: np.ndarray,
                           indices: Sequence[int]) -> dict[str, Any]:
    if not indices:
        return {"frame_count": 0, "center_l2_mean": None, "center_l2_max": None,
                "size_l2_mean": None, "size_l2_max": None,
                "component_rms": None, "component_max_abs": None}
    diff = np.asarray(left, dtype=np.float64)[list(indices)] - np.asarray(right, dtype=np.float64)[list(indices)]
    center = np.linalg.norm(diff[:, :2], axis=-1)
    size = np.linalg.norm(diff[:, 2:], axis=-1)
    return {
        "frame_count": len(indices),
        "center_l2_mean": float(center.mean()), "center_l2_max": float(center.max()),
        "size_l2_mean": float(size.mean()), "size_l2_max": float(size.max()),
        "component_rms": float(np.sqrt(np.mean(diff * diff))),
        "component_max_abs": float(np.max(np.abs(diff))),
    }


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    runtime = read_json(RUNTIME)
    design = read_json(DESIGN_LOCK)
    protocol_sha = file_sha(PROTOCOL)
    require(runtime.get("status") == "locked" and runtime.get("GT_read") is False
            and runtime.get("count") == 8 and runtime.get("case_ids") == list(CASE_ALIASES)
            and runtime.get("scope") ==
            "CPU saved decoder step diagnostic; no new GT/model/GPU; all8 with four focal cases",
            "diagnostic runtime does not match the fixed eight-case no-GT scope")
    require(design.get("status") == "locked" and design.get("GT_read") is False
            and design.get("historically_exposed") is True and design.get("count") == 8
            and design.get("protocol_sha256") == protocol_sha,
            "design lock or fixed protocol binding differs")
    require(runtime.get("design_lock_sha256") == file_sha(DESIGN_LOCK),
            "runtime does not bind the current design lock")
    current = ROOT / "methods/CURRENT_METHOD.json"
    require(design.get("CURRENT_sha256") == file_sha(current), "current-method hash differs from design lock")
    require(runtime.get("pins") == EXPECTED_PINS, "locked source pin set differs")
    for relative, digest in EXPECTED_PINS.items():
        path = ROOT / relative
        require(path.is_file() and file_sha(path) == digest, f"pinned source changed: {relative}")

    prior = design.get("prior_bindings")
    prior_names = {"ROSTER_LOCK.json", "GLOBAL_PREDICTION_BARRIER.json", "REFERENCE_LOCK.json",
                   "TRACKING_GLOBAL_BARRIER.json", "TEACHER_GLOBAL_BARRIER.json",
                   "STUDENT_GLOBAL_BARRIER.json"}
    require(isinstance(prior, dict) and set(prior) == prior_names, "prior evidence binding set differs")
    for name, digest in prior.items():
        require(file_sha(OLD / name) == digest, f"sealed prior evidence changed: {name}")

    reference_path = runtime.get("reference_path")
    require(reference_path == "artifacts/stvg_native_instance_spatial_opd_v1/REFERENCE_LOCK.json"
            and runtime.get("reference_sha256") == file_sha(OLD / "REFERENCE_LOCK.json"),
            "runtime points to an unexpected reference lock")
    roster = read_json(OLD / "ROSTER_LOCK.json")
    reference = read_json(OLD / "REFERENCE_LOCK.json")
    barrier = read_json(OLD / "STUDENT_GLOBAL_BARRIER.json")
    require(roster.get("status") == "locked" and roster.get("GT_read") is False
            and roster.get("count") == 8 and reference.get("status") == "locked"
            and reference.get("GT_read") is False and reference.get("count") == 8
            and barrier.get("status") == "complete" and barrier.get("GT_read") is False
            and barrier.get("count") == 8,
            "fixed roster, reference, or student barrier is not sealed GT-free")

    case_ids = list(CASE_ALIASES)
    runtime_records = {row["case_id"]: row for row in runtime.get("records", [])}
    roster_records = {row["case_id"]: row for row in roster.get("records", [])}
    reference_records = {row["case_id"]: row for row in reference.get("records", [])}
    barrier_records = {row["case_id"]: row for row in barrier.get("files", [])}
    require(all(set(rows) == set(case_ids) for rows in
                (runtime_records, roster_records, reference_records, barrier_records)),
            "one of the sealed case catalogs differs from the fixed eight")
    for case_id in case_ids:
        row = runtime_records[case_id]
        expected_path = f"artifacts/stvg_native_instance_spatial_opd_v1/student/{case_id}.pt"
        student_path = ROOT / expected_path
        require(row.get("path") == expected_path and student_path.is_file()
                and student_path.stat().st_size == row.get("bytes")
                and file_sha(student_path) == row.get("sha256"),
                f"student artifact differs from locked bytes: {CASE_ALIASES[case_id]}")
        br = barrier_records[case_id]
        receipt_path_rel = f"artifacts/stvg_native_instance_spatial_opd_v1/student/{case_id}.receipt.json"
        require(br.get("path") == receipt_path_rel, "student barrier does not point to its fixed receipt")
        receipt_path = ROOT / receipt_path_rel
        require(receipt_path.is_file() and file_sha(receipt_path) == br.get("sha256"),
                f"student receipt differs from the original barrier: {CASE_ALIASES[case_id]}")
        receipt = read_json(receipt_path)
        require(receipt.get("status") == "sealed" and receipt.get("GT_read") is False
                and receipt.get("path") == expected_path and receipt.get("sha256") == row.get("sha256")
                and receipt.get("bytes") == row.get("bytes"),
                f"student receipt does not bind the runtime tensor: {CASE_ALIASES[case_id]}")
        require(roster_records[case_id].get("source_case_sha256") ==
                reference_records[case_id].get("source_case_sha256"),
                f"source-case binding differs: {CASE_ALIASES[case_id]}")

    hashes = {
        "diagnostic_runtime_sha256": file_sha(RUNTIME),
        "design_lock_sha256": file_sha(DESIGN_LOCK),
        "protocol_sha256": protocol_sha,
        "reference_lock_sha256": file_sha(OLD / "REFERENCE_LOCK.json"),
        "student_barrier_sha256": file_sha(OLD / "STUDENT_GLOBAL_BARRIER.json"),
        "source_barrier_sha256": file_sha(OLD / "GLOBAL_PREDICTION_BARRIER.json"),
    }
    refs = reference_records
    return runtime, refs, runtime_records, hashes


def read_case(case_id: str, runtime_record: Mapping[str, Any], ref: Mapping[str, Any], torch) -> tuple[dict[str, Any], dict[str, Any]]:
    path = ROOT / str(runtime_record["path"])
    record = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(record, dict) and record.get("case_id") == case_id
            and record.get("GT_read") is False, "student case identity or GT flag differs")
    require(record.get("reference_lock_sha256") == file_sha(OLD / "REFERENCE_LOCK.json")
            and record.get("reference_record", {}).get("case_id") == case_id
            and record.get("reference_record", {}).get("source_case_sha256") == ref.get("source_case_sha256")
            and record.get("reference_record", {}).get("anchors") == ref.get("anchors"),
            "student reference binding differs from the sealed lock")
    require(record.get("source_case_sha256") == ref.get("source_case_sha256"),
            "student source-case hash differs from frozen reference")
    source = tensor_array(record.get("source_boxes"), torch, "source boxes")
    frame_ids = [int(value) for value in record.get("frame_ids", [])]
    require(source.ndim == 2 and source.shape == (len(frame_ids), 4) and len(frame_ids) > 0
            and all(y > x for x, y in zip(frame_ids, frame_ids[1:])),
            "source boxes do not match the ordered physical-frame grid")
    anchors = []
    for anchor in ref.get("anchors", []):
        pos = anchor.get("position")
        box = anchor.get("box_cxcywh")
        require(isinstance(pos, int) and not isinstance(pos, bool) and 0 <= pos < len(frame_ids)
                and anchor.get("physical_frame_id") == frame_ids[pos]
                and isinstance(box, list) and len(box) == 4 and float(box[2]) > 0 and float(box[3]) > 0,
                "teacher anchor is not on the locked source-grid position")
        anchors.append({"position": pos, "box": np.asarray(box, dtype=np.float64)})
    require(len(anchors) <= 4, "reference exceeds the fixed four-anchor limit")
    anchor_positions = [a["position"] for a in anchors]
    unobserved_positions = [i for i in range(len(source)) if i not in set(anchor_positions)]
    fits = record.get("fits", {})
    receipts = record.get("receipts", {})
    readouts = record.get("readouts", {})
    require(isinstance(fits, Mapping) and isinstance(receipts, Mapping) and isinstance(readouts, Mapping)
            and set(receipts) == set(ARMS), "saved arms or receipts are incomplete")

    case = {
        "case_alias": CASE_ALIASES[case_id],
        "dataset": "HC2" if case_id.startswith("hc2_") else "VidSTG",
        "source_frame_count": len(frame_ids),
        "reference_status": ref.get("parsed_status"),
        "anchor_count": len(anchors),
        "arm_readbacks": [],
    }
    custody = {
        "case_id": case_id,
        "student_sha256": runtime_record["sha256"],
        "student_bytes": runtime_record["bytes"],
        "source_case_sha256": record["source_case_sha256"],
        "source_model_state_sha256": record.get("source_model_state_sha256"),
        "source_pixel_sha256": record.get("pixel_sha256"),
        "anchor_count": len(anchors),
    }
    for arm in ARMS:
        fit, receipt = fits.get(arm), receipts[arm]
        if fit is None:
            require(receipt.get("status") == "exact_no_update" and receipt.get("optimizer_steps") == 0
                    and receipt.get("final_redecode_calls") == 0,
                    "missing fit does not have an exact no-update receipt")
            case["arm_readbacks"].append({
                "arm": arm, "fit_present": False, "status": "exact_no_update",
                "reason_category": NOFIT_REASON.get(str(ref.get("reason")), "no_verified_reference"),
                "optimizer_steps": 0, "full_model_final_redecode_calls": 0, "steps": [],
            })
            continue

        require(fit.get("algorithm") == arm and fit.get("scope") == "joint"
                and fit.get("active_parameters") == 1792 and fit.get("declared_steps") == 5
                and fit.get("selected_step") == 5 and fit.get("GT_read") is False
                and fit.get("reference_role") == "teacher"
                and fit.get("reference_sha256") == file_sha(OLD / "REFERENCE_LOCK.json"),
                "fit differs from the fixed joint/teacher/five-step contract")
        require(fit.get("config") == record.get("configuration")
                and fit["config"].get("steps") == 5 and fit["config"].get("samples") == 32,
                "fit configuration differs from the saved five-step contract")
        if arm != "direct":
            require(fit.get("seed_key") == "caption_query_fixed_space|" + case_id,
                    "OPD deterministic per-case key differs")
        path = fit.get("path")
        require(isinstance(path, list) and len(path) == 5, "saved fit must contain five path entries")
        require(np.array_equal(tensor_array(path[0].get("boxes"), torch, "step-zero boxes"), source),
                "fit did not start at the saved source decoder boxes")
        require(receipt.get("status") == "actual_updated_full_redecode"
                and receipt.get("optimizer_steps") == fit.get("optimizer_steps")
                and receipt.get("final_redecode_calls") == 1,
                "fit receipt does not match its saved path")

        steps = []
        actual_update_count = 0
        for step_index, entry in enumerate(path):
            require(entry.get("step") == step_index, "saved path step order is incomplete")
            before = tensor_array(entry.get("boxes"), torch, "cached pre-step boxes")
            after_tensor = path[step_index + 1].get("boxes") if step_index < 4 else fit.get("final_cached_decoder_boxes")
            after = tensor_array(after_tensor, torch, "cached post-step boxes")
            delta = tensor_array(entry.get("actual_delta"), torch, "saved actual box delta")
            require(before.shape == after.shape == delta.shape == source.shape,
                    "saved boxes/delta do not match the frozen source grid")
            transition_error = float(np.max(np.abs(before + delta - after)))
            require(transition_error <= 3e-7, "saved box delta does not reconstruct the next cached step")
            did_update = (not bool(entry.get("skipped"))) if arm == "direct" else entry.get("update") is not None
            actual_update_count += int(did_update)
            before_iou = (iou_cxcywh(before[anchor_positions], np.stack([a["box"] for a in anchors])).tolist()
                          if anchors else [])
            after_iou = (iou_cxcywh(after[anchor_positions], np.stack([a["box"] for a in anchors])).tolist()
                         if anchors else [])
            before_drift = {
                "supervised": partition_displacement(before, source, anchor_positions),
                "unobserved": partition_displacement(before, source, unobserved_positions),
            }
            after_drift = {
                "supervised": partition_displacement(after, source, anchor_positions),
                "unobserved": partition_displacement(after, source, unobserved_positions),
            }
            movement = {
                "supervised": partition_displacement(after, before, anchor_positions),
                "unobserved": partition_displacement(after, before, unobserved_positions),
            }
            loss = entry.get("loss")
            if loss is not None and hasattr(loss, "item"):
                loss = loss.item()
            require(loss is None or (not isinstance(loss, bool) and math.isfinite(float(loss))),
                    "saved step loss is non-finite")
            no_info = None
            rounds = fit.get("rounds")
            if isinstance(rounds, list) and len(rounds) > step_index:
                no_info = bool(rounds[step_index].get("no_information"))
            steps.append({
                "step": step_index,
                "optimizer_updated": did_update,
                "no_information": no_info,
                "saved_loss": float(loss) if loss is not None else None,
                "teacher_anchor_iou_before_by_ordinal": [float(x) for x in before_iou],
                "teacher_anchor_iou_after_by_ordinal": [float(x) for x in after_iou],
                "teacher_anchor_iou_before_mean": float(np.mean(before_iou)) if before_iou else None,
                "teacher_anchor_iou_after_mean": float(np.mean(after_iou)) if after_iou else None,
                "source_drift_before": before_drift,
                "source_drift_after": after_drift,
                "per_step_box_movement": movement,
                "delta_reconstruction_max_abs": transition_error,
            })
        require(actual_update_count == fit.get("optimizer_steps"),
                "saved step update indicators differ from optimizer-step count")
        readout = readouts.get(arm)
        require(isinstance(readout, Mapping) and torch.is_tensor(readout.get("boxes"))
                and torch.equal(readout["boxes"].detach().cpu(), fit["final_cached_decoder_boxes"].detach().cpu()),
                "final full-model readout does not exactly match cached final decoder boxes")
        source_logits, final_logits = record.get("source_logits"), readout.get("logits")
        require(isinstance(source_logits, list) and isinstance(final_logits, list)
                and len(source_logits) == len(final_logits)
                and all(torch.equal(x.detach().cpu(), y.detach().cpu())
                        for x, y in zip(source_logits, final_logits)),
                "final full-model readout changed frozen source logits")
        case["arm_readbacks"].append({
            "arm": arm, "fit_present": True,
            "status": "fit_with_zero_optimizer_updates" if actual_update_count == 0 else "actual_fit",
            "optimizer_steps": actual_update_count, "declared_steps": 5,
            "full_model_final_redecode_calls": 1,
            "full_model_final_boxes_match_cache": True,
            "source_logits_unchanged": True,
            "steps": steps,
        })
    return case, custody


def run_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    require(not OUT.exists() and not CUSTODY.exists(), "refusing to overwrite prior diagnostic outputs")
    runtime, ref_records, runtime_records, bindings = validate_inputs()
    import torch
    require(not torch.cuda.is_initialized(), "CUDA was initialized before CPU-only diagnostic")
    cases, custody = [], []
    for case_id in runtime["case_ids"]:
        case, private = read_case(case_id, runtime_records[case_id], ref_records[case_id], torch)
        cases.append(case)
        custody.append(private)
    require(not torch.cuda.is_initialized(), "CUDA was initialized during CPU-only diagnostic")
    arms = [arm for case in cases for arm in case["arm_readbacks"]]
    summary = {
        "schema": "stvg-instance-step-diagnostic-readback-v1",
        "status": "pass", "GT_read": False, "model_replayed": False, "GPU_used": False,
        "historically_exposed_cohort": True,
        "scope": "fixed eight-source saved-fit step readback against frozen teacher anchors; no GT/model/GPU",
        "input_bindings": bindings,
        "counts": {
            "cases": len(cases), "arms": len(arms),
            "fitted_arms": sum(bool(row["fit_present"]) for row in arms),
            "exact_no_update_arms": sum(not bool(row["fit_present"]) for row in arms),
            "zero_optimizer_update_fits": sum(row["status"] == "fit_with_zero_optimizer_updates" for row in arms),
            "saved_steps": sum(len(row["steps"]) for row in arms),
            "optimizer_updates": sum(row["optimizer_steps"] for row in arms),
            "full_model_final_redecode_calls": sum(row["full_model_final_redecode_calls"] for row in arms),
        },
        "readout_semantics": {
            "boxes": "saved normalized cxcywh cached decoder boxes on the physical source-frame grid",
            "teacher_anchor_iou": "overlap with frozen R1 teacher anchors, not GT",
            "supervised_frames": "source-grid positions with a verified anchor in the locked reference",
            "unobserved_frames": "all remaining source-grid positions; drift is measured against saved source boxes",
            "step_boxes": "cached replay/head outputs; only one final full-model readout is saved per fitted arm",
        },
        "cases": cases,
        "limitations": [
            "Historical cohort was previously exposed; this is a mechanism diagnosis, not a fresh or confirmatory test.",
            "Per-step teacher-anchor overlap and source-box drift do not establish GT accuracy or object identity.",
            "No intermediate image/model decode was run; only saved cached step boxes and existing final readouts were checked.",
            "No best-step selection, hyperparameter change, or GT-based decision was made.",
        ],
        "reader_sha256": file_sha(Path(__file__).resolve()),
    }
    canonical = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    custody_readback = {
        "schema": "stvg-instance-step-diagnostic-custody-v1", "status": "pass", "GT_read": False,
        "private_case_ids_local_only": True,
        "readback_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "input_bindings": bindings, "records": custody,
        "reader_sha256": summary["reader_sha256"],
    }
    return summary, custody_readback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args()
    summary, custody = run_audit()
    for path, value in ((OUT, summary), (CUSTODY, custody)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": "pass", "cases": summary["counts"]["cases"],
                      "fitted_arms": summary["counts"]["fitted_arms"],
                      "saved_steps": summary["counts"]["saved_steps"],
                      "readback": str(OUT.relative_to(ROOT)),
                      "custody": str(CUSTODY.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
