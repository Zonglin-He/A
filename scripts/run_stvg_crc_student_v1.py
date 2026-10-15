#!/usr/bin/env python3
"""Source-reset student for the fixed-pool contrastive referent correction.

Only an Alternative decision exposes a geometry anchor. Source, Uncertain,
invalid, and explicitly unpairable slots do not create references. Historical
F/T and new-crop Top1 predictions are copied from their sealed parent outputs.
This module is inert on import; --contracts is CPU-only.
"""
from __future__ import annotations

import argparse
import collections
import copy
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_crc_common_v1 import BASE, C, OLD, PARENT, ROOT as PROJECT_ROOT
from scripts.stvg_crc_common_v1 import load_case, native_arm, read, save_pt, sha, verify_runtime, write

OWN_REL = "scripts/run_stvg_crc_student_v1.py"
SOURCE_ROSTER = ROOT / "artifacts/stvg_final_table1_v1/PREFLIGHT_ROSTER_LOCK.json"


def _case_id(row: Mapping[str, Any]) -> str:
    value = row.get("case_id", row.get("probe_alias"))
    if not isinstance(value, str) or not value:
        raise ValueError("roster row requires case_id or probe_alias")
    return value


def _index(rows: Sequence[Mapping[str, Any]], key: str = "case_id") -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"duplicate or missing {key}")
        result[value] = row
    return result


def _sig(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("position", "physical_frame_id", "box_cxcywh", "weight", "candidate_id")
    result = []
    for anchor in anchors:
        row = {}
        for field in fields:
            if field == "weight" and field not in anchor:
                row[field] = 1.0
            elif field == "candidate_id" and field not in anchor:
                row[field] = None
            else:
                row[field] = anchor.get(field)
        result.append(row)
    return result


def _same_value(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _file_record_index(barrier: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = barrier.get("files")
    if not isinstance(rows, list):
        raise ValueError("sealed barrier files must be a list")
    result = {}
    for row in rows:
        path = row.get("path")
        if not isinstance(path, str) or path in result:
            raise ValueError("barrier paths must be unique strings")
        result[path] = row
    return result


def _load_sealed_payload(path: Path, item: Mapping[str, Any], torch):
    if not path.is_file() or sha(path) != item.get("sha256"):
        raise RuntimeError(f"sealed payload byte hash mismatch: {path}")
    if "bytes" in item and path.stat().st_size != int(item["bytes"]):
        raise RuntimeError(f"sealed payload byte count mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _parent_artifacts(case_id: str, case: Mapping[str, Any], torch):
    """Read only prediction/reference metadata from sealed prompt009 artifacts."""
    student_barrier = read(PARENT / "STUDENT_GLOBAL_BARRIER.json")
    prediction_barrier = read(PARENT / "GLOBAL_PREDICTION_BARRIER.json")
    if (student_barrier.get("status") != "complete" or student_barrier.get("count") != 32
            or student_barrier.get("GT_read") is not False):
        raise RuntimeError("prompt009 student barrier is incomplete or not explicitly GT-free")
    if (prediction_barrier.get("status") != "complete" or prediction_barrier.get("count") != 32
            or prediction_barrier.get("GT_read") is not False):
        raise RuntimeError("prompt009 global prediction barrier is incomplete or not explicitly GT-free")
    path = PARENT / "student" / f"{case_id}.pt"
    rel = str(path.relative_to(ROOT))
    student_items = _file_record_index(student_barrier)
    prediction_items = _file_record_index(prediction_barrier)
    if rel not in student_items or rel not in prediction_items:
        raise RuntimeError(f"parent student output is absent from both required barriers: {case_id}")
    if student_items[rel].get("sha256") != prediction_items[rel].get("sha256"):
        raise RuntimeError("prompt009 student and global prediction barriers disagree")
    parent = _load_sealed_payload(path, student_items[rel], torch)
    if parent.get("case_id") != case_id or parent.get("GT_read") is not False:
        raise RuntimeError("prompt009 student payload identity/GT marker mismatch")
    if parent.get("source_case_sha256") != case.get("source_case_sha256"):
        raise RuntimeError("prompt009 student source case differs from current locked source")
    if list(map(int, parent.get("frame_ids", []))) != list(map(int, case.get("frame_ids", []))):
        raise RuntimeError("prompt009 student source grid differs from current source case")
    temporal_path = PARENT / "temporal" / f"{case_id}.pt"
    if not temporal_path.is_file() or sha(temporal_path) != parent.get("temporal_file_sha256"):
        raise RuntimeError("prompt009 selected temporal artifact is not bound by its student payload")
    temporal = torch.load(temporal_path, map_location="cpu", weights_only=False)
    if temporal.get("case_id", case_id) != case_id or temporal.get("GT_read") is not False:
        raise RuntimeError("parent temporal payload identity/GT marker mismatch")
    interval = temporal.get("selected_interval")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise RuntimeError("parent temporal output lacks a selected interval")
    if not _same_value(parent["arms"]["T"]["interval"], list(interval)):
        raise RuntimeError("parent T arm differs from its sealed temporal output")
    for key in ("S_new_crop", "TS_new_crop"):
        if key not in parent.get("arms", {}):
            raise RuntimeError(f"prompt009 parent lacks {key}")
    return parent, temporal, path, student_items[rel]


def _historical_fit_candidates(case_id: str, expected_sha: str):
    """Resolve an exact parent_file_sha through known historical student folders."""
    direct = [OLD / "round2" / "student" / f"{case_id}.pt",
              PARENT / "student" / f"{case_id}.pt"]
    # Older copies are searched only when the immediate prompt009 parent points
    # elsewhere. Paths are hashed before any deserialization.
    if not any(p.is_file() and sha(p) == expected_sha for p in direct):
        patterns = [
            OLD.glob(f"round*/student/{case_id}.pt"),
            OLD.glob(f"round*/*/student/{case_id}.pt"),
            C.glob(f"*/student/{case_id}.pt"),
            C.glob(f"*/*/student/{case_id}.pt"),
        ]
        for pattern in patterns:
            direct.extend(pattern)
    found = []
    seen = set()
    for path in direct:
        path = Path(path)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if sha(path) == expected_sha:
            found.append(path)
    return found


def _fit_is_reusable(fitted: Mapping[str, Any], source: Mapping[str, Any],
                     anchors: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
                     seed_key: str, torch, h) -> bool:
    if (not isinstance(fitted, Mapping) or fitted.get("GT_read") is not False
            or fitted.get("active_parameters") != 1792
            or fitted.get("algorithm") != "validated_loss_weighted_iou"
            or fitted.get("reward") != "original_iou"
            or fitted.get("reference_role") != "teacher"
            or fitted.get("seed_key") != seed_key
            or fitted.get("config") != dict(config)
            or fitted.get("declared_steps") != 5):
        return False
    old_ref = source.get("reference_records", {}).get("new_crop", {}).get("anchors")
    if not isinstance(old_ref, list) or not _same_value(_sig(old_ref), _sig(anchors)):
        return False
    if "new_crop" not in source.get("readouts", {}) or "S_new_crop" not in source.get("arms", {}):
        return False
    if not torch.equal(source["readouts"]["new_crop"]["boxes"].detach().cpu(),
                       fitted["final_cached_decoder_boxes"].detach().cpu()):
        return False
    import numpy as np
    expected_xyxy = h._xyxy(fitted["final_cached_decoder_boxes"])
    actual_xyxy = np.asarray(source["arms"]["S_new_crop"]["boxes"])
    if not np.array_equal(expected_xyxy, actual_xyxy):
        return False
    return True


def _resolve_historical_fit(case_id: str, parent: Mapping[str, Any], anchors,
                            config, seed_key: str, torch, h):
    """Follow sealed receipt hashes to the actual fit; never infer reuse from arms alone."""
    current = parent
    current_path = PARENT / "student" / f"{case_id}.pt"
    current_sha = sha(current_path)
    visited = set()
    while True:
        if current_sha in visited:
            return None
        visited.add(current_sha)
        receipt = current.get("receipts", {}).get("new_crop", {})
        fit = current.get("fits", {}).get("new_crop")
        if (receipt.get("actual_fit_count") == 1 and fit is not None
                and _fit_is_reusable(fit, current, anchors, config, seed_key, torch, h)):
            return dict(fit=copy.deepcopy(fit), readout=copy.deepcopy(current["readouts"]["new_crop"]),
                        student=current, source_path=current_path, source_sha256=current_sha)
        parent_sha = receipt.get("parent_file_sha256")
        if not parent_sha:
            return None
        matches = _historical_fit_candidates(case_id, str(parent_sha))
        if len(matches) != 1:
            return None
        current_path = matches[0]
        current_sha = str(parent_sha)
        current = torch.load(current_path, map_location="cpu", weights_only=False)
        if current.get("case_id") != case_id or current.get("source_case_sha256") != parent.get("source_case_sha256"):
            return None
        if current.get("GT_read") is not False:
            return None


def _validate_inputs(torch):
    runtime = verify_runtime()
    if OWN_REL not in {**runtime.get("pins", {}), **runtime.get("protected", {})}:
        raise RuntimeError("CRC RUNTIME must pin this exact student worker")
    reference_path = BASE / "REFERENCE_LOCK.json"
    ref_doc = read(reference_path)
    if (ref_doc.get("schema") != "stvg-crc-reference-lock-v1" or ref_doc.get("status") != "locked"
            or ref_doc.get("count") != 32 or ref_doc.get("slot_count") != 128
            or ref_doc.get("pair_job_count") != 49 or ref_doc.get("GT_read") is not False):
        raise RuntimeError("CRC REFERENCE_LOCK is not complete and locked")
    refs = _index(ref_doc.get("records", []))
    roster = []
    for row in read(BASE / "ROSTER_LOCK.json").get("records", []):
        roster.append(row)
    roster_index = {_case_id(row): row for row in roster}
    if len(roster_index) != 32 or set(roster_index) != set(refs):
        raise RuntimeError("CRC roster/reference case sets differ")
    for cid, row in refs.items():
        if row.get("GT_read") is not False or row.get("dataset") != roster_index[cid].get("dataset"):
            raise RuntimeError(f"CRC reference case metadata mismatch: {cid}")
        if row.get("source_case_sha256") != roster_index[cid].get("source_case_sha256"):
            raise RuntimeError(f"CRC source identity mismatch: {cid}")
        pair = row.get("Pair")
        if not isinstance(pair, Mapping) or not isinstance(pair.get("anchors"), list):
            raise RuntimeError(f"CRC Pair reference is malformed: {cid}")
        positions = [int(a["position"]) for a in pair["anchors"]]
        if len(positions) != len(set(positions)):
            raise RuntimeError(f"CRC Pair contains duplicate source positions: {cid}")
    barrier = read(BASE / "TEACHER_GLOBAL_BARRIER.json")
    if (barrier.get("status") != "complete" or barrier.get("count") != 49
            or barrier.get("GT_read") is not False):
        raise RuntimeError("CRC teacher barrier is incomplete")
    if ref_doc.get("teacher_barrier_sha256") != sha(BASE / "TEACHER_GLOBAL_BARRIER.json"):
        raise RuntimeError("CRC reference lock does not bind current teacher barrier")
    for path, key in ((BASE / "REFERENCE_LOCK.json", "reference_lock_sha256"),
                      (BASE / "TEACHER_GLOBAL_BARRIER.json", "teacher_barrier_sha256")):
        if runtime.get(key) not in (None, sha(path)):
            raise RuntimeError(f"CRC student runtime has a stale {key}")
    parent_student = read(PARENT / "STUDENT_GLOBAL_BARRIER.json")
    parent_teacher = read(PARENT / "TEACHER_GLOBAL_BARRIER.json")
    parent_prediction = read(PARENT / "GLOBAL_PREDICTION_BARRIER.json")
    for item in (parent_student, parent_teacher, parent_prediction):
        if item.get("status") != "complete" or item.get("GT_read") is not False:
            raise RuntimeError("prompt009 parent barrier is not complete and GT-free")
    if len(parent_student.get("files", [])) != 32 or len(parent_prediction.get("files", [])) != 42:
        raise RuntimeError("prompt009 prediction barriers have unexpected cardinality")
    for rel in ("scripts/run_stvg_table1_student004.py", "scripts/run_stvg_adjacent_motion_student_v1.py",
                "scripts/stvg_weightfree_coordinate_fit_v1.py"):
        # Runtime may encode these as pins or protected; if present, verify is already done by verify_runtime.
        if rel not in {**runtime.get("pins", {}), **runtime.get("protected", {})}:
            raise RuntimeError(f"CRC runtime must pin the reused implementation: {rel}")
    if str((SOURCE_ROSTER).relative_to(ROOT)) not in {**runtime.get("pins", {}), **runtime.get("protected", {})}:
        raise RuntimeError("CRC RUNTIME must pin the complete original source roster")
    return runtime, roster_index, refs, ref_doc, parent_student, parent_prediction


def _resolve_source_case(rec: Mapping[str, Any], torch, source_roster: Mapping[str, Mapping[str, Any]]):
    cid = _case_id(rec)
    load_row = dict(rec)
    load_row["probe_alias"] = cid
    case, path = load_case(load_row)
    if case.get("case_id") != cid or case.get("GT_read") is not False:
        raise RuntimeError(f"source case identity/GT marker mismatch: {cid}")
    if case.get("source_case_sha256") != rec.get("source_case_sha256"):
        raise RuntimeError(f"source case hash differs from CRC roster: {cid}")
    if cid not in source_roster:
        raise RuntimeError(f"missing original source worker roster row: {cid}")
    original = source_roster[cid]
    if (original.get("dataset") != rec.get("dataset")
            or original.get("query_key_sha256") != rec.get("query_key_sha256")):
        raise RuntimeError(f"original source roster identity differs for {cid}")
    return case, path, original


def contracts() -> dict[str, Any]:
    import numpy as np
    # Pure synthetic arithmetic only; torch/model/media are not imported.
    a = [{"position": 2, "physical_frame_id": 30, "box_cxcywh": [0.5, 0.5, 0.2, 0.4],
          "weight": 1.0, "candidate_id": 7, "teacher_output_sha256": "a" * 64}]
    b = [{"position": 2, "physical_frame_id": 30, "box_cxcywh": [0.5, 0.5, 0.2, 0.4],
          "weight": 1.0, "candidate_id": 7, "teacher_output_sha256": "b" * 64,
          "original_top1_source_sha256": "c" * 64}]
    assert _sig(a) == _sig(b)  # provenance differs, fit-defining geometry is identical.
    cfg = {"lr": 0.003535533905932738, "sigma": 0.0125, "tau": 0.2,
           "steps": 5, "writeback": 0.0625, "samples": 32}
    key = "caption_query_fixed_space|hc2_" + "1" * 64
    fake_fit = {"GT_read": False, "active_parameters": 1792,
                "algorithm": "validated_loss_weighted_iou", "reward": "original_iou",
                "reference_role": "teacher", "seed_key": key, "config": cfg, "declared_steps": 5}
    # Test exact reuse eligibility at the pure metadata/signature level through a minimal shim.
    class FakeTensor:
        def __init__(self, x): self.x = np.asarray(x)
        def detach(self): return self
        def cpu(self): return self
    class FakeTorch:
        @staticmethod
        def equal(x, y): return np.array_equal(x.x, y.x)
    class FakeHelper:
        @staticmethod
        def _xyxy(x):
            z = x.x
            return np.concatenate([z[:, :2] - z[:, 2:] / 2, z[:, :2] + z[:, 2:] / 2], axis=1)
    final = FakeTensor(np.array([[0.5, 0.5, 0.2, 0.4]], dtype=np.float32))
    fake_fit["final_cached_decoder_boxes"] = final
    source = {"reference_records": {"new_crop": {"anchors": a}},
              "readouts": {"new_crop": {"boxes": final}},
              "arms": {"S_new_crop": {"boxes": FakeHelper._xyxy(final)}}}
    assert _fit_is_reusable(fake_fit, source, b, cfg, key, FakeTorch, FakeHelper)
    assert not _fit_is_reusable(fake_fit, source, b, {**cfg, "lr": 0.1}, key, FakeTorch, FakeHelper)
    assert not _fit_is_reusable(fake_fit, source, b, cfg, key + "x", FakeTorch, FakeHelper)
    assert _same_value(_arm_copy("F", [1, 2]), _arm_copy("T", [1, 2]))
    assert not _same_value(_arm_copy("F", [1, 2]), _arm_copy("T", [2, 3]))
    return {"status": "pass", "check_count": 6,
            "checks": ["anchor_geometry_signature_ignores_provenance_only",
                       "historical_reuse_requires_exact_reference_config_and_seed",
                       "historical_reuse_requires_cached_full_readout_match",
                       "no_update_arm_interval_identity"],
            "GT_read": False, "CPU_only": True, "model_loaded": False, "GPU_used": False}


def _arm_copy(name: str, interval: Sequence[float]) -> dict[str, Any]:
    return {"boxes": [[0.1, 0.2, 0.3, 0.4]], "interval": list(interval)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(contracts(), sort_keys=True))
        return
    if not args.run:
        parser.error("select --contracts or --run")
    _run()


def _run():
    import numpy as np
    import torch
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    from scripts import run_stvg_table1_source_v1 as source_helper
    from scripts.run_stvg_table1_student004 import stable_fit_key
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
    from scripts.stvg_weightfree_coordinate_fit_v1 import fit_validated

    runtime, roster_index, refs, ref_doc, parent_student_barrier, parent_prediction_barrier = _validate_inputs(torch)
    source_roster_doc = read(SOURCE_ROSTER)
    if source_roster_doc.get("status") != "locked" or source_roster_doc.get("GT_read") is not False:
        raise RuntimeError("original source roster is not locked and explicitly GT-free")
    source_rows = _index(source_roster_doc.get("records", []), "probe_alias")
    for cid, row in roster_index.items():
        orig = source_rows.get(cid)
        if orig is None or orig.get("dataset") != row.get("dataset"):
            raise RuntimeError(f"CRC roster is outside source roster: {cid}")
        for key in ("query_key_sha256", "frame_ids"):
            if row.get(key) is not None and row.get(key) != orig.get(key):
                raise RuntimeError(f"CRC roster/source roster mismatch for {cid}: {key}")

    design = read(BASE / "DESIGN_LOCK.json")
    cfg_by_ds = design.get("spatial_configs")
    if not isinstance(cfg_by_ds, Mapping):
        raise RuntimeError("CRC DESIGN_LOCK lacks original spatial configs")
    refs_by_case = refs
    parent_student_items = _file_record_index(parent_student_barrier)
    parent_prediction_items = _file_record_index(parent_prediction_barrier)
    install_clean_loader()
    sys.addaudithook(__import__("scripts.stvg_native_instance_common_v1", fromlist=["guard"]).guard)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = current_source = None
    files = []
    total_fits = total_reuses = total_steps = 0
    started = time.perf_counter()
    for index, rec in enumerate(roster_index.values(), 1):
        cid = _case_id(rec)
        dest = BASE / "student" / f"{cid}.pt"
        receipt_path = dest.with_suffix(".receipt.json")
        if dest.exists() or receipt_path.exists():
            if not (dest.exists() and receipt_path.exists()):
                raise RuntimeError(f"partial pre-existing CRC output for {cid}; refusing re-entry")
            receipt = read(receipt_path)
            if (receipt.get("status") != "complete" or receipt.get("sha256") != sha(dest)
                    or receipt.get("reference_lock_sha256") != sha(BASE / "REFERENCE_LOCK.json")):
                raise RuntimeError(f"pre-existing CRC output is not sealed for {cid}")
            files.append(receipt["file"])
            continue
        case, source_path, source_row = _resolve_source_case(rec, torch, source_rows)
        parent, temporal, parent_path, parent_item = _parent_artifacts(cid, case, torch)
        pair_ref = refs_by_case[cid]["Pair"]
        anchors = pair_ref["anchors"]
        interval = list(temporal["selected_interval"])
        arms = {
            "F": native_arm(case),
            "T": native_arm(case, interval),
            "S_historical": copy.deepcopy(parent["arms"]["S_new_crop"]),
            "TS_historical": copy.deepcopy(parent["arms"]["TS_new_crop"]),
        }
        if not _same_value(arms["T"]["interval"], parent["arms"]["T"]["interval"]):
            raise RuntimeError(f"parent T interval mismatch for {cid}")
        fits: dict[str, Any] = {}
        readouts: dict[str, Any] = {}
        receipts: dict[str, Any] = {}
        fit_key = stable_fit_key(str(rec["dataset"]), str(rec["query_key_sha256"]))
        config = dict(cfg_by_ds[rec["dataset"]])
        config["steps"] = 5
        initial = None
        source_capture_s = 0.0

        if not anchors:
            arms["S_Pair"] = copy.deepcopy(arms["F"])
            arms["TS_Pair"] = copy.deepcopy(arms["T"])
            receipts["Pair"] = {"actual_fit_count": 0, "status": "no_pair_reference_no_update",
                                "fit_invocations": 0}
        else:
            reusable = _resolve_historical_fit(cid, parent, anchors, config, fit_key, torch, h)
            if reusable is not None:
                source_student = reusable["student"]
                historical_xyxy = h._xyxy(reusable["fit"]["final_cached_decoder_boxes"])
                if (not np.array_equal(historical_xyxy, np.asarray(parent["arms"]["S_new_crop"]["boxes"]))
                        or not np.array_equal(historical_xyxy, np.asarray(parent["arms"]["TS_new_crop"]["boxes"]))):
                    raise RuntimeError("exact-fit reuse source differs from the sealed historical S/TS arms")
                fits["Pair"] = reusable["fit"]
                readouts["Pair"] = reusable["readout"]
                arms["S_Pair"] = copy.deepcopy(parent["arms"]["S_new_crop"])
                arms["TS_Pair"] = copy.deepcopy(parent["arms"]["TS_new_crop"])
                receipts["Pair"] = {
                    "actual_fit_count": 0,
                    "status": "exact_historical_fit_readout_reuse",
                    "historical_source_path": str(reusable["source_path"].relative_to(ROOT)),
                    "historical_source_sha256": reusable["source_sha256"],
                    "historical_prediction_path": str(parent_path.relative_to(ROOT)),
                    "historical_prediction_sha256": sha(parent_path),
                    "historical_fit_config": config,
                    "historical_seed_key": fit_key,
                    "fit_invocations": 0,
                }
                total_reuses += 1
            else:
                transient = dict(case)
                transient["source"] = source_helper.SOURCE_MODEL_NAME[rec["dataset"]]
                transient["_source_case_sha256"] = case["source_case_sha256"]
                _, original_source_row = source_helper._validate_roster_record(source_row)
                transient["parent"] = original_source_row["source"]

                def catalog(record):
                    if (record.get("case_id") != cid
                            or record.get("source_case_sha256") != case["source_case_sha256"]):
                        raise RuntimeError("capture requested a different locked CRC source")
                    return transient, {}

                h.catalog = catalog
                capture_record = {"case_id": cid, "dataset": rec["dataset"],
                                  "source_case_sha256": case["source_case_sha256"]}
                (model, current_source, captured_case, _catalog_data, raw, frames, ids,
                 native_records, batch, base, native, expected_state, pixel_sha,
                 source_capture_s) = h._capture_source(
                    capture_record, model, current_source, torch, collections,
                    model_for, frames_for, raw_row, frozen_forward, prediction, state_hash)
                if captured_case["source_case_sha256"] != case["source_case_sha256"]:
                    raise RuntimeError("cold capture returned a different source case")
                if pixel_sha != case["pixel_sha256"] or expected_state != case["source_model_state_sha256"]:
                    raise RuntimeError("cold capture pixels or frozen model state differ from source case")
                if not np.array_equal(h._xyxy(native["boxes"]), arms["F"]["boxes"]):
                    raise RuntimeError("cold capture native boxes differ from frozen F")
                if not h._same_logits(native["logits"], case["zero"]["logits"]):
                    raise RuntimeError("cold capture native logits differ from frozen source")
                initial = {name: value.detach().clone() for name, value in base.initial.items()}
                checked_anchors = h._anchors(pair_ref, ids, torch)
                partial = BASE / "partial" / cid / "Pair"
                partial.mkdir(parents=True, exist_ok=False)

                def checkpoint(payload):
                    target = partial / f"step{int(payload['next_step'])}.pt"
                    if target.exists():
                        raise FileExistsError(target)
                    save_pt(target, payload)

                fitted = fit_validated(
                    base, {name: value.clone() for name, value in initial.items()},
                    checked_anchors, ids, fit_key, config, validate=True,
                    reference_sha256=sha(BASE / "REFERENCE_LOCK.json"),
                    checkpoint_callback=checkpoint)
                full_readout = h._readout(
                    model, batch, ids, native_records, fitted, raw["parses"]["subject"],
                    core, torch, transient_native_masks=transient_native_masks,
                    frozen_visual_host=frozen_visual_host,
                    frozen_attention_storage=frozen_attention_storage,
                    inactive_modules_on_host=inactive_modules_on_host,
                    query_subject=query_subject)
                if not h._same_logits(full_readout["logits"], native["logits"]):
                    raise RuntimeError("full Pair readout changed original native logits")
                if state_hash(model.state_dict()) != expected_state:
                    raise RuntimeError("full Pair readout did not restore frozen model state")
                if not torch.equal(full_readout["boxes"].detach().cpu(),
                                   fitted["final_cached_decoder_boxes"].detach().cpu()):
                    raise RuntimeError("full Pair readout differs from cached decoder output")
                fitted["historical_reuse"] = False
                fits["Pair"] = fitted
                readouts["Pair"] = full_readout
                spatial_boxes = h._xyxy(full_readout["boxes"])
                arms["S_Pair"] = {"boxes": spatial_boxes.copy(), "interval": list(case["Native_interval"])}
                arms["TS_Pair"] = {"boxes": spatial_boxes.copy(), "interval": interval}
                receipts["Pair"] = {
                    "actual_fit_count": 1,
                    "status": "actual_fit_full_redecode",
                    "fit_invocations": 1,
                    "accepted_steps": int(fitted["optimizer_steps"]),
                    "proposed_steps": int(fitted.get("proposed_optimizer_steps", fitted["optimizer_steps"])),
                    "config": config,
                    "seed_key": fit_key,
                    "reference_sha256": sha(BASE / "REFERENCE_LOCK.json"),
                }
                total_fits += 1
                total_steps += int(fitted["optimizer_steps"])
                del batch, base, native_records, frames
                gc.collect()
                torch.cuda.empty_cache()

        payload = {
            "schema": "stvg-crc-student-case-v1",
            "case_id": cid,
            "dataset": rec["dataset"],
            "source_case_sha256": case["source_case_sha256"],
            "source_file_sha256": sha(source_path),
            "source_model_state_sha256": case["source_model_state_sha256"],
            "frame_ids": list(map(int, case["frame_ids"])),
            "pixel_sha256": case["pixel_sha256"],
            "arms": arms,
            "fits": fits,
            "readouts": readouts,
            "receipts": receipts,
            "reference_records": {"Pair": pair_ref},
            "historical_prediction_sha256": sha(parent_path),
            "temporal_file_sha256": sha(PARENT / "temporal" / f"{cid}.pt"),
            "reference_lock_sha256": sha(BASE / "REFERENCE_LOCK.json"),
            "teacher_barrier_sha256": sha(BASE / "TEACHER_GLOBAL_BARRIER.json"),
            "parent_student_sha256": sha(parent_path),
            "source_capture_seconds": source_capture_s,
            "source_restored": True,
            "GT_read": False,
        }
        file_item = save_pt(dest, payload)
        file_item["actual_fit_count"] = int(receipts["Pair"]["actual_fit_count"])
        write(receipt_path, {
            "schema": "stvg-crc-student-receipt-v1",
            "status": "complete",
            "case_id": cid,
            "sha256": sha(dest),
            "file": file_item,
            "reference_lock_sha256": sha(BASE / "REFERENCE_LOCK.json"),
            "teacher_barrier_sha256": sha(BASE / "TEACHER_GLOBAL_BARRIER.json"),
            "GT_read": False,
        })
        files.append(file_item)
        write(BASE / "STUDENT_STATUS.json", {
            "status": "running", "done": len(files), "total": 32,
            "actual_fit_count": total_fits, "exact_historical_reuse_count": total_reuses,
            "accepted_optimizer_steps": total_steps, "GT_read": False,
        }, mutable=True)
        print("CRC student", cid, receipts["Pair"]["status"], flush=True)
        del payload, parent, temporal
        gc.collect()

    if len(files) != 32:
        raise RuntimeError("CRC student did not produce all 32 locked case outputs")
    write(BASE / "STUDENT_GLOBAL_BARRIER.json", {
        "schema": "stvg-crc-student-global-barrier-v1",
        "status": "complete", "count": 32, "files": files,
        "actual_fit_count": total_fits, "exact_historical_reuse_count": total_reuses,
        "accepted_optimizer_steps": total_steps,
        "reference_lock_sha256": sha(BASE / "REFERENCE_LOCK.json"),
        "teacher_barrier_sha256": sha(BASE / "TEACHER_GLOBAL_BARRIER.json"),
        "runtime_sha256": sha(BASE / "RUNTIME.json"),
        "wall_seconds": time.perf_counter() - started, "GT_read": False,
    })


if __name__ == "__main__":
    main()
