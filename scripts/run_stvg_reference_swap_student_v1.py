"""Matched same-frame Weight-free versus original-DINO reference fits.

This worker is intentionally inert on import.  Its executable path consumes
only the locked, label-free common-support reference rows and the already
sealed round2_prompt009 predictions.  It does not load or call the DINO
teacher, and it never reads evaluation annotations.
"""
from __future__ import annotations

import argparse
import collections
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stvg_reference_recovery_common_v1 as common

BASE = ROOT / "artifacts/stvg_reference_recovery_v1/Reference_swap"
PARENT = common.PARENT
OLD = common.OLD
ROSTER_PATH = BASE / "ROSTER_LOCK.json"
REFERENCE_PATH = BASE / "REFERENCE_LOCK.json"
DESIGN_PATH = BASE / "DESIGN_LOCK.json"
RUNTIME_PATH = BASE / "STUDENT_RUNTIME.json"
STUDENT_DIR = BASE / "student"
STUDENT_BARRIER_PATH = BASE / "STUDENT_GLOBAL_BARRIER.json"
DOMAINS = ("hc2", "vidstg")
ARMS = (
    "F", "T", "S_new_crop", "TS_new_crop",
    "S_WF_common", "TS_WF_common", "S_DINO_common", "TS_DINO_common",
)


def _read(path: Path) -> dict[str, Any]:
    result = common.read(path)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path.relative_to(ROOT)}")
    return result


def _cid(record: Mapping[str, Any]) -> str:
    value = record.get("case_id", record.get("probe_alias"))
    if not isinstance(value, str) or not value:
        raise ValueError("locked record lacks case_id/probe_alias")
    return value


def _assert_locked_gt_free(doc: Mapping[str, Any], label: str, *, count: int | None = None) -> None:
    if doc.get("status") != "locked" or doc.get("GT_read") is not False:
        raise ValueError(f"{label} must be locked and explicitly GT-free")
    if count is not None and int(doc.get("count", len(doc.get("records", [])))) != count:
        raise ValueError(f"{label} count must be {count}")


def _verify_runtime() -> dict[str, Any]:
    runtime = _read(RUNTIME_PATH)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("STUDENT_RUNTIME must be locked and explicitly GT-free")
    pins = runtime.get("pins")
    protected = runtime.get("protected", {})
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise ValueError("STUDENT_RUNTIME pins/protected must be mappings")
    for rel, expected in {**pins, **protected}.items():
        path = (ROOT / rel).resolve()
        if not path.is_file() or common.sha(path) != expected:
            raise ValueError(f"STUDENT_RUNTIME source hash mismatch: {rel}")
    return runtime


def _parent_prediction_files() -> dict[str, dict[str, Any]]:
    """Validate the immutable parent barrier and return its 32 prediction paths."""
    barrier_path = PARENT / "STUDENT_GLOBAL_BARRIER.json"
    barrier = _read(barrier_path)
    if barrier.get("status") != "complete" or barrier.get("GT_read") is not False:
        raise ValueError("round2_prompt009 student barrier is not complete/GT-free")
    if int(barrier.get("count", -1)) != 32:
        raise ValueError("round2_prompt009 barrier must contain exactly 32 predictions")
    allowed_metadata = {
        str((PARENT / name).relative_to(ROOT))
        for name in (
            "REFERENCE_LOCK.json", "TEACHER_GLOBAL_BARRIER.json", "TEACHER_JOB_LOCK.json",
            "CANDIDATE_INPUT_BARRIER.json", "ROOT_INPUT_READBACK.json",
            "ROOT_TEMPORAL_READBACK.json", "ROOT_SPATIAL_READBACK.json", "RUNTIME.json",
        )
    }
    allowed_metadata.update({
        str((OLD / "DEVELOPMENT_ROSTER_LOCK.json").relative_to(ROOT)),
        str((OLD / "TEMPORAL_SELECTION_LOCK.json").relative_to(ROOT)),
        str((OLD / "DESIGN_LOCK.json").relative_to(ROOT)),
    })
    result: dict[str, dict[str, Any]] = {}
    for item in barrier.get("files", []):
        rel = item.get("path")
        if not isinstance(rel, str):
            raise ValueError("parent barrier contains a nonstring path")
        path = (ROOT / rel).resolve()
        if not path.is_file() or common.sha(path) != item.get("sha256"):
            raise ValueError(f"parent barrier SHA mismatch: {rel}")
        if item.get("bytes") is not None and path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"parent barrier byte-count mismatch: {rel}")
        if not rel.endswith(".pt"):
            if rel not in allowed_metadata:
                raise ValueError(f"parent barrier contains unapproved metadata: {rel}")
            continue
        cid = path.stem
        expected_path = (PARENT / "student" / f"{cid}.pt").resolve()
        if path != expected_path or cid in result:
            raise ValueError(f"unexpected or duplicate parent prediction: {rel}")
        result[cid] = {"path": path, "sha256": item["sha256"], "bytes": path.stat().st_size}
    if len(result) != 32:
        raise ValueError("parent barrier must list exactly 32 unique student predictions")
    return result


def _parent_reference_index() -> dict[str, dict[str, Any]]:
    doc = _read(PARENT / "REFERENCE_LOCK.json")
    _assert_locked_gt_free(doc, "parent Weight-free reference lock", count=32)
    records = doc.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise ValueError("parent Weight-free reference lock must contain 32 records")
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        cid = str(row.get("case_id", ""))
        new_crop = row.get("new_crop")
        if (not cid or cid in result or not isinstance(new_crop, dict)
                or not isinstance(new_crop.get("anchors"), list)):
            raise ValueError("malformed or duplicate parent new_crop reference")
        result[cid] = row
    return result


def _box_is_valid(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        cx, cy, width, height = map(float, box)
    except (TypeError, ValueError):
        return False
    return (all(math.isfinite(v) for v in (cx, cy, width, height))
            and width > 0 and height > 0
            and 0 <= cx - width / 2 < cx + width / 2 <= 1
            and 0 <= cy - height / 2 < cy + height / 2 <= 1)


def _anchor_positions(anchors: Any, frame_ids: list[int], label: str) -> list[int]:
    if not isinstance(anchors, list):
        raise ValueError(f"{label} must be a list")
    positions: list[int] = []
    for row in anchors:
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a non-object anchor")
        pos = row.get("position")
        fid = row.get("physical_frame_id")
        weight = row.get("weight")
        if (isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < len(frame_ids)
                or isinstance(fid, bool) or not isinstance(fid, int) or fid != frame_ids[pos]):
            raise ValueError(f"{label} has an invalid source-grid/physical-frame binding")
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight)) or float(weight) <= 0):
            raise ValueError(f"{label} has a nonpositive/nonfinite reliability weight")
        if not _box_is_valid(row.get("box_cxcywh")):
            raise ValueError(f"{label} has invalid normalized cxcywh geometry")
        positions.append(pos)
    if positions != sorted(set(positions)):
        raise ValueError(f"{label} positions must be strictly increasing and unique")
    return positions


def _validate_common_record(record: Mapping[str, Any], parent_ref: Mapping[str, Any],
                            frame_ids: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wf = record.get("common_wf_anchors")
    dino = record.get("common_dino_anchors")
    wf_positions = _anchor_positions(wf, frame_ids, "common_wf_anchors")
    dino_positions = _anchor_positions(dino, frame_ids, "common_dino_anchors")
    if wf_positions != dino_positions:
        raise ValueError("WF and DINO common support positions differ")
    for wf_row, dino_row in zip(wf, dino):
        for key in ("position", "physical_frame_id", "weight"):
            if wf_row[key] != dino_row[key]:
                raise ValueError(f"WF/DINO matched support differs at {key}")

    parent_anchors = parent_ref.get("new_crop", {}).get("anchors", [])
    _anchor_positions(parent_anchors, frame_ids, "parent new_crop anchors")
    parent_by_position = {int(row["position"]): row for row in parent_anchors}
    for row in wf:
        pos = int(row["position"])
        old = parent_by_position.get(pos)
        if old is None:
            raise ValueError("common WF anchor is not present in the original full-pool reference")
        if (int(old["physical_frame_id"]) != int(row["physical_frame_id"])
                or float(old.get("weight", 1.0)) != float(row["weight"])
                or list(old["box_cxcywh"]) != list(row["box_cxcywh"])):
            raise ValueError("common WF anchor differs from its original full-pool anchor")
        if ("original_wf_candidate_id" in row
                and row["original_wf_candidate_id"] != old.get("candidate_id")):
            raise ValueError("common WF anchor candidate ID differs from the parent reference")
        if ("original_wf_teacher_output_sha256" in row
                and row["original_wf_teacher_output_sha256"] != old.get("teacher_output_sha256")):
            raise ValueError("common WF anchor teacher-output hash differs from the parent reference")

    rejected = record.get("dino_rejected_positions", [])
    unknown = record.get("dino_unknown_positions", [])
    if not isinstance(rejected, list) or not isinstance(unknown, list):
        raise ValueError("DINO rejected/unknown position summaries must be lists")
    common_set = set(wf_positions)
    if common_set.intersection(map(int, rejected)) or common_set.intersection(map(int, unknown)):
        raise ValueError("DINO rejected/unknown positions entered the common support")
    decisions = record.get("dino_decisions")
    if decisions is not None:
        if not isinstance(decisions, list):
            raise ValueError("dino_decisions must be a list when present")
        decision_positions: list[int] = []
        accepted: dict[int, Mapping[str, Any]] = {}
        actual_rejected: set[int] = set()
        actual_unknown: set[int] = set()
        for item in decisions:
            pos = item.get("position")
            fid = item.get("physical_frame_id")
            status = item.get("status")
            if (isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < len(frame_ids)
                    or isinstance(fid, bool) or not isinstance(fid, int) or fid != frame_ids[pos]
                    or status not in ("accepted", "rejected", "unknown")):
                raise ValueError("DINO decision has malformed position/frame/status metadata")
            decision_positions.append(pos)
            if status == "accepted":
                accepted[pos] = item
            elif status == "rejected":
                actual_rejected.add(pos)
            else:
                actual_unknown.add(pos)
            if status != "accepted" and pos in common_set:
                raise ValueError("a rejected/unknown DINO decision entered common support")
        if len(decision_positions) != len(set(decision_positions)):
            raise ValueError("DINO decisions contain duplicate positions")
        if not common_set.issubset(accepted):
            raise ValueError("a nonaccepted DINO observation entered the common support")
        if set(map(int, rejected)) != actual_rejected or set(map(int, unknown)) != actual_unknown:
            raise ValueError("DINO rejected/unknown summaries differ from per-position decisions")
        decision_by_pos = accepted
        for anchor in dino:
            decision = decision_by_pos[int(anchor["position"])]
            selected = decision.get("selected_box_cxcywh")
            digest = decision.get("output_sha256")
            if "teacher_output_path" in decision:
                output_path = ((BASE if decision["teacher_output_path"].startswith("teacher/") else ROOT) / decision["teacher_output_path"]).resolve()
                if output_path.parent != (BASE / "teacher").resolve():
                    raise ValueError("DINO decision escapes its sealed teacher directory")
                digest = decision["teacher_output_sha256"]
                if common.sha(output_path) != digest:
                    raise ValueError("DINO teacher output SHA mismatch")
                observed = _read(output_path)
                if (observed.get("decision_status") != "accepted"
                        or observed.get("job_id") != decision["job_id"]
                        or observed.get("position") != anchor["position"]
                        or observed.get("physical_frame_id") != anchor["physical_frame_id"]
                        or observed.get("GT_read") is not False):
                    raise ValueError("DINO sealed output decision/frame mismatch")
                selected = observed["selected_box_cxcywh"]
            if selected != anchor.get("box_cxcywh"):
                raise ValueError("DINO common geometry differs from its accepted teacher decision")
            if digest is not None and anchor.get("teacher_output_sha256") != digest:
                raise ValueError("DINO common anchor is not bound to its accepted teacher output")
    return wf, dino


def _locked_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]],
                               dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    runtime = _verify_runtime()
    design = _read(DESIGN_PATH)
    roster_doc = _read(ROSTER_PATH)
    reference_doc = _read(REFERENCE_PATH)
    _assert_locked_gt_free(design, "E1 design lock")
    _assert_locked_gt_free(roster_doc, "E1 roster lock", count=32)
    _assert_locked_gt_free(reference_doc, "E1 reference lock", count=32)
    if bool(design.get("GT_used_for_selection", False)):
        raise ValueError("E1 design lock cannot record GT-based selection")
    roster = roster_doc.get("records")
    references = reference_doc.get("records")
    if not isinstance(roster, list) or len(roster) != 32 or not isinstance(references, list) or len(references) != 32:
        raise ValueError("E1 roster/reference membership must each contain 32 records")
    roster_index = {_cid(row): row for row in roster}
    reference_index = {str(row.get("case_id", "")): row for row in references}
    if len(roster_index) != 32 or len(reference_index) != 32 or set(roster_index) != set(reference_index):
        raise ValueError("E1 roster and reference case membership differ")
    parent_refs = _parent_reference_index()
    if set(parent_refs) != set(roster_index):
        raise ValueError("E1 case membership differs from the parent Weight-free scope")
    parent_files = _parent_prediction_files()
    for cid, row in roster_index.items():
        ref = reference_index[cid]
        if ref.get("dataset") != row.get("dataset") or row.get("dataset") not in DOMAINS:
            raise ValueError(f"E1 dataset mismatch for {cid}")
        if not isinstance(ref.get("source_case_sha256"), str) or len(ref["source_case_sha256"]) != 64:
            raise ValueError(f"E1 reference lock lacks a source-case hash for {cid}")
        if cid not in parent_files:
            raise ValueError(f"missing sealed parent student prediction for {cid}")
        parent_ref = parent_refs[cid]
        if parent_ref.get("dataset") != row.get("dataset"):
            raise ValueError(f"parent reference dataset mismatch for {cid}")
        if (parent_ref.get("source_case_sha256") is not None
                and parent_ref.get("source_case_sha256") != ref.get("source_case_sha256")):
            raise ValueError(f"parent reference source SHA mismatch for {cid}")
    return runtime, roster, reference_index, parent_refs, parent_files


def _full_readout(h, model, batch, frame_ids, records, fit, query, core, torch,
                  transient_native_masks, frozen_visual_host, frozen_attention_storage,
                  inactive_modules_on_host, query_subject):
    return h._readout(
        model=model,
        batch=batch,
        frame_ids=frame_ids,
        records=records,
        fit=fit,
        query=query,
        core=core,
        torch=torch,
        transient_native_masks=transient_native_masks,
        frozen_visual_host=frozen_visual_host,
        frozen_attention_storage=frozen_attention_storage,
        inactive_modules_on_host=inactive_modules_on_host,
        query_subject=query_subject,
    )


def _fit_reference(role: str, raw_anchors: list[dict[str, Any]], base, initial,
                   frame_ids: list[int], dataset: str, query_key: str,
                   reference_sha: str, h, torch, fitter, locked_config,
                   case_id: str) -> dict[str, Any]:
    anchors = h._anchors({"anchors": raw_anchors}, frame_ids, torch)
    cfg = locked_config(dataset, "current5")
    role_label = ("weight_free_common_same_frame_geometry" if role == "WF_common"
                  else "original_dino_same_frame_geometry")
    checkpoint_dir = BASE / "partial" / case_id / role
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    def checkpoint(value: Mapping[str, Any]) -> None:
        item = copy.deepcopy(dict(value))
        item["reference_role"] = role_label
        item["reference_sha256"] = reference_sha
        item["case_id"] = case_id
        item["GT_read"] = False
        common.save_pt(checkpoint_dir / f"step{int(value['next_step'])}.pt", item)

    fit = fitter(
        base=base,
        initial={name: tensor.clone() for name, tensor in initial.items()},
        anchors=anchors,
        frame_ids=frame_ids,
        key=query_key,
        config=cfg,
        profile="current5",
        dataset=dataset,
        validate=True,
        reference_sha256=reference_sha,
        checkpoint_callback=checkpoint,
    )
    if fit.get("GT_read") is not False or fit.get("profile") != "current5":
        raise ValueError("frozen fitter returned an unexpected profile/GT declaration")
    # The frozen helper labels every input as the Weight-free teacher.  This
    # worker corrects only the descriptive metadata; fit math is unchanged.
    fit["reference_role"] = role_label
    fit["candidate_ids"] = [row.get("candidate_id") for row in raw_anchors]
    fit["reference_support_count"] = len(raw_anchors)
    return fit


def _run() -> dict[str, Any]:
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
    from scripts.stvg_reference_recovery_fit_v1 import fit_reference_recovery, locked_config

    runtime, roster, ref_index, parent_ref_index, parent_files = _locked_inputs()
    sys.addaudithook(common.guard)
    install_clean_loader()
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    reference_sha = common.sha(REFERENCE_PATH)
    runtime_sha = common.sha(RUNTIME_PATH)
    if STUDENT_BARRIER_PATH.exists():
        raise FileExistsError("E1 student global barrier already exists; refusing to overwrite")
    STUDENT_DIR.mkdir(parents=True, exist_ok=True)
    for rec in roster:
        cid = _cid(rec)
        output = STUDENT_DIR / f"{cid}.pt"
        if output.exists() or output.with_suffix(".receipt.json").exists():
            raise FileExistsError(f"existing E1 student artifact blocks fresh run: {cid}")

    parent_ref_lock = _read(PARENT / "REFERENCE_LOCK.json")
    parent_ref_lock_sha = common.sha(PARENT / "REFERENCE_LOCK.json")
    if parent_ref_lock.get("status") != "locked" or parent_ref_lock.get("GT_read") is not False:
        raise ValueError("parent Weight-free reference lock changed after preflight")
    if parent_ref_lock.get("count") != 32:
        raise ValueError("parent Weight-free reference lock no longer has 32 cases")

    original_roster = {row["probe_alias"]: row for row in _read(ROOT / "artifacts/stvg_reference_recovery_v1/Optimizer_replay/ROSTER_LOCK.json")["records"]}
    model = current_source = None
    files: list[dict[str, Any]] = []
    fit_count = optimizer_steps = full_readout_calls = 0
    started = time.perf_counter()
    for rec in roster:
        cid = _cid(rec)
        dataset = str(rec["dataset"])
        ref = ref_index[cid]
        case, source_path = common.load_case(rec)
        if (case.get("GT_read") is not False or case.get("source_case_sha256") != ref["source_case_sha256"]
                or case.get("dataset") != dataset):
            raise ValueError(f"source case binding mismatch for {cid}")
        frame_ids = list(map(int, case.get("frame_ids", [])))
        if not frame_ids or any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
            raise ValueError(f"source frame grid is malformed for {cid}")
        wf_raw, dino_raw = _validate_common_record(ref, parent_ref_index[cid], frame_ids)

        parent_path = parent_files[cid]["path"]
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        parent_sha = common.sha(parent_path)
        if (parent.get("GT_read") is not False or parent.get("case_id") != cid
                or parent.get("dataset") != dataset
                or parent.get("source_case_sha256") != case["source_case_sha256"]):
            raise ValueError(f"sealed parent prediction identity mismatch for {cid}")
        parent_arms = parent.get("arms", {})
        for name in ("F", "T", "S_new_crop", "TS_new_crop"):
            if name not in parent_arms:
                raise ValueError(f"parent prediction lacks required arm {name} for {cid}")
        arms = {
            "F": copy.deepcopy(parent_arms["F"]),
            "T": copy.deepcopy(parent_arms["T"]),
            "S_new_crop": copy.deepcopy(parent_arms["S_new_crop"]),
            "TS_new_crop": copy.deepcopy(parent_arms["TS_new_crop"]),
        }
        if (not np.array_equal(np.asarray(arms["S_new_crop"]["boxes"]), np.asarray(arms["TS_new_crop"]["boxes"]))
                or list(arms["S_new_crop"]["interval"]) != list(case["Native_interval"])
                or list(arms["TS_new_crop"]["interval"]) != list(arms["T"]["interval"])):
            raise ValueError(f"parent full-pool spatial/temporal controls are inconsistent for {cid}")

        capture_rec = {"case_id": cid, "dataset": dataset,
                       "source_case_sha256": case["source_case_sha256"]}
        transient = dict(
            case,
            source=source_helper.SOURCE_MODEL_NAME[dataset],
            _source_case_sha256=case["source_case_sha256"],
        )
        _, official_row = source_helper._validate_roster_record(original_roster[cid])
        transient["parent"] = official_row["source"]
        # The legacy capture implementation and its mutable catalog live in
        # the adjacent-motion student module (`h`).  `source_helper` only
        # contributes the locked source-name mapping and roster validator.
        previous_catalog = h.catalog
        try:
            def locked_catalog(requested):
                if any(requested.get(key) != value for key, value in capture_rec.items()):
                    raise ValueError("source capture requested a different locked source case")
                return transient, {}
            h.catalog = locked_catalog
            capture = h._capture_source(
                capture_rec, model, current_source, torch, collections,
                model_for, frames_for, raw_row, frozen_forward, prediction, state_hash,
            )
        finally:
            h.catalog = previous_catalog
        (model, current_source, captured_case, _catalog_data, source_row, frames,
         captured_ids, records, batch, base, native, expected_model_sha,
         pixel_sha, capture_seconds) = capture
        if (captured_case.get("source_case_sha256") != case["source_case_sha256"]
                or captured_ids != frame_ids or pixel_sha != case.get("pixel_sha256")
                or expected_model_sha != case.get("source_model_state_sha256")):
            raise ValueError(f"frozen source capture readback mismatch for {cid}")
        if parent.get("source_model_state_sha256") != expected_model_sha or parent.get("pixel_sha256") != pixel_sha:
            raise ValueError(f"parent source model/pixel binding mismatch for {cid}")
        if (not np.array_equal(np.asarray(arms["F"]["boxes"]), h._xyxy(native["boxes"]))
                or list(arms["F"]["interval"]) != list(case["Native_interval"])
                or not np.array_equal(np.asarray(arms["T"]["boxes"]), np.asarray(arms["F"]["boxes"]))):
            raise ValueError(f"parent F/T geometry does not match the captured Source for {cid}")

        source_initial = {key: tensor.detach().cpu().clone() for key, tensor in base.initial.items()}
        if not wf_raw:
            # Equal common-support maps imply both new fits are absent.  Keep
            # explicit no-update arms as exact F/T copies; the original full
            # Weight-free new_crop arms above remain a separate control.
            arms["S_WF_common"] = copy.deepcopy(arms["F"])
            arms["TS_WF_common"] = copy.deepcopy(arms["T"])
            arms["S_DINO_common"] = copy.deepcopy(arms["F"])
            arms["TS_DINO_common"] = copy.deepcopy(arms["T"])
            fits = {"WF_common": None, "DINO_common": None}
            readouts = {"WF_common": None, "DINO_common": None}
            readout_call_count = 0
        else:
            query_key = stable_fit_key(dataset, str(rec["query_key_sha256"]))
            cfg = locked_config(dataset, "current5")
            fits, readouts = {}, {}
            for role, raw_anchors in (("WF_common", wf_raw), ("DINO_common", dino_raw)):
                base.restore(source_initial)
                fit = _fit_reference(role, raw_anchors, base, source_initial, frame_ids,
                                     dataset, query_key, reference_sha, h, torch,
                                     fit_reference_recovery, locked_config, cid)
                if (fit.get("seed_key") != query_key or fit.get("config") != cfg
                        or fit.get("validation_enabled") is not True
                        or fit.get("optimizer_state_reset_each_query") is not True):
                    raise ValueError(f"frozen fit config/seed/reset mismatch for {cid}/{role}")
                base.restore(source_initial)
                out = _full_readout(
                    h, model, batch, frame_ids, records, fit, source_row["parses"]["subject"],
                    core, torch, transient_native_masks, frozen_visual_host,
                    frozen_attention_storage, inactive_modules_on_host, query_subject,
                )
                if (state_hash(model.state_dict()) != expected_model_sha
                        or not h._same_logits(out["logits"], native["logits"])
                        or not torch.equal(out["boxes"].detach().cpu(), fit["final_cached_decoder_boxes"].detach().cpu())):
                    raise ValueError(f"full-model readout/source restoration mismatch for {cid}/{role}")
                fit["full_model_readout_calls"] = 1
                fit["source_model_state_sha256"] = expected_model_sha
                fits[role] = fit
                readouts[role] = out
                s_name = "S_WF_common" if role == "WF_common" else "S_DINO_common"
                ts_name = "TS_WF_common" if role == "WF_common" else "TS_DINO_common"
                arms[s_name] = {"boxes": h._xyxy(out["boxes"]), "interval": list(arms["F"]["interval"])}
                arms[ts_name] = {"boxes": h._xyxy(out["boxes"]), "interval": list(arms["T"]["interval"])}
                fit_count += 1
                optimizer_steps += int(fit["optimizer_steps"])
                readout_call_count += 1
            if state_hash(model.state_dict()) != expected_model_sha:
                raise ValueError(f"Source model was not restored after matched fits for {cid}")

        if set(arms) != set(ARMS):
            raise ValueError(f"student arm set differs from the locked E1 schema for {cid}")
        source_restore_status = "pass" if wf_raw else "pass_source_capture_no_new_fit"
        payload = {
            "schema": "stvg-reference-swap-student-case-v1",
            "case_id": cid,
            "dataset": dataset,
            "arms": arms,
            "fits": fits,
            "readouts": readouts,
            "reference_records": copy.deepcopy(ref),
            "reference_lock_sha256": reference_sha,
            "frame_ids": frame_ids,
            "source_case_sha256": case["source_case_sha256"],
            "source_file_sha256": common.sha(source_path),
            "source_model_state_sha256": expected_model_sha,
            "pixel_sha256": pixel_sha,
            "parent_student_sha256": parent_sha,
            "source_capture_seconds": float(capture_seconds),
            "source_restored": True,
            "source_restore_status": source_restore_status,
            "source_capture_readback": {
                "status": "pass",
                "source_model_state_sha256": expected_model_sha,
                "pixel_sha256": pixel_sha,
                "source_logits_equal_cold_case": True,
                "source_boxes_equal_cold_case": True,
                "GT_read": False,
            },
            "source_cold_readback": copy.deepcopy(parent.get("source_cold_readback")),
            "reference_support": {
                "common_position_count": len(wf_raw),
                "common_positions": [int(row["position"]) for row in wf_raw],
                "empty_common_support": not bool(wf_raw),
                "wf_geometry_source": "parent_new_crop_same_frame_geometry",
                "dino_geometry_source": "locked_original_dino_same_frame_geometry",
            },
            "readout_call_count": readout_call_count,
            "actual_fit_count": int(bool(wf_raw)) * 2,
            "optimizer_steps": sum(int(value["optimizer_steps"]) for value in fits.values() if value is not None),
            "runtime_sha256": runtime_sha,
            "GT_read": False,
        }
        output = STUDENT_DIR / f"{cid}.pt"
        saved = common.save_pt(output, payload)
        receipt = {
            "schema": "stvg-reference-swap-case-receipt-v1",
            "status": "complete",
            "case_id": cid,
            "dataset": dataset,
            "sha256": saved["sha256"],
            "bytes": saved["bytes"],
            "reference_lock_sha256": reference_sha,
            "runtime_sha256": runtime_sha,
            "parent_student_sha256": parent_sha,
            "source_case_sha256": case["source_case_sha256"],
            "actual_fit_count": payload["actual_fit_count"],
            "optimizer_steps": payload["optimizer_steps"],
            "readout_call_count": readout_call_count,
            "source_restored": True,
            "GT_read": False,
        }
        common.write(output.with_suffix(".receipt.json"), receipt)
        files.append(saved)
        full_readout_calls += readout_call_count
        common.write(BASE / "STATUS.json", {
            "schema": "stvg-reference-swap-status-v1",
            "status": "running", "done": len(files), "total": 32,
            "actual_fit_count": fit_count, "optimizer_steps": optimizer_steps,
            "GT_read": False,
        }, mutable=True)
        print(f"reference swap {cid}: common={len(wf_raw)} fits={payload['actual_fit_count']}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    barrier = {
        "schema": "stvg-reference-swap-student-barrier-v1",
        "status": "complete",
        "count": 32,
        "files": files,
        "actual_fit_count": fit_count,
        "optimizer_steps": optimizer_steps,
        "full_model_readout_calls": full_readout_calls,
        "runtime_sha256": runtime_sha,
        "reference_lock_sha256": reference_sha,
        "parent_student_barrier_sha256": common.sha(PARENT / "STUDENT_GLOBAL_BARRIER.json"),
        "wall_seconds": time.perf_counter() - started,
        "GT_read": False,
    }
    common.write(STUDENT_BARRIER_PATH, barrier)
    common.write(BASE / "STATUS.json", {
        "schema": "stvg-reference-swap-status-v1",
        "status": "complete", "done": 32, "total": 32,
        "actual_fit_count": fit_count, "optimizer_steps": optimizer_steps,
        "full_model_readout_calls": barrier["full_model_readout_calls"],
        "student_barrier_sha256": common.sha(STUDENT_BARRIER_PATH),
        "GT_read": False,
    }, mutable=True)
    return barrier


def _self_test() -> dict[str, Any]:
    """Small structural tests; never opens experiment payloads or models."""
    # Verify the real module ownership used by the E2 source-capture ABI.  This
    # imports Python modules only; it does not load a checkpoint or payload.
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    from scripts import run_stvg_table1_source_v1 as source_helper

    if not callable(getattr(h, "_capture_source", None)) or not hasattr(h, "catalog"):
        raise AssertionError("E2 source capture API is not exposed by the helper module")
    if (not callable(getattr(source_helper, "_validate_roster_record", None))
            or not isinstance(getattr(source_helper, "SOURCE_MODEL_NAME", None), dict)):
        raise AssertionError("Table 1 source metadata API is not exposed by its module")
    original_catalog = h.catalog
    catalog_calls: list[dict[str, Any]] = []
    try:
        def synthetic_catalog(requested):
            catalog_calls.append(dict(requested))
            return {"case_id": requested["case_id"]}, {"synthetic": True}

        h.catalog = synthetic_catalog
        got_case, got_meta = h.catalog({"case_id": "CPU_FIXTURE"})
        assert got_case == {"case_id": "CPU_FIXTURE"} and got_meta == {"synthetic": True}
        assert catalog_calls == [{"case_id": "CPU_FIXTURE"}]
    finally:
        h.catalog = original_catalog

    frame_ids = [100, 110, 120]
    wf = [{"position": 1, "physical_frame_id": 110, "box_cxcywh": [0.5, 0.5, 0.2, 0.2],
           "weight": 0.75, "candidate_id": "wf-1"}]
    dino = [{"position": 1, "physical_frame_id": 110, "box_cxcywh": [0.52, 0.5, 0.2, 0.2],
             "weight": 0.75, "candidate_id": "dino-1"}]
    parent = {"new_crop": {"anchors": [copy.deepcopy(wf[0])]}}
    row = {"common_wf_anchors": wf, "common_dino_anchors": dino,
           "dino_rejected_positions": [0], "dino_unknown_positions": [2]}
    assert _validate_common_record(row, parent, frame_ids) == (wf, dino)
    invalid = copy.deepcopy(row)
    invalid["common_dino_anchors"][0]["weight"] = 0.5
    try:
        _validate_common_record(invalid, parent, frame_ids)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched common reliability weight was accepted")
    invalid = copy.deepcopy(row)
    invalid["common_dino_anchors"][0]["position"] = 2
    try:
        _validate_common_record(invalid, parent, frame_ids)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched common support position was accepted")
    empty = {"common_wf_anchors": [], "common_dino_anchors": [],
             "dino_rejected_positions": [0], "dino_unknown_positions": [1, 2]}
    assert _validate_common_record(empty, parent, frame_ids) == ([], [])
    return {"status": "pass", "checks": 7, "GT_read": False,
            "model_loaded": False, "GPU_used": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), sort_keys=True))
        return
    if args.preflight:
        runtime, roster, refs, parent_refs, parent_files = _locked_inputs()
        # CPU-only source-case loading binds sparse physical frame IDs without
        # constructing the model or decoding video.
        sys.addaudithook(common.guard)
        by_id = {_cid(row): row for row in roster}
        summaries = {"cases": len(by_id), "parent_predictions": len(parent_files),
                     "common_support_cases": 0, "empty_common_support_cases": 0,
                     "anchor_pairs": 0}
        for cid, ref in refs.items():
            case, _ = common.load_case(by_id[cid])
            if case.get("source_case_sha256") != ref.get("source_case_sha256") or case.get("GT_read") is not False:
                raise ValueError(f"source case binding mismatch for {cid}")
            frame_ids = list(map(int, case.get("frame_ids", [])))
            wf, dino = _validate_common_record(ref, parent_refs[cid], frame_ids)
            summaries["common_support_cases"] += int(bool(wf))
            summaries["empty_common_support_cases"] += int(not wf)
            summaries["anchor_pairs"] += len(wf)
        print(json.dumps({"status": "pass", "runtime_sha256": common.sha(RUNTIME_PATH),
                          **summaries, "GT_read": False, "model_loaded": False,
                          "GPU_used": False}, sort_keys=True))
        return
    _run()


if __name__ == "__main__":
    main()
