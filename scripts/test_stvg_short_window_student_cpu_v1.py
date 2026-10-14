"""Synthetic CPU-only contracts for the finite short-window student worker."""
from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stvg_short_window_student_v1 as worker


def _fixture():
    case_id = "hc2_000"
    centers = [
        {"slot": slot, "source_grid_position": slot * 2,
         "center_physical_frame_id": physical}
        for slot, physical in enumerate((100, 200, 300, 400))
    ]
    tracklets = []
    for center in centers:
        for candidate_id in (1, 2, 3):
            tracklets.append({
                "tracklet_id": f"{case_id}:center{center['center_physical_frame_id']}:proposal{candidate_id}",
                "candidate_id": candidate_id,
                "center_physical_frame_id": center["center_physical_frame_id"],
            })

    def anchor(slot, candidate_id=2):
        physical = centers[slot]["center_physical_frame_id"]
        return {
            "position": centers[slot]["source_grid_position"],
            "physical_frame_id": physical,
            "box_cxcywh": [0.5, 0.5, 0.2, 0.3],
            "candidate_id": candidate_id,
            "selected_candidate_id": candidate_id,
            "tracklet_id": f"{case_id}:center{physical}:proposal{candidate_id}",
            "teacher_job_id": f"{case_id}_window{slot:02d}_base",
            "teacher_output_sha256": "d" * 64,
            "is_Source": False,
            "association_support": {"mutual_nearest": True} if slot == 0 else None,
        }

    refs = {
        "local_R1": {
            "no_update": False, "eligible_for_update": True, "reason": "synthetic",
            "selected_candidate_id": None, "selected_candidate_id_is_case_constant": False,
            "selected_candidate_ids_by_position": {"0": 2, "4": 2},
            "anchors": [anchor(0), anchor(2)],
        },
        "associated_R1": {
            "no_update": False, "eligible_for_update": True, "reason": "synthetic",
            "selected_candidate_id": 2, "selected_candidate_id_is_case_constant": True,
            "selected_candidate_ids_by_position": {"0": 2},
            "anchors": [anchor(0)],
        },
    }
    record = {
        "case_id": case_id, "GT_read": False, "references": refs,
    }
    return case_id, centers, tracklets, record, anchor


def _base_teacher_jobs(case_id: str):
    return {
        f"{case_id}_window{slot:02d}_base": {
            "job_id": f"{case_id}_window{slot:02d}_base",
            "case_id": case_id,
            "center_slot": slot,
            "kind": "base",
        }
        for slot in range(4)
    }


def _top_level_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    return set(imports)


def _calls_private_old_fit(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "_fit_one":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "_fit_one":
                return True
    return False


def _synthetic_locked_documents():
    ids = list(worker.CASE_IDS)
    manifest_cases = []
    tracklet_cases = []
    references = []
    roster = []
    digests = {
        str(worker.MANIFEST_PATH): "1" * 64,
        str(worker.TRACKLET_BARRIER_PATH): "2" * 64,
        str(worker.FEATURE_BARRIER_PATH): "3" * 64,
        str(worker.TEACHER_BARRIER_PATH): "4" * 64,
        str(worker.TEACHER_JOB_LOCK_PATH): "5" * 64,
        str(worker.REFERENCE_LOCK_PATH): "6" * 64,
        str(worker.ROSTER_PATH): "7" * 64,
    }
    for case_no, cid in enumerate(ids):
        dataset = "hc2" if cid.startswith("hc2_") else "vidstg"
        model_sha = f"{case_no + 1:x}" * 64
        source_sha = f"{case_no + 9:x}" * 64
        track_sha = f"{case_no + 17:x}" * 64
        rgb_sha = f"{case_no + 25:x}" * 64
        track_path = f"artifacts/stvg_short_event_tracklets_v1/tracklets/{cid}.json"
        rgb_path = f"artifacts/stvg_short_event_tracklets_v1/private_rgb/{cid}.npz"
        digests[str(worker.ROOT / track_path)] = track_sha
        digests[str(worker.ROOT / rgb_path)] = rgb_sha
        centers = []
        tracklets = []
        for slot in range(4):
            physical = 100 + slot
            proposal_rows = []
            for candidate_id in (1, 2):
                tracklet_id = f"{cid}:center{physical}:proposal{candidate_id}"
                tracklet = {
                    "tracklet_id": tracklet_id,
                    "candidate_id": candidate_id,
                    "center_physical_frame_id": physical,
                }
                proposal_rows.append(tracklet)
                tracklets.append(tracklet)
            centers.append({
                "slot": slot,
                "source_grid_position": slot,
                "center_physical_frame_id": physical,
                "matched_tracklets": proposal_rows,
            })
        manifest_case = {
            "case_id": cid,
            "dataset": dataset,
            "source_case_sha256": source_sha,
            "source_model_state_sha256": model_sha,
            "source_grid_count": 4,
            "source_grid_frame_ids_sha256": "8" * 64,
            "canonical_request_sha256": "9" * 64,
            "canonical_video_sha256": "a" * 64,
            "GT_read": False,
            "centers": centers,
        }
        manifest_cases.append(manifest_case)
        tracklet_cases.append({
            "case_id": cid, "path": track_path, "sha256": track_sha,
            "rgb_npz_path": rgb_path, "rgb_npz_sha256": rgb_sha,
        })
        refs = {}
        for role in worker.ROLES:
            has_anchor = case_no == 0 and role == "local_R1"
            anchor = ({
                "position": 0,
                "physical_frame_id": 100,
                "box_cxcywh": [0.5, 0.5, 0.2, 0.3],
                "selected_candidate_id": 2,
                "candidate_id": 2,
                "tracklet_id": f"{cid}:center100:proposal2",
                "teacher_job_id": f"{cid}_window00_base",
                "teacher_output_sha256": "d" * 64,
                "is_Source": False,
                **({"association_support": {"mutual_nearest": True}}
                   if role == "associated_R1" else {}),
            } if has_anchor else None)
            refs[role] = {
                "no_update": not has_anchor,
                "eligible_for_update": has_anchor,
                "reason": "synthetic",
                "selected_candidate_id": 2 if has_anchor else None,
                "selected_candidate_id_is_case_constant": True,
                "selected_candidate_ids_by_position": {"0": 2} if has_anchor else {},
                "anchors": [anchor] if has_anchor else [],
            }
        references.append({
            "case_id": cid,
            "dataset": dataset,
            "source_case_sha256": source_sha,
            "source_model_state_sha256": model_sha,
            "manifest_case_sha256": worker._canonical_sha(manifest_case),
            "tracklet_case_sha256": track_sha,
            "canonical_request_sha256": "9" * 64,
            "canonical_video_sha256": "a" * 64,
            "rgb_npz_sha256": rgb_sha,
            "references": refs,
            "GT_read": False,
        })
        roster.append({
            "case_id": cid, "dataset": dataset,
            "source_case_path": f"artifacts/stvg_mllm_candidate_feedback_v1/source_cases/{cid}.pt",
            "source_case_sha256": source_sha,
            "cohort": "old8",
        })
    manifest = {
        "status": "prepared_manifest_not_executed", "GT_read": False,
        "cases": manifest_cases,
    }
    tracklet_barrier = {
        "status": "complete", "count": 8, "case_ids": ids,
        "case_records": tracklet_cases, "GPU_used": False,
        "model_loaded": False, "teacher_called": False, "GT_read": False,
    }
    feature_barrier = {
        "status": "sealed", "count": 8,
        "files": {cid: {"sha256": "c" * 64} for cid in ids}, "GT_read": False,
    }
    teacher_barrier = {
        "status": "complete", "count": 64,
        "files": [{"path": f"synthetic/{i}.json", "sha256": "d" * 64, "bytes": 1}
                  for i in range(64)],
        "GT_read": False,
    }
    teacher_job_lock = {
        "status": "locked", "count": 64,
        "records": [
            {
                "job_id": f"{cid}_window{slot:02d}_{kind}",
                "case_id": cid,
                "center_slot": slot,
                "kind": kind,
            }
            for cid in ids
            for slot in range(4)
            for kind in ("base", "id_rotation")
        ],
        "GT_read": False,
    }
    reference_lock = {
        "schema": "stvg-short-window-reference-lock-v1",
        "status": "locked", "count": 8,
        "manifest_sha256": digests[str(worker.MANIFEST_PATH)],
        "tracklet_barrier_sha256": digests[str(worker.TRACKLET_BARRIER_PATH)],
        "feature_barrier_sha256": digests[str(worker.FEATURE_BARRIER_PATH)],
        "teacher_global_barrier_sha256": digests[str(worker.TEACHER_BARRIER_PATH)],
        "teacher_job_lock_sha256": digests[str(worker.TEACHER_JOB_LOCK_PATH)],
        "records": references, "GT_read": False,
    }
    from scripts.stvg_video_global_spatial_fit_fp32_geometry001 import SOURCE_BINDINGS
    pin_paths = set(worker.REQUIRED_STATIC_PINS) | set(worker._old_t_pin_paths())
    pin_paths.add("scripts/stvg_video_global_spatial_fit_fp32_geometry001.py")
    pin_paths.update(SOURCE_BINDINGS)
    pins = {rel: "e" * 64 for rel in pin_paths}
    protected = {"methods/CURRENT_METHOD.json": "f" * 64}
    runtime = {
        "status": "locked", "GT_read": False,
        "reference_lock_sha256": digests[str(worker.REFERENCE_LOCK_PATH)],
        "steps": 5, "active_parameters": 1792, "expected_fit_count": 1,
        "pins": pins, "protected": protected,
    }
    docs = {
        worker.ROSTER_PATH.name: {"status": "locked", "GT_read": False, "records": roster},
        worker.MANIFEST_PATH.name: manifest,
        worker.TRACKLET_BARRIER_PATH.name: tracklet_barrier,
        worker.FEATURE_BARRIER_PATH.name: feature_barrier,
        worker.TEACHER_BARRIER_PATH.name: teacher_barrier,
        worker.TEACHER_JOB_LOCK_PATH.name: teacher_job_lock,
        worker.REFERENCE_LOCK_PATH.name: reference_lock,
        worker.RUNTIME_PATH.name: runtime,
    }
    return docs, digests, pins, protected


def run_contracts() -> dict:
    checks = []

    # Import-time safety: model/video frameworks are only imported inside the
    # explicitly launched worker, never while constructing the module.
    top = _top_level_imports(Path(worker.__file__))
    forbidden = {"torch", "transformers", "cv2", "decord"}
    assert not (top & forbidden), f"top-level worker imports forbidden runtimes: {top & forbidden}"
    assert not _calls_private_old_fit(Path(worker.__file__)), "worker called the old fit algorithm"
    checks.append("inert_import_no_model_video_frameworks")

    case_id, centers, tracklets, record, anchor = _fixture()
    count = worker._validate_reference_row(
        record, case_id=case_id, frame_ids=list(range(7)), centers=centers, tracklets=tracklets,
        teacher_output_digests={"d" * 64},
        teacher_jobs=_base_teacher_jobs(case_id),
    )
    assert count == 2
    checks.append("two_nonempty_roles_count_two_fits")

    # `position` is already the source-grid index. Confirm it maps to the
    # physical frame at the selected QA center before calling OPD.
    torch = importlib.import_module("torch")
    assert not torch.cuda.is_initialized()
    mapped = worker._anchors_from_locked_positions(
        record["references"]["local_R1"], centers, [100, 150, 200, 250, 300, 350, 400], torch,
    )
    assert [item["position"] for item in mapped] == [0, 4]
    assert [item["physical_frame_id"] for item in mapped] == [100, 300]
    checks.append("source_grid_positions_map_to_locked_physical_centers")

    # Anchors are valid at slots 0 and 2; duplicate slots and Source-only
    # candidate 1 are rejected before any optimizer can be called.
    bad_duplicate = json.loads(json.dumps(record))
    bad_duplicate["references"]["local_R1"]["anchors"].append(anchor(0, 3))
    try:
        worker._validate_reference_row(
            bad_duplicate, case_id=case_id, frame_ids=list(range(7)), centers=centers, tracklets=tracklets,
            teacher_output_digests={"d" * 64},
            teacher_jobs=_base_teacher_jobs(case_id),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate center anchor was accepted")
    bad_source = json.loads(json.dumps(record))
    bad_source["references"]["local_R1"]["anchors"][0] = anchor(0, 1)
    try:
        worker._validate_reference_row(
            bad_source, case_id=case_id, frame_ids=list(range(7)), centers=centers, tracklets=tracklets,
            teacher_output_digests={"d" * 64},
            teacher_jobs=_base_teacher_jobs(case_id),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Source candidate was accepted as an update reference")
    checks.append("duplicate_and_source_references_rejected")

    empty_record = json.loads(json.dumps(record))
    for role in worker.ROLES:
        empty_record["references"][role].update(
            no_update=True, eligible_for_update=False, selected_candidate_id=None, anchors=[],
        )
    empty_count = worker._validate_reference_row(
        empty_record, case_id=case_id, frame_ids=[], centers=centers, tracklets=tracklets,
        teacher_output_digests={"d" * 64},
        teacher_jobs=_base_teacher_jobs(case_id),
    )
    assert empty_count == 0
    native = np.asarray([[0.1, 0.2, 0.8, 0.9]], dtype=np.float32)
    f_arm = {"boxes": native.copy(), "interval": [100, 401]}
    t_arm = {"boxes": native.copy(), "interval": [200, 400]}
    arms, receipts = worker._no_update_arms({"F": f_arm, "T": t_arm}, empty_record["references"])
    assert set(arms) == {"F", "T", "local_R1", "T_local_R1", "associated_R1", "T_associated_R1"}
    for role in worker.ROLES:
        np.testing.assert_array_equal(arms[role]["boxes"], f_arm["boxes"])
        assert arms[role]["interval"] == f_arm["interval"]
        np.testing.assert_array_equal(arms[f"T_{role}"]["boxes"], t_arm["boxes"])
        assert arms[f"T_{role}"]["interval"] == t_arm["interval"]
        assert receipts[role]["optimizer_steps"] == receipts[f"T_{role}"]["optimizer_steps"] == 0
    checks.append("empty_roles_are_exact_F_T_zero_fit_reuse")

    config = {"lr": 0.01, "sigma": 0.025, "tau": 0.05, "steps": 99,
              "writeback": 0.0625, "samples": 32}
    spec = worker._fit_spec(case_id, "local_R1", config, record["references"]["local_R1"])
    assert spec["key"] == "caption_query_fixed_space|hc2_000"
    assert spec["scope"] == "joint" and spec["reward_kind"] == "iou"
    assert spec["steps"] == spec["config"]["steps"] == 5
    assert spec["active_parameters"] == 1792
    assert spec["reference_role"] == "teacher"
    assert spec["reference_sha256"] == worker._canonical_sha(record["references"]["local_R1"])
    associated_spec = worker._fit_spec(
        case_id, "associated_R1", config, record["references"]["associated_R1"],
    )
    assert associated_spec["key"] == spec["key"]
    checks.append("nonempty_fit_spec_is_original_gaussian_iou_5step_1792")

    zero_update_fit = {
        "rounds": [{"step": step, "updated": False, "no_information": True}
                   for step in range(worker.STEPS)],
        "optimizer_steps": 0,
        "declared_steps": worker.STEPS,
        "selected_step": worker.STEPS,
        "active_parameters": worker.ACTIVE_PARAMETERS,
    }
    assert worker._validate_fit_result(zero_update_fit, role="local_R1") == 0
    two_update_fit = json.loads(json.dumps(zero_update_fit))
    two_update_fit["rounds"][1]["updated"] = True
    two_update_fit["rounds"][4]["updated"] = True
    two_update_fit["optimizer_steps"] = 2
    assert worker._validate_fit_result(two_update_fit, role="associated_R1") == 2
    inconsistent_fit = json.loads(json.dumps(two_update_fit))
    inconsistent_fit["optimizer_steps"] = 5
    try:
        worker._validate_fit_result(inconsistent_fit, role="associated_R1")
    except RuntimeError:
        pass
    else:
        raise AssertionError("optimizer steps inconsistent with original round updates were accepted")
    checks.append("nonempty_equal_reward_may_complete_five_rounds_with_zero_updates")

    # The public arm set excludes Direct/motion-loss arms and the history T
    # baseline is the only temporal source; T_* are defined as fit + exact T.
    assert worker.ROLES == ("local_R1", "associated_R1")
    assert worker.STEPS == 5 and worker.ACTIVE_PARAMETERS == 1792
    checks.append("fixed_phase2_arm_and_optimizer_scope")

    # Full preflight with synthetic locked JSON and fake SHA sources. This
    # traverses the same status/schema/cross-binding checks without opening any
    # real roster, lock, teacher output, case tensor, image, or label.
    docs, digests, pins, protected = _synthetic_locked_documents()
    original_barrier = worker.OUT_BARRIER_PATH
    try:
        worker.OUT_BARRIER_PATH = worker.BASE / ".synthetic-student-barrier-absent.json"
        with patch.object(worker, "read", side_effect=lambda path: docs[Path(path).name]), \
             patch.object(worker, "sha", side_effect=lambda path: digests[str(Path(path))]), \
             patch.object(worker, "_sha_path", side_effect=lambda rel: (pins | protected).get(rel)):
            preflight = worker._read_locked_inputs()
        assert len(preflight[0]) == 8 and preflight[4]["expected_fit_count"] == 1
    finally:
        worker.OUT_BARRIER_PATH = original_barrier
    checks.append("complete_synthetic_lock_and_runtime_preflight")

    return {
        "schema": "stvg-short-window-student-cpu-contracts-v1",
        "status": "pass",
        "checks": checks,
        "check_count": len(checks),
        "synthetic_only": True,
        "actual_roster_or_reference_lock_read": False,
        "source_case_or_tracklet_payload_read": False,
        "video_or_RGB_read": False,
        "GT_read": False,
        "checkpoint_or_tokenizer_loaded": False,
        "GPU_used": False,
        "optimizer_executed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_contracts(), sort_keys=True, indent=2))
