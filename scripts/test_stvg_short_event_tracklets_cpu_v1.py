"""CPU-only synthetic contracts for the short event-local plan builder.

This file never opens videos, decodes RGB, reads labels, runs LK, or loads a
model.  Pass ``--metadata-dryrun`` to additionally call the builder against its
already sealed no-GT metadata inputs; this still does not write a manifest.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "scripts/prepare_stvg_short_event_tracklets_v1.py"
SPEC = importlib.util.spec_from_file_location("stvg_short_event_tracklets_plan", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not import the short event-local manifest builder")
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)
RUNNER_PATH = ROOT / "scripts/run_stvg_short_event_tracklets_v1.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("stvg_short_event_tracklet_runner", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError("could not import the short event-local CPU runner")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


def check_synthetic_contracts() -> list[str]:
    passed: list[str] = []

    tied = PLAN._nearest_unique_grid_centers([0, 4], [0, 1, 2, 3])
    if [row["center_physical_frame_id"] for row in tied] != [0, 1, 2, 3]:
        raise AssertionError("equal-distance center snapping must break ties toward lower IDs")
    passed.append("uniform_center_snapping_and_lower_id_tie_break")

    try:
        PLAN._nearest_unique_grid_centers([0, 100], [0, 1, 2, 3, 100])
    except ValueError as exc:
        if "duplicate" not in str(exc):
            raise
    else:
        raise AssertionError("duplicate snapped centers must fail closed")
    passed.append("duplicate_center_snap_rejected")

    source_box = [0.20, 0.25, 0.60, 0.75]
    source, source_rule = PLAN._seed_at_center(1, {}, source_box)
    if source != source_box or source_rule != "source_native_box_at_center":
        raise AssertionError("Source must use the actual native box at that center")

    fixed_region = [0.05, 0.10, 0.25, 0.40]
    region, region_rule = PLAN._seed_at_center(
        3,
        {"anchor_box_xyxy": fixed_region},
        [0.70, 0.70, 0.95, 0.95],
    )
    if region != fixed_region or region_rule != "existing_anchor_box_xyxy_copied_unchanged_to_each_center":
        raise AssertionError("non-Source proposal must be copied unchanged, without Source translation")
    passed.append("source_center_box_and_fixed_region_seed_rules")

    invalid, invalid_rule = PLAN._seed_at_center(
        4,
        {"anchor_box_xyxy": [-0.1, 0.1, 0.2, 0.4]},
        source_box,
    )
    if invalid is not None or invalid_rule != "existing_region_proposal_invalid_no_clip":
        raise AssertionError("invalid region proposal must remain explicit and unclipped")
    try:
        PLAN.track_short_window_lk(
            [None], [0], 0, [-0.1, 0.1, 0.2, 0.4], [0, 1]
        )
    except ValueError as exc:
        if "must not be passed to LK" not in str(exc):
            raise
    else:
        raise AssertionError("invalid anchor must be rejected before the LK helper is imported")
    passed.append("invalid_region_preserved_without_clipping")

    return passed


def check_lock_then_preflight_contract() -> str:
    with tempfile.TemporaryDirectory(prefix="stvg-short-cpu-lock-contract-") as temp_dir:
        lock_path = Path(temp_dir) / ".worker.lock"
        callback_calls: list[str] = []

        def mock_no_video_preflight() -> dict[str, object]:
            if not lock_path.is_file():
                raise AssertionError("lock must be created before preflight")
            callback_calls.append("preflight")
            return {"status": "mock_pass", "video_decoded": False, "GT_read": False}

        lock_stream, result = RUNNER._acquire_lock_and_preflight(lock_path, mock_no_video_preflight)
        try:
            if result != {"status": "mock_pass", "video_decoded": False, "GT_read": False}:
                raise AssertionError("mock preflight result changed")
            try:
                RUNNER._acquire_lock_and_preflight(
                    lock_path,
                    lambda: callback_calls.append("competing_preflight"),
                )
            except FileExistsError:
                pass
            else:
                raise AssertionError("a competing runner must be rejected by O_EXCL")
        finally:
            lock_stream.close()
        if callback_calls != ["preflight"]:
            raise AssertionError("competing run reached preflight unexpectedly")
    return "lock_created_before_preflight_and_competitor_rejected"


def check_direct_runner_import_contract() -> str:
    wrapper = RUNNER._load_track_wrapper()
    if not callable(wrapper) or wrapper.__name__ != "track_short_window_lk":
        raise AssertionError("direct-runner root path must resolve the frozen wrapper")
    return "project_root_added_before_wrapper_import"


def check_metadata_dryrun() -> dict[str, object]:
    manifest = PLAN.build_manifest()
    if manifest["case_count"] != 8 or manifest["center_count"] != 32:
        raise AssertionError("dryrun must preserve eight cases and four centers per case")
    if manifest["status"] != "prepared_manifest_not_executed":
        raise AssertionError("dryrun must remain plan-only")
    if not all(manifest[name] is False for name in ("GT_read", "RGB_read", "model_loaded", "GPU_used", "video_decoded")):
        raise AssertionError("plan-only flags must remain false")

    tracklet_ids: set[str] = set()
    for case in manifest["cases"]:
        if len(case["centers"]) != 4 or case["GT_read"] is not False:
            raise AssertionError("every case must contain four no-GT center plans")
        expected_candidates = list(case["fixed_displayed_candidate_ids"])
        if not expected_candidates or expected_candidates[0] != 1:
            raise AssertionError("candidate IDs must preserve the fixed Source-first list")
        for center in case["centers"]:
            if [item["candidate_id"] for item in center["matched_tracklets"]] != expected_candidates:
                raise AssertionError("each center must preserve the exact candidate list and order")
            expected_frames = list(range(
                max(case["canonical_physical_coverage_half_open"][0], center["center_physical_frame_id"] - 2),
                min(case["canonical_physical_coverage_half_open"][1] - 1, center["center_physical_frame_id"] + 2) + 1,
            ))
            if center["physical_frame_ids"] != expected_frames:
                raise AssertionError("window must be the clipped physical +/-2 range")
            if center["local_frame_indices"] != list(range(
                center["local_frame_indices"][0],
                center["local_frame_indices"][0] + len(center["local_frame_indices"]),
            )):
                raise AssertionError("canonical local indices must be consecutive within a window")
            center_records = center["matched_tracklets"]
            for tracklet in center_records:
                if tracklet["tracklet_id"] in tracklet_ids:
                    raise AssertionError("each center/proposal pair must have a unique local ID")
                tracklet_ids.add(tracklet["tracklet_id"])
                if tracklet["cross_window_identity_claim"] is not False:
                    raise AssertionError("windows must not assert cross-window identity")
                if tracklet["physical_frame_ids"] != expected_frames or tracklet["local_frame_indices"] != center["local_frame_indices"]:
                    raise AssertionError("all proposals at a center must share the exact source window")
                if tracklet["anchor_index"] != expected_frames.index(center["center_physical_frame_id"]):
                    raise AssertionError("tracklet anchor index must point to the center physical frame")

    return {
        "case_count": manifest["case_count"],
        "center_count": manifest["center_count"],
        "tracklet_proposal_count": manifest["tracklet_proposal_count"],
        "candidate_counts": [len(case["fixed_displayed_candidate_ids"]) for case in manifest["cases"]],
        "unique_local_tracklet_ids": len(tracklet_ids),
        "GT_read": False,
        "RGB_read": False,
        "video_decoded": False,
        "model_loaded": False,
        "GPU_used": False,
        "manifest_written": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dryrun", action="store_true")
    args = parser.parse_args()
    result: dict[str, object] = {
        "synthetic_contracts_passed": check_synthetic_contracts(),
        "lock_contract_passed": check_lock_then_preflight_contract(),
        "direct_runner_import_contract_passed": check_direct_runner_import_contract(),
    }
    if args.metadata_dryrun:
        result["metadata_dryrun"] = check_metadata_dryrun()
    print(json.dumps(result, sort_keys=True))
