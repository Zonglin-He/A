"""CPU-only synthetic contracts for the matched-support reference worker.

This file contains no case loader, checkpoint access, annotation access,
inference, scoring, or GPU initialization.  It exercises only pure validation
and arm-construction helpers from the new worker.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stvg_matched_support_reference_v1 as worker


def _anchor(role: str, position: int, physical_id: int, slot: int,
            candidate_id: int | None) -> dict:
    box = [0.4 + 0.01 * position, 0.5, 0.2, 0.25]
    return {
        "position": position,
        "physical_frame_id": physical_id,
        "slot": slot,
        "box_cxcywh": box,
        "candidate_id": candidate_id,
        "selected_candidate_id": candidate_id,
        "is_Source": candidate_id == 1,
        "offline_label_assisted": role in ("pool_oracle", "gt_oracle"),
        "teacher_job_id": f"hc2_000_window{slot:02d}_base",
        "teacher_output_sha256": "a" * 64,
    }


def _synthetic_case() -> tuple[dict, list[dict]]:
    centers = [
        {"slot": 0, "source_grid_position": 4, "center_physical_frame_id": 104},
        {"slot": 2, "source_grid_position": 9, "center_physical_frame_id": 109},
    ]
    candidate = {"source": 1, "r1": 2, "pool_oracle": 3, "gt_oracle": None}
    references = {
        role: {
            "offline_diagnostic": True,
            "no_update": False,
            "eligible_for_update": True,
            "reason": "synthetic_matched_support",
            "anchors": [
                _anchor(role, 4, 104, 0, candidate[role]),
                _anchor(role, 9, 109, 2, candidate[role]),
            ],
        }
        for role in worker.ROLES
    }
    # A GT oracle is a box reference, not a candidate ID or tracklet.
    for anchor in references["gt_oracle"]["anchors"]:
        anchor["tracklet_id"] = None
    # Candidate-based references bind to a fixed candidate tracklet.
    for role in ("source", "r1", "pool_oracle"):
        for anchor in references[role]["anchors"]:
            anchor["tracklet_id"] = f"{role}_{anchor['slot']}"
    return references, centers


def _must_fail(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except (AssertionError, KeyError, TypeError, ValueError):
        return
    raise AssertionError("invalid synthetic contract was accepted")


def run_contracts() -> dict:
    checks = []

    refs, centers = _synthetic_case()
    support = worker._reference_support_signature(refs, centers, case_id="hc2_000")
    assert support == [(4, 104), (9, 109)]
    worker._validate_role_provenance(
        refs,
        {f"hc2_000_window{slot:02d}_base": "a" * 64 for slot in (0, 2)},
        case_id="hc2_000",
    )
    checks.append("four_roles_share_exact_ordered_source_grid_and_physical_support")

    swapped = copy.deepcopy(refs)
    swapped["r1"]["anchors"][0]["physical_frame_id"] = 105
    _must_fail(worker._reference_support_signature, swapped, centers, case_id="hc2_000")
    missing = copy.deepcopy(refs)
    missing["gt_oracle"]["anchors"].pop()
    missing["gt_oracle"].update(no_update=False, eligible_for_update=True)
    _must_fail(worker._reference_support_signature, missing, centers, case_id="hc2_000")
    checks.append("rejects_mismatched_or_missing_role_support")

    config = {"lr": 0.01, "sigma": 0.2, "tau": 0.5, "steps": 5}
    specs = {role: worker._fit_spec("hc2_000", role, config, refs[role])
             for role in worker.ROLES}
    assert {spec["key"] for spec in specs.values()} == {"caption_query_fixed_space|hc2_000"}
    assert {tuple(sorted(spec["config"].items())) for spec in specs.values()} == {
        tuple(sorted({**config, "steps": 5}.items()))
    }
    assert {role: spec["reference_role"] for role, spec in specs.items()} == worker.HELPER_ROLES
    assert len({spec["reference_sha256"] for spec in specs.values()}) == 4
    checks.append("same_locked_seed_and_config_with_role_isolated_resume_bindings")

    no_refs = {role: {"anchors": [], "no_update": True, "eligible_for_update": False}
               for role in worker.ROLES}
    f_arms = {
        "F": {"boxes": [[0.1, 0.2, 0.3, 0.4]], "interval": [100, 110]},
        "T": {"boxes": [[0.1, 0.2, 0.3, 0.4]], "interval": [102, 109]},
    }
    arms, receipts = worker._no_update_arms(f_arms, no_refs)
    assert arms["F"] == f_arms["F"] and arms["T"] == f_arms["T"]
    assert all(arms[role] == f_arms["F"] for role in worker.ROLES)
    assert all(arms[f"T_{role}"] == f_arms["T"] for role in worker.ROLES)
    assert all(receipts[role]["optimizer_steps"] == 0
               and receipts[role]["final_redecode_calls"] == 0 for role in worker.ROLES)
    assert all(receipts[f"T_{role}"]["exact_T_reuse"] for role in worker.ROLES)
    checks.append("empty_support_is_exact_F_T_no_update_with_zero_fit_and_readout")

    worker._require_digest_binding("synthetic_source_sha", "a" * 64, "a" * 64)
    _must_fail(worker._require_digest_binding, "synthetic_source_sha", "a" * 64, "b" * 64)
    checks.append("source_hash_binding_accepts_exact_and_rejects_swapped_digest")

    return {
        "schema": "stvg-matched-support-reference-cpu-contracts-v1",
        "status": "pass",
        "checks": checks,
        "check_count": len(checks),
        "synthetic_only": True,
        "model_loaded": False,
        "checkpoint_read": False,
        "GT_read": False,
        "media_read": False,
        "GPU_used": False,
    }


def main() -> None:
    print(json.dumps(run_contracts(), sort_keys=True))


if __name__ == "__main__":
    main()
