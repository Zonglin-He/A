"""Continue the bounded Spatial-Lite study after its structural lock exists.

This is an explicit-dispatch controller. Importing it or running --contracts
does not launch workers. A later operator must pass --dispatch after L000/L001/
L002 and the root STRUCTURAL_SELECTION_LOCK are complete.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stvg_lite_event_common_v1 as common

BASE = common.BASE
SEARCH = common.SEARCH
COMP = common.COMP
STRUCTURAL_LOCK = BASE / "STRUCTURAL_SELECTION_LOCK.json"
TEMPORAL_CANDIDATE = BASE / "temporal" / "revision003" / "TEMPORAL_LOCK.json"
TEMPORAL_RUNTIME_LOCK = BASE / "temporal" / "revision003" / "TEMPORAL_LITE_RUNTIME_LOCK.json"
TEMPORAL_CPU_BARRIER = BASE / "temporal" / "revision003" / "CPU_OUTPUT_BARRIER.json"
TEMPORAL_ROOT_READBACK = BASE / "temporal" / "revision003" / "ROOT_READBACK.json"
TEMPORAL_PREDICTION_BARRIER = BASE / "temporal" / "revision003" / "PREDICTION_GLOBAL_BARRIER.json"
TEMPORAL_GT_EXPOSURE = BASE / "temporal" / "revision003" / "POSTSEAL_GT_EXPOSURE.json"
TEMPORAL_SCORE_ROWS = BASE / "temporal" / "revision003" / "POSTSEAL_ROWS.json"
TEMPORAL_SCORE_SUMMARY = BASE / "temporal" / "revision003" / "POSTSEAL_SUMMARY.json"
TEMPORAL_SCORE_PAIRED = BASE / "temporal" / "revision003" / "POSTSEAL_PAIRED.json"
TEMPORAL_SCORE_BARRIER = BASE / "temporal" / "revision003" / "POSTSEAL_SCORE_BARRIER.json"
TEMPORAL_CPU_COMPLETION = BASE / "temporal" / "revision003" / "CPU_COMPLETION.json"
SELECTED_TEMPORAL_LOCK = BASE / "SELECTED_TEMPORAL_LOCK.json"
STAGE_RUNTIME_LOCK = BASE / "STAGE_EXECUTION_RUNTIME.json"
FINAL_RECIPE_LOCK = BASE / "FINAL_RECIPE_LOCK.json"

# This dispatcher requires its own preconfigured user unit. Its monitor can
# signal only this name, and it refuses to run from another cgroup.
UNIT = "stvg-lite-event-final-search-001.service"
GIB = 1024 ** 3
LR_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
STRUCTURAL_IDS = ("L000", "L001", "L002")


def _finite_interval(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(float(x)) for x in value)
            and float(value[1]) > float(value[0]))


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def expected_structural_binding_paths():
    paths = set()
    for trial_id in STRUCTURAL_IDS:
        trial = BASE / "trials" / trial_id
        for filename in ("ROOT_READBACK.json", "CPU_COMPLETION.json",
                         "POSTSEAL_ROWS.json", "MATCHED_EFFICIENCY_READBACK.json"):
            paths.add(str((trial / filename).relative_to(ROOT)))
    for filename in ("POSTSEAL_ROWS.json", "CPU_COMPLETION.json"):
        paths.add(str((BASE / "temporal" / "revision003" / filename).relative_to(ROOT)))
    return paths


def validate_structural_bindings(bindings):
    expected = expected_structural_binding_paths()
    if not isinstance(bindings, dict):
        raise ValueError("structural selection bindings must be a mapping")
    if set(bindings) != expected:
        missing = sorted(expected - set(bindings))
        extra = sorted(set(bindings) - expected)
        raise ValueError(f"structural selection bindings must be the exact 14-path set; missing={missing}, extra={extra}")
    if not all(_is_sha256(value) for value in bindings.values()):
        raise ValueError("structural selection bindings must contain SHA-256 values")
    return True


def validate_temporal_candidate_chain(*, candidate, runtime, cpu_barrier, root_readback,
                                      prediction, gt_exposure, score_rows, score_barrier,
                                      completion, hashes, selection_bindings):
    """Verify the scored Temporal candidate still matches its entire sealed chain.

    The values are supplied as parsed lock documents plus their current file
    hashes so the same logic can be exercised with fully synthetic witnesses.
    """
    required_hashes = {
        "candidate", "runtime", "cpu_barrier", "root_readback", "prediction",
        "gt_exposure", "score_rows", "score_barrier", "completion", "design",
        "summary", "paired",
    }
    if set(hashes) != required_hashes or not all(_is_sha256(x) for x in hashes.values()):
        raise ValueError("Temporal chain hash map is incomplete or malformed")
    if (runtime.get("status") != "locked" or runtime.get("GT_read") is not False
            or runtime.get("execution_authorized") is not True):
        raise ValueError("Temporal runtime lock is not locked, authorized, and GT-free")
    if not isinstance(selection_bindings, dict):
        raise ValueError("structural selection bindings are not a mapping")
    runtime_pins = runtime.get("pins")
    if not isinstance(runtime_pins, dict):
        raise ValueError("Temporal runtime lock lacks source pins")
    for field, relative in (
        ("runner_sha256", "scripts/run_stvg_temporal_lite128003.py"),
        ("fit_helper_sha256", "scripts/stvg_temporal_lite_fit_v1.py"),
        ("readback_script_sha256", "scripts/readback_stvg_temporal_lite128003.py"),
    ):
        if cpu_barrier.get(field) != runtime_pins.get(relative):
            raise ValueError(f"Temporal CPU barrier does not bind runtime-pinned source: {relative}")
    if (candidate.get("schema") != "stvg-temporal-lite-candidate-temporal-lock-v1"
            or candidate.get("status") != "complete"
            or candidate.get("candidate_variant") != "temporal_lite"
            or candidate.get("not_a_deployment_or_scored_method_selection") is not True
            or candidate.get("count") != 128 or candidate.get("GT_read") is not False
            or candidate.get("cpu_output_barrier_sha256") != hashes["cpu_barrier"]
            or candidate.get("runtime_lock_sha256") != hashes["runtime"]):
        raise ValueError("Temporal candidate lock is not bound to its CPU barrier/runtime")
    cpu_bindings = cpu_barrier.get("bindings")
    if (cpu_barrier.get("schema") != "stvg-temporal-lite-cpu-output-barrier-v1"
            or cpu_barrier.get("status") != "complete" or cpu_barrier.get("count") != 128
            or cpu_barrier.get("GT_read") is not False or cpu_barrier.get("CPU_only") is not True
            or cpu_barrier.get("runtime_lock_sha256") != hashes["runtime"]
            or not isinstance(cpu_bindings, dict)
            or runtime.get("input_bindings") != cpu_bindings
            or cpu_bindings.get("design_lock_sha256") != hashes["design"]):
        raise ValueError("Temporal CPU output barrier is incomplete or not bound to runtime")
    if (root_readback.get("schema") != "stvg-temporal-lite-independent-root-readback-v1"
            or root_readback.get("status") != "pass" or root_readback.get("count") != 128
            or root_readback.get("GT_read") is not False
            or root_readback.get("CPU_only") is not True
            or root_readback.get("actual_scoring_performed") is not False
            or root_readback.get("cpu_output_barrier_sha256") != hashes["cpu_barrier"]
            or root_readback.get("temporal_lock_sha256") != hashes["candidate"]
            or root_readback.get("runtime_lock_sha256") != hashes["runtime"]
            or root_readback.get("input_bindings") != cpu_barrier.get("bindings")):
        raise ValueError("Temporal independent root readback is incomplete or bound to other inputs")
    if (prediction.get("schema") != "stvg-temporal-lite-pre-gt-prediction-barrier-v1"
            or prediction.get("status") != "complete" or prediction.get("count") != 128
            or prediction.get("GT_read") is not False
            or prediction.get("scoring_performed") is not False
            or prediction.get("cpu_output_barrier_sha256") != hashes["cpu_barrier"]
            or prediction.get("root_readback_sha256") != hashes["root_readback"]
            or prediction.get("temporal_lock_sha256") != hashes["candidate"]
            or prediction.get("runtime_lock_sha256") != hashes["runtime"]):
        raise ValueError("Temporal pre-GT prediction barrier does not bind the candidate/root/runtime")

    expected_score_bindings = {
        "prediction_barrier_sha256": hashes["prediction"],
        "cpu_output_barrier_sha256": hashes["cpu_barrier"],
        "root_readback_sha256": hashes["root_readback"],
        "candidate_temporal_lock_sha256": hashes["candidate"],
        "runtime_lock_sha256": hashes["runtime"],
        "design_lock_sha256": hashes["design"],
    }
    if (gt_exposure.get("schema") != "stvg-temporal-lite-postseal-gt-exposure-v1"
            or gt_exposure.get("status") != "after_full_128_prediction_barrier_and_independent_root_readback"
            or gt_exposure.get("count") != 128 or gt_exposure.get("GT_read") is not True
            or gt_exposure.get("bindings") != expected_score_bindings):
        raise ValueError("Temporal GT exposure receipt does not bind the sealed candidate chain")
    if (score_rows.get("schema") != "stvg-temporal-lite-postseal-v1"
            or score_rows.get("status") != "complete"
            or score_rows.get("case_count") != 128 or score_rows.get("row_count") != 768
            or score_rows.get("GT_read") is not True
            or score_rows.get("bootstrap_executed") is not False
            or score_rows.get("bindings") != expected_score_bindings):
        raise ValueError("Temporal scored rows do not bind the candidate and full pre-GT chain")
    if (score_barrier.get("schema") != "stvg-temporal-lite-postseal-score-barrier-v1"
            or score_barrier.get("status") != "complete"
            or score_barrier.get("count") != 768 or score_barrier.get("case_count") != 128
            or score_barrier.get("arm_count") != 6 or score_barrier.get("GT_read") is not True
            or score_barrier.get("bootstrap_executed") is not False
            or score_barrier.get("rows_path") != str(TEMPORAL_SCORE_ROWS.relative_to(ROOT))
            or score_barrier.get("summary_path") != str(TEMPORAL_SCORE_SUMMARY.relative_to(ROOT))
            or score_barrier.get("paired_path") != str(TEMPORAL_SCORE_PAIRED.relative_to(ROOT))
            or score_barrier.get("rows_sha256") != hashes["score_rows"]
            or score_barrier.get("summary_sha256") != hashes["summary"]
            or score_barrier.get("paired_sha256") != hashes["paired"]
            or score_barrier.get("gt_exposure_sha256") != hashes["gt_exposure"]
            or score_barrier.get("prediction_bindings") != expected_score_bindings
            or score_barrier.get("scorer_sha256") != runtime_pins.get("scripts/score_stvg_temporal_lite128003.py")):
        raise ValueError("Temporal post-seal score barrier does not bind rows/exposure/predictions")
    if (completion.get("schema") != "stvg-temporal-lite-postseal-cpu-completion-v1"
            or completion.get("status") != "complete" or completion.get("count") != 128
            or completion.get("row_count") != 768 or completion.get("arm_count") != 6
            or completion.get("GT_read") is not True
            or completion.get("CPU_only") is not True
            or completion.get("bootstrap_executed") is not False
            or completion.get("score_barrier_sha256") != hashes["score_barrier"]):
        raise ValueError("Temporal CPU completion receipt does not bind the post-seal score barrier")

    expected_temporal_bindings = {
        str(TEMPORAL_SCORE_ROWS.relative_to(ROOT)): hashes["score_rows"],
        str(TEMPORAL_CPU_COMPLETION.relative_to(ROOT)): hashes["completion"],
    }
    if any(selection_bindings.get(path) != digest
           for path, digest in expected_temporal_bindings.items()):
        raise ValueError("structural selection does not bind the current Temporal score/completion")

    candidate_rows = candidate.get("records")
    cpu_rows = cpu_barrier.get("files")
    readback_rows = root_readback.get("files")
    prediction_rows = prediction.get("files")
    def index(rows, label):
        if (not isinstance(rows, list) or len(rows) != 128
                or not all(isinstance(row, dict) for row in rows)):
            raise ValueError(f"Temporal {label} must contain 128 rows")
        result = {row.get("case_id"): row for row in rows}
        if None in result or len(result) != 128:
            raise ValueError(f"Temporal {label} has missing/duplicate case ids")
        return result
    crows = index(candidate_rows, "candidate records")
    brows = index(cpu_rows, "CPU barrier records")
    rrows = index(readback_rows, "root readback records")
    prows = index(prediction_rows, "prediction barrier records")
    if not (set(crows) == set(brows) == set(rrows) == set(prows)):
        raise ValueError("Temporal candidate/barrier case membership differs")
    for case_id, row in crows.items():
        cache_sha = row.get("source_temporal_sha256")
        if (row.get("GT_read") is not False or not _is_sha256(cache_sha)
                or row.get("dataset") != brows[case_id].get("dataset")
                or row.get("source_case_sha256") != brows[case_id].get("source_case_sha256")
                or row.get("source_file_sha256") != brows[case_id].get("source_file_sha256")
                or cache_sha != brows[case_id].get("temporal_cache_sha256")):
            raise ValueError(f"Temporal candidate source/cache binding differs from CPU barrier: {case_id}")
        for other in (rrows[case_id], prows[case_id]):
            if (other.get("dataset") != row.get("dataset")
                    or other.get("source_case_sha256") != row.get("source_case_sha256")
                    or other.get("source_file_sha256") != row.get("source_file_sha256")
                    or other.get("temporal_cache_sha256") != cache_sha):
                raise ValueError(f"Temporal candidate source/cache binding differs from readback: {case_id}")
    return True


def _verify_temporal_candidate_chain(selection):
    """Read and bind the scored candidate to every pre/post-score seal."""
    paths = {
        "candidate": TEMPORAL_CANDIDATE,
        "runtime": TEMPORAL_RUNTIME_LOCK,
        "cpu_barrier": TEMPORAL_CPU_BARRIER,
        "root_readback": TEMPORAL_ROOT_READBACK,
        "prediction": TEMPORAL_PREDICTION_BARRIER,
        "gt_exposure": TEMPORAL_GT_EXPOSURE,
        "score_rows": TEMPORAL_SCORE_ROWS,
        "summary": TEMPORAL_SCORE_SUMMARY,
        "paired": TEMPORAL_SCORE_PAIRED,
        "score_barrier": TEMPORAL_SCORE_BARRIER,
        "completion": TEMPORAL_CPU_COMPLETION,
        "design": BASE / "DESIGN_LOCK.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Temporal candidate chain is incomplete: {path}")
    hashes = {name: common.sha(path) for name, path in paths.items()}
    docs = {name: common.read(path) for name, path in paths.items()}
    validate_temporal_candidate_chain(
        candidate=docs["candidate"], runtime=docs["runtime"],
        cpu_barrier=docs["cpu_barrier"], root_readback=docs["root_readback"],
        prediction=docs["prediction"], gt_exposure=docs["gt_exposure"],
        score_rows=docs["score_rows"], score_barrier=docs["score_barrier"],
        completion=docs["completion"], hashes=hashes,
        selection_bindings=selection.get("bindings", {}))
    return docs["candidate"]


def build_selected_temporal_lock(old_lock, candidate_lock, selected_temporal, *, expected_count=128):
    """Adapt a temporal candidate to the pre-existing worker record schema.

    Every old record is copied and only ``selected_interval`` is replaced.
    Existing per-record input hashes and any other old-schema fields survive.
    """
    if selected_temporal not in {"temporal_lite", "offset_trust"}:
        raise ValueError("selected_temporal must be temporal_lite or offset_trust")
    if old_lock.get("status") != "locked" or old_lock.get("GT_read") is not False:
        raise ValueError("original temporal input lock is not a GT-free locked input")
    if candidate_lock.get("status") != "complete" or candidate_lock.get("GT_read") is not False:
        raise ValueError("temporal candidate output is incomplete or has invalid scope")
    if (candidate_lock.get("candidate_variant") != "temporal_lite"
            or candidate_lock.get("not_a_deployment_or_scored_method_selection") is not True):
        raise ValueError("temporal candidate lock schema/scope is not the expected CPU candidate")
    old_rows = old_lock.get("records")
    new_rows = candidate_lock.get("records")
    count = old_lock.get("count")
    if (not isinstance(count, int) or count != expected_count or len(old_rows or []) != count
            or candidate_lock.get("count") != count or len(new_rows or []) != count):
        raise ValueError(f"temporal lock counts do not match the locked {expected_count}-source cohort")
    old_by_id = {row.get("case_id"): row for row in old_rows}
    new_by_id = {row.get("case_id"): row for row in new_rows}
    if (None in old_by_id or None in new_by_id or len(old_by_id) != count
            or len(new_by_id) != count or set(old_by_id) != set(new_by_id)):
        raise ValueError("old and candidate temporal record membership differs")

    output_rows = []
    for old_row in old_rows:
        cid = old_row["case_id"]
        candidate = new_by_id[cid]
        # Do not replace old input provenance with candidate output hashes.
        old_temporal_sha = old_row.get("source_temporal_sha256")
        candidate_temporal_sha = candidate.get("source_temporal_sha256")
        if not _is_sha256(old_temporal_sha) or not _is_sha256(candidate_temporal_sha):
            raise ValueError(f"source temporal SHA-256 is missing or malformed for {cid}")
        if old_temporal_sha != candidate_temporal_sha:
            raise ValueError(f"source temporal hash changed for {cid}")
        selected_key = ("temporal_lite_interval" if selected_temporal == "temporal_lite"
                        else "baseline_offset_trust_interval")
        interval = candidate.get(selected_key)
        if not _finite_interval(interval):
            raise ValueError(f"selected temporal interval is invalid for {cid}")
        row = copy.deepcopy(old_row)
        row["selected_interval"] = list(interval)
        output_rows.append(row)

    result = copy.deepcopy(old_lock)
    result["records"] = output_rows
    return result


def scope_decision(full_means_pp, event_means_pp, full_cost, event_cost, budget_pp=0.1):
    """Apply the two-domain loss guard, then equal-domain macro mean.

    Full-grid is the safe reference. Event support is eligible only if neither
    domain loses more than the locked budget. Among eligible scopes, the larger
    equal-domain mean wins; exact score ties retain full-grid.
    """
    domains = ("hc2", "vidstg")
    if set(full_means_pp) != set(domains) or set(event_means_pp) != set(domains):
        raise ValueError("scope comparison requires exactly hc2 and vidstg means")
    deltas = {ds: float(event_means_pp[ds]) - float(full_means_pp[ds]) for ds in domains}
    if not all(math.isfinite(value) for value in deltas.values()):
        raise ValueError("scope comparison contains a non-finite domain mean")
    budget = float(budget_pp)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("scope loss budget must be finite and nonnegative")
    event_eligible = all(value >= -budget - 1e-12 for value in deltas.values())
    full_macro = sum(float(full_means_pp[ds]) for ds in domains) / 2
    event_macro = sum(float(event_means_pp[ds]) for ds in domains) / 2
    if not event_eligible:
        selected = "full_grid"
    elif event_macro > full_macro + 1e-12:
        selected = "event_support"
    elif event_macro < full_macro - 1e-12:
        selected = "full_grid"
    else:
        # Scope is decided only from the two-domain guard and equal-domain
        # score mean. Exact score ties retain full-grid as the safe default.
        selected = "full_grid"
    return {
        "selected_response_scope": selected,
        "domain_delta_pp_event_minus_full": deltas,
        "full_equal_domain_mean_pp": full_macro,
        "event_equal_domain_mean_pp": event_macro,
        "event_within_both_domain_budgets": event_eligible,
        "per_domain_loss_budget_pp": budget,
        "full_cost": copy.deepcopy(full_cost),
        "event_cost": copy.deepcopy(event_cost),
        "decision_rule": "event support must pass both per-domain loss guards and improve the equal-domain mean; exact mean ties retain full-grid",
    }


def choose_lr_by_domain(scores_by_factor):
    """Select one of the five locked LR multipliers independently per domain."""
    if set(scores_by_factor) != {"hc2", "vidstg"}:
        raise ValueError("LR search must contain both domains")
    factor_keys = {str(value) for value in LR_FACTORS}
    selected = {}
    summaries = {}
    for domain in ("hc2", "vidstg"):
        values = scores_by_factor[domain]
        if set(values) != factor_keys:
            raise ValueError(f"{domain} LR search does not contain the fixed five factors")
        parsed = {float(key): float(value) for key, value in values.items()}
        if not all(math.isfinite(value) for value in parsed.values()):
            raise ValueError(f"{domain} LR search has a non-finite score")
        maximum = max(parsed.values())
        tied = [factor for factor, value in parsed.items() if abs(value - maximum) <= 1e-12]
        winner = min(tied, key=lambda factor: (abs(factor - 1.0), factor))
        selected[domain] = winner
        summaries[domain] = {
            "scores_pp_by_factor": {str(key): parsed[key] for key in LR_FACTORS},
            "selected_factor": winner,
            "selected_mean_vIoU_pp": parsed[winner],
            "tie_rule": "closest to center 1.0, then lower factor",
        }
    return selected, summaries


def _expected_trial_inputs(multiplier):
    final = common.read(COMP / "FINAL_PARAMETER_LOCK.json")
    configs = {
        ds: dict(final["configs"][ds], lr=final["configs"][ds]["lr"] * float(multiplier))
        for ds in ("hc2", "vidstg")
    }
    settings = {ds: copy.deepcopy(final["response_settings"][ds]) for ds in configs}
    return configs, settings


def _trial_identity_is_exact(trial_path, recipe, response_scope, temporal_sha, selection_sha,
                             selected_temporal):
    """Reusable-center guard: all model/protocol/input identity must match."""
    design = common.read(trial_path / "DESIGN_LOCK.json")
    if not (design.get("recipe") == recipe and design.get("response_scope") == response_scope
            and float(design.get("lr_multiplier", math.nan)) == 1.0
            and design.get("structural_selection_sha256") == selection_sha
            and design.get("selected_temporal") == selected_temporal
            and design.get("selected_temporal_lock_sha256") == temporal_sha
            and design.get("continuation_trial") is True):
        return False
    expected_configs, expected_settings = _expected_trial_inputs(1.0)
    if design.get("configs") != {ds: {"M2": cfg} for ds, cfg in expected_configs.items()}:
        return False
    if design.get("response_settings") != {
            ds: {"M1": settings, "M2": settings} for ds, settings in expected_settings.items()}:
        return False
    if common.sha(trial_path / "TEMPORAL_LOCK.json") != temporal_sha:
        return False
    if common.sha(trial_path / "REFERENCE_LOCK.json") != common.sha(SEARCH / "REFERENCE_LOCK.json"):
        return False
    runtime_path = trial_path / "RUNTIME.json"
    if not runtime_path.is_file():
        return False
    runtime = common.read(runtime_path)
    if (runtime.get("status") != "locked" or runtime.get("GT_read") is not False
            or runtime.get("pins", {}).get(str(STAGE_RUNTIME_LOCK.relative_to(ROOT)))
            != common.sha(STAGE_RUNTIME_LOCK)
            or runtime.get("pins", {}).get(str(Path(__file__).resolve().relative_to(ROOT)))
            != common.sha(Path(__file__))):
        return False
    for filename, expected_status in (("ROOT_READBACK.json", "pass"),
                                      ("POSTSEAL_ROWS.json", "complete"),
                                      ("MATCHED_EFFICIENCY_READBACK.json", "complete")):
        path = trial_path / filename
        if not path.is_file() or common.read(path).get("status") != expected_status:
            return False
    return True


def _attach_stage_pins(trial_runtime, stage_runtime_sha256, script_sha256):
    """Bind each continuation trial to its additional source-closure lock."""
    if trial_runtime.get("status") != "locked" or trial_runtime.get("GT_read") is not False:
        raise ValueError("continuation trial runtime must be locked and GT-free")
    result = copy.deepcopy(trial_runtime)
    pins = dict(result.get("pins", {}))
    pins[str(STAGE_RUNTIME_LOCK.relative_to(ROOT))] = stage_runtime_sha256
    pins[str(Path(__file__).resolve().relative_to(ROOT))] = script_sha256
    result["pins"] = pins
    return result


def _stage_runtime_payload(selection, temporal_sha):
    """Build a separate source/runtime closure without mutating root's lock."""
    if not TEMPORAL_RUNTIME_LOCK.is_file():
        raise FileNotFoundError("reviewed Temporal-Lite runtime lock is missing")
    temporal_runtime = common.read(TEMPORAL_RUNTIME_LOCK)
    if (temporal_runtime.get("status") != "locked"
            or temporal_runtime.get("GT_read") is not False
            or temporal_runtime.get("execution_authorized") is not True
            or not isinstance(temporal_runtime.get("pins"), dict)):
        raise ValueError("Temporal-Lite runtime lock is not a reviewed, authorized source closure")
    pin_paths = {
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/select_stvg_lite_event004.py",
        "scripts/run_stvg_temporal_lite128003.py",
        "scripts/readback_stvg_temporal_lite128003.py",
        "scripts/score_stvg_temporal_lite128003.py",
    }
    pin_paths.update(temporal_runtime["pins"])
    pin_paths.add("scripts/stvg_lite_recovery004_common.py")
    pin_paths.update({"scripts/review_stvg_lite_event002.py", "scripts/audit_stvg_spatial_lite_fit002.py", "scripts/stvg_lite_recovery002_common.py"})
    pins = {}
    for relative in sorted(pin_paths):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"stage source pin is missing: {relative}")
        digest = common.sha(path)
        if relative in temporal_runtime["pins"] and temporal_runtime["pins"][relative] != digest:
            raise ValueError(f"Temporal-Lite runtime source pin changed: {relative}")
        pins[relative] = digest

    # Pin the decisions and exact inputs from which the selected temporal
    # adapter was built. This is separate from Top001's source closure.
    artifact_paths = (
        STRUCTURAL_LOCK, TEMPORAL_CANDIDATE, TEMPORAL_RUNTIME_LOCK,
        TEMPORAL_CPU_BARRIER, TEMPORAL_ROOT_READBACK, TEMPORAL_PREDICTION_BARRIER,
        TEMPORAL_GT_EXPOSURE, TEMPORAL_SCORE_ROWS, TEMPORAL_SCORE_SUMMARY,
        TEMPORAL_SCORE_PAIRED, TEMPORAL_SCORE_BARRIER, TEMPORAL_CPU_COMPLETION,
        SELECTED_TEMPORAL_LOCK, SEARCH / "TEMPORAL_LOCK.json",
        SEARCH / "REFERENCE_LOCK.json", COMP / "FINAL_PARAMETER_LOCK.json",
        BASE / "DESIGN_LOCK.json", common.RUNTIME,
    )
    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(f"stage input lock is missing: {path}")
        pins[str(path.relative_to(ROOT))] = common.sha(path)
    if pins[str(SELECTED_TEMPORAL_LOCK.relative_to(ROOT))] != temporal_sha:
        raise ValueError("selected temporal lock hash changed while building stage runtime")
    return {
        "schema": "stvg-lite-event-stage-execution-runtime-v1",
        "status": "locked",
        "stage": "post-structural-scope-and-lr-search",
        "GT_read": False,
        "execution_authorized": True,
        "structural_selection_lock_sha256": selection["_sha256"],
        "selected_temporal_lock_sha256": temporal_sha,
        "temporal_runtime_lock_sha256": common.sha(TEMPORAL_RUNTIME_LOCK),
        "pins": pins,
        "scope": "finite development continuation only; does not promote CURRENT or establish reclocked confirmation",
    }


def _lock_stage_runtime(selection, temporal_sha):
    payload = _stage_runtime_payload(selection, temporal_sha)
    if STAGE_RUNTIME_LOCK.exists():
        if common.read(STAGE_RUNTIME_LOCK) != payload:
            raise FileExistsError("STAGE_EXECUTION_RUNTIME.json exists with different contents")
    else:
        common.write(STAGE_RUNTIME_LOCK, payload)
    for relative, digest in common.read(STAGE_RUNTIME_LOCK)["pins"].items():
        if not (ROOT / relative).is_file() or common.sha(ROOT / relative) != digest:
            raise ValueError(f"stage runtime pin changed before dispatch: {relative}")
    return common.sha(STAGE_RUNTIME_LOCK)


def make_final_recipe_lock(*, selection, temporal_lock_sha256, selected_recipe,
                           selected_scope, configs, response_settings,
                           selected_factors, lr_summaries, scope_summary,
                           trial_map, diagnostic_scope=None):
    """Build the sole final development-recipe lock, pending reclocked review."""
    if selected_recipe not in {"full", "lite_backtrack", "lite_fullstep"}:
        raise ValueError("unsupported selected spatial recipe")
    if selected_scope not in {"full_grid", "event_support"}:
        raise ValueError("unsupported selected response scope")
    if set(configs) != {"hc2", "vidstg"} or set(response_settings) != {"hc2", "vidstg"}:
        raise ValueError("final recipe must include plain per-domain configs/settings")
    if selected_recipe == "full" and selected_scope != "full_grid":
        raise ValueError("Full spatial recipe remains full-grid response protected")
    return {
        "schema": "stvg-lite-event-final-recipe-v1",
        "status": "locked",
        "completion_scope": "pending_reclocked_confirmation",
        "method_validation_status": "pending_reclocked_confirmation",
        "scope": "development-selected recipe only; requires new-reference closed-loop confirmation",
        "selected_spatial_trial": selection["selected_spatial_trial"],
        "selected_spatial_recipe": selected_recipe,
        "selected_response_scope": selected_scope,
        "configs": copy.deepcopy(configs),
        "response_settings": copy.deepcopy(response_settings),
        "selected_temporal": selection["selected_temporal"],
        "selected_temporal_lock_sha256": temporal_lock_sha256,
        "structural_selection_lock_sha256": selection.get("_sha256"),
        "structural_selection": {key: copy.deepcopy(selection[key]) for key in (
            "selected_spatial_trial", "selected_spatial_recipe", "selected_temporal",
            "structural_loss_budget_pp", "temporal_delta_pp", "temporal_means_pp") if key in selection},
        "scope_comparison": copy.deepcopy(scope_summary),
        "unpromoted_lite_diagnostic": copy.deepcopy(diagnostic_scope),
        "lr_factors_by_domain": copy.deepcopy(selected_factors),
        "lr_search": copy.deepcopy(lr_summaries),
        "trial_map": copy.deepcopy(trial_map),
        "GT_read": True,
        "GT_used_for_online_reference_or_step": False,
        "CURRENT_promoted": False,
        "suite_final": False,
        "confirmation_status": "pending_reclocked_confirmation",
        "bootstrap_executed": False,
    }


def contracts():
    """Pure synthetic controller checks; no locks or result artifacts are read."""
    fake_structural_bindings = {
        path: f"{index % 10}" * 64
        for index, path in enumerate(sorted(expected_structural_binding_paths()))
    }
    assert len(fake_structural_bindings) == 14
    assert validate_structural_bindings(fake_structural_bindings)
    incomplete_structural_bindings = dict(fake_structural_bindings)
    incomplete_structural_bindings.pop(next(iter(incomplete_structural_bindings)))
    try:
        validate_structural_bindings(incomplete_structural_bindings)
    except ValueError:
        structural_missing_binding_rejected = True
    else:
        raise AssertionError("structural binding checker accepted a missing required path")

    temporal_hashes = {
        "candidate": "1" * 64,
        "runtime": "2" * 64,
        "cpu_barrier": "3" * 64,
        "root_readback": "4" * 64,
        "prediction": "5" * 64,
        "gt_exposure": "6" * 64,
        "score_rows": "7" * 64,
        "score_barrier": "8" * 64,
        "completion": "9" * 64,
        "design": "a" * 64,
        "summary": "b" * 64,
        "paired": "c" * 64,
    }
    temporal_pins = {
        "scripts/run_stvg_temporal_lite128003.py": "d" * 64,
        "scripts/stvg_temporal_lite_fit_v1.py": "e" * 64,
        "scripts/readback_stvg_temporal_lite128003.py": "f" * 64,
        "scripts/score_stvg_temporal_lite128003.py": "0" * 64,
    }
    temporal_input_bindings = {
        "original-lock": "1" * 64,
        "design_lock_sha256": temporal_hashes["design"],
    }
    temporal_runtime = {
        "status": "locked", "GT_read": False, "execution_authorized": True,
        "pins": temporal_pins, "input_bindings": temporal_input_bindings,
    }
    temporal_cpu = {
        "schema": "stvg-temporal-lite-cpu-output-barrier-v1",
        "status": "complete", "count": 128, "GT_read": False, "CPU_only": True,
        "runtime_lock_sha256": temporal_hashes["runtime"],
        "bindings": temporal_input_bindings,
        "runner_sha256": temporal_pins["scripts/run_stvg_temporal_lite128003.py"],
        "fit_helper_sha256": temporal_pins["scripts/stvg_temporal_lite_fit_v1.py"],
        "readback_script_sha256": temporal_pins["scripts/readback_stvg_temporal_lite128003.py"],
    }
    temporal_records = []
    temporal_files = []
    temporal_prediction_files = []
    for index in range(128):
        case_id = f"SYNTH-{index:03d}"
        dataset = "hc2" if index < 64 else "vidstg"
        case_sha = f"{index % 10}" * 64
        source_sha = f"{(index + 1) % 10}" * 64
        cache_sha = "a" * 64
        temporal_records.append({
            "case_id": case_id, "dataset": dataset,
            "source_case_sha256": case_sha, "source_file_sha256": source_sha,
            "source_temporal_sha256": cache_sha, "GT_read": False,
        })
        common_file_fields = {
            "case_id": case_id, "dataset": dataset,
            "source_case_sha256": case_sha, "source_file_sha256": source_sha,
            "temporal_cache_sha256": cache_sha,
        }
        temporal_files.append(dict(common_file_fields))
        temporal_prediction_files.append(dict(common_file_fields))
    temporal_cpu["files"] = [dict(row) for row in temporal_files]
    temporal_candidate = {
        "schema": "stvg-temporal-lite-candidate-temporal-lock-v1",
        "status": "complete", "candidate_variant": "temporal_lite",
        "not_a_deployment_or_scored_method_selection": True,
        "count": 128, "GT_read": False,
        "cpu_output_barrier_sha256": temporal_hashes["cpu_barrier"],
        "runtime_lock_sha256": temporal_hashes["runtime"],
        "records": temporal_records,
    }
    temporal_root = {
        "schema": "stvg-temporal-lite-independent-root-readback-v1",
        "status": "pass", "count": 128, "GT_read": False, "CPU_only": True,
        "actual_scoring_performed": False,
        "cpu_output_barrier_sha256": temporal_hashes["cpu_barrier"],
        "temporal_lock_sha256": temporal_hashes["candidate"],
        "runtime_lock_sha256": temporal_hashes["runtime"],
        "input_bindings": temporal_input_bindings, "files": temporal_files,
    }
    temporal_prediction = {
        "schema": "stvg-temporal-lite-pre-gt-prediction-barrier-v1",
        "status": "complete", "count": 128, "GT_read": False,
        "scoring_performed": False,
        "cpu_output_barrier_sha256": temporal_hashes["cpu_barrier"],
        "root_readback_sha256": temporal_hashes["root_readback"],
        "temporal_lock_sha256": temporal_hashes["candidate"],
        "runtime_lock_sha256": temporal_hashes["runtime"],
        "files": temporal_prediction_files,
    }
    score_bindings = {
        "prediction_barrier_sha256": temporal_hashes["prediction"],
        "cpu_output_barrier_sha256": temporal_hashes["cpu_barrier"],
        "root_readback_sha256": temporal_hashes["root_readback"],
        "candidate_temporal_lock_sha256": temporal_hashes["candidate"],
        "runtime_lock_sha256": temporal_hashes["runtime"],
        "design_lock_sha256": temporal_hashes["design"],
    }
    temporal_gt_exposure = {
        "schema": "stvg-temporal-lite-postseal-gt-exposure-v1",
        "status": "after_full_128_prediction_barrier_and_independent_root_readback",
        "count": 128, "GT_read": True, "bindings": score_bindings,
    }
    temporal_rows = {
        "schema": "stvg-temporal-lite-postseal-v1", "status": "complete",
        "case_count": 128, "row_count": 768, "GT_read": True,
        "bootstrap_executed": False, "bindings": score_bindings,
    }
    temporal_score_barrier = {
        "schema": "stvg-temporal-lite-postseal-score-barrier-v1", "status": "complete",
        "count": 768, "case_count": 128, "arm_count": 6,
        "GT_read": True, "bootstrap_executed": False,
        "rows_path": str(TEMPORAL_SCORE_ROWS.relative_to(ROOT)),
        "summary_path": str(TEMPORAL_SCORE_SUMMARY.relative_to(ROOT)),
        "paired_path": str(TEMPORAL_SCORE_PAIRED.relative_to(ROOT)),
        "rows_sha256": temporal_hashes["score_rows"],
        "summary_sha256": temporal_hashes["summary"],
        "paired_sha256": temporal_hashes["paired"],
        "gt_exposure_sha256": temporal_hashes["gt_exposure"],
        "prediction_bindings": score_bindings,
        "scorer_sha256": temporal_pins["scripts/score_stvg_temporal_lite128003.py"],
    }
    temporal_completion = {
        "schema": "stvg-temporal-lite-postseal-cpu-completion-v1",
        "status": "complete", "count": 128, "row_count": 768, "arm_count": 6,
        "GT_read": True, "CPU_only": True, "bootstrap_executed": False,
        "score_barrier_sha256": temporal_hashes["score_barrier"],
    }
    temporal_selection_bindings = dict(fake_structural_bindings)
    temporal_selection_bindings[str(TEMPORAL_SCORE_ROWS.relative_to(ROOT))] = temporal_hashes["score_rows"]
    temporal_selection_bindings[str(TEMPORAL_CPU_COMPLETION.relative_to(ROOT))] = temporal_hashes["completion"]
    temporal_chain_kwargs = {
        "candidate": temporal_candidate, "runtime": temporal_runtime,
        "cpu_barrier": temporal_cpu, "root_readback": temporal_root,
        "prediction": temporal_prediction, "gt_exposure": temporal_gt_exposure,
        "score_rows": temporal_rows, "score_barrier": temporal_score_barrier,
        "completion": temporal_completion, "hashes": temporal_hashes,
        "selection_bindings": temporal_selection_bindings,
    }
    assert validate_temporal_candidate_chain(**temporal_chain_kwargs)
    tampered_hashes = dict(temporal_hashes, candidate="f" * 64)
    try:
        validate_temporal_candidate_chain(**dict(temporal_chain_kwargs, hashes=tampered_hashes))
    except ValueError:
        candidate_hash_tamper_rejected = True
    else:
        raise AssertionError("Temporal candidate chain accepted a changed candidate-lock hash")
    tampered_source_candidate = copy.deepcopy(temporal_candidate)
    tampered_source_candidate["records"][0]["source_temporal_sha256"] = "b" * 64
    try:
        validate_temporal_candidate_chain(**dict(temporal_chain_kwargs, candidate=tampered_source_candidate))
    except ValueError:
        source_temporal_sha_tamper_rejected = True
    else:
        raise AssertionError("Temporal chain accepted a source_temporal SHA mismatch")

    old = {
        "status": "locked", "count": 2, "GT_read": False,
        "records": [
            {"case_id": "Q1", "selected_interval": [1, 5], "source_temporal_sha256": "a" * 64, "GT_read": False},
            {"case_id": "Q2", "selected_interval": [2, 8], "source_temporal_sha256": "b" * 64, "GT_read": False},
        ],
    }
    candidate = {
        "status": "complete", "candidate_variant": "temporal_lite",
        "not_a_deployment_or_scored_method_selection": True, "count": 2, "GT_read": False,
        "records": [
            {"case_id": "Q1", "selected_interval": [2, 6],
             "temporal_lite_interval": [2, 6], "baseline_offset_trust_interval": [1, 5],
             "source_temporal_sha256": "a" * 64, "GT_read": False},
            {"case_id": "Q2", "selected_interval": [3, 9],
             "temporal_lite_interval": [3, 9], "baseline_offset_trust_interval": [2, 8],
             "source_temporal_sha256": "b" * 64, "GT_read": False},
        ],
    }
    lite_temporal = build_selected_temporal_lock(old, candidate, "temporal_lite", expected_count=2)
    offset_temporal = build_selected_temporal_lock(old, candidate, "offset_trust", expected_count=2)
    assert lite_temporal["records"][0]["selected_interval"] == [2, 6]
    assert lite_temporal["records"][1]["selected_interval"] == [3, 9]
    assert offset_temporal["records"][1]["selected_interval"] == [2, 8]
    assert offset_temporal["records"][0]["selected_interval"] == [1, 5]
    assert lite_temporal["records"][0]["source_temporal_sha256"] == "a" * 64
    assert set(lite_temporal["records"][0]) == set(old["records"][0])

    decision = scope_decision(
        {"hc2": 30.0, "vidstg": 20.0}, {"hc2": 30.02, "vidstg": 20.02},
        {"decoder_calls": 30, "median_fit_seconds": 2.0},
        {"decoder_calls": 20, "median_fit_seconds": 1.5}, budget_pp=0.1)
    assert decision["selected_response_scope"] == "event_support"
    guarded = scope_decision(
        {"hc2": 30.0, "vidstg": 20.0}, {"hc2": 30.0, "vidstg": 19.89},
        {"decoder_calls": 30, "median_fit_seconds": 2.0},
        {"decoder_calls": 10, "median_fit_seconds": 1.0}, budget_pp=0.1)
    assert guarded["selected_response_scope"] == "full_grid"
    tied = scope_decision(
        {"hc2": 30.0, "vidstg": 20.0}, {"hc2": 30.0, "vidstg": 20.0},
        {"decoder_calls": 30, "median_fit_seconds": 2.0},
        {"decoder_calls": 10, "median_fit_seconds": 1.0}, budget_pp=0.1)
    assert tied["selected_response_scope"] == "full_grid"

    factors, summaries = choose_lr_by_domain({
        "hc2": {"0.8": 1.0, "0.9": 2.0, "1.0": 2.0, "1.1": 1.5, "1.2": 1.1},
        "vidstg": {"0.8": 1.0, "0.9": 1.2, "1.0": 1.4, "1.1": 1.6, "1.2": 1.8},
    })
    assert factors == {"hc2": 1.0, "vidstg": 1.2}
    assert summaries["hc2"]["selected_mean_vIoU_pp"] == 2.0

    fake_selection = {
        "selected_spatial_trial": "L001", "selected_spatial_recipe": "lite_backtrack",
        "selected_temporal": "temporal_lite", "structural_loss_budget_pp": 0.1,
        "temporal_delta_pp": {"hc2": 0.0, "vidstg": 0.0},
        "temporal_means_pp": {"offset_trust": {}, "temporal_lite": {}}, "_sha256": "selection-hash",
    }
    lock = make_final_recipe_lock(
        selection=fake_selection, temporal_lock_sha256="temporal-hash",
        selected_recipe="lite_backtrack", selected_scope="event_support",
        configs={"hc2": {"lr": 0.005}, "vidstg": {"lr": 0.03}},
        response_settings={"hc2": {"mu": 0.125}, "vidstg": {"mu": 0.125}},
        selected_factors=factors, lr_summaries=summaries,
        scope_summary=decision, trial_map={"scope": ["SCOPE_FG", "SCOPE_ES"]})
    assert lock["status"] == "locked"
    assert lock["completion_scope"] == lock["method_validation_status"] == "pending_reclocked_confirmation"
    assert lock["confirmation_status"] == "pending_reclocked_confirmation"
    assert lock["suite_final"] is False and lock["CURRENT_promoted"] is False
    assert set(lock["configs"]) == {"hc2", "vidstg"}
    fake_runtime = {"status": "locked", "GT_read": False, "pins": {"root-pin": "keep"},
                    "protected": {"protected-pin": "unchanged"}}
    attached = _attach_stage_pins(fake_runtime, "stage-sha", "script-sha")
    assert attached["pins"]["root-pin"] == "keep"
    assert attached["pins"][str(STAGE_RUNTIME_LOCK.relative_to(ROOT))] == "stage-sha"
    assert attached["pins"][str(Path(__file__).resolve().relative_to(ROOT))] == "script-sha"
    assert attached["protected"] == fake_runtime["protected"]
    assert fake_runtime["pins"] == {"root-pin": "keep"}
    return {"status": "pass", "checks": 21, "CPU_only": True,
            "GT_read": False, "GPU_used": False, "worker_started": False,
            "old_temporal_record_schema_preserved": True,
            "two_domain_scope_guard_and_macro": True,
            "independent_domain_five_point_lr": True,
            "scope_exact_tie_keeps_full_grid": True,
            "structural_binding_set_exact_14_and_missing_path_rejected": structural_missing_binding_rejected,
            "temporal_candidate_cpu_root_prediction_score_chain_validated": True,
            "candidate_hash_tamper_rejected": candidate_hash_tamper_rejected,
            "source_temporal_sha_tamper_rejected": source_temporal_sha_tamper_rejected,
            "stage_pins_attach_without_mutating_root_runtime": True,
            "final_parameters_locked_confirmation_pending": True}


def _verify_structural_lock():
    common.verify()
    if not STRUCTURAL_LOCK.is_file():
        raise FileNotFoundError("root STRUCTURAL_SELECTION_LOCK is not ready; dispatch must wait")
    selection = common.read(STRUCTURAL_LOCK)
    if (selection.get("status") != "locked" or selection.get("count") != 128
            or selection.get("GT_read") is not True
            or selection.get("CURRENT_promoted") is not False
            or selection.get("bootstrap_executed") is not False):
        raise ValueError("structural selection lock schema/scope is invalid")
    if selection.get("selected_spatial_trial") not in STRUCTURAL_IDS:
        raise ValueError("selected spatial trial must be L000/L001/L002")
    if selection.get("selected_temporal") not in {"temporal_lite", "offset_trust"}:
        raise ValueError("selected temporal variant is invalid")
    expected_budget = common.read(BASE / "DESIGN_LOCK.json").get("structural_selection_max_per_domain_loss_pp")
    if selection.get("structural_loss_budget_pp") != expected_budget:
        raise ValueError("root structural loss budget differs from the predeclared study lock")
    validate_structural_bindings(selection.get("bindings"))
    selected_trial = selection["selected_spatial_trial"]
    selected_design = common.read(BASE / "trials" / selected_trial / "DESIGN_LOCK.json")
    if selected_design.get("recipe") != selection.get("selected_spatial_recipe"):
        raise ValueError("selected spatial recipe does not match its locked trial")
    for relative, digest in selection["bindings"].items():
        path = ROOT / relative
        if not path.is_file() or common.sha(path) != digest:
            raise ValueError(f"structural selection binding changed: {relative}")
    selection["_sha256"] = common.sha(STRUCTURAL_LOCK)
    return selection


def _write_selected_temporal(selection, candidate_lock=None):
    old = common.read(SEARCH / "TEMPORAL_LOCK.json")
    candidate = candidate_lock if candidate_lock is not None else _verify_temporal_candidate_chain(selection)
    selected = build_selected_temporal_lock(old, candidate, selection["selected_temporal"])
    if SELECTED_TEMPORAL_LOCK.exists():
        if common.read(SELECTED_TEMPORAL_LOCK) != selected:
            raise FileExistsError("SELECTED_TEMPORAL_LOCK exists with different contents")
    else:
        common.write(SELECTED_TEMPORAL_LOCK, selected)
    return selected, common.sha(SELECTED_TEMPORAL_LOCK)


def _prepare_trial(trial_id, recipe, response_scope, multiplier, selection, temporal_sha):
    if FINAL_RECIPE_LOCK.exists():
        raise FileExistsError("FINAL_RECIPE_LOCK already exists; refusing another selection run")
    path = common.make_trial(
        trial_id, recipe, response_scope=response_scope, multiplier=multiplier,
        temporal_path=SELECTED_TEMPORAL_LOCK)
    design = common.read(path / "DESIGN_LOCK.json")
    design.update(
        structural_selection_sha256=selection["_sha256"],
        selected_temporal=selection["selected_temporal"],
        selected_temporal_lock_sha256=temporal_sha,
        continuation_trial=True,
    )
    common.write(path / "DESIGN_LOCK.json", design, mutable=True)
    runtime_path = path / "RUNTIME.json"
    runtime = common.read(runtime_path)
    runtime["pins"][str(STRUCTURAL_LOCK.relative_to(ROOT))] = selection["_sha256"]
    runtime["pins"][str(SELECTED_TEMPORAL_LOCK.relative_to(ROOT))] = temporal_sha
    runtime["pins"][str((path / "DESIGN_LOCK.json").relative_to(ROOT))] = common.sha(path / "DESIGN_LOCK.json")
    runtime = _attach_stage_pins(runtime, common.sha(STAGE_RUNTIME_LOCK), common.sha(Path(__file__)))
    common.write(runtime_path, runtime, mutable=True)
    return path


def _run_trial(trial_id, recipe, response_scope, multiplier, selection, temporal_sha,
               launcher, *, reuse_path=None):
    if reuse_path is not None:
        if not _trial_identity_is_exact(reuse_path, recipe, response_scope, temporal_sha,
                                        selection["_sha256"], selection["selected_temporal"]):
            raise ValueError("attempted center reuse with non-identical locked inputs")
        return {"trial_id": reuse_path.name, "path": str(reuse_path.relative_to(ROOT)),
                "multiplier": float(multiplier), "reused": True}
    path = _prepare_trial(trial_id, recipe, response_scope, multiplier, selection, temporal_sha)
    launcher.child(trial_id + "_student", "run_stvg_lite_event_student_v1.py", trial_id, gpu=True)
    launcher.child(trial_id + "_root", "review_stvg_lite_event002.py", trial_id)
    launcher.child(trial_id + "_score", "measure_stvg_lite_event_v1.py", trial_id)
    score = common.read(path / "POSTSEAL_ROWS.json")
    if score.get("status") != "complete" or score.get("count") != 128 or score.get("GT_read") is not True:
        raise RuntimeError(f"trial score is not complete: {trial_id}")
    return {"trial_id": trial_id, "path": str(path.relative_to(ROOT)),
            "multiplier": float(multiplier), "reused": False}


def _score_means(trial_path):
    result = common.read(trial_path / "POSTSEAL_ROWS.json")
    if result.get("status") != "complete" or result.get("count") != 128:
        raise RuntimeError(f"missing complete 128-source score: {trial_path}")
    means = {}
    for domain in ("hc2", "vidstg"):
        values = [row["metrics"]["m_vIoU"] for row in result["rows"]
                  if row["dataset"] == domain and row["arm"] == "TS_M2"]
        if len(values) != 64 or any(value is None or not math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"incomplete TS_M2 {domain} scores in {trial_path}")
        means[domain] = 100 * math.fsum(float(value) for value in values) / 64
    return means


def _measured_cost(trial_path):
    result = common.read(trial_path / "MATCHED_EFFICIENCY_READBACK.json")
    if result.get("status") != "complete" or result.get("count") != 128:
        raise RuntimeError(f"missing matched efficiency readback: {trial_path}")
    records = result["records"]
    fit_records = [row["fit_cost"] for row in records if row["fit_cost"] is not None]
    calls = [sum(int(count) for count in fit["decode_counts"].values()) for fit in fit_records]
    times = [float(fit["fit_wall_cuda_synchronized_seconds"]) for fit in fit_records]
    if not calls or len(calls) != len(times) or any(not math.isfinite(value) for value in times):
        raise RuntimeError(f"invalid matched efficiency rows: {trial_path}")
    return {"decoder_calls": sum(calls), "median_fit_seconds": statistics.median(times),
            "fitted_query_count": len(fit_records)}


def _assert_dispatch_unit(launcher):
    launcher.resource.UNIT = UNIT
    launcher.resource.BASE = BASE / "execution"
    line = next(row for row in Path("/proc/self/cgroup").read_text().splitlines() if row.startswith("0::"))
    if not line[3:].endswith("/" + UNIT):
        raise RuntimeError(f"dispatch must run inside its own unit {UNIT}")
    properties = dict(
        item.split("=", 1)
        for item in subprocess.check_output(
            ["systemctl", "--user", "show", UNIT,
             "--property=MemoryHigh,MemoryMax,MemorySwapMax"], text=True).splitlines()
        if "=" in item)
    expected = {"MemoryHigh": 16 * GIB, "MemoryMax": 20 * GIB, "MemorySwapMax": GIB}
    if {name: int(properties[name]) for name in expected} != expected:
        raise RuntimeError(f"{UNIT} must have the locked 16/20 GiB and 1 GiB swap limits")
    sample = launcher.resource.sample()
    if sample["available"] < 8 * GIB or sample["gpu_free"] < 26 * GIB or sample["disk_free"] < 30 * GIB:
        raise RuntimeError("host/GPU/disk admission threshold not met for this owned search unit")
    return sample


def run_dispatch():
    from scripts.stvg_lite_recovery004_common import verify_recovery004
    verify_recovery004()
    """Run after explicit --dispatch, after external structural selection is sealed."""
    import threading
    from scripts import launch_stvg_lite_event_v1 as launcher

    if FINAL_RECIPE_LOCK.exists():
        raise FileExistsError("FINAL_RECIPE_LOCK already exists; refusing another selection run")
    selection = _verify_structural_lock()
    candidate_lock = _verify_temporal_candidate_chain(selection)
    selected_temporal, temporal_sha = _write_selected_temporal(selection, candidate_lock)
    stage_runtime_sha = _lock_stage_runtime(selection, temporal_sha)
    # Keep the root's stored old record schema; only selected intervals differ.
    old_keys = [set(row) for row in common.read(SEARCH / "TEMPORAL_LOCK.json")["records"]]
    if any(set(row) != keys for row, keys in zip(selected_temporal["records"], old_keys)):
        raise AssertionError("selected temporal lock did not preserve the standard record schema")

    admission = _assert_dispatch_unit(launcher)
    stop = threading.Event()
    monitor = threading.Thread(target=launcher.resource.monitor, args=(stop,), daemon=True)
    monitor.start()
    trial_map = {"scope": {}, "lr": {}}
    scope_summary = None
    diagnostic_scope = None
    selected_recipe = selection["selected_spatial_recipe"]
    selected_scope = "full_grid"

    try:
        if selected_recipe == "full":
            # The selected route is Full. Lite-fullstep is recorded only as an
            # unselected mechanism diagnostic and cannot enter the final lock.
            diagnostic_recipe = "lite_fullstep"
            full_id, event_id = "DIAG_FG", "DIAG_ES"
            full_trial = _run_trial(full_id, diagnostic_recipe, "full_grid", 1.0,
                                    selection, temporal_sha, launcher)
            event_trial = _run_trial(event_id, diagnostic_recipe, "event_support", 1.0,
                                     selection, temporal_sha, launcher)
            trial_map["scope"] = {"full_grid": full_trial, "event_support": event_trial}
            full_path, event_path = ROOT / full_trial["path"], ROOT / event_trial["path"]
            decision = scope_decision(
                _score_means(full_path), _score_means(event_path),
                _measured_cost(full_path), _measured_cost(event_path),
                budget_pp=selection["structural_loss_budget_pp"])
            scope_summary = decision
            diagnostic_scope = {"recipe": diagnostic_recipe, "promoted": False, **decision}
            selected_scope = "full_grid"
            lr_recipe = "full"
            lr_scope = "full_grid"
        else:
            if selected_recipe not in {"lite_backtrack", "lite_fullstep"}:
                raise ValueError(f"unsupported selected Lite recipe: {selected_recipe}")
            full_id, event_id = "SCOPE_FG", "SCOPE_ES"
            full_trial = _run_trial(full_id, selected_recipe, "full_grid", 1.0,
                                    selection, temporal_sha, launcher)
            event_trial = _run_trial(event_id, selected_recipe, "event_support", 1.0,
                                     selection, temporal_sha, launcher)
            trial_map["scope"] = {"full_grid": full_trial, "event_support": event_trial}
            full_path, event_path = ROOT / full_trial["path"], ROOT / event_trial["path"]
            decision = scope_decision(
                _score_means(full_path), _score_means(event_path),
                _measured_cost(full_path), _measured_cost(event_path),
                budget_pp=selection["structural_loss_budget_pp"])
            scope_summary = decision
            selected_scope = decision["selected_response_scope"]
            lr_recipe = selected_recipe
            lr_scope = selected_scope

        # At most the exact factor-1.0 scope trial may be reused. The explicit
        # identity check includes recipe, scope, configs/settings, temporal and
        # reference hashes, plus the structural-selection lock.
        candidate_center = None
        if selected_recipe != "full":
            candidate_center = ROOT / trial_map["scope"][selected_scope]["path"]
            if not _trial_identity_is_exact(candidate_center, lr_recipe, lr_scope,
                                            temporal_sha, selection["_sha256"], selection["selected_temporal"]):
                candidate_center = None

        factor_scores = {"hc2": {}, "vidstg": {}}
        expected_configs_by_factor = {}
        expected_settings = None
        for factor in LR_FACTORS:
            if factor == 1.0 and candidate_center is not None:
                trial = _run_trial(candidate_center.name, lr_recipe, lr_scope, factor,
                                   selection, temporal_sha, launcher, reuse_path=candidate_center)
                trial_map["lr"][str(factor)] = trial
            else:
                tid = {0.8: "LR_080", 0.9: "LR_090", 1.0: "LR_100",
                       1.1: "LR_110", 1.2: "LR_120"}[factor]
                trial = _run_trial(tid, lr_recipe, lr_scope, factor,
                                   selection, temporal_sha, launcher)
                trial_map["lr"][str(factor)] = trial
            path = ROOT / trial["path"]
            means = _score_means(path)
            for domain in ("hc2", "vidstg"):
                factor_scores[domain][str(factor)] = means[domain]
            expected_configs, expected_settings = _expected_trial_inputs(factor)
            expected_configs_by_factor[str(factor)] = expected_configs

        selected_factors, lr_summaries = choose_lr_by_domain(factor_scores)
        base_final = common.read(COMP / "FINAL_PARAMETER_LOCK.json")
        configs = {}
        response_settings = {}
        for domain in ("hc2", "vidstg"):
            factor = selected_factors[domain]
            configs[domain] = dict(base_final["configs"][domain],
                                   lr=base_final["configs"][domain]["lr"] * factor)
            response_settings[domain] = copy.deepcopy(base_final["response_settings"][domain])

        final_lock = make_final_recipe_lock(
            selection=selection,
            temporal_lock_sha256=temporal_sha,
            selected_recipe=selected_recipe,
            selected_scope=selected_scope,
            configs=configs,
            response_settings=response_settings,
            selected_factors=selected_factors,
            lr_summaries=lr_summaries,
            scope_summary=scope_summary,
            trial_map=trial_map,
            diagnostic_scope=diagnostic_scope)
        final_lock["lr_search"]["factor_scores_pp"] = factor_scores
        final_lock["lr_search"]["configs_by_factor"] = expected_configs_by_factor
        final_lock["dispatch_unit"] = UNIT
        final_lock["dispatch_admission"] = admission
        final_lock["stage_execution_runtime_sha256"] = stage_runtime_sha
        final_lock["stage_execution_runtime_path"] = str(STAGE_RUNTIME_LOCK.relative_to(ROOT))
        final_lock["continuation_script_sha256"] = common.sha(Path(__file__))
        final_lock["selected_temporal_lock_path"] = str(SELECTED_TEMPORAL_LOCK.relative_to(ROOT))
        final_lock["selected_temporal_lock_records_schema"] = "legacy records with selected_interval replaced; source hashes preserved"
        common.write(FINAL_RECIPE_LOCK, final_lock)
        return final_lock
    finally:
        stop.set()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contracts", action="store_true",
                      help="run synthetic CPU control-flow checks only")
    mode.add_argument("--dispatch", action="store_true",
                      help="explicitly dispatch the bounded continuation after root selection is sealed")
    args = parser.parse_args(argv)
    if args.contracts:
        print(json.dumps(contracts(), indent=2, sort_keys=True))
        return 0
    result = run_dispatch()
    print(json.dumps({"status": result["status"],
                      "final_recipe_lock": str(FINAL_RECIPE_LOCK.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

