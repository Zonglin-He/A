"""Run one locked 128-source Soft EP coordinate-search trial without GT."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stvg_soft_ep_coordinate_common_v1 import BASE, load_case, read, save_pt, sha, verify_runtime, write, guard

TRIALS = BASE / "trials"
ROSTER_PATH = BASE / "ROSTER_LOCK.json"
SOURCE_BARRIER_PATH = BASE / "SOURCE_CAPTURE_BARRIER.json"
REFERENCE_PATH = BASE / "REFERENCE_LOCK.json"
TEMPORAL_PATH = BASE / "TEMPORAL_LOCK.json"
ARM_NAMES = ("F", "T", "S_M0", "TS_M0", "S_M1", "TS_M1", "S_M2", "TS_M2")
MODES = ("M0", "M1", "M2")


def _case_id(row):
    value = row.get("case_id", row.get("probe_alias"))
    if not isinstance(value, str) or not value:
        raise ValueError("roster/reference/temporal row lacks case identity")
    return value


def _records(path, label):
    doc = read(path)
    if doc.get("status") != "locked" or doc.get("GT_read") is not False:
        raise RuntimeError(f"{label} must be locked and explicitly GT-free")
    rows = doc.get("records")
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} records must be a list")
    return doc, rows


def _lock_file(trial_dir: Path, name: str) -> Path:
    local = trial_dir / name
    global_path = BASE / name
    if local.is_file():
        return local
    if global_path.is_file():
        return global_path
    raise FileNotFoundError(f"missing locked {name} at trial or study root")


def _assert_runtime(trial_dir: Path, trial_id: str, design_path: Path,
                    roster_path: Path, source_barrier_path: Path,
                    reference_path: Path, temporal_path: Path):
    runtime_path = trial_dir / "RUNTIME.json"
    runtime = read(runtime_path)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise RuntimeError("trial RUNTIME.json must be locked with GT_read=false")
    if runtime.get("trial_id", trial_id) != trial_id:
        raise RuntimeError("trial runtime ID mismatch")
    pins = runtime.get("pins")
    protected = runtime.get("protected")
    if not isinstance(pins, dict) or not isinstance(protected, dict):
        raise RuntimeError("trial runtime must bind pins and protected inputs")
    hashes = {**pins, **protected}
    for rel, expected in hashes.items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"runtime source/protected binding mismatch: {rel}")
    required_code = (
        "scripts/run_stvg_soft_ep_coordinate_trial003.py",
        "scripts/stvg_soft_ep_coordinate_fit002.py",
    )
    for rel in required_code:
        if pins.get(rel) != sha(ROOT / rel):
            raise RuntimeError(f"runtime must pin this worker source: {rel}")
    required_files = (design_path, roster_path, source_barrier_path, reference_path, temporal_path)
    for path in required_files:
        rel = str(path.relative_to(ROOT))
        if hashes.get(rel) != sha(path):
            raise RuntimeError(f"trial runtime does not bind required lock/input: {rel}")
    binding_fields = {
        "design_lock_sha256": sha(design_path),
        "roster_lock_sha256": sha(roster_path),
        "source_capture_barrier_sha256": sha(source_barrier_path),
        "reference_lock_sha256": sha(reference_path),
        "temporal_lock_sha256": sha(temporal_path),
    }
    for name, value in binding_fields.items():
        if name in runtime and runtime[name] != value:
            raise RuntimeError(f"runtime {name} mismatch")
    return runtime, runtime_path


def _barrier_filemap(path: Path, expected_count: int, label: str):
    doc = read(path)
    if (doc.get("status") != "complete" or doc.get("count") != expected_count
            or doc.get("GT_read") is not False):
        raise RuntimeError(f"{label} barrier is not complete/GT-free at expected count")
    rows = doc.get("files")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError(f"{label} barrier file count mismatch")
    out = {}
    for row in rows:
        key = row.get("case_id", row.get("probe_alias", row.get("path")))
        if key in out:
            raise RuntimeError(f"duplicate {label} barrier entry: {key}")
        out[key] = row
    return doc, out


def _validate_config(config, dataset, mode):
    from scripts import stvg_target_teacher_student_v1 as core
    if dataset not in ("hc2", "vidstg") or mode not in MODES:
        raise ValueError("unsupported locked dataset/mode")
    expected = core.load_original_opd_config(dataset)
    if not isinstance(config, dict) or set(config) != set(expected):
        raise ValueError(f"{dataset}/{mode} config schema differs from original OPD")
    if config.get("samples") != 32 or config.get("writeback") != expected["writeback"]:
        raise ValueError(f"{dataset}/{mode} changes fixed writeback/sample count")
    tunable = {"lr", "sigma", "tau", "steps"}
    if any(config[k] != expected[k] for k in set(expected) - tunable):
        raise ValueError(f"{dataset}/{mode} changed a fixed OPD setting")
    for name, low, high in (("lr", 1e-6, 1.5), ("sigma", 1e-4, 0.4), ("tau", 1e-4, 2.0)):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not low <= float(value) <= high:
            raise ValueError(f"{dataset}/{mode} {name} outside locked search bounds")
    if isinstance(config.get("steps"), bool) or not isinstance(config.get("steps"), int) or not 1 <= config["steps"] <= 8:
        raise ValueError(f"{dataset}/{mode} steps outside [1, 8]")
    if mode != "M0":
        from scripts.stvg_soft_ep_coordinate_fit002 import validate_config
        validate_config(config, mode)
    return dict(config)


def _validate_interval(interval, bounds, label):
    import numpy as np
    values = np.asarray(interval, dtype=np.float64)
    limits = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2,) or limits.shape != (2,) or not np.isfinite(values).all() or not np.isfinite(limits).all():
        raise ValueError(f"{label} interval/bounds must be finite pairs")
    if not limits[0] <= values[0] < values[1] <= limits[1]:
        raise ValueError(f"{label} interval is outside source physical-time bounds")
    return [float(values[0]), float(values[1])]


def _validate_locks(trial_id: str):
    trial_dir = TRIALS / trial_id
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"root must prepare locked trial directory: {trial_dir}")
    design_path = trial_dir / "DESIGN_LOCK.json"
    roster_path = ROSTER_PATH
    source_barrier_path = SOURCE_BARRIER_PATH
    reference_path = _lock_file(trial_dir, "REFERENCE_LOCK.json")
    temporal_path = _lock_file(trial_dir, "TEMPORAL_LOCK.json")
    runtime, runtime_path = _assert_runtime(trial_dir, trial_id, design_path, roster_path,
                                            source_barrier_path, reference_path, temporal_path)
    design = read(design_path)
    if design.get("status") != "locked" or design.get("GT_read") is not False:
        raise RuntimeError("DESIGN_LOCK must be locked and explicitly GT-free")
    if design.get("trial_id", trial_id) != trial_id:
        raise RuntimeError("design lock trial ID mismatch")
    roster_doc, roster = _records(roster_path, "ROSTER_LOCK")
    refs_doc, ref_rows = _records(reference_path, "REFERENCE_LOCK")
    temporal_doc, temporal_rows = _records(temporal_path, "TEMPORAL_LOCK")
    source_doc, source_files = _barrier_filemap(source_barrier_path, 128, "SOURCE_CAPTURE")
    if roster_doc.get("count") != 128 or len(roster) != 128:
        raise RuntimeError("expected exactly 128 locked source cases")
    if refs_doc.get("count", 128) != 128 or len(ref_rows) != 128:
        raise RuntimeError("expected exactly 128 locked reference records")
    if temporal_doc.get("count", 128) != 128 or len(temporal_rows) != 128:
        raise RuntimeError("expected exactly 128 locked temporal records")
    aliases = [_case_id(row) for row in roster]
    if len(set(aliases)) != 128:
        raise RuntimeError("roster case identities are not unique")
    ref_map = {_case_id(row): row for row in ref_rows}
    temporal_map = {_case_id(row): row for row in temporal_rows}
    if len(ref_map) != 128 or len(temporal_map) != 128 or set(ref_map) != set(aliases) or set(temporal_map) != set(aliases):
        raise RuntimeError("reference/temporal lock case set differs from roster")
    if set(source_files) != set(aliases):
        # Some source barriers index files by relative path. Re-index with each roster binding.
        by_path = {row.get("path"): row for row in source_doc["files"]}
        if set(by_path) != {row.get("source_case_path") for row in roster}:
            raise RuntimeError("source capture barrier cases/paths differ from roster")
        source_files = { _case_id(row): by_path[row["source_case_path"]] for row in roster }
    configs = design.get("configs")
    response_settings = design.get("response_settings")
    reuse_matrix = design.get("family_reuse_from")
    if not all(isinstance(x, dict) for x in (configs, response_settings, reuse_matrix)):
        raise RuntimeError("design lock must declare configs, response_settings, family_reuse_from")
    if set(configs) != {"hc2", "vidstg"} or set(reuse_matrix) != {"hc2", "vidstg"}:
        raise RuntimeError("design lock must cover both locked domains")
    validated_configs = {}
    validated_settings = {}
    for dataset in ("hc2", "vidstg"):
        if set(configs[dataset]) != set(MODES) or set(reuse_matrix[dataset]) != set(MODES):
            raise RuntimeError(f"{dataset} design must define M0/M1/M2")
        validated_configs[dataset] = {mode: _validate_config(configs[dataset][mode], dataset, mode) for mode in MODES}
        if set(response_settings.get(dataset, {})) != {"M1", "M2"}:
            raise RuntimeError(f"{dataset} response settings must define M1/M2 only")
        validated_settings[dataset] = copy.deepcopy(response_settings[dataset])
        for mode in ("M1", "M2"):
            if not isinstance(validated_settings[dataset][mode], dict):
                raise RuntimeError(f"invalid response settings for {dataset}/{mode}")
        for mode in MODES:
            reuse_id = reuse_matrix[dataset][mode]
            if reuse_id is not None and (not isinstance(reuse_id, str) or reuse_id == trial_id):
                raise RuntimeError(f"invalid prior trial ID for {dataset}/{mode}")
    for row in roster:
        cid = _case_id(row)
        ref = ref_map[cid]
        temporal = temporal_map[cid]
        if ref.get("dataset", row.get("dataset")) != row.get("dataset") or temporal.get("dataset", row.get("dataset")) != row.get("dataset"):
            raise RuntimeError(f"domain binding mismatch for {cid}")
        expected_source_sha = source_files[cid].get("source_case_sha256")
        if expected_source_sha is not None and ref.get("source_case_sha256") != expected_source_sha:
            raise RuntimeError(f"reference source hash mismatch for {cid}")
        if expected_source_sha is not None and temporal.get("source_case_sha256") not in (None, expected_source_sha):
            raise RuntimeError(f"temporal source hash mismatch for {cid}")
        if not isinstance(ref.get("anchors"), list):
            raise RuntimeError(f"reference anchors missing for {cid}")
        if "selected_interval" not in temporal:
            raise RuntimeError(f"selected interval missing for {cid}")
        src = source_files[cid]
        expected_path = row.get("source_case_path")
        if src.get("path") != expected_path:
            raise RuntimeError(f"source path binding mismatch for {cid}")
        if expected_path is None or not isinstance(src.get("sha256"), str):
            raise RuntimeError(f"source file binding missing for {cid}")
    return trial_dir, runtime_path, runtime, design_path, design, roster_doc, roster, ref_map, temporal_map, source_doc, source_files, validated_configs, validated_settings


def _stable_fit_key(row):
    from scripts.run_stvg_table1_student004 import stable_fit_key
    return stable_fit_key(str(row["dataset"]), str(row["query_key_sha256"]))


def _barrier_index(trial_dir: Path):
    barrier_path = trial_dir / "STUDENT_GLOBAL_BARRIER.json"
    doc, filemap = _barrier_filemap(barrier_path, 128, "prior STUDENT")
    if doc.get("stage") not in (None, "development", "coordinate_trial"):
        raise RuntimeError("prior trial student barrier stage is not reusable")
    return doc, filemap


def _load_prior_trial_context(prior_id: str, current_runtime_hashes: dict):
    """Verify immutable prior-trial metadata once and cache its case index.

    Per-case prediction/receipt/source/reference/fit checks remain in
    ``_verify_prior_trial`` so this optimization only avoids rescanning the
    same prior runtime pins and global barrier for every reused case.
    """
    prior_dir = TRIALS / prior_id
    design_path = prior_dir / "DESIGN_LOCK.json"
    runtime_path = prior_dir / "RUNTIME.json"
    barrier_path = prior_dir / "STUDENT_GLOBAL_BARRIER.json"
    if not all(path.is_file() for path in (design_path, runtime_path, barrier_path)):
        raise RuntimeError(f"reuse trial {prior_id} lacks a complete design/runtime/barrier")

    runtime_rel = str(runtime_path.relative_to(ROOT))
    barrier_rel = str(barrier_path.relative_to(ROOT))
    old_runtime_sha = sha(runtime_path)
    old_barrier_sha = sha(barrier_path)
    if current_runtime_hashes.get(runtime_rel) != old_runtime_sha:
        raise RuntimeError(f"current runtime does not bind prior trial runtime: {runtime_rel}")
    if current_runtime_hashes.get(barrier_rel) != old_barrier_sha:
        raise RuntimeError(f"current runtime does not bind prior trial barrier: {barrier_rel}")

    old_runtime = read(runtime_path)
    if old_runtime.get("status") != "locked" or old_runtime.get("GT_read") is not False:
        raise RuntimeError(f"reuse trial {prior_id} runtime is not locked/GT-free")
    oldpins = old_runtime.get("pins", {})
    oldprotected = old_runtime.get("protected", {})
    if not isinstance(oldpins, dict) or not isinstance(oldprotected, dict):
        raise RuntimeError(f"reuse trial {prior_id} runtime pin schema invalid")
    for rel, expected in {**oldpins, **oldprotected}.items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"reuse trial {prior_id} pin mismatch: {rel}")

    design_sha = sha(design_path)
    if oldpins.get(str(design_path.relative_to(ROOT))) != design_sha and oldprotected.get(str(design_path.relative_to(ROOT))) != design_sha:
        raise RuntimeError(f"reuse trial {prior_id} runtime does not bind its design lock")
    old_design = read(design_path)
    if old_design.get("status") != "locked" or old_design.get("GT_read") is not False:
        raise RuntimeError(f"reuse trial {prior_id} design is not locked/GT-free")
    old_barrier, files = _barrier_index(prior_dir)
    return dict(prior_dir=prior_dir, design=old_design, barrier=old_barrier,
                files=files, barrier_sha256=old_barrier_sha,
                runtime_sha256=old_runtime_sha, design_sha256=design_sha)


def _cached_prior_trial_context(prior_id: str, current_runtime_hashes: dict,
                                prior_context_cache: dict):
    context = prior_context_cache.get(prior_id)
    if context is None:
        context = _load_prior_trial_context(prior_id, current_runtime_hashes)
        prior_context_cache[prior_id] = context
    return context


def _verify_prior_trial(prior_id: str, dataset: str, mode: str, current_row: dict,
                        current_case: dict, reference: dict, config: dict,
                        settings: dict | None, key: str, reference_lock_sha256: str,
                        current_runtime_hashes: dict, prior_context_cache: dict):
    import torch
    context = _cached_prior_trial_context(prior_id, current_runtime_hashes,
                                          prior_context_cache)
    prior_dir = context["prior_dir"]
    old_design = context["design"]
    if old_design.get("configs", {}).get(dataset, {}).get(mode) != config:
        raise RuntimeError(f"reuse trial {prior_id} config differs for {dataset}/{mode}")
    if mode in ("M1", "M2") and old_design.get("response_settings", {}).get(dataset, {}).get(mode) != settings:
        raise RuntimeError(f"reuse trial {prior_id} response settings differ for {dataset}/{mode}")
    old_barrier, files = context["barrier"], context["files"]
    rel = str((prior_dir / "student" / f"{_case_id(current_row)}.pt").relative_to(ROOT))
    entry = files.get(_case_id(current_row))
    if entry is None:
        entry = next((x for x in old_barrier["files"] if x.get("path") == rel), None)
    if entry is None or entry.get("path") != rel:
        raise RuntimeError(f"reuse trial {prior_id} lacks {_case_id(current_row)} output")
    path = ROOT / rel
    receipt_path = path.with_suffix(".receipt.json")
    if (entry.get("receipt_path") != str(receipt_path.relative_to(ROOT))
            or not receipt_path.is_file() or sha(receipt_path) != entry.get("receipt_sha256")):
        raise RuntimeError(f"reuse trial {prior_id} receipt file binding mismatch: {rel}")
    if not path.is_file() or sha(path) != entry.get("sha256"):
        raise RuntimeError(f"reuse trial {prior_id} output SHA mismatch: {rel}")
    receipt = read(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("sha256") != sha(path) or receipt.get("GT_read") is not False:
        raise RuntimeError(f"reuse trial {prior_id} output receipt mismatch: {rel}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("GT_read") is not False or payload.get("case_id") != _case_id(current_row):
        raise RuntimeError(f"reuse trial {prior_id} payload identity/GT flag mismatch")
    checks = {
        "dataset": dataset,
        "source_case_sha256": current_case["source_case_sha256"],
        "source_model_state_sha256": current_case["source_model_state_sha256"],
        "pixel_sha256": current_case["pixel_sha256"],
        "frame_ids": list(map(int, current_case["frame_ids"])),
        "reference_records": reference,
        "reference_lock_sha256": reference_lock_sha256,
    }
    for name, expected in checks.items():
        actual = payload.get(name)
        if name == "frame_ids":
            actual = list(map(int, actual or []))
        if actual != expected:
            raise RuntimeError(f"reuse trial {prior_id} payload {name} mismatch")
    fit = payload.get("fits", {}).get(mode)
    readout = payload.get("readouts", {}).get(mode)
    if not isinstance(fit, dict) or not isinstance(readout, dict):
        raise RuntimeError(f"reuse trial {prior_id} lacks complete fit/readout for {mode}")
    if "reused_fit_from_sha256" in fit:
        raise RuntimeError("fit reuse chains are forbidden; family_reuse_from must name an actual-fit trial")
    if fit.get("config") != config or fit.get("seed_key") != key or fit.get("GT_read") is not False:
        raise RuntimeError(f"reuse trial {prior_id} fit configuration/seed mismatch")
    if mode in ("M1", "M2"):
        if fit.get("mode") != mode or fit.get("settings") != settings:
            raise RuntimeError(f"reuse trial {prior_id} fit response settings/mode mismatch")
    elif fit.get("algorithm") != "validated_loss_weighted_iou" or fit.get("reference_sha256") != reference_lock_sha256:
        raise RuntimeError(f"reuse trial {prior_id} M0 is not the sealed validated loss fit")
    boxes = fit.get("final_cached_decoder_boxes")
    if boxes is None or not torch.equal(readout.get("boxes").detach().cpu(), boxes.detach().cpu()):
        raise RuntimeError(f"reuse trial {prior_id} fit/readout cached boxes differ")
    return path, payload, copy.deepcopy(fit), copy.deepcopy(readout)


def _no_partial_outputs(trial_dir: Path, roster):
    if (trial_dir / "STUDENT_GLOBAL_BARRIER.json").exists():
        raise FileExistsError("student barrier already exists; no rerun over sealed output")
    if (trial_dir / "STATUS.json").exists():
        raise FileExistsError("student status already exists; inspect prior attempt before retry")
    for name in ("student", "partial"):
        directory = trial_dir / name
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"nonempty {name} output exists; preserving partial attempt")


def _build_arms(native_xyxy, native_interval, selected_interval, readouts, has_reference, xyxy_fn):
    import numpy as np
    native_boxes = np.asarray(native_xyxy).copy()
    arms = {
        "F": dict(boxes=native_boxes.copy(), interval=list(native_interval)),
        "T": dict(boxes=native_boxes.copy(), interval=list(selected_interval)),
    }
    for mode in MODES:
        if not has_reference:
            arms[f"S_{mode}"] = copy.deepcopy(arms["F"])
            arms[f"TS_{mode}"] = copy.deepcopy(arms["T"])
        else:
            output = readouts.get(mode)
            if not isinstance(output, dict) or output.get("boxes") is None:
                raise ValueError(f"missing fitted readout for {mode}")
            boxes = xyxy_fn(output["boxes"])
            arms[f"S_{mode}"] = dict(boxes=boxes.copy(), interval=list(native_interval))
            arms[f"TS_{mode}"] = dict(boxes=boxes.copy(), interval=list(selected_interval))
    if set(arms) != set(ARM_NAMES):
        raise AssertionError("student arm set mismatch")
    return arms


def contracts():
    import numpy as np
    from scripts import stvg_target_teacher_student_v1 as core
    checks = 0
    # Exercise the lock-entry symbol without requiring a real locked trial.
    assert isinstance(ROSTER_PATH, Path)
    assert "ROSTER_PATH" in _validate_locks.__code__.co_names
    assert "ROOSTER_PATH" not in _validate_locks.__code__.co_names
    checks += 1
    # Prove prior runtime/design/barrier verification is one-time per prior ID.
    global _load_prior_trial_context
    original_loader = _load_prior_trial_context
    loads = []
    def synthetic_loader(prior_id, current_runtime_hashes):
        loads.append((prior_id, current_runtime_hashes))
        return {"prior_id": prior_id}
    try:
        _load_prior_trial_context = synthetic_loader
        cache = {}
        runtime_hashes = {"locked": "synthetic"}
        first = _cached_prior_trial_context("prior-a", runtime_hashes, cache)
        second = _cached_prior_trial_context("prior-a", runtime_hashes, cache)
        assert first is second and len(loads) == 1
        assert loads[0] == ("prior-a", runtime_hashes)
        checks += 2
    finally:
        _load_prior_trial_context = original_loader
    ids = [f"e{i:03d}" for i in range(128)]
    assert len(set(ids)) == 128
    for dataset in ("hc2", "vidstg"):
        original = core.load_original_opd_config(dataset)
        for steps in (3, 5, 8):
            config = dict(original, steps=steps)
            for mode in MODES:
                assert _validate_config(config, dataset, mode) == config
                checks += 1
    source_boxes = np.arange(20, dtype=np.float64).reshape(5, 4) / 100.0
    f_interval = [1, 17]
    t_interval = [4, 21]
    empty = _build_arms(source_boxes, f_interval, t_interval, {}, False, lambda x: x)
    for mode in MODES:
        assert np.array_equal(empty[f"S_{mode}"]["boxes"], source_boxes)
        assert np.array_equal(empty[f"TS_{mode}"]["boxes"], source_boxes)
        assert empty[f"S_{mode}"]["interval"] == f_interval
        assert empty[f"TS_{mode}"]["interval"] == t_interval
        checks += 4
    readouts = {mode: {"boxes": source_boxes + 0.01} for mode in MODES}
    fitted = _build_arms(source_boxes, f_interval, t_interval, readouts, True, lambda x: x)
    for mode in MODES:
        assert np.array_equal(fitted[f"S_{mode}"]["boxes"], readouts[mode]["boxes"])
        assert np.array_equal(fitted[f"TS_{mode}"]["boxes"], readouts[mode]["boxes"])
        assert fitted[f"S_{mode}"]["interval"] == f_interval
        assert fitted[f"TS_{mode}"]["interval"] == t_interval
        checks += 4
    bounds = [0.0, 100.0]
    assert _validate_interval([12.5, 31.0], bounds, "synthetic") == [12.5, 31.0]
    checks += 1
    for invalid in ([0.0, 0.0], [-1.0, 2.0], [2.0, 1.0], [0.0, math.inf]):
        try:
            _validate_interval(invalid, bounds, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid synthetic interval accepted: {invalid}")
        checks += 1
    return dict(status="pass", checks=checks, scope="synthetic config/arm/cardinality/interval contracts",
                case_count=len(ids), GT_read=False, GPU_used=False, model_loaded=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_id", nargs="?")
    parser.add_argument("--contracts", action="store_true")
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(contracts(), sort_keys=True))
        return
    if not args.trial_id or "/" in args.trial_id or args.trial_id in {".", ".."}:
        parser.error("provide one simple trial id")

    import numpy as np
    import torch
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    from scripts import run_stvg_table1_source_v1 as source_helper
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
    from scripts.stvg_weightfree_coordinate_fit002 import fit_validated
    from scripts.stvg_soft_ep_coordinate_fit002 import fit as fit_generalized

    # Verify immutable locks and trial outputs before loading a model or touching video.
    verify_runtime()
    (trial_dir, runtime_path, runtime, design_path, design, roster_doc, roster,
     refs, temporal, source_doc, source_files, configs, settings) = _validate_locks(args.trial_id)
    _no_partial_outputs(trial_dir, roster)
    sys.addaudithook(guard)
    install_clean_loader()
    for rec in roster:
        cid = _case_id(rec)
        rel = rec["source_case_path"]
        entry = source_files[cid]
        p = ROOT / rel
        if sha(p) != entry["sha256"] or p.stat().st_size != int(entry.get("bytes", p.stat().st_size)):
            raise RuntimeError(f"source case file differs from sealed source barrier: {cid}")
        receipt = read(p.with_suffix(".receipt.json"))
        if (not (receipt.get("status") == "sealed" or (receipt.get("status") is None and receipt.get("schema") == "stvg-final-table1-source-receipt-v1" and receipt.get("query_key_sha256") == rec["query_key_sha256"])) or receipt.get("sha256") != entry["sha256"]
                or receipt.get("bytes", p.stat().st_size) != p.stat().st_size
                or receipt.get("GT_read") is not False or receipt.get("case_id") != cid
                or receipt.get("source_case_sha256") != entry["source_case_sha256"]):
            raise RuntimeError(f"source case receipt mismatch: {cid}")
        # Deserialize only the frozen source cache; common.load_case validates its hash and query key.
        case, actual_path = load_case(rec)
        if actual_path != p or case.get("source_case_sha256") != entry.get("source_case_sha256", case.get("source_case_sha256")):
            raise RuntimeError(f"source-case identity mismatch: {cid}")
        ref = refs[cid]
        if ref.get("source_case_sha256") not in (None, case["source_case_sha256"]):
            raise RuntimeError(f"reference/source binding mismatch: {cid}")
        trow = temporal[cid]
        _validate_interval(trow["selected_interval"], case["video_bounds"], f"{cid} selected temporal")
        if list(map(int, rec.get("frame_ids", []))) != list(map(int, case["frame_ids"])):
            raise RuntimeError(f"roster/source frame grid mismatch: {cid}")
        del case
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    source_filemap = {str(row["path"]): row for row in source_doc["files"]}
    ref_sha = sha(_lock_file(trial_dir, "REFERENCE_LOCK.json"))
    temporal_sha = sha(_lock_file(trial_dir, "TEMPORAL_LOCK.json"))
    roster_sha = sha(ROSTER_PATH)
    model = None
    current_source = None
    current_runtime_hashes = {**runtime.get("pins", {}), **runtime.get("protected", {})}
    prior_context_cache = {}
    final_files = []
    checkpoint_files = []
    source_capture_count = fit_invocations = final_readout_calls = reused_fit_count = 0
    mode_invocations = {mode: 0 for mode in MODES}
    mode_steps = {mode: 0 for mode in MODES}
    started = time.perf_counter()

    for number, rec in enumerate(roster, 1):
        cid = _case_id(rec)
        dataset = rec["dataset"]
        case, source_path = load_case(rec)
        ref_record = refs[cid]
        temporal_record = temporal[cid]
        selected_interval = _validate_interval(temporal_record["selected_interval"], case["video_bounds"], f"{cid} selected temporal")
        native_interval = list(case["Native_interval"])
        frame_ids = list(map(int, case["frame_ids"]))
        source_sha = case["source_case_sha256"]
        expected_state = case["source_model_state_sha256"]
        pixel_sha = case["pixel_sha256"]
        key = _stable_fit_key(rec)
        anchors_json = ref_record["anchors"]
        if not isinstance(anchors_json, list):
            raise RuntimeError(f"reference anchors malformed: {cid}")
        fit_needed = bool(anchors_json) and any(design["family_reuse_from"][dataset][mode] is None for mode in MODES)
        source_seconds = 0.0
        source_cold = None
        base = model_batch = native = raw = records = None
        initial = None
        cached_outputs = {}
        fits = {mode: None for mode in MODES}
        readouts = {mode: None for mode in MODES}
        reuse_receipts = {}

        if anchors_json:
            for mode in MODES:
                prior_id = design["family_reuse_from"][dataset][mode]
                if prior_id is None:
                    continue
                prior_path, prior_payload, fit_copy, readout_copy = _verify_prior_trial(
                    prior_id, dataset, mode, rec, case, ref_record,
                    configs[dataset][mode], settings[dataset].get(mode), key, ref_sha,
                    current_runtime_hashes, prior_context_cache,
                )
                fits[mode] = fit_copy
                fits[mode]["reused_fit_from_sha256"] = sha(prior_path)
                fits[mode]["reused_fit_from_trial"] = prior_id
                readouts[mode] = readout_copy
                reuse_receipts[mode] = dict(trial_id=prior_id, path=str(prior_path.relative_to(ROOT)),
                                             sha256=sha(prior_path), no_new_fit=True, GT_read=False)
                reused_fit_count += 1

        if fit_needed:
            transient = dict(case, source=source_helper.SOURCE_MODEL_NAME[dataset],
                             _source_case_sha256=source_sha)
            _, source_row = source_helper._validate_roster_record(rec)
            transient["parent"] = source_row["source"]
            old_catalog = h.catalog

            def _catalog(_record):
                assert _case_id(_record) == cid and _record["source_case_sha256"] == source_sha
                return transient, {}

            h.catalog = _catalog
            try:
                (model, current_source, _, _, raw, frames, captured_ids, records,
                 model_batch, base, native, got_state, got_pixel, source_seconds) = h._capture_source(
                    dict(case_id=cid, dataset=dataset, source_case_sha256=source_sha),
                    model, current_source, torch, collections,
                    model_for, frames_for, raw_row, frozen_forward, prediction, state_hash,
                )
            finally:
                h.catalog = old_catalog
            source_capture_count += 1
            if list(map(int, captured_ids)) != frame_ids or got_state != expected_state or got_pixel != pixel_sha:
                raise RuntimeError(f"hot source capture differs from frozen source cache: {cid}")
            if not np.array_equal(h._xyxy(native["boxes"]), np.asarray(case["native_boxes_xyxy_normalized"])):
                raise RuntimeError(f"source native boxes differ from frozen source cache: {cid}")
            if not h._same_logits(native["logits"], case["zero"]["logits"]):
                raise RuntimeError(f"source logits differ from frozen source cache: {cid}")
            if list(native["physical_interval"]) != native_interval:
                raise RuntimeError(f"source Native interval differs from frozen source cache: {cid}")
            initial = {name: value.detach().clone() for name, value in base.initial.items()}
            if any(not torch.equal(initial[name].detach().cpu(), value.detach().cpu())
                   for name, value in case["source_light_initial_state"].items()):
                raise RuntimeError(f"source OPD initialization differs from frozen source cache: {cid}")
            anchors = h._anchors(dict(anchors=anchors_json), frame_ids, torch)
            source_cold = dict(status="pass", native_boxes_logits_bitwise=True,
                               full_model_state_sha256=got_state, pixel_sha256=got_pixel,
                               native_interval=native_interval, frame_ids=frame_ids,
                               source_capture_seconds=source_seconds, GT_read=False)
            for mode in MODES:
                if mode in reuse_receipts or not anchors_json:
                    continue
                base.restore(initial)
                partial = trial_dir / "partial" / cid / mode
                if partial.exists():
                    raise FileExistsError(f"partial output exists; no automatic retry: {partial}")
                partial.mkdir(parents=True, exist_ok=False)

                def checkpoint(payload, partial_dir=partial):
                    destination = partial_dir / f"step{int(payload['next_step'])}.pt"
                    save_pt(destination, payload)

                config = configs[dataset][mode]
                if mode == "M0":
                    fit = fit_validated(base, initial, anchors, frame_ids, key, config,
                                        validate=True, reference_sha256=ref_sha,
                                        checkpoint_callback=checkpoint)
                else:
                    fit = fit_generalized(base, initial, anchors, frame_ids, key, config,
                                          mode, settings[dataset][mode], checkpoint=checkpoint)
                fit_invocations += 1
                mode_invocations[mode] += 1
                mode_steps[mode] += int(fit.get("optimizer_steps", 0))
                base.restore(initial)
                out = h._readout(model, model_batch, frame_ids, records, fit,
                                 raw["parses"]["subject"], core, torch,
                                 transient_native_masks=transient_native_masks,
                                 frozen_visual_host=frozen_visual_host,
                                 frozen_attention_storage=frozen_attention_storage,
                                 inactive_modules_on_host=inactive_modules_on_host,
                                 query_subject=query_subject)
                final_readout_calls += 1
                if state_hash(model.state_dict()) != expected_state:
                    raise RuntimeError(f"source model state changed after full readout: {cid}/{mode}")
                if not h._same_logits(out["logits"], native["logits"]):
                    raise RuntimeError(f"source temporal logits changed after full readout: {cid}/{mode}")
                if not torch.equal(out["boxes"].detach().cpu(), fit["final_cached_decoder_boxes"].detach().cpu()):
                    raise RuntimeError(f"full readout differs from cached final boxes: {cid}/{mode}")
                fits[mode] = fit
                readouts[mode] = out

        for mode in MODES:
            if not anchors_json:
                fits[mode] = None
                readouts[mode] = None
                continue
            if fits[mode] is None or readouts[mode] is None:
                raise RuntimeError(f"nonempty reference missing fit/readout: {cid}/{mode}")

        native_xyxy = np.asarray(case["native_boxes_xyxy_normalized"]).copy()
        arms = _build_arms(native_xyxy, native_interval, selected_interval,
                           readouts, bool(anchors_json), h._xyxy)

        checkpoints = {}
        for mode in MODES:
            partial = trial_dir / "partial" / cid / mode
            if partial.is_dir():
                rows = []
                for checkpoint_path in sorted(partial.glob("step*.pt"), key=lambda p: int(p.stem[4:])):
                    item = dict(path=str(checkpoint_path.relative_to(ROOT)), sha256=sha(checkpoint_path),
                                bytes=checkpoint_path.stat().st_size, mode=mode, case_id=cid)
                    rows.append(item)
                    checkpoint_files.append(item)
                checkpoints[mode] = rows
            else:
                checkpoints[mode] = []
        payload = dict(
            schema="stvg-soft-ep-coordinate-trial-case-v1",
            case_id=cid, dataset=dataset, arms=arms, fits=fits, readouts=readouts,
            reference_records=copy.deepcopy(ref_record), temporal_record=copy.deepcopy(temporal_record),
            selected_interval=selected_interval, frame_ids=frame_ids,
            source_case_sha256=source_sha, source_file_sha256=sha(source_path),
            source_model_state_sha256=expected_state, pixel_sha256=pixel_sha,
            source_cold_readback=source_cold, source_capture_seconds=source_seconds,
            source_capture_reused=False if source_cold is not None else True,
            source_capture_barrier_sha256=sha(SOURCE_BARRIER_PATH),
            roster_lock_sha256=roster_sha, reference_lock_sha256=ref_sha,
            temporal_lock_sha256=temporal_sha, design_lock_sha256=sha(design_path),
            runtime_sha256=sha(runtime_path), family_reuse=copy.deepcopy(reuse_receipts),
            checkpoints=checkpoints, actual_fit_count=sum(v is not None and "reused_fit_from_sha256" not in v for v in fits.values()),
            reused_fit_count=sum("reused_fit_from_sha256" in v for v in fits.values() if isinstance(v, dict)),
            actual_final_redecode_calls=sum(v is not None and mode not in reuse_receipts for mode, v in readouts.items()),
            source_restored=True, GT_read=False,
        )
        destination = trial_dir / "student" / f"{cid}.pt"
        item = save_pt(destination, payload)
        receipt = dict(status="complete", case_id=cid, sha256=sha(destination), file=item,
                       actual_fit_count=payload["actual_fit_count"],
                       reused_fit_count=payload["reused_fit_count"],
                       full_readout_calls=payload["actual_final_redecode_calls"],
                       checkpoint_files=checkpoints, GT_read=False)
        write(destination.with_suffix(".receipt.json"), receipt)
        final_files.append(dict(case_id=cid, path=item["path"], sha256=item["sha256"], bytes=item["bytes"],
                                receipt_path=str(destination.with_suffix(".receipt.json").relative_to(ROOT)),
                                receipt_sha256=sha(destination.with_suffix(".receipt.json")),
                                source_case_sha256=source_sha,
                                actual_fit_count=payload["actual_fit_count"],
                                reused_fit_count=payload["reused_fit_count"],
                                full_readout_calls=payload["actual_final_redecode_calls"]))
        write(trial_dir / "STATUS.json", dict(status="running", trial_id=args.trial_id,
              done=len(final_files), total=128, actual_fit_invocations=fit_invocations,
              mode_fit_invocations=mode_invocations, mode_optimizer_steps=mode_steps,
              source_capture_calls=source_capture_count, full_readout_calls=final_readout_calls,
              reused_fit_count=reused_fit_count, GT_read=False, time=time.time()), mutable=True)
        print("SOFT_EP_COORD_TRIAL", args.trial_id, cid, "anchors", len(anchors_json),
              "new_fits", payload["actual_fit_count"], "reuse", payload["reused_fit_count"], flush=True)
        del case
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    expected_paths = {str((trial_dir / "student" / f"{_case_id(row)}.pt").relative_to(ROOT)) for row in roster}
    assert len(final_files) == len(expected_paths) == 128
    assert {item["path"] for item in final_files} == expected_paths
    barrier = dict(
        schema="stvg-soft-ep-coordinate-trial-student-barrier-v1", status="complete",
        trial_id=args.trial_id, count=128, files=final_files,
        checkpoint_files=checkpoint_files,
        actual_fit_invocations=fit_invocations, mode_fit_invocations=mode_invocations,
        mode_optimizer_steps=mode_steps, source_capture_calls=source_capture_count,
        full_readout_calls=final_readout_calls, reused_fit_count=reused_fit_count,
        design_lock_sha256=sha(design_path), roster_lock_sha256=roster_sha,
        source_capture_barrier_sha256=sha(SOURCE_BARRIER_PATH),
        reference_lock_sha256=ref_sha, temporal_lock_sha256=temporal_sha,
        runtime_sha256=sha(runtime_path), wall_seconds=time.perf_counter() - started,
        GT_read=False,
    )
    write(trial_dir / "STUDENT_GLOBAL_BARRIER.json", barrier)
    print("SOFT_EP_COORD_TRIAL_COMPLETE", args.trial_id, fit_invocations,
          source_capture_count, final_readout_calls, len(checkpoint_files), flush=True)


if __name__ == "__main__":
    main()
