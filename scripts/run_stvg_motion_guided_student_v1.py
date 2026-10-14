"""Source-reset motion-guided OPD student and isolated reference-control worker."""
import collections
import copy
import gc
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import catalog, read, sha, write, inference_guard

BASE = ROOT / "artifacts/stvg_motion_guided_instance_opd_v1"
OLD = ROOT / "artifacts/stvg_temporal_caption_query_spatial_opd_v1"
EXPECTED_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(8)
}
OLD_IDS = {
    f"{domain}_{index:03d}"
    for domain in ("hc2", "vidstg")
    for index in range(4)
}
ALGORITHMS = ("direct", "iou", "diou")
CONTROL_ROLES = ("Source", "oldR1ID2", "oldAlternativeID3")


def _case_ids(rows):
    return [str(row["case_id"]) for row in rows]


def _read_locked_inputs(phase):
    assert phase in ("method", "reference_control")
    roster_path = BASE / "ROSTER_LOCK.json"
    lock_path = BASE / (
        "METHOD_REFERENCE_LOCK.json"
        if phase == "method"
        else "REFERENCE_CONTROL_REFERENCE_LOCK.json"
    )
    runtime_path = BASE / (
        "METHOD_STUDENT_RUNTIME.json"
        if phase == "method"
        else "REFERENCE_CONTROL_STUDENT_RUNTIME.json"
    )
    roster_doc = read(roster_path)
    assert roster_doc.get("status") == "locked" and roster_doc.get("GT_read") is False
    rows = roster_doc["records"]
    ids = _case_ids(rows)
    assert len(ids) == 16 and len(set(ids)) == 16 and set(ids) == EXPECTED_IDS
    assert ids == sorted(ids), "ROSTER_LOCK must use stable case_id order"
    roster_index = {rec["case_id"]: rec for rec in rows}
    assert all(rec["dataset"] in ("hc2", "vidstg") for rec in rows)
    assert all(rec["source_case_sha256"] for rec in rows)

    lock = read(lock_path)
    assert lock.get("status") == "locked" and lock.get("GT_read") is False
    refs = lock["records"]
    ref_ids = _case_ids(refs)
    if phase == "method":
        assert len(ref_ids) == 16 and len(set(ref_ids)) == 16 and set(ref_ids) == EXPECTED_IDS
    else:
        assert ref_ids == ["vidstg_002"], "reference_control is limited to vidstg_002"
    reference_index = {rec["case_id"]: rec for rec in refs}
    for cid, reference_record in reference_index.items():
        roster_record = roster_index[cid]
        assert reference_record["dataset"] == roster_record["dataset"]
        assert reference_record["source_case_sha256"] == roster_record["source_case_sha256"]

    runtime = read(runtime_path)
    assert runtime.get("status") == "locked"
    assert runtime.get("phase") == phase and runtime.get("GT_read") is False
    assert runtime.get("steps") == 5
    assert runtime.get("scope") == "joint"
    assert runtime.get("roster_lock_sha256") == sha(roster_path)
    assert runtime.get("reference_lock_sha256") == sha(lock_path)
    pins = runtime.get("pins", {})
    protected = runtime.get("protected", {})
    assert isinstance(pins, dict) and isinstance(protected, dict)
    own_worker = "scripts/run_stvg_motion_guided_student_v1.py"
    own_launcher = "scripts/launch_stvg_motion_guided_student_v1.py"
    assert own_worker in pins and own_launcher in pins
    assert "methods/CURRENT_METHOD.json" in protected
    required_old = [
        "artifacts/stvg_temporal_caption_query_spatial_opd_v1/STUDENT_GLOBAL_BARRIER.json",
        *[
            f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
            for cid in sorted(OLD_IDS)
        ],
    ]
    assert all(rel in pins for rel in required_old), "all reused old T outputs and barrier must be pinned"
    for rel, digest in {**pins, **protected}.items():
        assert sha(ROOT / rel) == digest, f"runtime pin mismatch: {rel}"

    if phase == "method":
        assert set(reference_index) == EXPECTED_IDS
        assert not (BASE / "method" / "partial").exists(), "partial fits exist; no resume is allowed"
        expected_fits = 0
        for cid in sorted(EXPECTED_IDS):
            refs0 = reference_index[cid]["references"]
            teacher = refs0["teacher"]
            assert isinstance(teacher.get("no_update"), bool)
            anchors = teacher.get("anchors", [])
            if not teacher["no_update"]:
                assert anchors, f"updating teacher reference has no anchors: {cid}"
                expected_fits += len(ALGORITHMS)
        assert runtime.get("expected_fit_count") == expected_fits
    else:
        assert not (BASE / "reference_control" / "partial").exists(), "partial fits exist; no resume is allowed"
        assert runtime.get("expected_fit_count") == len(CONTROL_ROLES)
        control_refs = reference_index["vidstg_002"]["references"]
        assert set(control_refs) == set(CONTROL_ROLES)
        assert all(control_refs[role].get("anchors") for role in CONTROL_ROLES)
    return rows, reference_index, runtime, lock_path


def _clone_state(state):
    return {name: value.detach().clone() for name, value in state.items()}


def _xyxy(boxes):
    import numpy as np
    values = boxes.detach().cpu().double().numpy()
    return np.concatenate(
        (values[:, :2] - values[:, 2:] / 2, values[:, :2] + values[:, 2:] / 2),
        axis=1,
    )


def _arm(boxes, interval):
    import numpy as np
    return {"boxes": np.asarray(boxes).copy(), "interval": list(interval)}


def _same_logits(left, right):
    import torch
    return len(left) == len(right) and all(
        torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in zip(left, right)
    )


def _validate_interval(interval, bounds, label):
    import numpy as np
    values = np.asarray(interval, dtype=np.float64)
    limits = np.asarray(bounds, dtype=np.float64)
    assert values.shape == (2,) and limits.shape == (2,)
    assert np.isfinite(values).all() and np.isfinite(limits).all()
    assert values[0] < values[1], f"invalid {label} interval"
    assert limits[0] <= values[0] and values[1] <= limits[1], (
        f"{label} interval is outside source physical-frame bounds"
    )


def _anchors(reference, frame_ids, torch):
    anchors = []
    for row in reference.get("anchors", []):
        position = int(row["position"])
        assert 0 <= position < len(frame_ids)
        raw = torch.tensor(row["box_cxcywh"], dtype=torch.float64)
        cx0,cy0,w0,h0=map(float,raw.tolist())
        assert 0 <= cx0-w0/2 < cx0+w0/2 <= 1 and 0 <= cy0-h0/2 < cy0+h0/2 <= 1
        box = raw.to(torch.float32)
        assert box.shape == (4,) and torch.isfinite(box).all()
        cx, cy, width, height = map(float, box.tolist())
        assert width > 0 and height > 0
        assert -1e-7 <= cx - width / 2 < cx + width / 2 <= 1+1e-7
        assert -1e-7 <= cy - height / 2 < cy + height / 2 <= 1+1e-7
        anchors.append({
            "position": position,
            "physical_frame_id": int(frame_ids[position]),
            "box": box,
            "weight": float(row.get("weight", 1.0)),
        })
    return anchors


def _load_old_temporal_case(case_id, source_sha, source_model_hash, frame_ids, pixel_sha,
                            native_xyxy, native_interval, projection_interval, bounds, torch):
    import numpy as np
    path = OLD / "student" / f"{case_id}.pt"
    assert case_id in OLD_IDS and path.is_file()
    old = torch.load(path, map_location="cpu", weights_only=False)
    assert old.get("case_id") == case_id and old.get("GT_read") is False
    assert old["source_case_sha256"] == source_sha
    assert list(map(int, old["frame_ids"])) == list(map(int, frame_ids))
    cold = old["source_cold_readback"]
    assert cold["status"] == "pass" and cold["source_boxes_logits_bitwise"] is True
    assert cold["full_model_state_sha256"] == source_model_hash
    assert cold["pixel_sha256"] == pixel_sha
    assert np.array_equal(np.asarray(old["arms"]["F"]["boxes"]), native_xyxy)
    assert list(old["arms"]["F"]["interval"]) == list(native_interval)
    assert np.array_equal(np.asarray(old["arms"]["Projection"]["boxes"]), native_xyxy)
    assert list(old["arms"]["Projection"]["interval"]) == list(projection_interval)
    assert np.array_equal(np.asarray(old["arms"]["T"]["boxes"]), native_xyxy)
    _validate_interval(old["arms"]["T"]["interval"], bounds, "old T")
    return old


def _readout(model, batch, frame_ids, records, fit, query, core, torch,
             transient_native_masks, frozen_visual_host, frozen_attention_storage,
             inactive_modules_on_host, query_subject):
    with (
        torch.no_grad(),
        transient_native_masks(model),
        frozen_visual_host(model),
        frozen_attention_storage(model),
        inactive_modules_on_host(model),
        query_subject(model, batch, query),
    ):
        return core._finalize_student_readout(
            model, batch, frame_ids, records, fit["final_state"]
        )


def _fit_one(phase, case_id, role, algorithm, base, initial, anchors, frame_ids,
             config, reference_sha, core, fit_spatial_reward_opd,
             direct_step_readback, torch):
    key = "caption_query_fixed_space|" + case_id
    partial = BASE / phase / "partial" / case_id / f"{role}_{algorithm}"
    assert not partial.exists(), f"partial output already exists; worker will not resume: {partial}"

    def checkpoint(payload):
        destination = partial / f"step{payload['next_step']}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        torch.save(payload, destination)

    fresh_initial = _clone_state(initial)
    cfg = dict(config)
    cfg["steps"] = 5
    if algorithm == "direct":
        fit = core._fit_direct(
            base, fresh_initial, anchors, frame_ids, key, cfg, scope="joint",
            checkpoint_callback=checkpoint,
        )
        fit["exact_step_readback"] = direct_step_readback(
            core, base, fresh_initial, frame_ids, key, fit, torch
        )
    else:
        fit = fit_spatial_reward_opd(
            base, fresh_initial, anchors, frame_ids, key, cfg, scope="joint",
            reward_kind=algorithm, checkpoint_callback=checkpoint,
            reference_role=role, reference_sha256=reference_sha,
        )
    assert int(fit["optimizer_steps"]) == 5
    assert int(fit["active_parameters"]) == 1792
    assert int(fit.get("declared_steps", 5)) == 5
    return fit


def _capture_source(rec, model, current_source, torch, collections_module,
                    model_for, frames_for, raw_row,
                    frozen_forward, prediction, state_hash):
    capture_started = time.perf_counter()
    case, catalog_data = catalog(rec)
    assert case["case_id"] == rec["case_id"]
    assert case["_source_case_sha256"] == rec["source_case_sha256"]
    if current_source != case["source"]:
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        model = model_for(case["source"])
        current_source = case["source"]
    expected = case["source_model_state_sha256"]
    assert model._fixed_full_state_hash == expected
    assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())
    row = raw_row(case)
    frames, frame_ids, _ = frames_for(rec["dataset"], row, collections_module.OrderedDict())
    frame_ids = list(map(int, frame_ids))
    assert frame_ids == list(map(int, case["frame_ids"]))
    pixel_sha = hashlib.sha256(frames.tobytes()).hexdigest()
    assert pixel_sha == case["pixel_sha256"]
    batch, records, base = frozen_forward(model, frames, row)
    native = prediction(base.zero["logits"], base.zero["boxes"], records, frame_ids)
    assert torch.equal(native["boxes"], case["zero"]["boxes"])
    assert len(native["logits"]) == len(case["zero"]["logits"])
    assert all(torch.equal(a, b) for a, b in zip(native["logits"], case["zero"]["logits"]))
    assert list(native["physical_interval"]) == list(case["Native_interval"])
    assert records == case["records"]
    for name, value in base.initial.items():
        assert torch.equal(value.detach().cpu(), case["source_light_initial_state"][name])
    assert state_hash(model.state_dict()) == expected
    _validate_interval(case["Native_interval"], case["video_bounds"], "Native")
    _validate_interval(case["Projection_interval"], case["video_bounds"], "Projection")
    return (
        model, current_source, case, catalog_data, row, frames, frame_ids, records,
        batch, base, native, expected, pixel_sha, time.perf_counter() - capture_started,
    )


def _main_method():
    import numpy as np
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
    from scripts.run_stvg_support_aware_spatial_v1 import _direct_step_readback
    from scripts.stvg_video_global_spatial_fit_fp32_geometry001 import fit_spatial_reward_opd

    phase = "method"
    rows, reference_index, runtime, lock_path = _read_locked_inputs(phase)
    output_dir = BASE / "method" / "student"
    assert not any(output_dir.glob("*.pt")), "existing method student outputs block restart"
    assert not (BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json").exists()
    assert not (BASE / "METHOD_STUDENT_FAILURE_CAPTURE.json").exists()
    install_clean_loader()
    sys.addaudithook(inference_guard)
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    lock_sha = sha(lock_path)
    model = None
    current_source = None
    output_files = []
    total_steps = 0
    total_fits = 0
    total_final_readouts = 0
    started_all = time.perf_counter()
    write(BASE / "METHOD_STUDENT_GPU_STATUS.json", {
        "status": "running", "stage": "source_reset_method", "done": 0,
        "total": len(rows), "pid": os.getpid(), "GT_read": False, "time": time.time(),
    }, mutable=True)

    for number, rec in enumerate(rows, 1):
        cid = rec["case_id"]
        destination = BASE / "method" / "student" / f"{cid}.pt"
        assert not destination.exists(), f"student result already exists: {destination}"
        teacher = reference_index[cid]["references"]["teacher"]
        no_update = bool(teacher["no_update"])
        if model is not None and current_source != catalog(rec)[0]["source"]:
            del model
            model=None
            gc.collect()
            torch.cuda.empty_cache()
        (
            model, current_source, case, catalog_data, row, frames, frame_ids,
            records, batch, base, native, expected, pixel_sha, source_capture_elapsed,
        ) = _capture_source(
            rec, model, current_source, torch, collections,
            model_for, frames_for, raw_row, frozen_forward, prediction, state_hash,
        )
        native_boxes = _xyxy(native["boxes"])
        native_interval = list(case["Native_interval"])
        projection_interval = list(case["Projection_interval"])
        bounds = list(case["video_bounds"])
        arms = {
            "F": _arm(native_boxes, native_interval),
            "Projection": _arm(native_boxes, projection_interval),
        }
        old = None
        if cid in OLD_IDS:
            old = _load_old_temporal_case(
                cid, rec["source_case_sha256"], expected, frame_ids, pixel_sha, native_boxes,
                native_interval, projection_interval, bounds, torch,
            )
            arms["T"] = copy.deepcopy(old["arms"]["T"])
        fits = {}
        readouts = {}
        receipts = {}
        anchors_all = {"teacher": teacher.get("anchors", [])}
        cfg = core.load_original_opd_config(rec["dataset"])
        cfg["steps"] = 5
        initial = _clone_state(base.initial)

        if no_update:
            for algorithm in ALGORITHMS:
                name = f"teacher_{algorithm}"
                arms[name] = copy.deepcopy(arms["F"])
                receipts[name] = {
                    "status": "explicit_no_update",
                    "lock_no_update": True,
                    "optimizer_steps": 0,
                    "final_redecode_calls": 0,
                    "teacher_weight_update": False,
                    "source_case_sha256": rec["source_case_sha256"],
                    "exact_F_reuse": True,
                }
                if old is not None:
                    tname = f"T_teacher_{algorithm}"
                    arms[tname] = copy.deepcopy(old["arms"]["T"])
                    assert np.array_equal(arms[tname]["boxes"], old["arms"]["T"]["boxes"])
                    assert arms[tname]["interval"] == old["arms"]["T"]["interval"]
                    receipts[tname] = {
                        "status": "exact_old_T_reuse",
                        "lock_no_update": True,
                        "optimizer_steps": 0,
                        "final_redecode_calls": 0,
                        "teacher_weight_update": False,
                        "source_case_sha256": rec["source_case_sha256"],
                        "exact_T_reuse": True,
                    }
                assert np.array_equal(arms[name]["boxes"], arms["F"]["boxes"])
                assert arms[name]["interval"] == arms["F"]["interval"]
        else:
            anchors = _anchors(teacher, frame_ids, torch)
            assert anchors
            for algorithm in ALGORITHMS:
                fit_start = time.perf_counter()
                fit = _fit_one(
                    "method", cid, "teacher", algorithm, base, initial, anchors,
                    frame_ids, cfg, lock_sha, core, fit_spatial_reward_opd,
                    _direct_step_readback, torch,
                )
                out = _readout(
                    model, batch, frame_ids, records, fit, row["parses"]["subject"],
                    core, torch, transient_native_masks, frozen_visual_host,
                    frozen_attention_storage, inactive_modules_on_host, query_subject,
                )
                assert torch.equal(out["boxes"].cpu(), fit["final_cached_decoder_boxes"].cpu())
                assert _same_logits(out["logits"], native["logits"])
                assert state_hash(model.state_dict()) == expected
                boxes = _xyxy(out["boxes"])
                name = f"teacher_{algorithm}"
                arms[name] = _arm(boxes, native_interval)
                if old is not None:
                    tname = f"T_teacher_{algorithm}"
                    arms[tname] = _arm(boxes, old["arms"]["T"]["interval"])
                    receipts[tname] = {
                        "status": "updated_spatial_plus_exact_old_T_interval",
                        "lock_no_update": False,
                        "optimizer_steps": int(fit["optimizer_steps"]),
                        "final_redecode_calls": 0,
                        "inherits_fit": name,
                        "old_T_interval_reused": True,
                    }
                fits[name] = fit
                readouts[name] = out
                total_steps += int(fit["optimizer_steps"])
                total_fits += 1
                total_final_readouts += 1
                receipts[name] = {
                    "status": "updated",
                    "lock_no_update": False,
                    "optimizer_steps": int(fit["optimizer_steps"]),
                    "final_redecode_calls": 1,
                    "teacher_weight_update": False,
                    "elapsed_seconds": time.perf_counter() - fit_start,
                }

        if no_update:
            assert fits == {} and readouts == {}
            assert all(receipts[f"teacher_{a}"]["optimizer_steps"] == 0 for a in ALGORITHMS)
            assert all(receipts[f"teacher_{a}"]["final_redecode_calls"] == 0 for a in ALGORITHMS)
        assert state_hash(model.state_dict()) == expected
        result = {
            "case_id": cid,
            "dataset": rec["dataset"],
            "source_case_sha256": rec["source_case_sha256"],
            "source_model_state_sha256": expected,
            "frame_ids": frame_ids,
            "pixel_sha256": pixel_sha,
            "Native_interval": native_interval,
            "Projection_interval": projection_interval,
            "arms": arms,
            "fits": fits,
            "readouts": readouts,
            "reference_anchors": anchors_all,
            "supervision_anchors": anchors_all,
            "initial_state": core._copy_cpu(initial),
            "source_boxes": base.zero["boxes"].cpu(),
            "source_logits": core._copy_cpu(base.zero["logits"]),
            "source_restored": True,
            "reference_lock_sha256": lock_sha,
            "reference_receipts": receipts,
            "source_cold_readback": {
                "status": "pass",
                "source_case_sha256": rec["source_case_sha256"],
                "source_model_state_sha256": expected,
                "pixel_sha256": pixel_sha,
                "frame_ids_match": True,
                "initial_state_match": True,
                "zero_boxes_logits_match": True,
                "source_restored": True,
            },
            "temporal_exact_reuse": (
                {
                    "status": "reused_exact_old_T",
                    "source_case_sha256": rec["source_case_sha256"],
                    "old_student_path": str((OLD / "student" / f"{cid}.pt").relative_to(ROOT)),
                    "old_student_sha256": sha(OLD / "student" / f"{cid}.pt"),
                    "old_T_boxes_equal_F": True,
                    "old_T_interval_reused": True,
                }
                if old is not None
                else {"status": "unavailable_no_new_T_created"}
            ),
            "configuration": {
                "key": "caption_query_fixed_space|" + cid,
                "scope": "joint",
                "steps": 5,
                "active_parameters": 1792,
                "config": cfg,
                "method_reference_lock_sha256": lock_sha,
            },
            "source_capture_calls": 1,
            "actual_fit_count": len(fits),
            "optimizer_steps": sum(int(fit["optimizer_steps"]) for fit in fits.values()),
            "actual_final_redecode_calls": len(readouts),
            "source_cold_elapsed_seconds": source_capture_elapsed,
            "GT_read": False,
            "online_carry": False,
            "teacher_weight_update": False,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        torch.save(result, destination)
        output_files.append({
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha(destination),
            "bytes": destination.stat().st_size,
        })
        write(BASE / "METHOD_STUDENT_GPU_STATUS.json", {
            "status": "running", "stage": "source_reset_method", "done": number,
            "total": len(rows), "pid": os.getpid(), "GT_read": False, "time": time.time(),
        }, mutable=True)
        del result, batch, frames, base, old
        gc.collect()
        torch.cuda.empty_cache()

    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    barrier = BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json"
    assert not barrier.exists()
    write(barrier, {
        "status": "complete",
        "count": len(output_files),
        "files": output_files,
        "actual_fit_count": total_fits,
        "optimizer_steps": total_steps,
        "actual_final_redecode_calls": total_final_readouts,
        "roster_lock_sha256": sha(BASE / "ROSTER_LOCK.json"),
        "reference_lock_sha256": lock_sha,
        "runtime_sha256": sha(BASE / "METHOD_STUDENT_RUNTIME.json"),
        "GT_read": False,
        "end_to_end_seconds": time.perf_counter() - started_all,
        "time": time.time(),
    })


def _main_reference_control():
    import numpy as np
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
    from scripts.run_stvg_support_aware_spatial_v1 import _direct_step_readback
    from scripts.stvg_video_global_spatial_fit_fp32_geometry001 import fit_spatial_reward_opd

    phase = "reference_control"
    rows, reference_index, runtime, lock_path = _read_locked_inputs(phase)
    output_dir = BASE / "reference_control" / "student"
    assert not any(output_dir.glob("*.pt")), "existing reference-control outputs block restart"
    assert not (BASE / "REFERENCE_CONTROL_STUDENT_GLOBAL_BARRIER.json").exists()
    assert not (BASE / "REFERENCE_CONTROL_STUDENT_FAILURE_CAPTURE.json").exists()
    rec = next(row for row in rows if row["case_id"] == "vidstg_002")
    install_clean_loader()
    sys.addaudithook(inference_guard)
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = None
    current_source = None
    output_files = []
    total_steps = 0
    total_fits = 0
    started_all = time.perf_counter()
    write(BASE / "REFERENCE_CONTROL_STUDENT_GPU_STATUS.json", {
        "status": "running", "stage": "reference_control", "done": 0,
        "total": 1, "pid": os.getpid(), "GT_read": False, "time": time.time(),
    }, mutable=True)

    cid = rec["case_id"]
    destination = BASE / "reference_control" / "student" / f"{cid}.pt"
    assert not destination.exists()
    (
        model, current_source, case, catalog_data, row, frames, frame_ids,
        records, batch, base, native, expected, pixel_sha, source_capture_elapsed,
    ) = _capture_source(
        rec, model, current_source, torch, collections,
        model_for, frames_for, raw_row, frozen_forward, prediction, state_hash,
    )
    lock_sha = sha(lock_path)
    arms = {
        "F": _arm(_xyxy(native["boxes"]), case["Native_interval"]),
        "Projection": _arm(_xyxy(native["boxes"]), case["Projection_interval"]),
    }
    old = _load_old_temporal_case(
        cid, rec["source_case_sha256"], expected, frame_ids, pixel_sha, arms["F"]["boxes"],
        case["Native_interval"], case["Projection_interval"], case["video_bounds"], torch,
    )
    arms["T"] = copy.deepcopy(old["arms"]["T"])
    references = reference_index[cid]["references"]
    fits = {}
    readouts = {}
    role_metadata = {}
    cfg = core.load_original_opd_config(rec["dataset"])
    cfg["steps"] = 5
    initial = _clone_state(base.initial)
    for role in CONTROL_ROLES:
        reference = references[role]
        anchors = _anchors(reference, frame_ids, torch)
        assert anchors, f"reference-control role has no anchors: {role}"
        fit_start = time.perf_counter()
        fit = _fit_one(
            "reference_control", cid, role, "iou", base, initial, anchors,
            frame_ids, cfg, lock_sha, core, fit_spatial_reward_opd,
            _direct_step_readback, torch,
        )
        out = _readout(
            model, batch, frame_ids, records, fit, row["parses"]["subject"],
            core, torch, transient_native_masks, frozen_visual_host, frozen_attention_storage,
            inactive_modules_on_host, query_subject,
        )
        assert torch.equal(out["boxes"].cpu(), fit["final_cached_decoder_boxes"].cpu())
        assert _same_logits(out["logits"], native["logits"])
        assert state_hash(model.state_dict()) == expected
        name = f"{role}_iou"
        arms[name] = _arm(_xyxy(out["boxes"]), case["Native_interval"])
        fits[name] = fit
        readouts[name] = out
        role_metadata[role] = {
            "anchors": references[role]["anchors"],
            "optimizer_steps": int(fit["optimizer_steps"]),
            "final_redecode_calls": 1,
            "source_role_update_intentionally_allowed": role == "Source",
            "lock_no_update_ignored_for_control": bool(reference.get("no_update", False)),
            "elapsed_seconds": time.perf_counter() - fit_start,
        }
        assert int(fit["optimizer_steps"]) == 5
        total_steps += int(fit["optimizer_steps"])
        total_fits += 1

    assert set(fits) == {f"{role}_iou" for role in CONTROL_ROLES}
    assert state_hash(model.state_dict()) == expected
    result = {
        "case_id": cid,
        "dataset": rec["dataset"],
        "source_case_sha256": rec["source_case_sha256"],
        "source_model_state_sha256": expected,
        "frame_ids": frame_ids,
        "pixel_sha256": pixel_sha,
        "Native_interval": list(case["Native_interval"]),
        "Projection_interval": list(case["Projection_interval"]),
        "source_cold_elapsed_seconds": source_capture_elapsed,
        "arms": arms,
        "fits": fits,
        "readouts": readouts,
        "reference_control_roles": role_metadata,
        "supervision_anchors": {k:v["anchors"] for k,v in references.items()},
        "initial_state": core._copy_cpu(initial),
        "source_boxes": base.zero["boxes"].cpu(),
        "source_logits": core._copy_cpu(base.zero["logits"]),
        "source_restored": True,
        "reference_lock_sha256": lock_sha,
        "source_cold_readback": {
            "status": "pass",
            "source_case_sha256": rec["source_case_sha256"],
            "source_model_state_sha256": expected,
            "pixel_sha256": pixel_sha,
            "frame_ids_match": True,
            "initial_state_match": True,
            "zero_boxes_logits_match": True,
            "source_restored": True,
        },
        "temporal_exact_reuse": {
            "status": "reused_exact_old_T",
            "source_case_sha256": rec["source_case_sha256"],
            "old_student_path": str((OLD / "student" / f"{cid}.pt").relative_to(ROOT)),
            "old_student_sha256": sha(OLD / "student" / f"{cid}.pt"),
            "old_T_boxes_equal_F": True,
            "old_T_interval_reused": True,
        },
        "configuration": {
            "key": "caption_query_fixed_space|" + cid,
            "scope": "joint",
            "steps": 5,
            "active_parameters": 1792,
            "config": cfg,
            "reference_lock_sha256": lock_sha,
            "Source_role_updated_for_diagnostic": True,
        },
        "source_capture_calls": 1,
        "actual_fit_count": total_fits,
        "optimizer_steps": total_steps,
        "actual_final_redecode_calls": len(readouts),
        "GT_read": False,
        "online_carry": False,
        "teacher_weight_update": False,
        "end_to_end_seconds": time.perf_counter() - started_all,
        "GPU_peak_bytes": torch.cuda.max_memory_allocated(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    torch.save(result, destination)
    output_files.append({
        "path": str(destination.relative_to(ROOT)),
        "sha256": sha(destination),
        "bytes": destination.stat().st_size,
    })
    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    barrier = BASE / "REFERENCE_CONTROL_STUDENT_GLOBAL_BARRIER.json"
    assert not barrier.exists()
    write(barrier, {
        "status": "complete",
        "count": len(output_files),
        "files": output_files,
        "actual_fit_count": total_fits,
        "optimizer_steps": total_steps,
        "actual_final_redecode_calls": len(readouts),
        "roster_lock_sha256": sha(BASE / "ROSTER_LOCK.json"),
        "reference_lock_sha256": lock_sha,
        "runtime_sha256": sha(BASE / "REFERENCE_CONTROL_STUDENT_RUNTIME.json"),
        "GT_read": False,
        "end_to_end_seconds": time.perf_counter() - started_all,
        "time": time.time(),
    })


def main(phase):
    assert phase in ("method", "reference_control")
    try:
        if phase == "method":
            _main_method()
        else:
            _main_reference_control()
    except BaseException as exc:
        write(BASE / f"{phase.upper()}_STUDENT_FAILURE_CAPTURE.json", {
            "status": "actual_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "GT_read": False,
            "time": time.time(),
        }, mutable=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_stvg_motion_guided_student_v1.py method|reference_control")
    main(sys.argv[1])
