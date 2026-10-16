"""Finite recovery for the already prepared coordinate128 teacher stage.

This binds the existing preparation artifacts, materializes the missing teacher
runtime, then runs only teacher -> references -> assembly -> CPU input readback.
It never reruns source, temporal, fit, or candidate-input generation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_native_instance_common_v1 import read, write, sha

BASE = ROOT / "artifacts/stvg_soft_ep_coordinate128_v1"
CONF = BASE / "confirmation"
RECOVERY = BASE / "recovery/teacher_runtime003"
EXEC = BASE / "execution_preparation003"
UNIT = "stvg-soft-ep-coordinate128-preparation-003.service"
GiB = 1024 ** 3
MODEL_RECEIPT_REL = "artifacts/stvg_target_teacher_distillation_v1/ROOT_OFFICIAL_MODEL_FILE_READBACK.json"
TEACHER_RUNTIME_REL = "artifacts/stvg_soft_ep_coordinate128_v1/confirmation/TEACHER_RUNTIME.json"
LOCK_REL = "artifacts/stvg_soft_ep_coordinate128_v1/recovery/teacher_runtime003/RECOVERY_LOCK.json"


def _rel_file(rel):
    candidate = Path(rel)
    if candidate.is_absolute():
        p = candidate.resolve()
        assert p.is_relative_to(ROOT.resolve()), rel
    else:
        assert ".." not in candidate.parts, rel
        p = (ROOT / candidate).resolve()
        assert p.is_relative_to(ROOT.resolve()), rel
    assert p.is_file() and not p.is_symlink(), rel
    return p


def _verify_runtime_file(rel):
    rt = read(ROOT / rel)
    assert rt.get("status") == "locked" and rt.get("GT_read") is False, rel
    merged = {**rt.get("pins", {}), **rt.get("protected", {})}
    assert len(merged) == len(rt.get("pins", {})) + len(rt.get("protected", {})), rel
    for path, expected in merged.items():
        assert sha(_rel_file(path)) == expected, (rel, path)
    return rt


def _safe_file_specs(input_barrier, job_lock):
    files = input_barrier["files"]
    assert len(files) == 2048 and len({x["path"] for x in files}) == len(files)
    f_by_path = {}
    for item in files:
        assert set(item) == {"path", "sha256", "bytes"}
        p = _rel_file(item["path"])
        assert isinstance(item["bytes"], int) and item["bytes"] > 0 and p.stat().st_size == item["bytes"], item["path"]
        assert len(item["sha256"]) == 64 and all(c in "0123456789abcdef" for c in item["sha256"])
        f_by_path[item["path"]] = {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}

    cases = input_barrier["case_records"]
    assert len(cases) == 64 and len({x["case_id"] for x in cases}) == 64
    case_files = []
    for rec in cases:
        assert {"case_id", "path", "sha256", "job_count"} <= set(rec)
        p = _rel_file(rec["path"])
        assert sha(p) == rec["sha256"]
        case_files.append({"case_id": rec["case_id"], "path": rec["path"], "sha256": rec["sha256"], "bytes": p.stat().st_size, "job_count": rec["job_count"]})

    jobs = job_lock["records"]
    assert len(jobs) == 256 and len({x["job_id"] for x in jobs}) == 256
    per_case = {}
    safe_jobs = []
    for job in jobs:
        assert job.get("variant") == "new_crop"
        cid = job["case_id"]
        per_case[cid] = per_case.get(cid, 0) + 1
        assert job.get("center_slot") in range(4)
        assert len(job.get("videos", [])) == 2
        video_specs = []
        for video in job["videos"]:
            path = video["path"]
            assert path in f_by_path and f_by_path[path]["sha256"] == video["sha256"]
            assert len(video.get("tensor_sha256", "")) == 64
            assert math.isfinite(float(video["fps"])) and float(video["fps"]) > 0
            video_specs.append({"path": path, "sha256": video["sha256"], "bytes": f_by_path[path]["bytes"], "tensor_sha256": video["tensor_sha256"], "fps": float(video["fps"]), "clip_role": video.get("clip_role")})
        safe_jobs.append({
            "job_id": job["job_id"], "case_id": cid, "dataset": job["dataset"],
            "variant": job["variant"], "center_slot": job["center_slot"],
            "prompt_sha256": job["prompt_sha256"],
            "videos": video_specs,
            "expected_output_path": f"artifacts/stvg_soft_ep_coordinate128_v1/confirmation/teacher/{job['job_id']}.json",
        })
    assert set(per_case) == {x["case_id"] for x in cases} and all(v == 4 for v in per_case.values())
    return files, case_files, safe_jobs


def prepare():
    assert not (CONF / "TEACHER_RUNTIME.json").exists(), "refusing to replace existing teacher runtime"
    assert not (CONF / "TEACHER_GLOBAL_BARRIER.json").exists(), "teacher barrier already exists; recovery must not overwrite"
    for p in (CONF / "teacher").glob("*.json") if (CONF / "teacher").exists() else ():
        raise RuntimeError(f"unexpected prior teacher output exists: {p.relative_to(ROOT)}")
    if RECOVERY.exists():
        leftovers = {p.name for p in RECOVERY.iterdir()}
        assert leftovers <= {"FAILURE.json", "FAILURE_003.json"} and "FAILURE.json" in leftovers, f"unexpected prior recovery003 artifacts: {sorted(leftovers)}"
        for oldname in leftovers:
            old_failure = read(RECOVERY / oldname)
            assert old_failure.get("schema") == "stvg-soft-ep-coordinate-preparation003-failure-v1" and old_failure.get("stage") == "preflight"
    assert not EXEC.exists(), "execution_preparation003 already exists; preserve and inspect before retry"

    prep = _verify_runtime_file("artifacts/stvg_soft_ep_coordinate128_v1/PREPARATION_RUNTIME.json")
    recovery002 = _verify_runtime_file("artifacts/stvg_soft_ep_coordinate128_v1/RECOVERY_RUNTIME002.json")
    assert prep["status"] == recovery002["status"] == "locked"

    source_barrier_path = BASE / "SOURCE_CAPTURE_BARRIER.json"
    source_barrier = read(source_barrier_path)
    assert source_barrier.get("status") in ("complete", "sealed") and source_barrier.get("count") == 128 and source_barrier.get("GT_read") is False
    assert read(BASE / "SOURCE_RUNTIME.json").get("status") == "locked"
    frozen = read(BASE / "FROZEN_METHOD_LOCK.json")
    assert frozen.get("status") == "locked" and frozen.get("confirmation_used_for_selection") is False and frozen.get("search_finished") is True
    assert isinstance(frozen.get("selection_sha256"), str) and len(frozen["selection_sha256"]) == 64

    conf = read(CONF / "CANDIDATE_INPUT_BARRIER.json")
    jobs_doc = read(CONF / "TEACHER_JOB_LOCK.json")
    temporal_bar = read(CONF / "TEMPORAL_TEACHER_GLOBAL_BARRIER.json")
    fit_bar = read(CONF / "TEMPORAL_FIT_BARRIER.json")
    assert temporal_bar.get("status") == "complete" and temporal_bar.get("count") == 64 and temporal_bar.get("GT_read") is False
    assert fit_bar.get("status") == "complete" and fit_bar.get("count") == 64 and fit_bar.get("GT_read") is False
    assert conf.get("status") == "complete" and conf.get("count") == 64 and conf.get("teacher_job_count") == 256 and conf.get("GT_read") is False
    assert jobs_doc.get("status") == "locked" and jobs_doc.get("GT_read") is False
    files, case_files, jobs = _safe_file_specs(conf, jobs_doc)
    assert len(jobs_doc["records"]) == 256

    # The previous independent input preparation receipt binds the full marked
    # tensor readback. Do not replay the completed input stage here.
    prelaunch_input = BASE / "ROOT_PRELAUNCH_INPUT_READBACK.json"
    if prelaunch_input.exists():
        input_readback_sha = sha(prelaunch_input)
        input_readback = read(prelaunch_input)
        assert input_readback.get("status") == "pass" and input_readback.get("GT_read") is False
    else:
        # The post-teacher reader remains mandatory and will independently read
        # the complete input/output set before this recovery returns success.
        input_readback_sha = None

    model_rel = MODEL_RECEIPT_REL
    model_receipt = read(ROOT / model_rel)
    assert model_receipt.get("status") == "pass" and model_receipt.get("repository") == "XiaowenZhang/stvg-r1-model-7b"
    assert model_receipt.get("count") == 15 and model_receipt.get("weight_bytes") == 16584414544
    assert model_receipt.get("CPU_only") is True and model_receipt.get("GT_read") is False and model_receipt.get("model_loaded") is False
    model_sha = sha(ROOT / model_rel)

    recovery_dir = RECOVERY
    recovery_dir.mkdir(parents=True, exist_ok=False)
    # Record the bounded engineering exposure without copying the prompt/query.
    write(recovery_dir / "CAPTURE.json", {
        "schema": "stvg-soft-ep-coordinate-preparation003-capture-v1",
        "status": "recorded_engineering_readback_deviation",
        "scope": "one first-job metadata summary during recovery preflight",
        "accidental_query_text_printed": True,
        "query_text_copied": False,
        "GT_or_annotations_read": False,
        "RGB_or_video_read": False,
        "model_or_weight_payload_read": False,
        "experiment_selection_or_method_changed": False,
        "subsequent_job_checks": "field-whitelisted metadata, hashes, bytes, and counts only",
    })

    # Recovery lock is deliberately path-and-hash metadata only. It excludes
    # prompts, raw outputs, image tensors, model bytes, and GT/annotation data.
    bindings = {
        "preparation_runtime_sha256": sha(BASE / "PREPARATION_RUNTIME.json"),
        "recovery_runtime002_sha256": sha(BASE / "RECOVERY_RUNTIME002.json"),
        "source_capture_barrier_sha256": sha(source_barrier_path),
        "source_runtime_sha256": sha(BASE / "SOURCE_RUNTIME.json"),
        "frozen_method_lock_sha256": sha(BASE / "FROZEN_METHOD_LOCK.json"),
        "roster_lock_sha256": sha(BASE / "ROSTER_LOCK.json"),
        "temporal_selection_lock_sha256": sha(BASE / "TEMPORAL_SELECTION_LOCK.json"),
        "variant_lock_sha256": sha(CONF / "VARIANT_LOCK.json"),
        "temporal_teacher_barrier_sha256": sha(CONF / "TEMPORAL_TEACHER_GLOBAL_BARRIER.json"),
        "temporal_fit_barrier_sha256": sha(CONF / "TEMPORAL_FIT_BARRIER.json"),
        "candidate_input_barrier_sha256": sha(CONF / "CANDIDATE_INPUT_BARRIER.json"),
        "teacher_job_lock_sha256": sha(CONF / "TEACHER_JOB_LOCK.json"),
        "official_model_receipt_sha256": model_sha,
        "teacher_source_sha256": sha(ROOT / "scripts/run_stvg_evidence_teacher_v1.py"),
        "stage_wrapper_sha256": sha(ROOT / "scripts/run_stvg_soft_ep_coordinate_inputs_v1.py"),
        "assembly_source_sha256": sha(ROOT / "scripts/assemble_stvg_soft_ep_coordinate_v1.py"),
        "independent_reader_sha256": sha(ROOT / "scripts/review_stvg_soft_ep_coordinate_inputs_v1.py"),
        "resource_guard_sha256": sha(ROOT / "scripts/launch_stvg_instance_identity_v1.py"),
        "resource_common_sha256": sha(ROOT / "scripts/stvg_instance_identity_common_v1.py"),
        "recovery_worker_sha256": sha(Path(__file__).resolve()),
        "input_readback_sha256": input_readback_sha,
    }
    lock = {
        "schema": "stvg-soft-ep-coordinate-preparation003-lock-v1",
        "status": "locked",
        "scope": "resume teacher then references, assemble, and independent CPU input readback; no source/temporal/fit/input rerun",
        "bindings": bindings,
        "source_count": 128,
        "new_input_case_count": 64,
        "candidate_input_file_count": len(files),
        "case_record_files": case_files,
        "candidate_input_files": [{"path": x["path"], "sha256": x["sha256"], "bytes": x["bytes"]} for x in files],
        "teacher_job_count": len(jobs),
        "teacher_jobs": jobs,
        "expected_teacher_output_count": len(jobs),
        "existing_teacher_output_count": 0,
        "official_model": {
            "repository": model_receipt["repository"],
            "revision": model_receipt["revision"],
            "receipt_path": model_rel,
            "receipt_sha256": model_sha,
            "file_count": model_receipt["count"],
            "weight_bytes": model_receipt["weight_bytes"],
        },
        "GT_read": False,
    }
    lock_path = recovery_dir / "RECOVERY_LOCK.json"
    write(lock_path, lock)

    # Carry both previously qualified source/runtime closures into the new
    # worker runtime, then add this recovery's exact execution inputs.
    pin_map = {}
    protected_map = {}
    for rt in (prep, recovery002):
        for path, value in rt.get("pins", {}).items():
            if path in pin_map:
                assert pin_map[path] == value, path
            pin_map[path] = value
        for path, value in rt.get("protected", {}).items():
            if path in protected_map:
                assert protected_map[path] == value, path
            protected_map[path] = value
    for path, value in list(protected_map.items()):
        if path in pin_map:
            assert pin_map[path] == value, (path, pin_map[path], value)
            del pin_map[path]
    extra = {
        "artifacts/stvg_soft_ep_coordinate128_v1/PREPARATION_RUNTIME.json": sha(BASE / "PREPARATION_RUNTIME.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/RECOVERY_RUNTIME002.json": sha(BASE / "RECOVERY_RUNTIME002.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/SOURCE_CAPTURE_BARRIER.json": sha(source_barrier_path),
        "artifacts/stvg_soft_ep_coordinate128_v1/SOURCE_RUNTIME.json": sha(BASE / "SOURCE_RUNTIME.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/FROZEN_METHOD_LOCK.json": sha(BASE / "FROZEN_METHOD_LOCK.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/ROSTER_LOCK.json": sha(BASE / "ROSTER_LOCK.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/TEMPORAL_SELECTION_LOCK.json": sha(BASE / "TEMPORAL_SELECTION_LOCK.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/confirmation/VARIANT_LOCK.json": sha(CONF / "VARIANT_LOCK.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/confirmation/TEMPORAL_TEACHER_GLOBAL_BARRIER.json": sha(CONF / "TEMPORAL_TEACHER_GLOBAL_BARRIER.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/confirmation/TEMPORAL_FIT_BARRIER.json": sha(CONF / "TEMPORAL_FIT_BARRIER.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/confirmation/CANDIDATE_INPUT_BARRIER.json": sha(CONF / "CANDIDATE_INPUT_BARRIER.json"),
        "artifacts/stvg_soft_ep_coordinate128_v1/confirmation/TEACHER_JOB_LOCK.json": sha(CONF / "TEACHER_JOB_LOCK.json"),
        model_rel: model_sha,
        "artifacts/stvg_soft_ep_coordinate128_v1/recovery/teacher_runtime003/CAPTURE.json": sha(recovery_dir / "CAPTURE.json"),
        LOCK_REL: sha(lock_path),
        "scripts/continue_stvg_soft_ep_coordinate_preparation003.py": sha(Path(__file__).resolve()),
        "scripts/run_stvg_soft_ep_coordinate_inputs_v1.py": sha(ROOT / "scripts/run_stvg_soft_ep_coordinate_inputs_v1.py"),
        "scripts/run_stvg_evidence_teacher_v1.py": sha(ROOT / "scripts/run_stvg_evidence_teacher_v1.py"),
        "scripts/assemble_stvg_soft_ep_coordinate_v1.py": sha(ROOT / "scripts/assemble_stvg_soft_ep_coordinate_v1.py"),
        "scripts/review_stvg_soft_ep_coordinate_inputs_v1.py": sha(ROOT / "scripts/review_stvg_soft_ep_coordinate_inputs_v1.py"),
        "scripts/launch_stvg_instance_identity_v1.py": sha(ROOT / "scripts/launch_stvg_instance_identity_v1.py"),
        "scripts/stvg_instance_identity_common_v1.py": sha(ROOT / "scripts/stvg_instance_identity_common_v1.py"),
    }
    for item in case_files:
        extra[item["path"]] = item["sha256"]
    for path, value in extra.items():
        if path in pin_map:
            assert pin_map[path] == value, (path, pin_map[path], value)
        if path in protected_map:
            assert protected_map[path] == value, (path, protected_map[path], value)
            continue
        pin_map[path] = value
    for path in protected_map:
        assert path not in pin_map, path

    teacher_runtime = {
        "schema": "stvg-soft-ep-coordinate-teacher-runtime003-v1",
        "status": "locked",
        "GT_read": False,
        "model_name": model_receipt["repository"],
        "model_revision": model_receipt["revision"],
        "checkpoint_path": "checkpoints/STVG-R1-7B",
        "model_readback_path": model_rel,
        "model_readback_sha256": model_sha,
        "generation": {
            "max_new_tokens": 96,
            "do_sample": False,
            "use_cache": True,
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "device_map": "auto",
            "max_memory": {"gpu": "20GiB", "cpu": "4GiB"},
            "manual_seed_set": False,
        },
        "parent_bindings": {
            "recovery_lock_sha256": sha(lock_path),
            "preparation_runtime_sha256": bindings["preparation_runtime_sha256"],
            "recovery_runtime002_sha256": bindings["recovery_runtime002_sha256"],
            "candidate_input_barrier_sha256": bindings["candidate_input_barrier_sha256"],
            "teacher_job_lock_sha256": bindings["teacher_job_lock_sha256"],
            "official_model_receipt_sha256": model_sha,
            "new_crop_teacher_job_count": len(jobs),
        },
        "pins": pin_map,
        "protected": protected_map,
    }
    write(CONF / "TEACHER_RUNTIME.json", teacher_runtime)
    assert sha(CONF / "TEACHER_RUNTIME.json")
    recovery_pins = dict(pin_map)
    recovery_protected = dict(protected_map)
    recovery_extra = {
        TEACHER_RUNTIME_REL: sha(CONF / "TEACHER_RUNTIME.json"),
        LOCK_REL: sha(lock_path),
        "artifacts/stvg_soft_ep_coordinate128_v1/recovery/teacher_runtime003/CAPTURE.json": sha(recovery_dir / "CAPTURE.json"),
        "scripts/continue_stvg_soft_ep_coordinate_preparation003.py": sha(Path(__file__).resolve()),
    }
    for path, value in recovery_extra.items():
        if path in recovery_protected:
            assert recovery_protected[path] == value, path
            continue
        if path in recovery_pins:
            assert recovery_pins[path] == value, path
        recovery_pins[path] = value
    for path in recovery_protected:
        assert path not in recovery_pins, path
    recovery_runtime = {
        "schema": "stvg-soft-ep-coordinate-recovery-runtime003-v1",
        "status": "locked",
        "scope": "teacher003 and CPU references/assembly/input readback continuation",
        "GT_read": False,
        "parent_bindings": {
            "preparation_runtime_sha256": bindings["preparation_runtime_sha256"],
            "recovery_runtime002_sha256": bindings["recovery_runtime002_sha256"],
            "recovery_lock_sha256": sha(lock_path),
            "teacher_runtime_sha256": sha(CONF / "TEACHER_RUNTIME.json"),
            "candidate_input_barrier_sha256": bindings["candidate_input_barrier_sha256"],
            "teacher_job_lock_sha256": bindings["teacher_job_lock_sha256"],
            "official_model_receipt_sha256": model_sha,
        },
        "pins": recovery_pins,
        "protected": recovery_protected,
    }
    write(BASE / "RECOVERY_RUNTIME003.json", recovery_runtime)
    assert sha(BASE / "RECOVERY_RUNTIME003.json")
    write(BASE / "PREPARATION_STATUS003.json", {
        "status": "prepared_for_teacher",
        "stage": "teacher",
        "teacher_job_count": len(jobs),
        "candidate_input_file_count": len(files),
        "runtime_sha256": sha(CONF / "TEACHER_RUNTIME.json"),
        "recovery_runtime_sha256": sha(BASE / "RECOVERY_RUNTIME003.json"),
        "recovery_lock_sha256": sha(lock_path),
        "GT_read": False,
        "time": time.time(),
    }, mutable=(BASE / "PREPARATION_STATUS003.json").exists())
    return lock, teacher_runtime


def _check_self_unit(resource):
    resource.BASE = EXEC
    resource.UNIT = UNIT
    resource.STAGE = "preparation003"
    sample = resource.sample()
    assert sample["available"] >= 8 * GiB and sample["gpu_free"] >= 26 * GiB and sample["disk_free"] >= 30 * GiB, sample
    props = dict(x.split("=", 1) for x in subprocess.check_output(
        ["systemctl", "--user", "show", UNIT, "--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID,ActiveState,SubState"],
        text=True, timeout=5).splitlines() if "=" in x)
    assert props.get("ActiveState") == "active" and int(props["MainPID"]) == os.getpid(), props
    assert [int(props[k]) for k in ("MemoryHigh", "MemoryMax", "MemorySwapMax")] == [16 * GiB, 20 * GiB, GiB], props
    return resource, sample, props


def _run_child(stage, script, interpreter, args, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with log_path.open("xb") as f:
        write(EXEC / f"LAUNCH_{stage}.json", {"status": "running", "stage": stage, "parent_pid": os.getpid(), "interpreter": str(interpreter.relative_to(ROOT)), "source_script": script, "time": time.time(), "GT_read": False})
        p = subprocess.Popen([str(interpreter), "-B", str(ROOT / script), *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        write(EXEC / f"WORKER_{stage}.json", {"status": "running", "stage": stage, "worker_pid": p.pid, "time": time.time(), "GT_read": False})
        code = p.wait()
    write(EXEC / f"WORKER_{stage}.json", {"status": "exited", "stage": stage, "worker_pid": p.pid, "exit_code": code, "time": time.time(), "GT_read": False}, mutable=True)
    assert code == 0, f"{stage} child failed with exit code {code}; see {log_path.relative_to(ROOT)}"


def _validate_teacher_barrier(runtime_sha, lock):
    barrier_path = CONF / "TEACHER_GLOBAL_BARRIER.json"
    b = read(barrier_path)
    assert b.get("status") == "complete" and b.get("count") == 256 and b.get("GT_read") is False
    assert b.get("runtime_sha256") == runtime_sha
    assert b.get("input_barrier_sha256") == lock["bindings"]["candidate_input_barrier_sha256"]
    assert b.get("job_lock_sha256") == lock["bindings"]["teacher_job_lock_sha256"]
    expected = {x["expected_output_path"] for x in lock["teacher_jobs"]}
    listed = {x["path"]: x for x in b["files"]}
    assert len(b["files"]) == 256 and set(listed) == expected
    for path, item in listed.items():
        p = _rel_file(path)
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], path
    return {"path": str(barrier_path.relative_to(ROOT)), "sha256": sha(barrier_path), "count": len(b["files"])}


def run():
    assert RECOVERY.exists() and (RECOVERY / "RECOVERY_LOCK.json").is_file()
    assert (CONF / "TEACHER_RUNTIME.json").is_file()
    lock = read(RECOVERY / "RECOVERY_LOCK.json")
    rt_path = CONF / "TEACHER_RUNTIME.json"
    rt = read(rt_path)
    assert lock.get("status") == "locked" and lock.get("GT_read") is False
    assert rt.get("status") == "locked" and rt.get("GT_read") is False
    assert rt["parent_bindings"]["recovery_lock_sha256"] == sha(RECOVERY / "RECOVERY_LOCK.json")
    assert rt["parent_bindings"]["candidate_input_barrier_sha256"] == sha(CONF / "CANDIDATE_INPUT_BARRIER.json")
    assert rt["parent_bindings"]["teacher_job_lock_sha256"] == sha(CONF / "TEACHER_JOB_LOCK.json")
    for path, expected in {**rt["pins"], **rt["protected"]}.items():
        assert sha(_rel_file(path)) == expected, path

    EXEC.mkdir(parents=True, exist_ok=False)
    write(EXEC / "LAUNCH.json", {"status": "running", "pid": os.getpid(), "unit": UNIT, "runtime_sha256": sha(rt_path), "recovery_lock_sha256": sha(RECOVERY / "RECOVERY_LOCK.json"), "GT_read": False, "time": time.time()})

    from scripts import launch_stvg_instance_identity_v1 as resource
    resource, sample, props = _check_self_unit(resource)
    write(EXEC / "ACTUAL_LIMITS.json", {"status": "pass", "unit": UNIT, "properties": props, "admission": sample, "GT_read": False, "time": time.time()})
    stop = threading.Event()
    monitor = threading.Thread(target=resource.monitor, args=(stop,), daemon=True)
    monitor.start()
    try:
        write(BASE / "PREPARATION_STATUS003.json", {"status": "running", "stage": "teacher", "teacher_job_count": 256, "GT_read": False, "time": time.time()}, mutable=True)
        _run_child("teacher", "scripts/run_stvg_soft_ep_coordinate_inputs_v1.py", ROOT / ".venv/stvg_native_distill_v1/bin/python", ["teacher"], EXEC / "teacher_worker.log")
        teacher_receipt = _validate_teacher_barrier(sha(rt_path), lock)
        write(BASE / "PREPARATION_STATUS003.json", {"status": "running", "stage": "references", "teacher_barrier_sha256": teacher_receipt["sha256"], "GT_read": False, "time": time.time()}, mutable=True)
        _run_child("references", "scripts/run_stvg_soft_ep_coordinate_inputs_v1.py", ROOT / ".conda/tubedetr/bin/python", ["references"], EXEC / "references_worker.log")
        write(BASE / "PREPARATION_STATUS003.json", {"status": "running", "stage": "assembly", "teacher_barrier_sha256": teacher_receipt["sha256"], "GT_read": False, "time": time.time()}, mutable=True)
        _run_child("assembly", "scripts/assemble_stvg_soft_ep_coordinate_v1.py", ROOT / ".conda/tubedetr/bin/python", [], EXEC / "assembly_worker.log")
        assembly = read(BASE / "ROOT_INPUT_ASSEMBLY.json")
        assert assembly.get("status") == "pass" and assembly.get("count") == 128 and assembly.get("GT_read") is False
        write(BASE / "PREPARATION_STATUS003.json", {"status": "running", "stage": "independent_input_readback", "teacher_barrier_sha256": teacher_receipt["sha256"], "GT_read": False, "time": time.time()}, mutable=True)
        _run_child("input_readback", "scripts/review_stvg_soft_ep_coordinate_inputs_v1.py", ROOT / ".conda/tubedetr/bin/python", [], EXEC / "input_readback_worker.log")
        readback_path = BASE / "ROOT_COMPLETE_INPUT_READBACK.json"
        readback = read(readback_path)
        assert readback.get("status") == "pass" and readback.get("count") == 128 and readback.get("GT_read") is False and readback.get("CPU_only") is True
        completion = {
            "schema": "stvg-soft-ep-coordinate-preparation003-completion-v1",
            "status": "complete_pending_coordinate_search_tail",
            "GT_read": False,
            "teacher_runtime_sha256": sha(rt_path),
            "recovery_lock_sha256": sha(RECOVERY / "RECOVERY_LOCK.json"),
            "teacher_barrier": teacher_receipt,
            "reference_lock_sha256": sha(BASE / "REFERENCE_LOCK.json"),
            "temporal_lock_sha256": sha(BASE / "TEMPORAL_LOCK.json"),
            "root_input_assembly_sha256": sha(BASE / "ROOT_INPUT_ASSEMBLY.json"),
            "root_complete_input_readback_sha256": sha(readback_path),
            "root_complete_input_readback_status": readback["status"],
            "case_count": readback["count"],
            "valid_teacher_job_count": readback["valid_count"],
            "invalid_teacher_job_count": readback["invalid_count"],
            "at_token_cap_count": readback["at_token_cap_count"],
            "time": time.time(),
        }
        write(RECOVERY / "COMPLETION.json", completion)
        write(EXEC / "WORKER_EXIT.json", {"status": "exited", "exit_code": 0, "pid": os.getpid(), "GT_read": False, "time": time.time()})
        write(BASE / "PREPARATION_STATUS003.json", {"status": "complete_pending_lr2_and_coordinate_search_tail", "stage": "root_handoff", "completion_sha256": sha(RECOVERY / "COMPLETION.json"), "GT_read": False, "time": time.time()}, mutable=True)
        return 0
    finally:
        stop.set()
        monitor.join(timeout=3)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "prepare-only":
        lock, runtime = prepare()
        print(json.dumps({"status": "prepared", "recovery_lock_sha256": sha(RECOVERY / "RECOVERY_LOCK.json"), "teacher_runtime_sha256": sha(CONF / "TEACHER_RUNTIME.json"), "input_files": len(lock["candidate_input_files"]), "jobs": len(lock["teacher_jobs"]), "GT_read": False}))
        return 0
    assert mode == "run"
    try:
        if not (CONF / "TEACHER_RUNTIME.json").exists():
            prepare()
        return run()
    except BaseException as exc:
        # Keep diagnostics to exception type and stage; never copy prompts,
        # response bodies, frame tensors, annotation data, or private text.
        stage = read(BASE / "PREPARATION_STATUS003.json").get("stage", "preflight") if (BASE / "PREPARATION_STATUS003.json").exists() else "preflight"
        write(RECOVERY / "FAILURE_003.json", {"schema": "stvg-soft-ep-coordinate-preparation003-failure-v1", "status": "failed", "stage": stage, "error_type": type(exc).__name__, "GT_read": False, "time": time.time()})
        write(BASE / "PREPARATION_STATUS003.json", {"status": "failed", "stage": stage, "error_type": type(exc).__name__, "GT_read": False, "time": time.time()}, mutable=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
