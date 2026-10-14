"""Fail-closed resource supervisor for the fixed-state spatial amplitude readout."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import read, sha, write

BASE = ROOT / "artifacts/stvg_spatial_causal_decomposition_v1/phase1"
UNIT = "stvg-spatial-amplitude-001.service"
GiB = 1024 ** 3


def runtime_path():
    return BASE / "RUNTIME.json"


def lock_path():
    return BASE / "PHASE1_LOCK.json"


def sample():
    line = next(
        row for row in Path("/proc/self/cgroup").read_text().splitlines()
        if row.startswith("0::")
    )
    assert line[3:].endswith("/" + UNIT), "supervisor must run in its own named unit"
    cgroup = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    stats = {
        key: int(value)
        for key, value in (
            row.split() for row in (cgroup / "memory.stat").read_text().splitlines()
        )
    }
    available = next(
        int(row.split()[1]) * 1024
        for row in Path("/proc/meminfo").read_text().splitlines()
        if row.startswith("MemAvailable:")
    )
    gpu_free = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=3,
    ).splitlines()[0]) * 1024 ** 2
    own_pids = {
        int(pid)
        for path in cgroup.rglob("cgroup.procs")
        for pid in path.read_text().split()
    }
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=3,
    )
    own_gpu_active = bool(
        own_pids & {int(pid.strip()) for pid in apps.splitlines() if pid.strip().isdigit()}
    )
    return {
        "current": int((cgroup / "memory.current").read_text()),
        "anon": stats.get("anon", 0) + stats.get("shmem", 0),
        "available": available,
        "gpu_free": gpu_free,
        "own_gpu_active": own_gpu_active,
        "disk_free": shutil.disk_usage(ROOT).free,
    }


def trigger(values):
    if values["current"] >= 18 * GiB:
        return "own_RAM_18GiB"
    if values["anon"] >= 14 * GiB:
        return "own_nonreclaimable_RAM_14GiB"
    if values["available"] < 6 * GiB:
        return "host_RAM_headroom_6GiB"
    if values["own_gpu_active"] and values["gpu_free"] < 4 * GiB:
        return "own_GPU_headroom_4GiB"
    if values["disk_free"] < 30 * GiB:
        return "disk_floor_30GiB"
    return None


def monitor(stop):
    errors = 0
    while not stop.wait(0.5):
        try:
            values = sample()
            errors = 0
            reason = trigger(values)
            if reason is None:
                continue
        except Exception as exc:
            errors += 1
            values = {"error_type": type(exc).__name__}
            reason = "resource_sampling_failed_three_consecutive_times"
            write(BASE / "STUDENT_RESOURCE_SAMPLING_ERROR.json", {
                "status": "engineering_sampling_error",
                "count": errors,
                "GT_read": False,
                "time": time.time(),
            }, mutable=True)
            if errors < 3:
                continue
        write(BASE / "STUDENT_RESOURCE_STOP_RECEIPT.json", {
            "status": "own_resource_guard_stop",
            "reason": reason,
            "sample": values,
            "unit": UNIT,
            "other_processes_signalled": False,
            "GT_read": False,
            "time": time.time(),
        }, mutable=True)
        subprocess.run(
            ["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGTERM", UNIT],
            timeout=5,
            check=True,
        )
        return


def _verify_locked_inputs():
    runtime = read(runtime_path())
    assert runtime.get("status") == "locked"
    assert runtime.get("phase") == "phase1" and runtime.get("GT_read") is False
    assert runtime.get("phase1_lock_sha256") == sha(lock_path())
    pins = runtime.get("pins", {})
    protected = runtime.get("protected", {})
    assert isinstance(pins, dict) and isinstance(protected, dict)
    for rel in (
        "scripts/run_stvg_spatial_update_amplitude_v1.py",
        "scripts/launch_stvg_spatial_amplitude_v1.py",
    ):
        assert rel in pins
    assert "methods/CURRENT_METHOD.json" in protected
    for rel, digest in {**pins, **protected}.items():
        assert sha(ROOT / rel) == digest, f"runtime pin mismatch: {rel}"


def main():
    _verify_locked_inputs()
    values = sample()
    assert values["available"] >= 8 * GiB
    assert values["gpu_free"] >= 8 * GiB
    assert values["disk_free"] >= 30 * GiB
    effective = subprocess.check_output(
        [
            "systemctl", "--user", "show", UNIT,
            "--property=MemoryHigh,MemoryMax,MemorySwapMax,MainPID,ActiveState,SubState",
        ],
        text=True,
    )
    properties = dict(row.split("=", 1) for row in effective.splitlines() if "=" in row)
    assert int(properties["MemoryHigh"]) == 16 * GiB
    assert int(properties["MemoryMax"]) == 20 * GiB
    assert int(properties["MemorySwapMax"]) == GiB
    write(BASE / "STUDENT_ACTUAL_LIVE_RESOURCE_LIMITS.json", {
        "status": "pass",
        "unit": UNIT,
        "properties": properties,
        "admission": values,
        "GT_read": False,
        "time": time.time(),
    })
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    log = BASE / "logs" / f"amplitude_worker_v1_{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("xb") as stream:
        child = subprocess.Popen(
            [
                str(ROOT / ".conda/tubedetr/bin/python"),
                "-B",
                str(ROOT / "scripts/run_stvg_spatial_update_amplitude_v1.py"),
            ],
            cwd=ROOT,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        write(BASE / "STUDENT_LAUNCH.json", {
            "status": "actual_launched",
            "controller_pid": os.getpid(),
            "worker_pid": child.pid,
            "runtime_sha256": sha(runtime_path()),
            "unit": UNIT,
            "GT_read": False,
            "time": time.time(),
        })
        exit_code = child.wait()
    stop.set()
    write(BASE / "STUDENT_WORKER_EXIT.json", {
        "status": "actual_worker_exited",
        "exit_code": exit_code,
        "unit": UNIT,
        "GT_read": False,
        "time": time.time(),
    })
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("usage: launch_stvg_spatial_amplitude_v1.py")
    raise SystemExit(main())
