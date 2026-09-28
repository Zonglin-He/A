"""One finite, resumable last-stage queue; no scheduler or method search.

Wait for the already-running HC ViTTA run AND independent preparation review.
Then run only the user-authorized remaining VidSTG experiments, sequentially.
Every worker retains its own exact source/reset/dev/full/hash barriers.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/decota_paper_execution_20260917"
OUT = BASE / "background_last"
PYTHON = ROOT / ".conda/tubedetr/bin/python"
READY = BASE / "vid_last/READY_FOR_BACKGROUND.json"
VID_SCRIPT = ROOT / "scripts/run_vid_last_paper_20260917.py"
VT_SCRIPT = ROOT / "scripts/run_vitta_paper_v1.py"
HC_COMPLETE = BASE / "vitta/vid_to_hc1/full/analysis/COMPLETION.json"
HC_PROCESS = BASE / "vitta/HC1_FULL_PROCESS.json"
LIBRARY = BASE / "runtime/nvidia580159/usr/lib/x86_64-linux-gnu"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def status(value):
    value.update(pid=os.getpid(), updated=time.time(), finite=True, recurring=False)
    tmp = OUT / "STATUS.json.tmp"
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(OUT / "STATUS.json")


def stages():
    vt = str(VT_SCRIPT)
    vd = str(VID_SCRIPT)
    # All short source/development checks precede any new full Vid inference.
    return [
        ("01_vitta_hc2_source64", [vt, "source", "--direction", "hc2_to_vid"]),
        ("02_vitta_vid_dev8", [vt, "run", "--direction", "hc2_to_vid", "--phase", "dev"]),
        ("03_vitta_vid_dev_score", [vt, "score", "--direction", "hc2_to_vid", "--phase", "dev"]),
        ("04_native_vid_dev", [vd, "native-run", "--phase", "dev"]),
        ("05_native_vid_dev_score", [vd, "native-score", "--phase", "dev"]),
        ("06_safety_vid_dev", [vd, "safety-run", "--phase", "dev"]),
        ("07_safety_vid_dev_score", [vd, "safety-score", "--phase", "dev"]),
        ("08_native_vid_full", [vd, "native-run", "--phase", "full"]),
        ("09_native_vid_full_score", [vd, "native-score", "--phase", "full"]),
        ("10_vitta_vid_full", [vt, "run", "--direction", "hc2_to_vid", "--phase", "full"]),
        ("11_vitta_vid_full_score", [vt, "score", "--direction", "hc2_to_vid", "--phase", "full"]),
        ("12_safety_vid_full", [vd, "safety-run", "--phase", "full"]),
        ("13_safety_vid_full_score", [vd, "safety-score", "--phase", "full"]),
    ]


def check_ready():
    r = read(READY)
    if r.get("status") != "passed":
        raise RuntimeError("Independent Vid-last review has not passed")
    if sha(VID_SCRIPT) != r["runner_sha256"]:
        raise RuntimeError("Vid-last runner differs from reviewed code")
    lock = Path(r["lock_path"]).resolve()
    if not lock.is_relative_to(BASE / "vid_last") or sha(lock) != r["lock_sha256"]:
        raise RuntimeError("Vid-last lock differs from independently reviewed lock")
    return r


def await_dependencies(deadline):
    dead_count = 0
    while True:
        if HC_COMPLETE.exists() and READY.exists():
            hc = read(HC_COMPLETE)
            if hc.get("status") != "scored_complete":
                raise RuntimeError("HC ViTTA score did not complete")
            for path, expected in hc["files"].items():
                if sha(path) != expected:
                    raise RuntimeError("HC ViTTA completion hash mismatch")
            return check_ready()
        if time.time() > deadline:
            raise TimeoutError("Three-hour dependency wait cap reached; no Vid inference launched")
        if not HC_COMPLETE.exists() and HC_PROCESS.exists():
            h = read(HC_PROCESS)
            if h.get("status") in {"failed", "error"}:
                raise RuntimeError("Preceding HC ViTTA run failed")
            try:
                os.kill(int(h["pid"]), 0)
                dead_count = 0
            except ProcessLookupError:
                dead_count += 1
            if dead_count >= 3:
                raise RuntimeError("HC coordinator exited without complete scored output")
        status(dict(status="waiting_for_HC_completion_and_independent_Vid_preparation",
                    HC_score_complete=HC_COMPLETE.exists(),
                    independent_Vid_ready=READY.exists(), no_Vid_inference_started=True))
        time.sleep(10)


def terminate_child(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def interrupted(signum, frame):
    raise InterruptedError(f"Coordinator interrupted by signal {signum}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    guard = (OUT / "coordinator.lock").open("a+")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    global_deadline = started + 168 * 3600
    stage_name = "dependencies"
    child = None
    signal.signal(signal.SIGTERM, interrupted)
    try:
        review = await_dependencies(started + 3 * 3600)
        vt_sha = sha(VT_SCRIPT)
        plan = dict(created=time.time(), script_sha256=sha(Path(__file__)),
                    independent_Vid_review=review, vitta_runner_sha256=vt_sha,
                    max_total_hours=168, stage_timeout_hours=36,
                    command_prefix=[str(PYTHON), "-u", "-B"],
                    stages=[dict(name=n, args=a) for n, a in stages()],
                    source_and_dev_before_full=True, recurring=False)
        plan_path = OUT / "LOCKED_PLAN.json"
        if plan_path.exists():
            old = read(plan_path)
            for field in ("script_sha256", "independent_Vid_review", "vitta_runner_sha256", "stages"):
                if plan[field] != old[field]:
                    raise RuntimeError("Resume plan changed: " + field)
        else:
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(LIBRARY)
        env["PYTHONUNBUFFERED"] = "1"
        for stage_name, args in stages():
            if time.time() - started > 168 * 3600:
                raise TimeoutError("Total finite background budget exceeded")
            if check_ready() != review:
                raise RuntimeError("Independent review changed after background plan lock")
            if sha(VT_SCRIPT) != vt_sha:
                raise RuntimeError("ViTTA runner changed after background plan lock")
            done_path = OUT / f"{stage_name}.done.json"
            if done_path.exists():
                saved = read(done_path)
                if saved["args"] != args or saved["exit_code"] != 0:
                    raise RuntimeError("Invalid completed-stage receipt")
                continue
            log_path = OUT / f"{stage_name}.log"
            tick = time.time()
            with log_path.open("a") as log:
                child = subprocess.Popen([str(PYTHON), "-u", "-B", *args], cwd=ROOT,
                                         env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                status(dict(status="running", stage=stage_name, child_pid=child.pid,
                            started=tick, log=str(log_path)))
                try:
                    remaining = global_deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError("Total finite background budget exceeded")
                    code = child.wait(timeout=min(36 * 3600, remaining))
                except subprocess.TimeoutExpired:
                    terminate_child(child)
                    child = None
                    raise TimeoutError("Stage cap exceeded: " + stage_name)
            child = None
            if code:
                raise RuntimeError(f"{stage_name} exited {code}; no later stages launched")
            done_path.write_text(json.dumps(dict(args=args, exit_code=code, started=tick,
                                                completed=time.time(), log=str(log_path)), indent=2) + "\n")
        status(dict(status="complete", stages_complete=len(stages()), started=started))
    except BaseException as exc:
        terminate_child(child)
        status(dict(status="failed", stage=stage_name, error=repr(exc), started=started))
        raise
    finally:
        guard.close()


if __name__ == "__main__":
    main()
