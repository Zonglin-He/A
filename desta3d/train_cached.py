"""Manual fixed-terminal cached fit. No PTD, native inference, or automatic gates.

Training starts only through the explicit `fit-cache` command. Every invocation
uses a new directory and writes its effective configuration, input hashes,
optimizer counters, terminal predictions and measured wall time.
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from vg_tta.optimizer_checkpoint import cpu_clone, restore_optimizer, validate_serialized_optimizer
from .cache import SealedCache, sha256
from .config import DirectionConfig
from .models import DirectionMixer, direction_loss


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def evaluate(model, cache, split, device, destination, guard):
    model.eval()
    entries = []
    with torch.no_grad():
        for index in range(len(cache.files[split])):
            guard()
            inputs, target = cache.get(split, index, device)
            prediction = model(inputs).cpu()
            # NumPy FP64 independently reduces terminal fields, not logged loss.
            p = prediction.numpy().astype(np.float64).ravel()
            t = target.cpu().numpy().astype(np.float64).ravel()
            denominator = np.linalg.norm(p) * np.linalg.norm(t)
            cosine = float(np.dot(p, t) / denominator) if denominator > 0 else None
            path = destination / "terminal_coefficients" / split / f"{index:04}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prediction, path)
            entries.append({"index": index, "cosine": cosine, "sha256": sha256(path)})
    values = [row["cosine"] for row in entries if row["cosine"] is not None]
    return {
        "rows": entries, "defined": len(values), "undefined": len(entries) - len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "count_gt_0_1": sum(x > 0.1 for x in values),
        "count_gt_0_3": sum(x > 0.3 for x in values),
    }


def fit(config, workspace, run_name):
    if config["kind"] != "cached_direction":
        raise ValueError("fit-cache requires a cached_direction config")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("run-name: 1–80 letters/digits/underscore/hyphen; no path separators")
    workspace = Path(workspace).resolve()
    root = workspace / "artifacts" / "manual_desta3d"
    root.mkdir(parents=True, exist_ok=True)
    # Serializes workbench fits. Also reject another visible CUDA compute process.
    with (root / ".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = root / run_name
        destination.mkdir(exist_ok=False)
        started = time.monotonic()
        status = "failed"
        device = config["training"]["device"]
        write_json(destination / "CONFIG.json", config)
        write_json(destination / "STARTED.json", {"time": time.time(), "pid": os.getpid(), "device": device})
        try:
            _fit(config, workspace, destination)
            status = "completed"
        except BaseException as error:
            write_json(destination / "FAILURE.json", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            write_json(destination / "RECEIPT.json", {
                "status": status, "device": device, "seconds_including_validation": time.monotonic() - started,
                "cap": None, "research_run": False, "PTD_loaded": False, "native_predictions": 0,
            })
    return destination


def _fit(config, workspace, destination):
    t = config["training"]
    minimum = int(t["minimum_free_gib"] * 2**30)

    def guard():
        if shutil.disk_usage(workspace).free < minimum:
            raise RuntimeError("Free disk fell below the configured floor")

    guard()
    if t["device"].startswith("cuda"):
        running = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if running:
            raise RuntimeError(f"CUDA already has a compute process: {running}")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    cache = SealedCache(config["cache"], workspace)
    # Reserve the full uncompressed terminal coefficient output before training.
    output_bytes = 0
    for split in ("train", "dev"):
        for i in range(len(cache.files[split])):
            _, target = cache.get(split, i, "cpu")
            output_bytes += target.numel() * 4 + 4096
    if shutil.disk_usage(workspace).free < minimum + output_bytes + 64 * 2**20:
        raise RuntimeError("Insufficient disk for complete terminal outputs and free-space floor")

    torch.set_num_threads(t["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(t["seed"])
    model = DirectionMixer(cache.basis, DirectionConfig(**config["model"])).to(t["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    source_files = [*Path(__file__).parent.glob("*.py"), Path(__file__).resolve().parents[1] / "vg_tta/optimizer_checkpoint.py"]
    write_json(destination / "INPUTS.json", {
        "input_sha256": cache.input_hashes,
        "code_sha256": {str(p): sha256(p) for p in source_files},
        "torch": torch.__version__, "numpy": np.__version__, "source_GT_cached_privilege": True,
        "train_queries": len(cache.files["train"]), "dev_queries": len(cache.files["dev"]),
        "fresh_data_read": False, "estimated_terminal_bytes": output_bytes,
    })
    initial = cpu_clone(model.state_dict())
    torch.save(initial, destination / "INITIAL.pt")
    sampler = random.Random(t["seed"])
    order, history = [], []
    for step in range(1, t["steps"] + 1):
        guard()
        optimizer.zero_grad(set_to_none=True)
        losses, indices = [], []
        for _ in range(t["batch_size"]):
            if not order:
                order = list(range(len(cache.files["train"])))
                sampler.shuffle(order)
            index = order.pop()
            indices.append(index)
            inputs, target = cache.get("train", index, t["device"])
            loss = direction_loss(model(inputs), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            (loss / t["batch_size"]).backward()
            losses.append(float(loss.detach()))
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Missing/nonfinite trainable gradient")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip_norm"]))
        optimizer.step()
        counters = [int(state["step"]) for state in optimizer.state.values()]
        if set(counters) != {step}:
            raise RuntimeError("Adam step counters diverged")
        row = {"step": step, "indices": indices, "loss": float(np.mean(losses)),
               "preclip_gradient_norm": grad_norm, "clipped": grad_norm > t["clip_norm"], "counters": counters}
        history.append(row)
        with (destination / "HISTORY.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if step == 1 or step % 100 == 0 or step == t["steps"]:
            print(f"step={step}/{t['steps']} loss={row['loss']:.6f} grad={grad_norm:.6f}", flush=True)

    state = cpu_clone(optimizer.state_dict())
    validate_serialized_optimizer(state)
    restored = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    restore_optimizer(restored, state)
    if not torch.equal(model.basis.cpu(), cache.basis.float()):
        raise RuntimeError("Frozen basis changed")
    torch.save({"mixer": cpu_clone(model.state_dict()), "optimizer": state, "steps": t["steps"],
                "sample_rng": sampler.getstate(), "sample_order": order, "config": config}, destination / "FINAL.pt")
    results = {s: evaluate(model, cache, s, t["device"], destination, guard) for s in ("train", "dev")}
    report = {"status": "completed_manual_fit", "results": results, "steps": t["steps"],
              "terminal_batch_loss": history[-1]["loss"], "last_gradient_norm": history[-1]["preclip_gradient_norm"],
              "clipped_steps": sum(r["clipped"] for r in history), "parameters": sum(p.numel() for p in model.parameters()),
              "automatic_native_or_promotion": False, "native_utility_measured": False}
    write_json(destination / "REPORT.json", report)
    files = {str(p.relative_to(destination)): sha256(p) for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "SEAL.json", {"files": files})
    write_json(destination / "COMPLETE.json", {"seal_sha256": sha256(destination / "SEAL.json"), "steps": t["steps"]})
