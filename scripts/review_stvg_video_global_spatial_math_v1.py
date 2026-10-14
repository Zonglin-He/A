"""Independent CPU arithmetic reader for sealed video-global spatial fits.

The reader accepts only complete ``source``, ``teacher``, or explicitly
offline ``oracle`` student barriers.  It validates all metadata and output
hashes before loading any case tensors.  It never invokes a model, image/video
reader, annotation loader, or optimizer-training function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "artifacts/stvg_video_conditioned_global_referent_v1"
PHASES = {"method", "oracle"}
EXPECTED_ROLES = {
    "method": {"source", "teacher"},
    "oracle": {"offline_oracle"},
}
TOLERANCES = {
    "mean_abs": 3e-5,
    "sample_abs": 3e-5,
    "reward_abs": 2e-4,
    "weight_abs": 2e-4,
    "loss_abs": 5e-4,
    "adam_parameter_abs": 1e-4,
    "adam_state_abs": 1e-4,
}
BOX_BOUND_TOLERANCE = 1e-7


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _safe_root_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a ROOT-relative POSIX path")
    p = Path(value)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        raise ValueError("absolute or traversing path in sealed inventory")
    lower = "/" + value.casefold().strip("/") + "/"
    if any(part in lower for part in ("/annotations/", "/ground_truth/", "/gt_labels/", "/labels/")):
        raise ValueError("annotation path is outside this fit-math reader scope")
    return value


def _verify_digest_path(relative: str, digest: Any, expected_bytes: Any = None) -> Path:
    relative = _safe_root_path(relative)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("malformed SHA256 in file receipt")
    path = ROOT / relative
    if not path.is_file() or _sha_file(path) != digest:
        raise RuntimeError(f"sealed file hash mismatch: {relative}")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise RuntimeError(f"sealed file byte count mismatch: {relative}")
    return path


def _normal_role(name: str) -> str:
    value = "".join(char for char in str(name).casefold() if char.isalpha())
    if value == "source":
        return "source"
    if value == "teacher":
        return "teacher"
    if value in {"oracle", "offlineoracle"}:
        return "offline_oracle"
    raise ValueError(f"unrecognized fixed spatial reference role: {name}")


def _required_source_paths(phase: str) -> set[str]:
    common = {
        "scripts/run_stvg_support_aware_spatial_v1.py",
        "scripts/stvg_target_teacher_student_v1.py",
        "vg_tta/decota_spatial_opd_tunable_v1.py",
        "vg_tta/tastvg_decota_critic_p0_v1.py",
    }
    if phase == "method":
        return common | {
            "scripts/prepare_stvg_video_global_references_v1.py",
            "scripts/run_stvg_video_global_student_v1.py",
            "scripts/stvg_video_global_spatial_fit_v1.py",
        }
    if phase == "oracle":
        return common | {
            "scripts/prepare_stvg_video_global_oracle_v1.py",
            "scripts/run_stvg_video_global_oracle_geometry001.py",
            "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py",
        }
    raise ValueError(f"unknown phase: {phase}")


def _verify_oracle_readback_chain(runtime: Mapping[str, Any]) -> str:
    """Verify the root's post-prediction Oracle exposure chain before reading labels."""
    receipt_path = BASE / "ROOT_READBACK_RUNTIME.json"
    receipt = _read_json(receipt_path)
    _require(receipt.get("status") == "locked"
             and receipt.get("scope") == "independent readback metadata and pre-GT method seal lineage"
             and receipt.get("GT_read") is False
             and receipt.get("Oracle_reference_is_GT_exposed") is True,
             "Oracle root readback lock is absent or has a different evaluation boundary")
    chain = receipt.get("chain_pins")
    _require(isinstance(chain, dict), "Oracle root readback lock has no explicit chain_pins map")
    required = {
        str((BASE / "METHOD_STUDENT_RUNTIME.json").relative_to(ROOT)),
        str((BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json").relative_to(ROOT)),
        str((BASE / "ORACLE_GT_EXPOSURE_RECEIPT.json").relative_to(ROOT)),
        str((BASE / "ORACLE_STUDENT_RUNTIME.json").relative_to(ROOT)),
        str((BASE / "ORACLE_REFERENCE_LOCK.json").relative_to(ROOT)),
    }
    _require(required <= set(chain),
             "Oracle root readback lock omits a method seal, exposure receipt, or phase lock")
    method_rel = str((BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json").relative_to(ROOT))
    method_runtime_rel = str((BASE / "METHOD_STUDENT_RUNTIME.json").relative_to(ROOT))
    exposure_rel = str((BASE / "ORACLE_GT_EXPOSURE_RECEIPT.json").relative_to(ROOT))
    for relative in required:
        _require(isinstance(chain[relative], str) and len(chain[relative]) == 64,
                 f"Oracle root readback has a malformed chain digest for {relative}")
    # Establish the unlabeled method seal before touching the Oracle reference
    # lock, whose anchor boxes contain the offline GT values.
    _verify_digest_path(method_rel, chain[method_rel])
    protected = receipt.get("protected")
    _require(isinstance(protected, dict) and protected,
             "Oracle root readback lock omits protected metadata pins")
    for relative, digest in protected.items():
        _verify_digest_path(relative, digest)

    method_barrier_path = BASE / "METHOD_STUDENT_GLOBAL_BARRIER.json"
    method_barrier = _read_json(method_barrier_path)
    _require(method_barrier.get("status") == "complete"
             and method_barrier.get("count") == 8
             and method_barrier.get("GT_read") is False,
             "unlabeled method phase was not completely sealed before Oracle GT")
    method_runtime_path = BASE / "METHOD_STUDENT_RUNTIME.json"
    _verify_digest_path(method_runtime_rel, chain[method_runtime_rel])
    method_runtime = _read_json(method_runtime_path)
    _require(method_runtime.get("status") == "locked"
             and method_runtime.get("GT_read") is False
             and method_barrier.get("runtime_sha256") == _sha_file(method_runtime_path),
             "unlabeled method barrier is not bound to its pre-GT runtime")
    method_files = method_barrier.get("files")
    _require(isinstance(method_files, list) and len(method_files) == 8,
             "unlabeled method barrier does not contain eight case outputs")
    method_prefix = f"{BASE.relative_to(ROOT)}/method/student/"
    method_case_ids = set()
    for row in method_files:
        relative = _safe_root_path(row.get("path"))
        _require(relative.startswith(method_prefix) and relative.endswith(".pt")
                 and "/" not in relative[len(method_prefix):],
                 "method seal includes an unexpected prediction path")
        case_id = relative[len(method_prefix):-3]
        _require(case_id not in method_case_ids, "method seal repeats a case output")
        method_case_ids.add(case_id)
        _verify_digest_path(relative, row.get("sha256"), row.get("bytes"))
    _require(len(method_case_ids) == 8, "method seal is not a complete eight-case barrier")

    _verify_digest_path(exposure_rel, chain[exposure_rel])
    exposure_path = BASE / "ORACLE_GT_EXPOSURE_RECEIPT.json"
    exposure = _read_json(exposure_path)
    _require(exposure.get("status") == "actual_offline_GT_read_after_complete_method_seal"
             and exposure.get("GT_read") is True
             and exposure.get("GT_used_in_unlabeled_method") is False
             and exposure.get("method_barrier_sha256") == _sha_file(method_barrier_path),
             "Oracle exposure receipt does not follow the sealed unlabeled method barrier")
    for relative in required - {method_rel, exposure_rel}:
        _verify_digest_path(relative, chain[relative])
    return _sha_file(receipt_path)


def _preflight(phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}")
    expected_gt = phase == "oracle"
    runtime_path = BASE / f"{phase.upper()}_STUDENT_RUNTIME.json"
    reference_path = BASE / f"{phase.upper()}_REFERENCE_LOCK.json"
    barrier_path = BASE / f"{phase.upper()}_STUDENT_GLOBAL_BARRIER.json"
    # Read only the execution lock and prediction barrier first.  In the
    # offline Oracle phase the reference lock contains GT-derived anchor
    # coordinates, so do not deserialize it until the full eight-case output
    # inventory is present and every output byte has passed its barrier hash.
    runtime = _read_json(runtime_path)

    _require(runtime.get("status") == "locked" and runtime.get("GT_read") is expected_gt,
             "runtime GT boundary does not match its explicit phase")
    if expected_gt:
        _require(runtime.get("offline_nondeployment") is True,
                 "oracle phase must be explicitly offline and nondeployment")
        _require(runtime.get("unlabeled_method_sealed_before_oracle") is True,
                 "oracle runtime does not attest that the unlabeled method was sealed first")

    barrier = _read_json(barrier_path)
    _require(barrier.get("status") == "complete" and barrier.get("count") == 8
             and barrier.get("GT_read") is expected_gt,
             "student global barrier is incomplete or has another GT boundary")
    if expected_gt:
        _require(barrier.get("offline_nondeployment") is True,
                 "oracle student barrier is not explicitly offline nondeployment")
    _require(barrier.get("runtime_sha256") == _sha_file(runtime_path),
             "student barrier is not bound to its runtime")

    pins = runtime.get("pins")
    protected = runtime.get("protected")
    _require(isinstance(pins, dict) and pins and isinstance(protected, dict) and protected,
             "runtime must pin source code and protected method metadata")
    required_source_paths = _required_source_paths(phase)
    # The actual pinned helper checks its embedded SOURCE_BINDINGS on every fit.
    # Verify that pinned helper before reading/importing its transitive hash list.
    helper_rel = ("scripts/stvg_video_global_spatial_fit_v1.py" if phase == "method"
                  else "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py")
    _require(helper_rel in pins, "runtime omits the actual execution helper")
    _verify_digest_path(helper_rel, pins[helper_rel])
    import importlib
    helper = importlib.import_module(helper_rel[:-3].replace("/", "."))
    transitive = helper.verify_source_bindings()
    pins = {**transitive, **pins}
    _require(required_source_paths <= set(pins),
             "runtime and pre-execution embedded helper omit a required source pin")
    for relative in required_source_paths:
        _verify_digest_path(relative, pins[relative])
    for relative, digest in protected.items():
        _verify_digest_path(relative, digest)

    file_rows = barrier.get("files")
    _require(isinstance(file_rows, list) and len(file_rows) == 8,
             "student barrier must seal exactly eight case outputs")
    expected_prefix = f"{BASE.relative_to(ROOT)}/{phase}/student/"
    verified_paths = {}
    file_index = set()
    for row in file_rows:
        _require(isinstance(row, dict), "malformed student output file row")
        relative = _safe_root_path(row.get("path"))
        _require(relative.startswith(expected_prefix) and relative.endswith(".pt")
                 and "/" not in relative[len(expected_prefix):],
                 "student barrier output path is outside the exact phase case directory")
        case_id = relative[len(expected_prefix):-3]
        _require(bool(case_id) and case_id not in file_index,
                 "student barrier repeats or omits a case filename")
        file_index.add(case_id)
        verified_paths[case_id] = _verify_digest_path(
            relative, row.get("sha256"), row.get("bytes"))

    reference_rel = str(reference_path.relative_to(ROOT))
    reference_digest = barrier.get("reference_lock_sha256")
    _require(isinstance(reference_digest, str) and len(reference_digest) == 64
             and pins.get(reference_rel) == reference_digest,
             "phase reference lock is not bound by both runtime and student barrier")

    oracle_readback_sha256 = None
    if expected_gt:
        oracle_readback_sha256 = _verify_oracle_readback_chain(runtime)

    # Runtime input pins can include the sealed reference file.  Check them
    # after prediction output sealing so the Oracle branch cannot read a GT
    # anchor lock before all phase outputs are complete.
    for relative, digest in pins.items():
        _verify_digest_path(relative, digest)
    references = _read_json(reference_path)
    _require(_sha_file(reference_path) == reference_digest,
             "reference lock bytes differ from the student barrier")
    _require(barrier.get("reference_lock_sha256") == reference_digest,
             "student barrier reference lock digest changed")
    _require(references.get("status") == "locked"
             and references.get("GT_read") is expected_gt,
             "phase reference lock is incomplete or its GT scope differs")
    if expected_gt:
        _require(references.get("offline_nondeployment") is True,
                 "oracle reference lock is not marked offline nondeployment")
    _require(runtime.get("expected_fit_count") == 8 * len(EXPECTED_ROLES[phase]) * 3,
             "runtime expected fit count differs from the fixed phase method matrix")
    _require(barrier.get("reference_lock_sha256") == _sha_file(reference_path),
             "student barrier is not bound to the phase reference lock")

    records = references.get("records")
    _require(isinstance(records, list) and len(records) == 8,
             "reference lock must contain all eight fixed cases")
    record_by_id = {}
    for record in records:
        _require(isinstance(record, dict) and isinstance(record.get("case_id"), str)
                 and record.get("dataset") in {"hc2", "vidstg"},
                 "reference lock has malformed case metadata")
        if record["case_id"] in record_by_id:
            raise RuntimeError("reference lock repeats a case ID")
        refs = record.get("references")
        _require(isinstance(refs, dict), "case has no separate reference roles")
        roles = {_normal_role(name) for name in refs}
        _require(roles == EXPECTED_ROLES[phase],
                 f"{phase} phase contains the wrong fixed reference role set")
        record_by_id[record["case_id"]] = record
    domains = {name: sum(row["dataset"] == name for row in records) for name in ("hc2", "vidstg")}
    _require(domains == {"hc2": 4, "vidstg": 4}, "locked eight-case roster is not balanced by domain")

    _require(set(verified_paths) == set(record_by_id),
             "student barrier does not seal the exact eight roster case outputs")

    return {"phase": phase, "GT_read": expected_gt,
            "offline_nondeployment": expected_gt, "runtime": runtime,
            "runtime_path": runtime_path, "references": references,
            "reference_path": reference_path, "record_by_id": record_by_id,
            "barrier": barrier, "barrier_path": barrier_path,
            "case_paths": verified_paths,
            "oracle_root_readback_runtime_sha256": oracle_readback_sha256,
            "phase_fit_count": 8 * len(EXPECTED_ROLES[phase]) * 3,
            "hashes": {"runtime": _sha_file(runtime_path),
                       "reference_lock": _sha_file(reference_path),
                       "student_barrier": _sha_file(barrier_path)}}


def _tensor_error(left: torch.Tensor, right: torch.Tensor) -> float:
    a, b = left.detach().cpu(), right.detach().cpu()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return math.inf
    if not a.numel():
        return 0.0
    return float((a.to(torch.float64) - b.to(torch.float64)).abs().max())


def _close(left: torch.Tensor, right: torch.Tensor, tolerance: float, name: str) -> float:
    error = _tensor_error(left, right)
    if error > tolerance:
        raise RuntimeError(f"{name} CPU reconstruction error {error:.9g} exceeds {tolerance:.9g}")
    return error


def _xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = cxcywh.unbind(-1)
    return torch.stack((cx - width / 2, cy - height / 2,
                        cx + width / 2, cy + height / 2), dim=-1)


def _iou(prediction: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    pred, target = _xyxy(prediction), _xyxy(evidence)
    intersection_wh = (torch.minimum(pred[..., 2:], target[..., 2:])
                      - torch.maximum(pred[..., :2], target[..., :2])).clamp_min(0)
    intersection = intersection_wh.prod(-1)
    pred_area = (pred[..., 2:] - pred[..., :2]).clamp_min(0).prod(-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp_min(0).prod(-1)
    return intersection / (pred_area + target_area - intersection).clamp_min(1e-7)


def _reward(prediction: torch.Tensor, evidence: torch.Tensor, objective: str) -> torch.Tensor:
    overlap = _iou(prediction, evidence)
    if objective == "iou":
        return overlap
    pred, target = _xyxy(prediction), _xyxy(evidence)
    pred_center = (pred[..., :2] + pred[..., 2:]) / 2
    target_center = (target[..., :2] + target[..., 2:]) / 2
    rho2 = (pred_center - target_center).square().sum(-1)
    enclosing_wh = (torch.maximum(pred[..., 2:], target[..., 2:])
                    - torch.minimum(pred[..., :2], target[..., :2])).clamp_min(0)
    c2 = enclosing_wh.square().sum(-1).clamp_min(1e-7)
    return overlap - rho2 / c2


def _cpu_noise(key: str, step: int, anchors: int) -> tuple[int, torch.Tensor]:
    seed = int(hashlib.sha256(f"spatial-opd-v1|{key}|{step}".encode()).hexdigest()[:15], 16)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((anchors, 16, 4), generator=generator, dtype=torch.float32)
    return seed, torch.cat((noise, -noise), dim=1)


def _likelihood(mean: torch.Tensor, rollout_mean: torch.Tensor,
                samples: torch.Tensor, weights: torch.Tensor, sigma: float) -> torch.Tensor:
    kl = (mean - rollout_mean).square().sum(-1) / (2 * sigma ** 2)
    logp = -(samples - mean.unsqueeze(-2)).square().sum(-1) / (2 * sigma ** 2)
    logp = logp - 2 * math.log(2 * math.pi * sigma ** 2)
    return (kl - ((weights - 1 / 32) * logp).sum(-1)).mean()


def _compare_tree(left: Any, right: Any, *, tolerance: float = 0.0,
                  errors: list[float] | None = None, name: str = "tree") -> None:
    if errors is None:
        errors = []
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise RuntimeError(f"{name}: tensor schema differs")
        if tolerance == 0:
            a, b = left.detach().cpu(), right.detach().cpu()
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                raise RuntimeError(f"{name}: tensors are not bitwise equal")
            errors.append(0.0)
            return
        errors.append(_close(left, right, tolerance, name))
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)):
            raise RuntimeError(f"{name}: mapping schema differs")
        for key in left:
            _compare_tree(left[key], right[key], tolerance=tolerance,
                          errors=errors, name=f"{name}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (type(left) is type(right) and len(left) == len(right)):
            raise RuntimeError(f"{name}: sequence schema differs")
        for index, (a, b) in enumerate(zip(left, right)):
            _compare_tree(a, b, tolerance=tolerance, errors=errors,
                          name=f"{name}[{index}]")
        return
    if isinstance(left, float) or isinstance(right, float):
        error = abs(float(left) - float(right))
        errors.append(error)
        if error > tolerance:
            raise RuntimeError(f"{name}: scalar error {error:.9g} exceeds {tolerance:.9g}")
        return
    if left != right:
        raise RuntimeError(f"{name}: scalar values differ")


def _anchors_from_case(case: Mapping[str, Any], refname: str,
                       frame_ids: Sequence[int]) -> tuple[list[int], torch.Tensor]:
    raw_rows = case["supervision_anchors"][refname]
    _require(isinstance(raw_rows, list) and len(raw_rows) <= 4,
             "supervision must contain zero to four sparse anchors")
    positions, boxes = [], []
    seen = set()
    for row in raw_rows:
        pos = row.get("position")
        _require(isinstance(pos, int) and not isinstance(pos, bool)
                 and 0 <= pos < len(frame_ids) and pos not in seen,
                 "sparse anchor position is invalid or duplicated")
        if "physical_frame_id" in row:
            _require(int(row["physical_frame_id"]) == int(frame_ids[pos]),
                     "sparse anchor is bound to another physical frame")
        box = torch.as_tensor(row.get("box_cxcywh"), dtype=torch.float32, device="cpu")
        _require(box.shape == (4,) and torch.isfinite(box).all(),
                 "sparse anchor box must be finite normalized cxcywh")
        cx, cy, width, height = [float(x) for x in box]
        _require(width > 0 and height > 0
                 and -BOX_BOUND_TOLERANCE <= cx - width / 2 < cx + width / 2 <= 1 + BOX_BOUND_TOLERANCE
                 and -BOX_BOUND_TOLERANCE <= cy - height / 2 < cy + height / 2 <= 1 + BOX_BOUND_TOLERANCE,
                 "sparse anchor box violates normalized xyxy geometry")
        _require(float(row.get("weight", 1.0)) == 1.0,
                 "original OPD requires unit sparse-anchor weights")
        positions.append(pos)
        boxes.append(box)
        seen.add(pos)
    if not positions:
        return [], torch.empty((0, 4), dtype=torch.float32)
    order = sorted(range(len(positions)), key=lambda index: positions[index])
    return ([positions[index] for index in order], torch.stack([boxes[index] for index in order]))


def _audit_opd_fit(fit: Mapping[str, Any], anchors: tuple[list[int], torch.Tensor],
                   frame_ids: Sequence[int], key: str, objective: str,
                   expected_initial: Mapping[str, Any]) -> dict[str, Any]:
    positions, evidence = anchors
    config = fit.get("config")
    _require(isinstance(config, dict) and config.get("steps") == 5
             and config.get("samples") == 32 and float(config.get("sigma", 0)) > 0
             and float(config.get("tau", 0)) > 0 and float(config.get("lr", 0)) > 0,
             "fit config is not the five-step, 32-sample original OPD config")
    _require(fit.get("scope") == "joint" and fit.get("active_parameters") == 1792
             and fit.get("declared_steps") == 5 and fit.get("selected_step") == 5
             and len(fit.get("path", [])) == 5 and len(fit.get("rounds", [])) == 5,
             "fit does not contain all five 1,792-parameter joint steps")
    _require(fit.get("observation_positions") == positions,
             "fit anchor positions differ from the sealed sparse reference")
    _require(fit.get("seed_key") == key,
             "fit Gaussian seed key differs from the locked case key")
    _compare_tree(fit.get("initial_state"), expected_initial, name="fit.initial_state")

    maxima = {"mean_abs": 0.0, "sample_abs": 0.0, "reward_abs": 0.0,
              "weight_abs": 0.0, "loss_abs": 0.0,
              "adam_parameter_abs": 0.0, "adam_state_abs": 0.0}
    optimizer_steps = 0
    if not positions:
        for index, entry in enumerate(fit["path"]):
            _require(entry.get("step") == index and entry.get("skipped") is True
                     and "rollout" not in entry,
                     "empty reference must skip each update without synthesizing a target")
            _compare_tree(entry.get("state"), fit["initial_state"], name=f"empty.path[{index}].state")
            if index + 1 < len(fit["path"]):
                _compare_tree(entry.get("state"), fit["path"][index + 1].get("state"),
                              name=f"empty.path[{index}].state_to_next")
                _compare_tree(entry.get("optimizer_state"), fit["path"][index + 1].get("optimizer_state"),
                              name=f"empty.path[{index}].optimizer_to_next")
            _require(not entry.get("optimizer_state", {}).get("state"),
                     "empty reference changed Adam state")
        _compare_tree(fit["final_state"], fit["initial_state"], name="empty.final_state")
        _require(int(fit.get("optimizer_steps", -1)) == 0,
                 "empty reference cannot count optimizer steps")
    else:
        for index, (entry, round_row) in enumerate(zip(fit["path"], fit["rounds"])):
            _require(entry.get("step") == index and entry.get("skipped") is False,
                     "nonempty fit omitted a rollout step")
            boxes = entry.get("boxes")
            _require(torch.is_tensor(boxes) and boxes.shape == (len(frame_ids), 4)
                     and torch.isfinite(boxes).all(), "fit omitted full finite pre-step boxes")
            mean = torch.logit(boxes.detach().cpu().float()[positions])
            policy = entry.get("rollout")
            _require(isinstance(policy, Mapping), "OPD step omitted its complete rollout")
            maxima["mean_abs"] = max(maxima["mean_abs"], _close(
                policy["mean"], mean, TOLERANCES["mean_abs"], f"step{index}.mean"))
            seed, noise = _cpu_noise(key, index, len(positions))
            _require(policy.get("seed") == seed, "Gaussian noise seed differs from the locked key/step")
            expected_samples = policy["mean"].detach().cpu().float()[:, None, :] + config["sigma"] * noise
            maxima["sample_abs"] = max(maxima["sample_abs"], _close(
                policy["samples"], expected_samples, TOLERANCES["sample_abs"], f"step{index}.samples"))
            expected_rewards = _reward(policy["samples"].detach().cpu().float().sigmoid(),
                                       evidence[:, None, :], objective)
            maxima["reward_abs"] = max(maxima["reward_abs"], _close(
                policy["rewards"], expected_rewards, TOLERANCES["reward_abs"], f"step{index}.rewards"))
            maxima["reward_abs"] = max(maxima["reward_abs"], _close(
                policy["used_rewards"], expected_rewards, TOLERANCES["reward_abs"], f"step{index}.used_rewards"))
            expected_weights = torch.softmax(expected_rewards / float(config["tau"]), dim=-1)
            maxima["weight_abs"] = max(maxima["weight_abs"], _close(
                policy["weights"], expected_weights, TOLERANCES["weight_abs"], f"step{index}.weights"))
            stored_used_rewards = policy.get("used_rewards")
            _require(torch.is_tensor(stored_used_rewards)
                     and policy.get("permutations") is None,
                     "fixed OPD must retain unshuffled rewards")
            informative = bool((stored_used_rewards != stored_used_rewards[:, :1]).any())
            _require(policy.get("informative") is informative,
                     "OPD informative/no-information status differs from exact reward variation")
            expected_loss = _likelihood(policy["mean"].detach().cpu().float(),
                policy["mean"].detach().cpu().float(), policy["samples"].detach().cpu().float(),
                expected_weights, float(config["sigma"]))
            loss_error = abs(float(entry["loss"]) - float(expected_loss))
            _require(loss_error <= TOLERANCES["loss_abs"],
                     f"step{index} Gaussian likelihood loss differs by {loss_error:.9g}")
            maxima["loss_abs"] = max(maxima["loss_abs"], loss_error)
            _require(abs(float(round_row.get("loss")) - float(entry["loss"])) <= 1e-7,
                     "fit path and round summary report different losses")

            if not informative:
                _require(entry.get("update") is None
                         and round_row.get("updated") is False
                         and round_row.get("no_information") is True,
                         "equal rewards must be a true no-update")
                _compare_tree(entry["state"], entry.get("update_state", entry["state"]),
                              name=f"step{index}.no_update_state")
                _compare_tree(entry.get("optimizer_before"), entry.get("optimizer_state"),
                              name=f"step{index}.no_update_adam")
            else:
                optimizer_steps += 1
                update = entry.get("update")
                _require(isinstance(update, Mapping) and torch.is_tensor(update.get("gradient"))
                         and torch.is_tensor(update.get("raw")),
                         "informative OPD step omitted the optimizer gradient/update evidence")
                maxima["adam_parameter_abs"], maxima["adam_state_abs"] = _audit_adam_transition(
                    entry, config, maxima["adam_parameter_abs"], maxima["adam_state_abs"], index)

            if index + 1 < len(fit["path"]):
                next_entry = fit["path"][index + 1]
                _compare_tree(entry.get("update_state", entry["state"]), next_entry.get("state"),
                              name=f"step{index}.state_to_next_pre_state")
                _compare_tree(entry.get("optimizer_state"),
                              next_entry.get("optimizer_before", next_entry.get("optimizer_state")),
                              name=f"step{index}.optimizer_to_next_pre_optimizer")
                actual_delta = entry.get("actual_delta")
                expected_delta = next_entry.get("boxes") - boxes
                _require(torch.is_tensor(actual_delta),
                         f"step{index} omitted its actual decoder movement")
                _close(actual_delta, expected_delta, 1e-7,
                       f"step{index}.decoder_delta_to_next_pre_boxes")

    _require(int(fit.get("optimizer_steps", -1)) == optimizer_steps,
             "fit optimizer step count differs from independently counted updates")
    if fit["path"]:
        final_entry = fit["path"][-1]
        expected_final_state = final_entry.get("update_state", final_entry.get("state"))
        _compare_tree(fit["final_state"], expected_final_state, name="fit.final_state_vs_last_step")
    return {"steps": len(fit["path"]), "optimizer_steps": optimizer_steps,
            "max_abs_errors": maxima, "objective": objective}


def _audit_adam_transition(entry: Mapping[str, Any], config: Mapping[str, Any],
                           max_parameter_error: float, max_state_error: float,
                           step_index: int) -> tuple[float, float]:
    before = entry["state"]
    names = list(before.keys())
    params = [torch.nn.Parameter(before[name].detach().cpu().clone()) for name in names]
    _require(sum(parameter.numel() for parameter in params) == 1792,
             "saved joint state order/size does not match 1,792 active parameters")
    optimizer = torch.optim.Adam(params, lr=float(config["lr"]), betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)
    optimizer_before = entry.get("optimizer_before")
    optimizer.load_state_dict(optimizer_before)
    gradient = entry["update"]["gradient"].detach().cpu().reshape(-1)
    expected_raw = entry["update"]["raw"].detach().cpu().reshape(-1)
    _require(gradient.numel() == expected_raw.numel() == 1792,
             "saved Adam gradient/update vector has the wrong joint parameter size")
    offset = 0
    before_flat = []
    for parameter in params:
        size = parameter.numel()
        before_flat.append(parameter.detach().clone().reshape(-1))
        parameter.grad = gradient[offset:offset + size].reshape_as(parameter).to(parameter)
        offset += size
    optimizer.step()
    after_flat = torch.cat([parameter.detach().reshape(-1) for parameter in params])
    before_flat_tensor = torch.cat(before_flat)
    actual_raw = after_flat - before_flat_tensor
    raw_error = _tensor_error(actual_raw, expected_raw)
    _require(raw_error <= TOLERANCES["adam_parameter_abs"],
             f"step{step_index} Adam parameter delta differs by {raw_error:.9g}")
    max_parameter_error = max(max_parameter_error, raw_error)
    expected_state = entry.get("update_state")
    _require(isinstance(expected_state, Mapping) and set(expected_state) == set(names),
             "OPD step omitted complete post-update parameter state")
    for name, parameter in zip(names, params):
        error = _tensor_error(parameter.detach(), expected_state[name])
        _require(error <= TOLERANCES["adam_parameter_abs"],
                 f"step{step_index} Adam state for {name} differs by {error:.9g}")
        max_parameter_error = max(max_parameter_error, error)
    state_error_box: list[float] = []
    _compare_tree(optimizer.state_dict(), entry.get("optimizer_state"),
                  tolerance=TOLERANCES["adam_state_abs"], errors=state_error_box,
                  name=f"step{step_index}.adam_state")
    max_state_error = max(max_state_error, max(state_error_box, default=0.0))
    return max_parameter_error, max_state_error


def _audit_direct_fit(fit: Mapping[str, Any], readout: Mapping[str, Any],
                      anchors: tuple[list[int], torch.Tensor],
                      expected_initial: Mapping[str, Any]) -> dict[str, Any]:
    path, trace = fit.get("path"), fit.get("exact_step_readback")
    _require(isinstance(path, list) and len(path) == 5
             and isinstance(trace, list) and len(trace) == len(path),
             "Direct fit lacks the complete five-step pre/post readback")
    _require(fit.get("declared_steps") == fit.get("selected_step") == 5,
             "Direct fit did not select the complete five-step path")
    positions, _ = anchors
    _require(fit.get("observation_positions") == positions,
             "Direct observation positions differ from the sealed sparse reference")
    _compare_tree(fit.get("initial_state"), expected_initial, name="direct.initial_state")
    _compare_tree(path[0].get("state"), fit.get("initial_state"),
                  name="direct.initial_state_vs_first_pre_state")
    for index, (entry, row) in enumerate(zip(path, trace)):
        _require(row.get("step") == index == entry.get("step"),
                 "Direct trace step order differs")
        _compare_tree(row.get("state_before"), entry.get("state"), name=f"direct{index}.state_before")
        _compare_tree(row.get("boxes_before"), entry.get("boxes"), name=f"direct{index}.boxes_before")
        expected_after_state = entry.get("update_state", entry.get("state"))
        _compare_tree(row.get("state_after"), expected_after_state, name=f"direct{index}.state_after")
        expected_optimizer_before = entry.get("optimizer_before", entry.get("optimizer_state"))
        _compare_tree(row.get("optimizer_state_before"), expected_optimizer_before,
                      name=f"direct{index}.optimizer_before")
        _compare_tree(row.get("optimizer_state_after"), entry.get("optimizer_state"),
                      name=f"direct{index}.optimizer_after")
        _require(row.get("no_update") is bool(entry.get("skipped", False)),
                 "Direct trace no-update flag differs from the fit path")
        if entry.get("skipped", False):
            _compare_tree(row.get("state_after"), row.get("state_before"),
                          name=f"direct{index}.skipped_state")
            _compare_tree(row.get("optimizer_state_after"), row.get("optimizer_state_before"),
                          name=f"direct{index}.skipped_optimizer")
        if index + 1 < len(path):
            _compare_tree(row.get("state_after"), trace[index + 1].get("state_before"),
                          name=f"direct{index}.state_to_next_pre_state")
            _compare_tree(row.get("optimizer_state_after"),
                          trace[index + 1].get("optimizer_state_before"),
                          name=f"direct{index}.optimizer_to_next_pre_optimizer")
            _compare_tree(row.get("boxes_after"), path[index + 1].get("boxes"),
                          name=f"direct{index}.boxes_after_next_pre")
    _compare_tree(trace[-1].get("boxes_after"), fit.get("final_cached_decoder_boxes"),
                  name="direct.final_boxes")
    _compare_tree(fit.get("final_state"),
                  path[-1].get("update_state", path[-1].get("state")),
                  name="direct.final_state_vs_last_step")
    _compare_tree(readout.get("boxes"), fit.get("final_cached_decoder_boxes"),
                  name="direct.full_model_final_readout")
    _require(int(fit.get("optimizer_steps", -1)) == sum(not row.get("skipped", False) for row in path),
             "Direct optimizer step count differs from its pre/post trace")
    return {"steps": len(path), "optimizer_steps": int(fit["optimizer_steps"]),
            "saved_full_model_readout_equals_cached_decoder": True,
            "model_replayed_by_reader": False}


def _audit_case(case: Mapping[str, Any], record: Mapping[str, Any],
                phase: str, reference_sha256: str) -> dict[str, Any]:
    cid = record["case_id"]
    expected_gt = phase == "oracle"
    _require(case.get("case_id") == cid and case.get("dataset") == record["dataset"]
             and case.get("source_case_sha256") == record.get("source_case_sha256"),
             "student case is not bound to its roster/source metadata")
    _require(case.get("GT_read") is expected_gt
             and case.get("source_restored") is True
             and case.get("reference_lock_sha256") == reference_sha256,
             "student case restoration, GT, or reference binding is invalid")
    if expected_gt:
        _require(case.get("GT_reference_offline_nondeployment") is True,
                 "oracle case is not marked offline nondeployment")
    else:
        _require(case.get("GT_reference_offline_nondeployment") is False,
                 "non-oracle case unexpectedly claims GT oracle supervision")
    frame_ids = case.get("frame_ids")
    _require(isinstance(frame_ids, list) and len(frame_ids) >= 2
             and all(isinstance(x, int) and not isinstance(x, bool) for x in frame_ids)
             and all(right > left for left, right in zip(frame_ids, frame_ids[1:])),
             "student case physical frame IDs are malformed")
    source_boxes = case.get("source_boxes")
    source_logits = case.get("source_logits")
    _require(torch.is_tensor(source_boxes) and source_boxes.shape == (len(frame_ids), 4)
             and torch.isfinite(source_boxes).all(),
             "case omitted the complete finite source decoder boxes")
    source_xyxy = _xyxy(source_boxes.detach().cpu().double())
    _require(((source_boxes > 0) & (source_boxes < 1)).all()
             and (source_boxes[:, 2:] > 0).all(),
             "source decoder boxes are outside the invertible normalized cxcywh chart")
    _require(isinstance(source_logits, (list, tuple)) and len(source_logits) == 2
             and all(torch.is_tensor(value) and torch.isfinite(value).all()
                     for value in source_logits),
             "case omitted the frozen two-offset source temporal logits")
    config_row = case.get("configuration", {})
    config = config_row.get("config")
    key = config_row.get("key")
    _require(config_row.get("scope") == "joint" and isinstance(config, dict)
             and config.get("steps") == 5 and config.get("samples") == 32
             and key == f"caption_query_fixed_space|{cid}",
             "case configuration does not preserve the locked five-step Gaussian key")
    from scripts import stvg_target_teacher_student_v1 as core
    original_config = core.load_original_opd_config(case["dataset"])
    original_config["steps"] = 5
    _require(config == original_config,
             "case configuration differs from the original per-domain OPD settings")
    fits, anchors_by_ref = case.get("fits"), case.get("supervision_anchors")
    readouts, arms = case.get("readouts"), case.get("arms")
    _require(isinstance(fits, dict) and isinstance(anchors_by_ref, dict)
             and isinstance(readouts, dict) and isinstance(arms, dict),
             "student case omitted its fit/readout/anchor maps")
    _compare_tree(torch.as_tensor(arms.get("F", {}).get("boxes"), dtype=torch.float64), source_xyxy,
                  name="source_arm_F_vs_native_source_boxes")
    ref_names = set(anchors_by_ref)
    roles = {_normal_role(name) for name in ref_names}
    _require(roles == EXPECTED_ROLES[phase], "case reference roles differ from the phase lock")
    locked_references = record.get("references")
    _require(isinstance(locked_references, Mapping),
             "reference lock case is missing its role-specific anchor records")
    locked_roles = {_normal_role(name) for name in locked_references}
    _require(locked_roles == EXPECTED_ROLES[phase] and set(locked_references) == ref_names,
             "case anchor roles do not match the sealed reference lock")
    for ref_name in ref_names:
        _compare_tree(anchors_by_ref[ref_name], locked_references[ref_name].get("anchors"),
                      name=f"{ref_name}.anchors_vs_reference_lock")
    expected_fit_keys = {f"{ref}_{method}" for ref in ref_names
                         for method in ("direct", "iou", "diou")}
    _require(set(fits) == expected_fit_keys and set(readouts) == expected_fit_keys,
             "case does not contain exactly Direct/IoU/DIoU fits and full readouts per reference")

    fit_checks = {}
    total_steps = 0
    for fit_name, fit in fits.items():
        ref_name, method = fit_name.rsplit("_", 1)
        anchors = _anchors_from_case(case, ref_name, frame_ids)
        if method == "direct":
            checked = _audit_direct_fit(fit, readouts[fit_name], anchors, case["initial_state"])
        else:
            checked = _audit_opd_fit(fit, anchors, frame_ids, key, method, case["initial_state"])
            _require(fit.get("reference_role") == _normal_role(ref_name)
                     and fit.get("reference_sha256") == reference_sha256,
                     "OPD fit is not bound to its role and sealed reference lock")
            _compare_tree(readouts[fit_name].get("boxes"), fit.get("final_cached_decoder_boxes"),
                          name=f"{fit_name}.full_model_final_readout")
            _require(fit.get("reward_objective") in {method, "diou_lambda_1" if method == "diou" else "iou"},
                     "fit objective label differs from the selected reward")
        _require(fit.get("config") == config and fit.get("scope") == "joint"
                 and fit.get("active_parameters") == 1792,
                 "fit deviates from the case's locked joint configuration")
        readout = readouts[fit_name]
        _compare_tree(readout.get("logits"), source_logits,
                      name=f"{fit_name}.full_model_temporal_logits_vs_source")
        _compare_tree(torch.as_tensor(arms.get(fit_name, {}).get("boxes"), dtype=torch.float64),
                      _xyxy(readout["boxes"].detach().cpu().double()),
                      name=f"{fit_name}.arm_boxes_vs_full_model_readout")
        _require(arms.get(fit_name, {}).get("interval") == readout.get("physical_interval"),
                 f"{fit_name} arm interval differs from its full-model readout")
        total_steps += int(fit.get("optimizer_steps", -1))
        fit_checks[fit_name] = checked
    _require(case.get("cost", {}).get("full_model_readouts") == len(readouts)
             and case.get("cost", {}).get("source_capture_calls") == 1,
             "case source capture/readout call counts differ")
    return {"dataset": case["dataset"], "fit_count": len(fits),
            "optimizer_steps": total_steps, "fits": fit_checks,
            "source_restoration_receipt": bool(case.get("source_restored")),
            "source_restoration_replayed_by_reader": False,
            "saved_full_model_readout_count": len(readouts),
            "saved_full_model_temporal_logits_preserved_count": len(readouts)}


def audit_phase(phase: str, *, write_report: bool = True) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU math reader must not initialize CUDA")
    gate = _preflight(phase)
    reference_sha = gate["hashes"]["reference_lock"]
    roster = gate["record_by_id"]
    counts = {"hc2": 0, "vidstg": 0}
    case_results = {}
    total_fits = 0
    total_optimizer_steps = 0
    for cid, path in gate["case_paths"].items():
        # All eight bytes were hash checked before any torch deserialization.
        case = torch.load(path, map_location="cpu", weights_only=False)
        old_fit_path = f"artifacts/stvg_temporal_caption_query_spatial_opd_v1/student/{cid}.pt"
        _require(gate["runtime"]["pins"].get(old_fit_path)
                 == case.get("temporal_exact_reuse_sha256"),
                 "case temporal baseline reuse SHA differs from its runtime input pin")
        result = _audit_case(case, roster[cid], phase, reference_sha)
        counts[result["dataset"]] += 1
        total_fits += result["fit_count"]
        total_optimizer_steps += result["optimizer_steps"]
        alias = ("HC" if result["dataset"] == "hc2" else "V") + f"{counts[result['dataset']]:02d}"
        case_results[alias] = result

    expected_fits = 8 * len(EXPECTED_ROLES[phase]) * 3
    _require(total_fits == expected_fits,
             "phase fit count differs from its Source/Teacher/Oracle method matrix")
    report = {
        "schema": "stvg-video-global-spatial-independent-cpu-math-v1",
        "status": "pass",
        "phase": phase,
        "cases": 8,
        "fit_count": total_fits,
        "method_matrix": {"reference_roles": sorted(EXPECTED_ROLES[phase]),
                           "algorithms_per_reference": ["direct", "iou", "diou"]},
        "optimizer_steps": total_optimizer_steps,
        "GT_read": gate["GT_read"],
        "GT_loader_called": False,
        "offline_nondeployment": gate["offline_nondeployment"],
        "model_loaded": False,
        "source_model_state_independently_reloaded": False,
        "GPU_initialized": False,
        "video_or_image_pixels_read": False,
        "tolerances": TOLERANCES,
        "case_readbacks": case_results,
        "provenance": {**gate["hashes"],
                       "oracle_root_readback_runtime_sha256": gate["oracle_root_readback_runtime_sha256"],
                       "reader_sha256": _sha_file(Path(__file__))},
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if write_report:
        output = BASE / phase / "ROOT_CPU_FIT_MATH_READBACK.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing readback: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    return report


def run_cpu_contracts() -> dict[str, Any]:
    """Exercise reader arithmetic and schema edges with synthetic CPU values only."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts require CUDA to remain uninitialized")

    from vg_tta.decota_spatial_opd_tunable_v1 import rollout
    from scripts.stvg_video_global_spatial_fit_v1 import (
        aligned_diou, fit_spatial_reward_opd,
    )
    from scripts import stvg_target_teacher_student_v1 as core

    frame_ids = [3, 8, 12]
    raw_anchors = [{"position": 1, "box_cxcywh": [.5, .4, .2, .3]}]
    positions, evidence = _anchors_from_case(
        {"supervision_anchors": {"source": raw_anchors}}, "source", frame_ids)
    assert positions == [1] and torch.equal(evidence, torch.tensor([[.5, .4, .2, .3]]))
    assert _normal_role("Source") == "source"
    assert _normal_role("offline_oracle") == "offline_oracle"
    assert _required_source_paths("method") >= {
        "scripts/run_stvg_video_global_student_v1.py",
        "scripts/stvg_video_global_spatial_fit_v1.py",
    }
    assert _required_source_paths("oracle") >= {
        "scripts/run_stvg_video_global_oracle_geometry001.py",
        "scripts/stvg_video_global_spatial_fit_fp32_geometry001.py",
    }
    assert 8 * len(EXPECTED_ROLES["method"]) * 3 == 48
    assert 8 * len(EXPECTED_ROLES["oracle"]) * 3 == 24

    key = "caption_query_fixed_space|synthetic_reader_contract"
    mean = torch.tensor([[0.1, -0.2, 0.3, -0.4],
                         [-0.3, 0.2, -0.1, 0.4]], dtype=torch.float32)
    target = torch.tensor([[.52, .46, .18, .24],
                           [.35, .63, .22, .16]], dtype=torch.float32)
    config = {"sigma": .025, "tau": .05, "samples": 32, "steps": 5,
              "lr": .01, "writeback": .0625}
    policy = rollout(mean, target, key, 2, shuffle=False, config=config)
    seed, noise = _cpu_noise(key, 2, len(mean))
    assert seed == policy["seed"] and torch.equal(
        mean[:, None, :] + config["sigma"] * noise, policy["samples"])

    action = policy["samples"].sigmoid()
    expected_iou = _reward(action, target[:, None, :], "iou")
    expected_diou = _reward(action, target[:, None, :], "diou")
    actual_iou = rollout(mean, target, key, 2, shuffle=False, config=config)["rewards"]
    actual_diou = aligned_diou(action, target[:, None, :])
    assert torch.allclose(expected_iou, actual_iou, atol=1e-7, rtol=0)
    assert torch.allclose(expected_diou, actual_diou, atol=1e-7, rtol=0)
    assert torch.isfinite(expected_iou).all() and torch.isfinite(expected_diou).all()

    # Direct readback must keep pre-update and post-update state/optimizer
    # separate even when every step is explicitly skipped.
    state = {"synthetic": torch.tensor([1.0])}
    boxes = torch.full((3, 4), .5, dtype=torch.float32)
    optimizer_state = {"state": {}, "param_groups": []}
    path, trace = [], []
    for index in range(5):
        path.append({"step": index, "state": state, "boxes": boxes,
                     "optimizer_state": optimizer_state, "skipped": True})
        trace.append({"step": index, "state_before": state, "state_after": state,
                      "boxes_before": boxes, "boxes_after": boxes,
                      "optimizer_state_before": optimizer_state,
                      "optimizer_state_after": optimizer_state, "no_update": True})
    _audit_direct_fit({"path": path, "exact_step_readback": trace,
                       "observation_positions": [1], "initial_state": state,
                       "final_state": state,
                       "final_cached_decoder_boxes": boxes, "optimizer_steps": 0,
                       "declared_steps": 5, "selected_step": 5},
                      {"boxes": boxes}, ([1], target[:1]), state)

    # Run the frozen student math on an in-memory synthetic base to exercise
    # the complete multi-anchor OPD audit, including five Adam transitions.
    synthetic_ids = [3, 8, 12, 20]
    synthetic_key = "caption_query_fixed_space|synthetic_fit_contract"
    fit_config = core.load_original_opd_config("hc2")
    fit_config["steps"] = 5
    fit_base = core._synthetic_student_base(len(synthetic_ids))
    synthetic_anchors = [
        {"position": 0, "physical_frame_id": synthetic_ids[0],
         "box": torch.tensor([.52, .48, .18, .22]), "weight": 1.0},
        {"position": 2, "physical_frame_id": synthetic_ids[2],
         "box": torch.tensor([.34, .62, .24, .16]), "weight": 1.0},
    ]
    reader_anchors = ([0, 2], torch.stack([row["box"] for row in synthetic_anchors]))
    iou_fit = fit_spatial_reward_opd(fit_base, fit_base.initial, synthetic_anchors,
        synthetic_ids, synthetic_key, fit_config, scope="joint", reward_kind="iou",
        reference_role="source", reference_sha256="c" * 64)
    _audit_opd_fit(iou_fit, reader_anchors, synthetic_ids, synthetic_key,
                   "iou", fit_base.initial)
    diou_fit = fit_spatial_reward_opd(fit_base, fit_base.initial, synthetic_anchors,
        synthetic_ids, synthetic_key, fit_config, scope="joint", reward_kind="diou",
        reference_role="source", reference_sha256="c" * 64)
    _audit_opd_fit(diou_fit, reader_anchors, synthetic_ids, synthetic_key,
                   "diou", fit_base.initial)

    # Zero-tolerance comparisons must not downcast away float64 differences.
    try:
        _compare_tree(torch.tensor([1.0], dtype=torch.float64),
                      torch.tensor([1.0 + 1e-12], dtype=torch.float64), name="precision")
    except RuntimeError:
        exact_precision = True
    else:
        raise AssertionError("zero-tolerance tensor comparison hid a float64 mismatch")

    return {"status": "pass", "CPU_only": True, "GT_read": False,
            "CUDA_initialized": False, "contracts": [
                "method/oracle reference-role aliases",
                "phase-specific pinned worker/helper and 48/24 fit-count contracts",
                "anchor rows bind through case frame IDs without requiring extra fields",
                "multi-anchor antithetic noise matches the frozen rollout",
                "independent IoU/DIoU reward arithmetic",
                "complete synthetic five-step IoU/DIoU optimizer readback",
                "Direct pre/post state and optimizer readback",
                "zero-tolerance comparisons preserve float64 differences"],
            "exact_precision_mismatch_rejected": exact_precision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", choices=sorted(PHASES))
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate barriers and file hashes without deserializing cases")
    parser.add_argument("--contracts", action="store_true",
                        help="run synthetic CPU-only reader contracts without reading artifacts")
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(run_cpu_contracts(), sort_keys=True))
        return
    if args.phase is None:
        parser.error("provide phase method/oracle or --contracts")
    if args.preflight_only:
        gate = _preflight(args.phase)
        print(json.dumps({"status": "preflight_pass", "phase": args.phase,
                          "cases": 8, "GT_read": gate["GT_read"],
                          "verified_case_bytes": len(gate["case_paths"])}, sort_keys=True))
        return
    report = audit_phase(args.phase)
    print(json.dumps({"status": report["status"], "phase": args.phase,
                      "cases": report["cases"], "fit_count": report["fit_count"],
                      "GT_read": report["GT_read"]}, sort_keys=True))


if __name__ == "__main__":
    main()
