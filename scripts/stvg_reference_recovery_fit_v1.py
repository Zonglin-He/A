"""Small, source-preserving spatial OPD fitter for the reference-recovery study.

This adapter keeps the existing Gaussian-IoU objective and the existing
1,792-parameter Query+LayerNorm replay.  Its only additions are explicit
support for the hash-locked original/current fit configurations and helpers
for the original O-split LN carry rule.  Importing it is inert.
"""
from __future__ import annotations

import copy
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERY = "spatial.query_residual"
LN_NUMEL = 1536
QUERY_NUMEL = 256
JOINT_NUMEL = QUERY_NUMEL + LN_NUMEL
ORIGINAL_CONFIG_PATH = "methods/decota_spatial_opd_v1/configs.json"
CURRENT_DESIGN_PATH = "artifacts/stvg_evidence_conditioned_v1/DESIGN_LOCK.json"
SOURCE_PATHS = (
    ORIGINAL_CONFIG_PATH,
    CURRENT_DESIGN_PATH,
    "scripts/stvg_target_teacher_student_v1.py",
    "scripts/stvg_weightfree_coordinate_fit_v1.py",
    "vg_tta/decota_spatial_opd_tunable_v1.py",
    "vg_tta/c1_enabling_tricks_v1.py",
    "vg_tta/spatial_online_state_v1.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """Return exact code/config custody hashes; caller runtime must pin them."""
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}


def _domain(dataset: str) -> str:
    aliases = {"hc2": "hc2", "hcstvg2": "hc2", "vidstg": "vidstg"}
    try:
        return aliases[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported target domain: {dataset!r}") from exc


def locked_config(dataset: str, profile: str) -> dict[str, Any]:
    """Load one of the two already locked config profiles, without inference."""
    import json
    from scripts import stvg_target_teacher_student_v1 as core

    domain = _domain(dataset)
    if profile == "original":
        return core.load_original_opd_config(domain)
    if profile == "current5":
        design_path = ROOT / CURRENT_DESIGN_PATH
        design = json.loads(design_path.read_text(encoding="utf-8"))
        if design.get("status") != "locked":
            raise ValueError("current five-step configuration is not locked")
        config = design.get("spatial_configs", {}).get(domain)
        if not isinstance(config, Mapping):
            raise ValueError("locked current design lacks a domain spatial config")
        return dict(config)
    raise ValueError("profile must be exactly 'original' or 'current5'")


def _validate_config(dataset: str, profile: str, config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("fit config must be a mapping")
    chosen = dict(config)
    expected = locked_config(dataset, profile)
    if chosen != expected:
        raise ValueError(f"config does not match locked {profile} profile for {_domain(dataset)}")
    if set(chosen) != {"lr", "sigma", "tau", "steps", "writeback", "samples"}:
        raise ValueError("fit config must keep the original six-field schema")
    if isinstance(chosen["steps"], bool) or not isinstance(chosen["steps"], int) or chosen["steps"] <= 0:
        raise ValueError("steps must be a positive integer from the selected lock")
    if chosen["samples"] != 32:
        raise ValueError("the original OPD policy uses exactly 32 antithetic samples")
    for key in ("lr", "sigma", "tau", "writeback"):
        value = chosen[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite numeric")
    if chosen["lr"] <= 0 or chosen["sigma"] <= 0 or chosen["tau"] <= 0:
        raise ValueError("lr, sigma, and tau must be positive")
    if not 0.0 <= float(chosen["writeback"]) <= 1.0:
        raise ValueError("writeback must lie in [0,1]")
    return chosen


def _copy_cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _copy_cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_copy_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_copy_cpu(child) for child in value)
    return copy.deepcopy(value)


def _same_schema(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    import torch
    if set(left) != set(right):
        raise ValueError("student state key schema changed")
    for name in left:
        if not torch.is_tensor(left[name]) or not torch.is_tensor(right[name]):
            raise ValueError("student state values must be tensors")
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
            raise ValueError(f"student state tensor ABI changed at {name}")


def _validate_state_roles(state: Mapping[str, Any]) -> None:
    import torch
    if QUERY not in state or state[QUERY].numel() != QUERY_NUMEL:
        raise ValueError("state must contain the 256-coordinate spatial Query residual")
    ln = {name: value for name, value in state.items() if name != QUERY}
    if sum(value.numel() for value in ln.values()) != LN_NUMEL:
        raise ValueError("all non-Query state must be exactly the 1,536 LayerNorm coordinates")
    if any(not name.startswith("spatial.layers.") for name in ln):
        raise ValueError("non-Query carry state must contain only spatial LayerNorm parameters")
    if not bool(torch.isfinite(state[QUERY]).all()) or any(not bool(torch.isfinite(v).all()) for v in ln.values()):
        raise ValueError("student state must be finite")


def initial_state(source_state: Mapping[str, Any], previous_committed: Mapping[str, Any] | None,
                  *, carry_ln: bool) -> dict[str, Any]:
    """Build an arrival state: Query always resets; only LN may carry."""
    import torch
    source = _copy_cpu(source_state)
    _validate_state_roles(source)
    if not isinstance(carry_ln, bool):
        raise ValueError("carry_ln must be an explicit Boolean")
    if carry_ln:
        if previous_committed is None:
            raise ValueError("LN carry requires a previous committed state")
        previous = _copy_cpu(previous_committed)
        _same_schema(source, previous)
        result = {name: value.clone() for name, value in source.items()}
        for name in result:
            if name != QUERY:
                result[name] = previous[name].clone().to(result[name])
    else:
        result = {name: value.clone() for name, value in source.items()}
    result[QUERY] = torch.zeros_like(source[QUERY])
    _validate_state_roles(result)
    return result


def commit_state(source_state: Mapping[str, Any], incoming_state: Mapping[str, Any],
                 fitted_final_state: Mapping[str, Any], writeback: float,
                 *, carry_ln: bool) -> dict[str, Any]:
    """Apply original O-split writeback, or discard fit state for episodic reset.

    For the carry arm, every non-Query coordinate is an LN parameter and uses
    ``incoming + alpha * (fit_final - incoming)``.  The Query residual is zero
    after every arrival/commit, exactly as in the existing O-split predictor.
    """
    import torch
    if isinstance(writeback, bool) or not isinstance(writeback, (float, int)) or not math.isfinite(float(writeback)):
        raise ValueError("writeback must be finite")
    source = _copy_cpu(source_state)
    incoming = _copy_cpu(incoming_state)
    final = _copy_cpu(fitted_final_state)
    _validate_state_roles(source)
    _validate_state_roles(incoming)
    _validate_state_roles(final)
    _same_schema(source, incoming)
    _same_schema(source, final)
    if not isinstance(carry_ln, bool):
        raise ValueError("carry_ln must be an explicit Boolean")
    if carry_ln:
        if not 0.0 < float(writeback) < 1.0:
            raise ValueError("LN carry uses the original interior writeback coefficient")
        result = {
            name: (torch.zeros_like(source[name]) if name == QUERY else
                   incoming[name] + (final[name] - incoming[name]) * float(writeback))
            for name in source
        }
    else:
        result = {name: value.clone() for name, value in source.items()}
        result[QUERY] = torch.zeros_like(source[QUERY])
    _validate_state_roles(result)
    return result


def fit_reference_recovery(base, initial, anchors: Sequence[Mapping[str, Any]],
                           frame_ids: Sequence[int], key: str, config: Mapping[str, Any], *,
                           profile: str, dataset: str, validate: bool,
                           reference_sha256: str,
                           checkpoint_callback=None) -> dict[str, Any]:
    """Run the same Gaussian-IoU/Adam fit with an explicit locked config.

    ``validate`` selects the existing label-free reference-IoU rollback gate.
    It must be held equal across paired E2 arms if the intent is to isolate the
    configuration package.  The returned trajectory distinguishes each
    before-state, proposed update, accepted/rolled-back after-state, and final
    LN writeback; no full model is loaded or scored here.
    """
    import torch
    from methods.decota_final_simplified_v1.tensors import detached
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts import stvg_native_instance_spatial_fit_v1 as native
    from scripts.stvg_weightfree_coordinate_fit_v1 import accept_reference_step, loss_per_anchor
    from vg_tta.decota_spatial_opd_tunable_v1 import mean_coordinates, optimizer_step, rollout
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

    if not isinstance(validate, bool):
        raise ValueError("validate must be an explicit Boolean")
    if not isinstance(key, str) or not key:
        raise ValueError("the locked source query key must be a nonempty string")
    if not isinstance(reference_sha256, str) or len(reference_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in reference_sha256):
        raise ValueError("reference_sha256 must be lowercase SHA-256")
    if not isinstance(frame_ids, (list, tuple)) or not frame_ids:
        raise ValueError("frame_ids must be the original nonempty ordered physical frame grid")
    ids = list(frame_ids)
    if any(isinstance(fid, bool) or not isinstance(fid, int) for fid in ids) or any(
            b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("physical frame IDs must be strictly increasing integers")
    cfg = _validate_config(dataset, profile, config)
    if not isinstance(anchors, (list, tuple)):
        raise ValueError("anchors must be a list/tuple; empty is explicit no-reference")
    native.verify_source_bindings()
    checked = native._validate_anchors(anchors, ids, torch)
    if len(checked) != len(anchors):
        raise ValueError("validated anchor count changed")

    started = time.perf_counter()
    base.restore(initial)
    replay, selected = core._trainable_replay(base, ids, key, "joint")
    origin = replay.state()
    _validate_state_roles(origin)
    parameters = [parameter for _, parameter in selected]
    if sum(parameter.numel() for parameter in parameters) != JOINT_NUMEL:
        raise RuntimeError("joint trainable state must remain Query256 + LayerNorm1536")
    if not torch.equal(origin[QUERY].detach().cpu(), torch.as_tensor(initial[QUERY]).detach().cpu()):
        raise RuntimeError("replay did not start from the explicit Query/LN arrival state")
    # Preserve the actual decoded boxes at arrival, before any optimizer step.
    # This remains available for empty-reference/no-update cases too.
    with torch.no_grad():
        initial_cached_boxes = replay.values()["boxes"].detach().cpu().clone()

    active = [anchor for anchor in checked if float(anchor["weight"]) > 0.0]
    positions = [int(anchor["position"]) for anchor in active]
    evidence = (torch.stack([anchor["box"].detach().to(base.zero["boxes"]) for anchor in active])
                if active else base.zero["boxes"].new_empty((0, 4)))
    weights = (base.zero["boxes"].new_tensor([anchor["weight"] for anchor in active]).detach()
               if active else base.zero["boxes"].new_empty((0,)))
    optimizer = torch.optim.Adam(parameters, lr=cfg["lr"], betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)

    path: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    proposals = accepted = 0
    stopped = False
    try:
        for step in range(cfg["steps"]):
            before_state = replay.state()
            current_boxes = replay.values()["boxes"]
            before_boxes = current_boxes.detach().clone()
            optimizer_before = core._copy_cpu(optimizer.state_dict())
            entry: dict[str, Any] = {
                "step": step,
                "boxes_before": detached(before_boxes, "cpu"),
                "state_before": detached(before_state, "cpu"),
                "optimizer_before": optimizer_before,
            }
            if not active or stopped:
                reason = "no_positive_weight_anchors" if not active else "stopped_after_rejected_step"
                after_state = replay.state()
                with torch.no_grad():
                    after_boxes = replay.values()["boxes"].detach().clone()
                entry.update(proposed=False, accepted=False, rollback=False, skipped=True,
                             reason=reason, boxes_proposed=detached(before_boxes, "cpu"),
                             state_proposed=detached(before_state, "cpu"),
                             optimizer_proposed=optimizer_before,
                             boxes_after=detached(after_boxes, "cpu"),
                             state_after=detached(after_state, "cpu"),
                             optimizer_after=core._copy_cpu(optimizer.state_dict()))
                path.append(entry)
                round_rows.append({"step": step, "proposed": False, "accepted": False,
                                   "rollback": False, "no_information": True, "reason": reason,
                                   "loss": None, "reference_mean_before": None,
                                   "reference_mean_proposed": None})
                if checkpoint_callback is not None:
                    checkpoint_callback({"next_step": step + 1, "path": path,
                                         "rounds": round_rows,
                                         "state": detached(after_state, "cpu"),
                                         "optimizer_state": core._copy_cpu(optimizer.state_dict()),
                                         "accepted_optimizer_steps": accepted,
                                         "proposed_optimizer_steps": proposals,
                                         "GT_read": False})
                continue

            # Keep the decoder graph on the optimization path.  ``before_boxes``
            # is the detached audit snapshot; ``current_boxes`` carries grads.
            means = mean_coordinates(current_boxes[positions])
            policy = rollout(means, evidence, key, step, config=cfg)
            per_anchor = loss_per_anchor(means, policy, cfg["sigma"])
            # This is the original mean-over-anchors objective when Top1
            # reliability is one for every anchor.
            loss = (weights * per_anchor).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite original per-anchor OPD loss")
            before_iou = aligned_iou(before_boxes[positions], evidence).detach()
            update = optimizer_step(optimizer, loss, parameters, bool(policy["informative"]))
            proposed_state = replay.state()
            with torch.no_grad():
                proposed_boxes = replay.values()["boxes"].detach().clone()
            optimizer_proposed = core._copy_cpu(optimizer.state_dict())
            proposed_iou = aligned_iou(proposed_boxes[positions], evidence).detach()
            is_proposal = update is not None
            approved = bool(is_proposal and (not validate or accept_reference_step(
                before_iou, proposed_iou, weights)))
            if is_proposal:
                proposals += 1
                if approved:
                    accepted += 1
                else:
                    replay.restore(before_state)
                    optimizer.load_state_dict(copy.deepcopy(optimizer_before))
                    stopped = True
                    if not core._tree_equal(core._copy_cpu(optimizer.state_dict()), optimizer_before):
                        raise AssertionError("rejected proposal did not restore the Adam state")
                    if not core._tree_equal(detached(replay.state(), "cpu"), detached(before_state, "cpu")):
                        raise AssertionError("rejected proposal did not restore the student state")

            after_state = replay.state()
            with torch.no_grad():
                after_boxes = replay.values()["boxes"].detach().clone()
            optimizer_after = core._copy_cpu(optimizer.state_dict())
            entry.update(
                skipped=False,
                loss=float(loss.detach()),
                per_anchor_loss=per_anchor.detach().cpu(),
                reliability=weights.detach().cpu(),
                rollout=detached(policy, "cpu"),
                update=None if update is None else detached(update, "cpu"),
                boxes_proposed=detached(proposed_boxes, "cpu"),
                state_proposed=detached(proposed_state, "cpu"),
                optimizer_proposed=optimizer_proposed,
                reference_iou_before=before_iou.detach().cpu(),
                reference_iou_proposed=proposed_iou.detach().cpu(),
                proposed=is_proposal,
                accepted=approved,
                rollback=bool(is_proposal and not approved),
                validation_enabled=validate,
                boxes_after=detached(after_boxes, "cpu"),
                state_after=detached(after_state, "cpu"),
                optimizer_after=optimizer_after,
            )
            path.append(entry)
            round_rows.append({
                "step": step,
                "proposed": is_proposal,
                "accepted": approved,
                "rollback": bool(is_proposal and not approved),
                "no_information": not bool(policy["informative"]),
                "loss": float(loss.detach()),
                "reference_mean_before": float((weights * before_iou).sum() / weights.sum()),
                "reference_mean_proposed": float((weights * proposed_iou).sum() / weights.sum()),
            })
            if checkpoint_callback is not None:
                checkpoint_callback({"next_step": step + 1, "path": path,
                                     "rounds": round_rows,
                                     "state": detached(after_state, "cpu"),
                                     "optimizer_state": optimizer_after,
                                     "accepted_optimizer_steps": accepted,
                                     "proposed_optimizer_steps": proposals,
                                     "GT_read": False})

        final_state = replay.state()
        with torch.no_grad():
            final_boxes = replay.values()["boxes"].detach().cpu()
        result = {
            "schema": "stvg-reference-recovery-fit-v1",
            "profile": profile,
            "dataset": _domain(dataset),
            "scope": "joint",
            "active_parameters": JOINT_NUMEL,
            "config": cfg,
            "validation_enabled": validate,
            "initial_state": detached(origin, "cpu"),
            "initial_cached_decoder_boxes": initial_cached_boxes,
            "fit_final_state": detached(final_state, "cpu"),
            "final_state": detached(final_state, "cpu"),
            "final_cached_decoder_boxes": final_boxes,
            "path": path,
            "rounds": round_rows,
            "optimizer_steps": accepted,
            "proposed_optimizer_steps": proposals,
            "rejected_optimizer_steps": proposals - accepted,
            "declared_steps": cfg["steps"],
            "seed_key": key,
            "observation_positions": positions,
            "reference_sha256": reference_sha256,
            "reference_role": "fixed_weight_free_top1_teacher",
            "optimizer_state_reset_each_query": True,
            "query_residual_reset_at_arrival": bool(torch.count_nonzero(origin[QUERY].detach().cpu()) == 0),
            "GT_read": False,
            "wall_seconds": time.perf_counter() - started,
        }
        return result
    finally:
        replay.restore(origin)
        base.restore(initial)


def transition_record(source_state: Mapping[str, Any], incoming: Mapping[str, Any],
                      fit_result: Mapping[str, Any], *, carry_ln: bool,
                      writeback: float) -> dict[str, Any]:
    """Emit explicit before/fit-after/writeback tensors for stream auditing."""
    import torch
    source = _copy_cpu(source_state)
    arrived = _copy_cpu(incoming)
    fitted = _copy_cpu(fit_result["fit_final_state"])
    _validate_state_roles(source)
    _validate_state_roles(arrived)
    _validate_state_roles(fitted)
    _same_schema(source, arrived)
    _same_schema(source, fitted)
    next_state = commit_state(source, arrived, fitted, writeback, carry_ln=carry_ln)
    ln_names = [name for name in source if name != QUERY]
    actual_delta = {
        name: (next_state[name] - arrived[name]).detach().cpu().clone()
        for name in ln_names
    }
    fitted_delta = {
        name: (fitted[name] - arrived[name]).detach().cpu().clone()
        for name in ln_names
    }
    if carry_ln:
        for name in ln_names:
            expected_next = arrived[name] + (fitted[name] - arrived[name]) * float(writeback)
            if not torch.equal(next_state[name], expected_next):
                raise AssertionError("recorded LN writeback differs from original alpha formula")
    else:
        if any(bool(torch.count_nonzero(delta)) for delta in actual_delta.values()):
            raise AssertionError("episodic mode must apply zero cross-query LN writeback")
    return {
        "schema": "stvg-reference-recovery-state-transition-v1",
        "carry_ln": bool(carry_ln),
        "writeback_alpha": float(writeback),
        "source_state_before_stream": source,
        "incoming_state_before_fit": arrived,
        "fit_final_state_before_writeback": fitted,
        "fitted_ln_delta": fitted_delta,
        "actual_ln_writeback_delta": actual_delta,
        "next_stream_state": next_state,
        "query_residual_next": next_state[QUERY].clone(),
        "actual_query_writeback_is_zero": bool(torch.count_nonzero(next_state[QUERY]) == 0),
        "GT_read": False,
    }


def contracts() -> dict[str, Any]:
    """Meaningful synthetic CPU exercise of config, fit, reset, and carry ABI."""
    import torch
    from scripts import stvg_target_teacher_student_v1 as core

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contracts must not initialize CUDA")
    original_hc = locked_config("hc2", "original")
    original_vid = locked_config("vidstg", "original")
    current_hc = locked_config("hc2", "current5")
    current_vid = locked_config("vidstg", "current5")
    assert original_hc["steps"] == 40 and original_vid["steps"] == 10
    assert current_hc["steps"] == current_vid["steps"] == 5
    assert original_hc["writeback"] == current_hc["writeback"] == .0625
    assert original_vid["writeback"] == current_vid["writeback"] == .125

    base = core._synthetic_student_base(8)
    source = _copy_cpu(base.initial)
    _validate_state_roles(source)
    q = source[QUERY]
    prior = {name: value.clone() for name, value in source.items()}
    for name in prior:
        if name != QUERY:
            prior[name] = prior[name] + .25
    prior[QUERY] = torch.ones_like(prior[QUERY])
    carry_initial = initial_state(source, prior, carry_ln=True)
    reset_initial = initial_state(source, prior, carry_ln=False)
    assert torch.count_nonzero(carry_initial[QUERY]) == 0
    assert torch.count_nonzero(reset_initial[QUERY]) == 0
    assert all(torch.equal(carry_initial[name], prior[name]) for name in source if name != QUERY)
    assert all(torch.equal(reset_initial[name], source[name]) for name in source if name != QUERY)

    fitted = {name: value.clone() for name, value in carry_initial.items()}
    for name in fitted:
        if name != QUERY:
            fitted[name] = fitted[name] + .1
    committed = commit_state(source, carry_initial, fitted, .0625, carry_ln=True)
    expected = {name: (torch.zeros_like(source[name]) if name == QUERY else
                       carry_initial[name] + .0625 * (fitted[name] - carry_initial[name]))
                for name in source}
    assert all(torch.equal(committed[name], expected[name]) for name in source)
    episodic = commit_state(source, carry_initial, fitted, .0625, carry_ln=False)
    assert all(torch.equal(episodic[name], source[name]) for name in source)

    # A complete tiny CPU fit uses the exact existing rollout/loss/Adam path.
    anchor_position = 2
    anchor_box = source[QUERY].new_tensor([.51, .47, .20, .18])
    # Synthetic source geometry is not necessarily this anchor; it is only a
    # valid normalized cxcywh input for exercising the optimizer interface.
    anchor = [{"position": anchor_position, "physical_frame_id": anchor_position,
               "box": anchor_box, "weight": 1.0}]
    cpu_replay, _ = core._trainable_replay(
        base, list(range(8)), "caption_query_fixed_space|synthetic", "joint")
    with torch.no_grad():
        expected_initial_boxes = cpu_replay.values()["boxes"].detach().cpu().clone()
    cpu_replay.restore(source)
    base.restore(source)
    current_fit = fit_reference_recovery(
        base, source, anchor, list(range(8)), "caption_query_fixed_space|synthetic",
        current_hc, profile="current5", dataset="hc2", validate=False,
        reference_sha256="a" * 64)
    assert current_fit["declared_steps"] == 5 and len(current_fit["path"]) == 5
    assert current_fit["active_parameters"] == JOINT_NUMEL
    assert current_fit["optimizer_state_reset_each_query"] is True
    assert torch.equal(base.initial[QUERY], q)
    transition = transition_record(source, carry_initial, current_fit,
                                   carry_ln=True, writeback=.0625)
    assert torch.equal(current_fit["initial_cached_decoder_boxes"],
                       expected_initial_boxes)
    assert torch.equal(current_fit["final_state"][QUERY], current_fit["fit_final_state"][QUERY])
    assert transition["actual_query_writeback_is_zero"]
    assert all(torch.allclose(
        transition["actual_ln_writeback_delta"][name],
        .0625 * (current_fit["fit_final_state"][name] - carry_initial[name]),
        rtol=0.0, atol=1e-7)
        for name in source if name != QUERY)

    empty_fit = fit_reference_recovery(
        base, source, [], list(range(8)), "caption_query_fixed_space|synthetic-empty",
        current_hc, profile="current5", dataset="hc2", validate=True,
        reference_sha256="b" * 64)
    assert empty_fit["optimizer_steps"] == empty_fit["proposed_optimizer_steps"] == 0
    assert empty_fit["initial_cached_decoder_boxes"].shape == empty_fit["final_cached_decoder_boxes"].shape
    assert torch.equal(empty_fit["initial_cached_decoder_boxes"],
                       empty_fit["final_cached_decoder_boxes"])
    assert torch.equal(empty_fit["fit_final_state"][QUERY], source[QUERY])
    assert all(torch.equal(empty_fit["fit_final_state"][name], source[name])
               for name in source if name != QUERY)
    assert not torch.cuda.is_initialized()
    return {
        "schema": "stvg-reference-recovery-fit-cpu-contracts-v1",
        "status": "pass",
        "checks": [
            "locked original HC40/Vid10 and current5 profiles",
            "Query reset with exactly 1536 LN carry coordinates",
            "original LN writeback equation and episodic no-writeback",
            "current5 synthetic fit through existing Gaussian rollout and Adam",
            "empty-reference exact no-update behavior",
        ],
        "fit_profile": "current5_hc2_synthetic",
        "optimizer_steps": current_fit["optimizer_steps"],
        "declared_steps": current_fit["declared_steps"],
        "active_parameters": current_fit["active_parameters"],
        "CPU_only": True,
        "GT_read": False,
        "model_loaded": False,
        "time": time.time(),
    }


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", action="store_true")
    args = parser.parse_args()
    if not args.contracts:
        parser.error("only the CPU synthetic --contracts command is provided")
    print(json.dumps(contracts(), ensure_ascii=False, sort_keys=True))
