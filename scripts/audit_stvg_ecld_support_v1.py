#!/usr/bin/env python3
"""CPU-only, GT-free audit of reward support in the sealed D prompt009 new_crop fits.

This reader deliberately opens only the development roster/locks/barriers, the
selected student PTs and their exact round2 parents, and the fixed temporal
selection lock. It does not read score/postseal/GT receipts, RGB, video, or
annotations. The saved tensor checkpoints are loaded with a NumPy-only
weights_only allowlist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OLD = ROOT / 'artifacts/stvg_evidence_constrained_v1'
D = OLD / 'round2_prompt009'
PARENT = OLD / 'round2'
C = ROOT / 'artifacts/stvg_weightfree_coordinate_v1'
DEFAULT_OUT = ROOT / 'artifacts/stvg_evidence_conditioned_v1/p0/SUPPORT_READBACK003.json'
DEFAULT_CONTRACTS_OUT = ROOT / 'artifacts/stvg_evidence_conditioned_v1/p0/CPU_CONTRACTS005.json'

CODE_PATHS = (
    'scripts/run_stvg_evidence_student009.py',
    'scripts/stvg_evidence_spatial_phase009.py',
    'scripts/stvg_weightfree_coordinate_fit_v1.py',
    'vg_tta/decota_spatial_opd_tunable_v1.py',
    'methods/decota_final_simplified_v1/objectives.py',
    'methods/decota_final_simplified_v1/tensors.py',
    'scripts/stvg_target_teacher_student_v1.py',
    'scripts/stvg_native_instance_spatial_fit_v1.py',
    'scripts/stvg_native_instance_common_v1.py',
    # These transitive imports were used by the historical worker but are not
    # present in RUNTIME006's pin map. Hash the current files and disclose the
    # historical pin gap; do not pretend the runtime bound them.
    'scripts/stvg_evidence_common_v1.py',
    'vg_tta/tastvg_decota_critic_p0_v1.py',
    'vg_tta/c1_enabling_tricks_v1.py',
    'vg_tta/decota_optimizer_posterior_r1_v1.py',
)

META_PATHS = (
    'methods/CURRENT_METHOD.json',
    'artifacts/stvg_evidence_constrained_v1/DEVELOPMENT_ROSTER_LOCK.json',
    'artifacts/stvg_evidence_constrained_v1/FROZEN_METHOD_LOCK.json',
    'artifacts/stvg_evidence_constrained_v1/TEMPORAL_SELECTION_LOCK.json',
    'artifacts/stvg_weightfree_coordinate_v1/SPATIAL_SELECTION_LOCK.json',
    'artifacts/stvg_evidence_constrained_v1/round2_prompt009/RUNTIME006.json',
    'artifacts/stvg_evidence_constrained_v1/round2_prompt009/VARIANT_LOCK.json',
    'artifacts/stvg_evidence_constrained_v1/round2_prompt009/REFERENCE_LOCK.json',
    'artifacts/stvg_evidence_constrained_v1/round2_prompt009/STUDENT_GLOBAL_BARRIER.json',
    'artifacts/stvg_evidence_constrained_v1/round2/REFERENCE_LOCK.json',
    'artifacts/stvg_evidence_constrained_v1/round2/STUDENT_GLOBAL_BARRIER.json',
)

REWARD_TOL = 3e-6
PROB_TOL = 2e-6
LOSS_TOL = 2e-4
IOU_TOL = 3e-6
STATE_TOL = 2e-7


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path.relative_to(ROOT)}')
    return value


def _rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def _safe_torch_load(path: Path):
    import numpy as np
    import torch

    # The PTs are locally-created, hash-verified project artifacts. Keep
    # weights_only=True and allow only the NumPy globals used by their arrays.
    allow = [np.core.multiarray._reconstruct, np.core.multiarray.scalar,
             np.ndarray, np.dtype]
    for dtype in (np.float16, np.float32, np.float64, np.int8, np.int16,
                  np.int32, np.int64, np.uint8, np.bool_):
        allow.append(type(np.dtype(dtype)))
    with torch.serialization.safe_globals(allow):
        return torch.load(Path(path), map_location='cpu', weights_only=True)


def _arr(x):
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _sigmoid(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def _cxcywh_to_xyxy(boxes):
    import numpy as np
    a = np.asarray(boxes, dtype=np.float64)
    out = np.empty_like(a)
    out[..., 0] = a[..., 0] - 0.5 * a[..., 2]
    out[..., 1] = a[..., 1] - 0.5 * a[..., 3]
    out[..., 2] = a[..., 0] + 0.5 * a[..., 2]
    out[..., 3] = a[..., 1] + 0.5 * a[..., 3]
    return out


def iou_xyxy_numpy(a, b):
    """Broadcasting aligned IoU, matching the original xyxy conversion/clamps."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lt = np.maximum(a[..., :2], b[..., :2])
    rb = np.minimum(a[..., 2:], b[..., 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    awh = np.maximum(a[..., 2:] - a[..., :2], 0.0)
    bwh = np.maximum(b[..., 2:] - b[..., :2], 0.0)
    area_a = awh[..., 0] * awh[..., 1]
    area_b = bwh[..., 0] * bwh[..., 1]
    union = np.maximum(area_a + area_b - inter, 1e-7)
    return inter / union


def iou_cxcywh_numpy(a, b):
    return iou_xyxy_numpy(_cxcywh_to_xyxy(a), _cxcywh_to_xyxy(b))


def _softmax(logits):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _tree_equal(a, b):
    import numpy as np
    try:
        import torch
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    except Exception:
        pass
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return (math.isnan(float(a)) and math.isnan(float(b))) or float(a) == float(b)
    return a == b


def _semantic_anchors(anchors):
    # The continuation worker only updates teacher receipt metadata on exact
    # reuse; geometry, weights and positions must be identical.
    return [{k: v for k, v in a.items() if k != 'teacher_output_sha256'} for a in anchors]


def _safe_pin_path(raw: str) -> Path | None:
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    try:
        p.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return None
    return p


def _runtime_pin_inventory(runtime):
    """Keep the entire pin map, but do not reopen visual payload pins."""
    entries = []
    reverified = 0
    visual_suffixes = {'.npy', '.npz', '.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mkv', '.avi'}
    always_rehash = set(CODE_PATHS) | set(META_PATHS)
    for raw, expected in sorted(runtime.get('pins', {}).items()):
        p = _safe_pin_path(raw)
        if p is None:
            entries.append(dict(path_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                                expected_sha256=expected, status='external_path_not_opened'))
            continue
        rel = _rel(p)
        if not p.exists():
            raise AssertionError(f'RUNTIME006 pin missing: {rel}')
        suffix = p.suffix.lower()
        should_hash = rel in always_rehash or (suffix in {'.py', '.md'} and p.is_file())
        if suffix in visual_suffixes or 'private_rgb' in rel or 'teacher_inputs' in rel:
            should_hash = False
        if should_hash:
            actual = _sha(p)
            if actual != expected:
                raise AssertionError(f'RUNTIME006 pin mismatch: {rel}')
            reverified += 1
            entries.append(dict(path=rel, expected_sha256=expected,
                                actual_sha256=actual, bytes=p.stat().st_size,
                                status='rehashed'))
        else:
            entries.append(dict(path=rel, expected_sha256=expected,
                                bytes=p.stat().st_size if p.is_file() else None,
                                status='metadata_only_not_opened'))
    protected_entries = []
    for raw, expected in sorted(runtime.get('protected', {}).items()):
        p = _safe_pin_path(raw)
        if p is None or not p.exists() or _sha(p) != expected:
            raise AssertionError(f'RUNTIME006 protected pin mismatch: {raw}')
        protected_entries.append(dict(path=_rel(p), expected_sha256=expected,
                                      actual_sha256=expected, bytes=p.stat().st_size,
                                      status='protected_rehashed'))
    entries.extend(protected_entries)
    return dict(expected_pin_count=len(runtime.get('pins', {})),
                expected_protected_count=len(runtime.get('protected', {})),
                expected_entry_count=len(runtime.get('pins', {})) + len(runtime.get('protected', {})),
                rehashed_count=reverified,
                protected_rehashed_count=len(protected_entries),
                metadata_only_count=sum(e['status'] == 'metadata_only_not_opened' for e in entries),
                external_not_opened_count=sum(e['status'] == 'external_path_not_opened' for e in entries),
                entries=entries)


def _source_bindings(runtime):
    bindings = []
    pins = runtime.get('pins', {})
    for rel in META_PATHS + CODE_PATHS:
        p = ROOT / rel
        if not p.exists():
            raise AssertionError(f'missing source input: {rel}')
        actual = _sha(p)
        expected = pins.get(rel)
        # Some seals predate a project-wide runtime pin for the lock file; keep
        # its direct byte hash and disclose that it is a scope-binding input.
        if expected is not None and expected != actual:
            raise AssertionError(f'input differs from RUNTIME006 pin: {rel}')
        bindings.append(dict(path=rel, sha256=actual, bytes=p.stat().st_size,
                             runtime_pin_sha256=expected,
                             pinned_in_runtime=expected is not None))
    return bindings


def _anchor_scalar_metrics(anchor, source_boxes_xyxy, fit_path, cfg, anchor_i):
    import numpy as np
    from scipy.special import logsumexp

    pos = int(anchor['position'])
    evidence = np.asarray(anchor['box_cxcywh'], dtype=np.float64)
    ref_xyxy = _cxcywh_to_xyxy(evidence)
    source_xyxy = np.asarray(source_boxes_xyxy[pos], dtype=np.float64)
    source_iou = float(iou_xyxy_numpy(source_xyxy, ref_xyxy))

    if fit_path.get('skipped'):
        return dict(anchor_ordinal=anchor_i, source_grid_position=pos,
                    physical_frame_id=int(anchor['physical_frame_id']),
                    reliability=float(anchor['weight']), source_reference_iou=source_iou,
                    novelty_one_minus_source_iou=float(1.0 - source_iou),
                    skipped=True, skip_reason=fit_path.get('reason'))

    rollout = fit_path['rollout']
    z = _arr(rollout['samples'])[anchor_i].astype(np.float64)
    rewards_saved = _arr(rollout['rewards'])[anchor_i].astype(np.float64)
    used_saved = _arr(rollout['used_rewards'])[anchor_i].astype(np.float64)
    q_saved = _arr(rollout['weights'])[anchor_i].astype(np.float64)
    samples = _sigmoid(z)
    rewards = iou_cxcywh_numpy(samples, evidence)
    q = _softmax(used_saved / float(cfg['tau']))
    variance = float(np.var(rewards, ddof=0))
    rmin, rmax = float(np.min(rewards)), float(np.max(rewards))
    positive = int(np.count_nonzero(rewards > 0.0))
    above = int(np.count_nonzero(rewards > 1e-6))
    ess = float(1.0 / np.sum(q * q))
    kl = float(np.sum(q * np.log(np.maximum(q, np.finfo(np.float64).tiny) * len(q))))

    # The stored action policy is the centered score-function estimate for a
    # forward KL to the current diagonal Gaussian. q, actions and rmean are
    # detached in the original implementation.
    mu = _arr(rollout['mean'])[anchor_i].astype(np.float64)
    rmean = _arr(rollout['mean'])[anchor_i].astype(np.float64)
    coord_grad = ((mu - rmean) / (float(cfg['sigma']) ** 2)
                  - np.sum((q - 1.0 / len(q))[:, None] * (z - mu[None, :]), axis=0)
                  / (float(cfg['sigma']) ** 2))
    coord_grad *= float(fit_path['reliability'][anchor_i]) / len(_arr(rollout['mean']))

    before_boxes = _arr(fit_path['boxes']).astype(np.float64)
    proposed_boxes = _arr(fit_path['proposed_boxes']).astype(np.float64)
    actual_after_boxes = _arr(fit_path['after_boxes']).astype(np.float64)
    iou_before = float(iou_cxcywh_numpy(before_boxes[pos], evidence))
    iou_proposed = float(iou_cxcywh_numpy(proposed_boxes[pos], evidence))
    iou_after = float(iou_cxcywh_numpy(actual_after_boxes[pos], evidence))
    return dict(anchor_ordinal=anchor_i, source_grid_position=pos,
                physical_frame_id=int(anchor['physical_frame_id']),
                reliability=float(anchor['weight']), source_reference_iou=source_iou,
                novelty_one_minus_source_iou=float(1.0 - source_iou),
                reward_variance_population=variance, reward_range=rmax-rmin,
                reward_min=rmin, reward_max=rmax,
                positive_reward_samples=positive, reward_gt_1e6_samples=above,
                reward_var_gt_0=bool(variance > 0.0),
                reward_var_gt_1e6=bool(variance > 1e-6),
                effective_sample_size=ess, kl_q_to_uniform=kl,
                anchor_reward_informative=bool(np.any(used_saved != used_saved[:1])),
                coordinate_gradient_l2=float(np.linalg.norm(coord_grad)),
                reference_iou_before=iou_before, reference_iou_proposed=iou_proposed,
                reference_iou_actual_after=iou_after,
                stored_reward_max_abs_error=float(np.max(np.abs(rewards - rewards_saved))),
                stored_used_reward_max_abs_error=float(np.max(np.abs(used_saved - rewards_saved))),
                stored_q_max_abs_error=float(np.max(np.abs(q - q_saved))),
                sample_count=int(len(rewards)))


def _audit_fit(fit, anchors, source_boxes_xyxy, expected_config, provenance, case_id):
    import numpy as np

    assert fit['GT_read'] is False and fit['scope'] == 'joint'
    assert fit['reward'] == 'original_iou' and fit['reference_role'] == 'teacher'
    assert fit['active_parameters'] == 1792
    assert fit['config'] == expected_config
    assert fit['declared_steps'] == expected_config['steps'] == 5
    assert fit['reference_sha256'] == provenance['reference_sha256']
    assert len(fit['path']) == expected_config['steps']
    positions = [int(a['position']) for a in anchors]
    assert fit['observation_positions'] == positions
    assert len(fit['reliability']) == len(anchors)
    max_reward_error = 0.0
    max_q_error = 0.0
    max_loss_error = 0.0
    max_iou_error = 0.0
    max_raw_delta_error = 0.0
    anchor_steps = []
    step_records = []
    proposed = accepted = rejected = no_information = skipped = 0
    for step, row in enumerate(fit['path']):
        assert row['step'] == step
        if row.get('skipped'):
            skipped += 1
            assert row['reason'] == 'stopped_after_rejected_step'
            step_records.append(dict(step=step, skipped=True, reason=row['reason']))
            continue
        rollout = row['rollout']
        samples = _arr(rollout['samples']).astype(np.float64)
        saved_rewards = _arr(rollout['rewards']).astype(np.float64)
        saved_used = _arr(rollout['used_rewards']).astype(np.float64)
        saved_q = _arr(rollout['weights']).astype(np.float64)
        means = _arr(rollout['mean']).astype(np.float64)
        assert samples.shape == (len(anchors), 32, 4)
        assert saved_rewards.shape == saved_used.shape == saved_q.shape == (len(anchors), 32)
        assert rollout.get('permutations') is None
        assert np.array_equal(saved_rewards, saved_used)
        assert np.isfinite(samples).all() and np.isfinite(saved_rewards).all()
        # Antithetic Gaussian actions: each saved pair sums to 2*the saved mean.
        antithetic_error = float(np.max(np.abs(samples[:, :16] + samples[:, 16:] - 2 * means[:, None, :])))
        assert antithetic_error < 4e-6, (case_id, step, antithetic_error)
        independent = np.empty_like(saved_rewards)
        for ai, anchor in enumerate(anchors):
            actions = _sigmoid(samples[ai])
            independent[ai] = iou_cxcywh_numpy(actions, np.asarray(anchor['box_cxcywh'], dtype=np.float64))
        reward_error = float(np.max(np.abs(independent - saved_rewards)))
        max_reward_error = max(max_reward_error, reward_error)
        assert reward_error <= REWARD_TOL, (case_id, step, reward_error)
        q_expected = _softmax(saved_used / float(expected_config['tau']))
        q_error = float(np.max(np.abs(q_expected - saved_q)))
        max_q_error = max(max_q_error, q_error)
        assert q_error <= PROB_TOL, (case_id, step, q_error)
        informative = bool(np.any(saved_used != saved_used[:, :1]))
        assert bool(rollout['informative']) == informative
        if not informative:
            no_information += 1

        # Independently rebuild the forward-KL plus centered score-function
        # loss in float64, including the outer mean over K anchors.
        kl = np.sum((means - means) ** 2, axis=-1) / (2 * expected_config['sigma'] ** 2)
        logp = (-np.sum((samples - means[:, None, :]) ** 2, axis=-1)
                / (2 * expected_config['sigma'] ** 2)
                - 2 * math.log(2 * math.pi * expected_config['sigma'] ** 2))
        per_anchor = kl - np.sum((saved_q - 1.0 / 32.0) * logp, axis=-1)
        reliability = _arr(row['reliability']).astype(np.float64)
        anchor_loss_error = float(np.max(np.abs(per_anchor - _arr(row['anchor_loss']).astype(np.float64))))
        loss_recalc = float(np.mean(reliability * per_anchor))
        loss_error = abs(loss_recalc - float(row['loss']))
        max_loss_error = max(max_loss_error, anchor_loss_error, loss_error)
        assert anchor_loss_error <= LOSS_TOL and loss_error <= LOSS_TOL, (case_id, step, anchor_loss_error, loss_error)

        positions = [int(a['position']) for a in anchors]
        evidence = np.asarray([a['box_cxcywh'] for a in anchors], dtype=np.float64)
        for arr_key, saved_key in [('boxes', 'reference_iou_before'),
                                   ('proposed_boxes', 'reference_iou_proposed')]:
            box_rows = _arr(row[arr_key]).astype(np.float64)[positions]
            calc = iou_cxcywh_numpy(box_rows, evidence)
            stored = _arr(row[saved_key]).astype(np.float64)
            err = float(np.max(np.abs(calc - stored)))
            max_iou_error = max(max_iou_error, err)
            assert err <= IOU_TOL, (case_id, step, arr_key, err)
        after_calc = iou_cxcywh_numpy(_arr(row['after_boxes']).astype(np.float64)[positions], evidence)
        proposed_flag = bool(row['proposed'])
        accepted_flag = bool(row['accepted'])
        if proposed_flag:
            proposed += 1
            if accepted_flag:
                accepted += 1
                weighted_gain = float(np.sum(reliability * (after_calc - _arr(row['reference_iou_before']).astype(np.float64)))
                                      / np.sum(reliability))
                assert weighted_gain >= -1e-7
            else:
                rejected += 1
                assert _tree_equal(row['state'], row['update_state'])
                assert _tree_equal(row['optimizer_before'], row['optimizer_after'])
        else:
            assert not accepted_flag
            assert _tree_equal(row['state'], row['update_state'])
            assert _tree_equal(row['optimizer_before'], row['optimizer_after'])
        state_keys = list(row['state'])
        assert sum(_arr(row['state'][k]).size for k in state_keys) == 1792
        if proposed_flag:
            grad = _arr(row['update']['gradient']).astype(np.float64).reshape(-1)
            raw = _arr(row['update']['raw']).astype(np.float64).reshape(-1)
            assert grad.size == raw.size == 1792 and np.isfinite(grad).all() and np.isfinite(raw).all()
            state_delta_proposed = np.concatenate([
                (_arr(row['proposed_state'][k]).astype(np.float64) - _arr(row['state'][k]).astype(np.float64)).reshape(-1)
                for k in state_keys])
            raw_err = float(np.max(np.abs(state_delta_proposed - raw)))
            max_raw_delta_error = max(max_raw_delta_error, raw_err)
            assert raw_err <= STATE_TOL, (case_id, step, raw_err)
            delta_actual = np.concatenate([
                (_arr(row['update_state'][k]).astype(np.float64) - _arr(row['state'][k]).astype(np.float64)).reshape(-1)
                for k in state_keys])
            if accepted_flag:
                assert float(np.max(np.abs(delta_actual - raw))) <= STATE_TOL
                assert _tree_equal(row['optimizer_after'], row['optimizer_proposed'])
            else:
                assert np.count_nonzero(delta_actual) == 0
            nonzero_raw = int(np.count_nonzero(raw))
            nonzero_actual = int(np.count_nonzero(delta_actual))
        else:
            grad = raw = np.zeros(1792, dtype=np.float64)
            nonzero_raw = nonzero_actual = 0

        per_anchor_rows = []
        for ai, anchor in enumerate(anchors):
            am = _anchor_scalar_metrics(anchor, source_boxes_xyxy, row, expected_config, ai)
            am['step'] = step
            am['reward_variance_population'] = float(np.var(independent[ai], ddof=0))
            am['reward_range'] = float(np.max(independent[ai]) - np.min(independent[ai]))
            am['reward_min'] = float(np.min(independent[ai]))
            am['reward_max'] = float(np.max(independent[ai]))
            am['positive_reward_samples'] = int(np.count_nonzero(independent[ai] > 0))
            am['reward_gt_1e6_samples'] = int(np.count_nonzero(independent[ai] > 1e-6))
            am['reward_var_gt_0'] = bool(am['reward_variance_population'] > 0.0)
            am['reward_var_gt_1e6'] = bool(am['reward_variance_population'] > 1e-6)
            # Bounds loss-side reliability from the exact fit row, not calibrated confidence.
            am['reliability_in_fit'] = float(reliability[ai])
            per_anchor_rows.append(am)
            anchor_steps.append(am)
        q_ess = [float(1.0 / np.sum(q_expected[i] ** 2)) for i in range(len(anchors))]
        q_kl = [float(np.sum(q_expected[i] * np.log(np.maximum(q_expected[i], np.finfo(float).tiny) * 32)))
                for i in range(len(anchors))]
        step_records.append(dict(
            step=step, skipped=False, loss=float(row['loss']),
            proposal=proposed_flag, accepted=accepted_flag,
            informative=informative,
            nonzero_raw_parameter_deltas=nonzero_raw,
            nonzero_accepted_parameter_deltas=nonzero_actual,
            gradient_l2=float(np.linalg.norm(grad)),
            gradient_nonzero_count=int(np.count_nonzero(grad)),
            raw_update_l2=float(np.linalg.norm(raw)),
            max_antithetic_pair_error=antithetic_error,
            max_iou_reward_error=reward_error,
            max_q_probability_error=q_error,
            max_loss_error=max(anchor_loss_error, loss_error),
            per_anchor=per_anchor_rows))

    assert proposed == fit['proposed_optimizer_steps']
    assert accepted == fit['optimizer_steps']
    assert rejected == fit['rejected_optimizer_steps']
    assert proposed == accepted + rejected
    return dict(steps=step_records, proposed_steps=proposed,
                accepted_steps=accepted, rejected_steps=rejected,
                no_information_rounds=no_information, skipped_rounds=skipped,
                max_reward_recalc_error=max_reward_error,
                max_q_recalc_error=max_q_error,
                max_loss_recalc_error=max_loss_error,
                max_reference_iou_recalc_error=max_iou_error,
                max_state_delta_recalc_error=max_raw_delta_error)


def audit(output_path: Path = DEFAULT_OUT):
    import numpy as np
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was already initialized; this reader refuses to continue')
    roster_path = OLD / 'DEVELOPMENT_ROSTER_LOCK.json'
    temporal_lock_path = OLD / 'TEMPORAL_SELECTION_LOCK.json'
    frozen_path = OLD / 'FROZEN_METHOD_LOCK.json'
    spatial_lock_path = C / 'SPATIAL_SELECTION_LOCK.json'
    runtime_path = D / 'RUNTIME006.json'
    variant_path = D / 'VARIANT_LOCK.json'
    ref_path = D / 'REFERENCE_LOCK.json'
    student_barrier_path = D / 'STUDENT_GLOBAL_BARRIER.json'
    parent_barrier_path = PARENT / 'STUDENT_GLOBAL_BARRIER.json'

    runtime = _read_json(runtime_path)
    assert runtime['status'] == 'locked' and runtime['GT_read'] is False
    assert runtime.get('confirmation_GT_read') is False
    temporal_lock = _read_json(temporal_lock_path)
    frozen = _read_json(frozen_path)
    variant = _read_json(variant_path)
    assert temporal_lock['status'] == frozen['status'] == variant['status'] == 'locked'
    assert temporal_lock['variant'] == frozen['temporal_variant'] == 'offset_trust'
    assert temporal_lock['GT_read'] is False and variant['GT_read'] is False
    assert frozen['confirmation_used_for_selection'] is False
    assert variant['temporal_selection_sha256'] == _sha(temporal_lock_path)
    assert variant['variants'] == ['old_crop', 'new_crop', 'old_scene', 'new_scene']
    spatial_lock = _read_json(spatial_lock_path)
    assert spatial_lock['status'] == 'locked' and spatial_lock['GT_read'] is False
    assert spatial_lock['confirmation_used_for_selection'] is False

    roster = _read_json(roster_path)
    ref_lock = _read_json(ref_path)
    student_barrier = _read_json(student_barrier_path)
    parent_barrier = _read_json(parent_barrier_path)
    assert roster['status'] == ref_lock['status'] == 'locked'
    assert roster['GT_read'] is False and ref_lock['GT_read'] is False
    assert len(roster['records']) == ref_lock.get('count', len(ref_lock['records'])) == 32
    assert student_barrier['status'] == 'complete' and student_barrier['GT_read'] is False
    assert parent_barrier['status'] == 'complete' and parent_barrier['GT_read'] is False
    assert student_barrier['count'] == parent_barrier['count'] == 32
    assert len(student_barrier['files']) == len(parent_barrier['files']) == 32

    roster_by_id = {r.get('probe_alias', r.get('case_id')): r for r in roster['records']}
    refs = {r['case_id']: r for r in ref_lock['records']}
    parent_refs = {r['case_id']: r for r in _read_json(PARENT / 'REFERENCE_LOCK.json')['records']}
    student_items = {Path(f['path']).stem: f for f in student_barrier['files']}
    parent_items = {Path(f['path']).stem: f for f in parent_barrier['files']}
    expected_ids = set(roster_by_id)
    assert expected_ids == set(refs) == set(student_items) == set(parent_items) == set(parent_refs)
    assert len(expected_ids) == 32

    runtime_pin_readback = _runtime_pin_inventory(runtime)
    source_bindings = _source_bindings(runtime)
    selected_config = spatial_lock['configs']
    cases = []
    counts = {
        'cases': 0, 'cases_with_new_crop_anchors': 0,
        'cases_without_new_crop_anchors': 0,
        'new_crop_anchor_count': 0,
        'actual_new_crop_fits': 0, 'exact_parent_reused_fits': 0,
        'no_reference_no_update': 0, 'parent_reused_no_fit_no_anchor': 0,
        'fit_rounds_audited': 0, 'proposed_optimizer_steps': 0,
        'accepted_optimizer_steps': 0, 'rejected_optimizer_steps': 0,
        'no_information_rounds': 0, 'skipped_rounds': 0,
        'anchor_round_rows': 0,
        'anchor_round_reward_variance_gt_1e6': 0,
        'anchor_round_reward_variance_gt_0': 0,
        'anchor_round_any_positive_reward': 0,
        'anchor_round_any_reward_gt_1e6': 0,
    }
    max_errors = dict(reward=0.0, q=0.0, loss=0.0, reference_iou=0.0,
                      state_delta=0.0, parent_copy=0.0)
    selected_cfg_by_domain = selected_config

    # Hash-check all relevant data files before loading any PT.
    for item in student_barrier['files'] + parent_barrier['files']:
        p = ROOT / item['path']
        assert p.is_file() and p.stat().st_size == item['bytes']
        assert _sha(p) == item['sha256']

    for case_id in sorted(expected_ids):
        roster_row = roster_by_id[case_id]
        curr_item, parent_item = student_items[case_id], parent_items[case_id]
        curr_path, parent_path = ROOT / curr_item['path'], ROOT / parent_item['path']
        d = _safe_torch_load(curr_path)
        p = _safe_torch_load(parent_path)
        assert d['case_id'] == p['case_id'] == case_id
        assert d['dataset'] == p['dataset'] == roster_row['dataset']
        assert d['GT_read'] is False and p['GT_read'] is False and d['source_restored'] is True
        assert d['source_case_sha256'] == p['source_case_sha256']
        assert d['source_case_sha256'] == refs[case_id]['source_case_sha256']
        assert d['frame_ids'] == p['frame_ids']
        assert d['pixel_sha256'] == p['pixel_sha256']
        anchors = refs[case_id]['new_crop']['anchors']
        assert d['reference_records']['new_crop']['anchors'] == anchors
        assert _semantic_anchors(anchors) == _semantic_anchors(parent_refs[case_id]['new_crop']['anchors']) or not d['receipts']['new_crop']['status'].startswith('exact_parent')
        rec = d['receipts']['new_crop']
        status = rec['status']
        fit = d['fits'].get('new_crop')
        src_boxes = _arr(d['arms']['F']['boxes']).astype(np.float64)
        native_boxes = _arr(d['arms']['T']['boxes']).astype(np.float64)
        assert src_boxes.shape == native_boxes.shape and np.array_equal(src_boxes, native_boxes)
        assert src_boxes.ndim == 2 and src_boxes.shape[1] == 4
        frames = list(map(int, d['frame_ids']))
        assert src_boxes.shape[0] == len(frames)
        positions = [int(a['position']) for a in anchors]
        for a in anchors:
            pos = int(a['position'])
            assert 0 <= pos < len(frames) and int(a['physical_frame_id']) == frames[pos]
            box = np.asarray(a['box_cxcywh'], dtype=np.float64)
            assert box.shape == (4,) and np.isfinite(box).all()
            assert np.all(box[:2] >= 0) and np.all(box[:2] <= 1)
            assert np.all(box[2:] > 0) and np.all(box[2:] <= 1)
            assert 0 < float(a['weight']) <= 1
        config = selected_cfg_by_domain['hc2' if d['dataset'] == 'hc2' else 'vidstg']
        case_row = dict(case_id=case_id, dataset=d['dataset'],
                        source_case_sha256=d['source_case_sha256'],
                        student_path=curr_item['path'], student_sha256=curr_item['sha256'],
                        student_bytes=curr_item['bytes'], parent_path=parent_item['path'],
                        parent_sha256=parent_item['sha256'], receipt_status=status,
                        anchor_count=len(anchors), reference_sha256=_sha(ref_path),
                        source_capture_seconds=d.get('source_capture_seconds'),
                        unknown=not bool(anchors), fit_status='unknown_no_reference' if not anchors else status,
                        anchors=[])
        counts['cases'] += 1
        counts['new_crop_anchor_count'] += len(anchors)
        if anchors:
            counts['cases_with_new_crop_anchors'] += 1
        else:
            counts['cases_without_new_crop_anchors'] += 1

        if status == 'actual_fit_full_redecode':
            assert anchors and fit is not None and fit['reference_sha256'] == _sha(ref_path)
            assert rec['actual_fit_count'] == 1 and rec['accepted_steps'] == fit['optimizer_steps']
            assert rec['config'] == config
            provenance = dict(reference_sha256=_sha(ref_path))
            fit_result = _audit_fit(fit, anchors, src_boxes, config, provenance, case_id)
            counts['actual_new_crop_fits'] += 1
        elif status == 'exact_parent_source_reference_geometry_weights_config_seed_reuse':
            assert rec['actual_fit_count'] == 0
            assert rec['parent_file_sha256'] == parent_item['sha256']
            assert _sha(parent_path) == rec['parent_file_sha256']
            parent_fit = p['fits'].get('new_crop')
            if fit is None:
                assert parent_fit is None and not anchors
                assert _tree_equal(d['arms']['S_new_crop'], d['arms']['F'])
                assert _tree_equal(d['arms']['TS_new_crop'], d['arms']['T'])
                counts['parent_reused_no_fit_no_anchor'] += 1
                fit_result = None
                case_row['fit_status'] = 'inherited_no_fit_no_anchor'
            else:
                assert anchors and parent_fit is not None
                assert _semantic_anchors(anchors) == _semantic_anchors(parent_refs[case_id]['new_crop']['anchors'])
                assert _tree_equal(fit, parent_fit)
                assert _tree_equal(d['readouts']['new_crop'], p['readouts']['new_crop'])
                assert _tree_equal(d['arms']['S_new_crop'], p['arms']['S_new_crop'])
                assert _tree_equal(d['arms']['TS_new_crop'], p['arms']['TS_new_crop'])
                parent_ref_sha = _sha(PARENT / 'REFERENCE_LOCK.json')
                provenance = dict(reference_sha256=parent_ref_sha)
                fit_result = _audit_fit(fit, anchors, src_boxes, config, provenance, case_id)
                counts['exact_parent_reused_fits'] += 1
                case_row['fit_status'] = 'exact_parent_reused_fit'
                case_row['fit_reference_sha256'] = fit['reference_sha256']
                assert fit['reference_sha256'] == parent_ref_sha
        elif status == 'no_reference_no_update':
            assert not anchors and fit is None and rec['actual_fit_count'] == 0
            assert _tree_equal(d['arms']['S_new_crop'], d['arms']['F'])
            assert _tree_equal(d['arms']['TS_new_crop'], d['arms']['T'])
            counts['no_reference_no_update'] += 1
            fit_result = None
            case_row['fit_status'] = 'explicit_no_reference_no_update'
        else:
            raise AssertionError(f'unknown new_crop receipt status for {case_id}: {status}')

        if anchors and fit is not None:
            assert len(fit['path']) == 5
            fit_row = fit_result
            for step in fit_row['steps']:
                if not step['skipped']:
                    counts['fit_rounds_audited'] += 1
                    counts['proposed_optimizer_steps'] += int(step['proposal'])
                    counts['accepted_optimizer_steps'] += int(step['accepted'])
                    counts['rejected_optimizer_steps'] += int(step['proposal'] and not step['accepted'])
                    counts['no_information_rounds'] += int(not step['informative'])
                    counts['anchor_round_rows'] += len(step['per_anchor'])
                    for am in step['per_anchor']:
                        counts['anchor_round_reward_variance_gt_1e6'] += int(am['reward_var_gt_1e6'])
                        counts['anchor_round_reward_variance_gt_0'] += int(am['reward_var_gt_0'])
                        counts['anchor_round_any_positive_reward'] += int(am['positive_reward_samples'] > 0)
                        counts['anchor_round_any_reward_gt_1e6'] += int(am['reward_gt_1e6_samples'] > 0)
            counts['skipped_rounds'] += fit_row['skipped_rounds']
            max_errors['reward'] = max(max_errors['reward'], fit_row['max_reward_recalc_error'])
            max_errors['q'] = max(max_errors['q'], fit_row['max_q_recalc_error'])
            max_errors['loss'] = max(max_errors['loss'], fit_row['max_loss_recalc_error'])
            max_errors['reference_iou'] = max(max_errors['reference_iou'], fit_row['max_reference_iou_recalc_error'])
            max_errors['state_delta'] = max(max_errors['state_delta'], fit_row['max_state_delta_recalc_error'])
            case_row['fit_diagnostics'] = fit_row
            # A per-anchor fixed-reference support summary across optimizer rounds.
            for ai, anchor in enumerate(anchors):
                per = [s['per_anchor'][ai] for s in fit_row['steps'] if not s['skipped']]
                case_row['anchors'].append(dict(
                    anchor_ordinal=ai,
                    source_grid_position=int(anchor['position']),
                    physical_frame_id=int(anchor['physical_frame_id']),
                    reference_weight=float(anchor['weight']),
                    source_reference_iou=float(per[0]['source_reference_iou']),
                    novelty_one_minus_source_iou=float(per[0]['novelty_one_minus_source_iou']),
                    rounds=[dict(step=s['step'], reward_variance_population=s['reward_variance_population'],
                                 reward_range=s['reward_range'], positive_reward_samples=s['positive_reward_samples'],
                                 reward_gt_1e6_samples=s['reward_gt_1e6_samples'],
                                 reward_var_gt_0=s['reward_var_gt_0'], reward_var_gt_1e6=s['reward_var_gt_1e6'],
                                 effective_sample_size=s['effective_sample_size'],
                                 kl_q_to_uniform=s['kl_q_to_uniform'],
                                 informative=s['anchor_reward_informative'],
                                 coordinate_gradient_l2=s['coordinate_gradient_l2'],
                                 reference_iou_before=s['reference_iou_before'],
                                 reference_iou_proposed=s['reference_iou_proposed'],
                                 reference_iou_actual_after=s['reference_iou_actual_after']) for s in per]))
        cases.append(case_row)

    assert counts['cases'] == 32
    assert counts['cases_with_new_crop_anchors'] + counts['cases_without_new_crop_anchors'] == 32
    assert counts['actual_new_crop_fits'] == 7
    assert counts['exact_parent_reused_fits'] == 16
    assert counts['no_reference_no_update'] == 2
    assert counts['parent_reused_no_fit_no_anchor'] == 7
    assert counts['fit_rounds_audited'] == counts['proposed_optimizer_steps'] + counts['no_information_rounds']
    assert counts['accepted_optimizer_steps'] + counts['rejected_optimizer_steps'] == counts['proposed_optimizer_steps']
    assert not torch.cuda.is_initialized()

    out = {
        'schema': 'stvg-ecld-p0-support-readback-v1',
        'status': 'pass',
        'created_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'scope': 'independent CPU replay of saved new_crop action rewards and fixed-reference optimizer records; no GT/postseal/media access',
        'GT_read': False, 'postseal_or_score_read': False, 'RGB_or_video_read': False,
        'model_or_GPU_run': False, 'CPU_only': True,
        'interpretation': {
            'support_is_not_semantic_correctness': True,
            'reward_variance_is_not_a_gradient_reachability_or_direction_guarantee': True,
            'anchor_references_are_imperfect_teacher_pseudo_targets': True,
            'development_historical_exposure': {
                'temporal_selection_used_development_GT': bool(temporal_lock['selection_using_development_GT']),
                'spatial_selection_used_development_GT': bool(spatial_lock.get('selected_using_development_GT')),
                'confirmation_used_for_selection': False,
                'this_readback_did_not_open_or_use_GT': True,
            },
            'mechanics': {
                'reward': 'original aligned IoU of sigmoid(logit-space action) against the selected anchor box',
                'samples': '32 fixed antithetic Gaussian actions per anchor and optimizer round',
                'policy': 'q = softmax(IoU / tau); q and actions are detached in the score-function term',
                'control_variate': 'per-anchor KL(mean || rollout_mean) minus sum((q - 1/32) * logp); exact all-equal q is uniform and optimizer_step returns without Adam update',
                'reliability': 'loss is mean(reliability_i * per_anchor_loss_i), so weights are external, unnormalized by sum(reliability); global informative flag is exact reward inequality across all anchors',
                'validation': 'an additional label-free gate accepts only if weighted mean anchor IoU does not fall by more than 1e-7; rejected proposal restores state and Adam state and stops',
                'scope': '1792 spatial parameters; independent reader checks saved gradients/update vectors and state deltas but does not independently run decoder backward',
            },
            'potential_next_mechanism': 'A pre-registered eligible-support or per-anchor informative gate may be tested as a new method; this audit does not claim one is effective and does not treat reward variance as an identity signal.',
        },
        'source_bindings': {
            'scope_hashes': [{'path': x['path'], 'sha256': x['sha256'], 'bytes': x['bytes'],
                              'runtime_pin_sha256': x['runtime_pin_sha256'],
                              'pinned_in_runtime': x['pinned_in_runtime']} for x in source_bindings],
            'current_source_hash_inventory_complete': True,
            'historical_runtime_import_pin_coverage_complete': all(x['pinned_in_runtime'] for x in source_bindings if x['path'] in CODE_PATHS),
            'unpinned_historical_imports_hash_read_but_not_bound_by_RUNTIME006': [
                dict(path=x['path'], sha256=x['sha256']) for x in source_bindings
                if x['path'] in CODE_PATHS and not x['pinned_in_runtime']],
            'provenance_limit': 'The selected saved fits and math are independently re-read; the historical RUNTIME006 pin set omits some transitive Python imports listed above, so this readback does not claim byte-complete historical runtime provenance.',
            'runtime_pin_inventory': runtime_pin_readback,
            'input_hashes': [dict(path=rel, sha256=_sha(ROOT / rel), bytes=(ROOT / rel).stat().st_size)
                             for rel in META_PATHS],
            'student_global_barrier_sha256': _sha(student_barrier_path),
            'parent_student_global_barrier_sha256': _sha(parent_barrier_path),
            'runtime006_sha256': _sha(runtime_path),
            'development_roster_count': 32,
            'reference_lock_count': 32,
            'student_payload_count': 32,
            'parent_payload_count': 32,
            'student_payloads': [dict(case_id=c['case_id'], path=c['student_path'], sha256=c['student_sha256'], bytes=c['student_bytes']) for c in cases],
        },
        'counts': counts,
        'max_recalculation_errors': max_errors,
        'tolerances': {'reward_iou': REWARD_TOL, 'probability': PROB_TOL,
                       'loss': LOSS_TOL, 'reference_iou': IOU_TOL,
                       'state_delta': STATE_TOL},
        'cases': cases,
        'implementation_sha256': _sha(Path(__file__)),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
    }
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite: {_rel(output_path)}')
    tmp = output_path.with_name(output_path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, output_path)
    return out


def contracts():
    import numpy as np
    # Constant reward -> uniform q, zero population variance/range, ESS=K,
    # KL(q||uniform)=0 and zero centered gradient when rollout mean is current.
    K, sigma = 32, 0.2
    mu = np.array([[0.2, -0.4, 0.1, 0.3]], dtype=np.float64)
    z = np.repeat(mu[:, None, :], K, axis=1)
    ref = np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float64)
    rewards = iou_cxcywh_numpy(_sigmoid(z[0]), ref)
    q = _softmax(rewards / 0.7)
    assert np.var(rewards, ddof=0) == 0.0 and np.ptp(rewards) == 0.0
    assert np.array_equal(q, np.full(K, 1.0 / K))
    grad = ((mu[0] - mu[0]) / sigma**2
            - np.sum((q - 1.0 / K)[:, None] * (z[0] - mu[0]), axis=0) / sigma**2)
    assert np.array_equal(grad, np.zeros(4))
    assert float(1.0 / np.sum(q*q)) == float(K)
    assert abs(float(np.sum(q*np.log(q*K)))) < 1e-15

    # External confidence multiplies the per-anchor loss outside the action
    # policy; it is not normalized by sum(weights). K-anchor mean contributes 1/K.
    per = np.array([2.0, 4.0], dtype=np.float64)
    ext = np.array([0.25, 0.75], dtype=np.float64)
    raw_loss = float(np.mean(ext * per))
    scaled_loss = float(np.mean((3.0 * ext) * per))
    normalized_loss = float(np.sum(ext * per) / np.sum(ext))
    assert raw_loss == 1.75 and scaled_loss == 3.0 * raw_loss and raw_loss != normalized_loss

    # Zero overlap is finite and exactly zero; one pixel-independent small box
    # pair checks float32/float64 conversion error against the same formula.
    nonoverlap = iou_cxcywh_numpy(np.array([0.1, 0.1, 0.05, 0.05]),
                                 np.array([0.9, 0.9, 0.05, 0.05]))
    assert float(nonoverlap) == 0.0
    test_boxes = np.array([[0.501, 0.499, 0.17, 0.13], [0.4, 0.6, 0.08, 0.2]], dtype=np.float64)
    test_refs = np.array([[0.5, 0.5, 0.16, 0.12], [0.405, 0.59, 0.1, 0.18]], dtype=np.float64)
    iou64 = iou_cxcywh_numpy(test_boxes, test_refs)
    iou32 = iou_cxcywh_numpy(test_boxes.astype(np.float32).astype(np.float64),
                             test_refs.astype(np.float32).astype(np.float64))
    err32_64 = float(np.max(np.abs(iou64 - iou32)))
    assert err32_64 < 2e-6
    # Exact reward inequality is the historical informative gate; no >1e-6 threshold.
    tiny = np.array([0.0, np.nextafter(0.0, 1.0)], dtype=np.float64)
    assert np.any(tiny != tiny[:1]) and not np.any((tiny - tiny[:1]) > 1e-6)
    return dict(schema='stvg-ecld-p0-cpu-contracts-v1', status='pass', CPU_only=True,
                GT_read=False, model_called=False, GPU_used=False,
                checks=['constant_reward_zero_centered_gradient_and_zero_update',
                        'external_anchor_weight_is_unnormalized_outer_mean',
                        'zero_overlap_is_finite_exact_zero',
                        'float32_vs_float64_iou_conversion_error',
                        'informative_gate_is_exact_inequality_not_1e-6_threshold'],
                float32_vs_float64_max_iou_error=err32_64,
                formula_scope='loss-level/support algebra only; no decoder/autograd claim')


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--contracts', action='store_true')
    group.add_argument('--audit', action='store_true')
    ap.add_argument('--output', default=None, help='Output JSON path; refusing to overwrite')
    args = ap.parse_args()
    if args.contracts:
        result = contracts()
        out = Path(args.output) if args.output else DEFAULT_CONTRACTS_OUT
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise FileExistsError(f'refusing to overwrite: {_rel(out)}')
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    else:
        result = audit(Path(args.output) if args.output else DEFAULT_OUT)
    print(json.dumps({'status': result['status'], 'counts': result.get('counts'),
                      'output': args.output or str(DEFAULT_OUT if args.audit else DEFAULT_CONTRACTS_OUT)},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
