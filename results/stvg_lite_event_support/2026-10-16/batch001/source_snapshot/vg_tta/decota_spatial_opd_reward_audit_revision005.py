"""CPU precision supplement for detached IoU, without changing any GPU operation.

Preserve the original float64 absolute failure. Independent CPU float32 IoU
retains 3e-6. A per-action interval encloses the independently measured sigmoid
rounding and the subsequent float32 corner/area/division arithmetic. This is
not a new reward, clipping rule, or claim of a CUDA transcendental proof.
"""
import numpy as np
import torch
from vg_tta.decota_fixed_full_audit_v1 import top1_support, overlap, vector
from vg_tta.decota_spatial_opd_precision_audit_revision002 import softmax_check
from vg_tta.decota_spatial_opd_precision_audit_revision001 import gradient_check
from vg_tta.decota_spatial_opd_chart_revision003 import audit as previous_audit, chart_audit

REVISION = 'detached_reward_precision_revision005'
THRESHOLD = 3e-6
U = float(np.finfo(np.float32).eps / 2)
TINY = float(np.finfo(np.float32).tiny)


def interval(lo, hi=None):
    return np.asarray(lo, float), np.asarray(lo if hi is None else hi, float)


def rounded(lo, hi):
    # Includes subnormal error; float64 nextafter makes endpoint rounding outward.
    error = U * np.maximum(abs(lo), abs(hi)) + TINY
    return np.nextafter(lo-error, -np.inf), np.nextafter(hi+error, np.inf)


def add(a, b): return rounded(a[0]+b[0], a[1]+b[1])
def sub(a, b): return rounded(a[0]-b[1], a[1]-b[0])
def mul(a, b):
    candidates = np.stack([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]])
    return rounded(candidates.min(0), candidates.max(0))
def minimum(a, b): return np.minimum(a[0], b[0]), np.minimum(a[1], b[1])
def maximum(a, b): return np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
def product(a): return mul((a[0][..., 0], a[1][..., 0]), (a[0][..., 1], a[1][..., 1]))
def positive(a): return maximum(a, interval(0.))


def corners(a):
    center = a[0][..., :2], a[1][..., :2]
    half = mul((a[0][..., 2:], a[1][..., 2:]), interval(.5))
    return sub(center, half), add(center, half)


def reward_check(samples, evidence, saved):
    x = np.asarray(samples, np.float32)
    e = np.asarray(evidence, np.float32)
    saved = np.asarray(saved, float)
    assert x.ndim == 3 and x.shape[-2:] == (32, 4) and e.shape == (x.shape[0], 4)
    assert saved.shape == x.shape[:-1] and np.isfinite(x).all() and np.isfinite(e).all()
    assert np.isfinite(saved).all() and (e[:, 2:] > 0).all()
    a32 = torch.sigmoid(torch.tensor(x, dtype=torch.float32)).numpy()
    # Independent float32 geometry; no imported production IoU/box functions.
    ec = e[:, None, :]
    pl, ph = a32[..., :2] - a32[..., 2:]*np.float32(.5), a32[..., :2] + a32[..., 2:]*np.float32(.5)
    el, eh = ec[..., :2] - ec[..., 2:]*np.float32(.5), ec[..., :2] + ec[..., 2:]*np.float32(.5)
    intersection = np.maximum(np.minimum(ph, eh)-np.maximum(pl, el), np.float32(0)).prod(-1, dtype=np.float32)
    pa = np.maximum(ph-pl, np.float32(0)).prod(-1, dtype=np.float32)
    ea = np.maximum(eh-el, np.float32(0)).prod(-1, dtype=np.float32)
    denom = np.maximum(pa+ea-intersection, np.float32(1e-7))
    ref32 = (intersection/denom).astype(float)
    same = abs(saved-ref32)
    assert float(same.max()) < THRESHOLD, 'Independent CPU32 IoU fails unchanged 3e-6 absolute check'
    a64 = 1/(1+np.exp(-x.astype(float)))
    exact = np.asarray([[overlap(a, b) for a in aa] for aa, b in zip(a64, e)])
    action_interval = interval(np.minimum(a32.astype(float), a64), np.maximum(a32.astype(float), a64))
    p0, p1 = corners(action_interval)
    e0, e1 = corners(interval(ec.astype(float)))
    area_i = product(positive(sub(minimum(p1, e1), maximum(p0, e0))))
    area_p, area_e = product(positive(sub(p1, p0))), product(positive(sub(e1, e0)))
    denominator = maximum(sub(add(area_p, area_e), area_i), interval(float(np.float32(1e-7))))
    assert (denominator[0] > 0).all()
    low, high = rounded(area_i[0]/denominator[1], area_i[1]/denominator[0])
    assert np.all(ref32 >= low) and np.all(ref32 <= high), 'Independent CPU32 reward escaped interval arithmetic'
    # Exact float64 oracle is observational, never substituted in fitting.
    geometry_bound = np.maximum(abs(low-exact), abs(high-exact))
    bound = geometry_bound + same + 32*np.finfo(np.float64).eps
    mixed = abs(saved-exact)
    assert np.all(mixed <= bound), 'Mixed-precision reward exceeds measured sigmoid/derived geometry bound'
    return dict(max_original_GPU32_vs_float64_error=float(mixed.max()),
        original_absolute_3e_minus6_passed=bool(mixed.max() < THRESHOLD),
        max_independent_CPU32_vs_GPU32_error=float(same.max()),
        matched_precision_absolute_threshold=THRESHOLD,
        max_measured_CPU32_sigmoid_vs_float64_error=float(abs(a32.astype(float)-a64).max()),
        max_float32_geometry_interval_bound=float(geometry_bound.max()),
        max_cross_precision_bound=float(bound.max()),
        max_cross_precision_bound_fraction=float((mixed/bound).max()),
        unit_roundoff=U, interval_rounding='outward float64 endpoints with float32 u/tiny at each arithmetic operation',
        CUDA_transcendental_kernel_not_independently_proved=True,
        production_reward_actions_and_weights_unchanged=True)


def reward_failure(exc):
    visited = set()
    while exc is not None and id(exc) not in visited:
        visited.add(id(exc)); tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename.endswith('/decota_spatial_opd_tunable_audit_v1.py') and tb.tb_lineno == 21:
                return True
            tb = tb.tb_next
        exc = exc.__context__
    return False


def audit(z, expert):
    try:
        return previous_audit(z, expert)
    except AssertionError as exc:
        if not reward_failure(exc): raise
    cfg = z['config']; steps = cfg['steps']; names = list(z['initial']); hist = z['path']; support = top1_support(expert)
    assert z['selected_step'] == steps and len(hist) == steps+1 and len(z['rounds']) == steps
    assert z['positions'] == [p for p, _ in support] and z['active_parameters'] == 1792 and not z['GT_used']
    assert np.array_equal(vector(z['state'], names), vector(hist[steps]['state'], names))
    assert np.array_equal(z['final'].numpy(), hist[steps]['boxes'].numpy())
    m = np.zeros(1792); v = np.zeros(1792); updates = 0; erradam = 0.
    rewards = []; gradients = []; weights = []
    evidence = np.array([e for _, e in support], np.float32)
    for k, (h, r) in enumerate(zip(hist, z['rounds'])):
        a = vector(h['state'], names); nxt = vector(hist[k+1]['state'], names)
        if not support:
            assert not r['updated'] and np.array_equal(a, nxt); continue
        rr = r['rollout']; samples = rr['samples'].numpy().astype(float)
        mean = rr['mean'].numpy().astype(float); w = rr['weights'].numpy().astype(float)
        used = rr['used_rewards'].numpy().astype(float); original = rr['rewards'].numpy().astype(float)
        rewards.append(reward_check(samples, evidence, original))
        if rr['permutations'] is not None:
            assert np.array_equal(used, np.take_along_axis(original, rr['permutations'].numpy(), 1))
        else: assert np.array_equal(used, original)
        weights.append(softmax_check(used, w, cfg['tau']))
        current = r['mean_before'].numpy().astype(float)
        gradients.append(gradient_check(current, mean, samples, w, r['autograd_mean_gradient'].numpy().astype(float), cfg['sigma']))
        assert r['analytic_gradient_error'] < 2e-5
        assert r['updated'] == rr['informative'] and r['no_information'] == (not rr['informative'])
        if not r['updated']:
            assert np.array_equal(a, nxt); continue
        update = h['update']; g = update['gradient'].numpy().astype(float); updates += 1
        m = .9*m+.1*g; v = .999*v+.001*g*g
        delta = -cfg['lr']*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
        raw = update['raw'].numpy(); error = float(abs(delta-raw).max())
        assert error < 4e-6; erradam = max(erradam, error)
        assert np.array_equal(nxt-a, raw) and update['names'] == names
    assert updates == z['gradient_calls'] and rewards and any(not r['original_absolute_3e_minus6_passed'] for r in rewards)
    return dict(status='pass', revision=REVISION,
        original_absolute_cross_precision_reward_audit_passed=False, original_failed_line=21,
        original_failure_preserved=True, reward_precision_checks=rewards,
        precision_checks=gradients, softmax_precision_checks=weights,
        max_detached_reward_error=max(r['max_original_GPU32_vs_float64_error'] for r in rewards),
        max_mean_gradient_error=max(r['max_original_GPU_vs_float64_error'] for r in gradients),
        max_Adam_error=erradam, gradient_calls=updates, rounds=steps, final_step=steps,
        authorized_chart_extension=chart_audit(z), GT_read=False,
        independent_decoder_Jacobian=False, algorithm_or_saved_gradient_changed=False)
