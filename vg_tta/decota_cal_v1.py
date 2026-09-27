"""Candidate event-support TTA. No label-reader or ground-truth input here."""
import copy
import hashlib
import math
import numpy as np
import torch


def cpu_state(head):
    return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}


def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


def calibrated(signal, alpha, beta):
    from scipy.special import expit, logit
    a = np.asarray(signal, float)
    if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('Invalid frozen probability')
    return expit(alpha*logit(np.clip(a, 1e-6, 1-1e-6))+beta)


def constraints(probability, high, low):
    p = np.asarray(probability, float); n = len(p)
    if not 0 < low < high < 1 or n < 4 or not np.isfinite(p).all():
        raise ValueError('Invalid event judgments')
    pos, neg = p >= high, p <= low
    ii, jj = np.indices((n, n)); legal = ii < jj
    pp = np.flatnonzero(pos); nn = np.flatnonzero(neg)
    if not len(pp):
        return np.zeros_like(legal), dict(reason='no_positive', positive=pp.tolist(), negative=nn.tolist())
    compatible = legal & (ii <= pp[0]) & (jj >= pp[-1])
    counts = np.r_[0, np.cumsum(neg)]
    compatible &= (counts[jj+1]-counts[ii]) == 0
    return compatible, dict(reason='active' if compatible.any() else 'incompatible_judgments',
        positive=pp.tolist(), negative=nn.tolist(), unknown=int((~(pos|neg)).sum()),
        compatible_count=int(compatible.sum()))


def envelope_joint(logits, positions, n):
    """Exact two-offset envelope PMF, by endpoint ownership (no CDF subtraction)."""
    if len(logits) != 2 or len(positions) != 2:
        raise ValueError('Exactly two disjoint official offsets required')
    if sorted(positions[0]+positions[1]) != list(range(n)):
        raise ValueError('Offset grid must partition full physical grid')
    joints, cdfs, starts, ends = [], [], [], []
    for z, pos in zip(logits, positions):
        z = z.reshape(-1, 2).double(); m = len(pos)
        if z.shape != (m, 2) or m < 2 or not bool(torch.isfinite(z).all()):
            raise ValueError('Bad logits')
        v = z[:, 0, None]+z[None, :, 1]
        valid = torch.triu(torch.ones_like(v, dtype=torch.bool), 1)
        j = torch.zeros_like(v).masked_scatter(valid, v[valid].softmax(0))
        idx = torch.tensor(pos, device=z.device)
        full = z.new_zeros(n, n).index_put((idx[:, None], idx[None, :]), j)
        start = full.cumsum(1)
        end = full.flip(0).cumsum(0).flip(0)
        cdf = start.flip(0).cumsum(0).flip(0)
        joints.append(full); starts.append(start); ends.append(end); cdfs.append(cdf)
    q = joints[0]*cdfs[1]+joints[1]*cdfs[0]+starts[0]*ends[1]+starts[1]*ends[0]
    if not bool(torch.isfinite(q).all()) or abs(float(q.detach().sum())-1) > 1e-8:
        raise FloatingPointError('Envelope normalization failed')
    return q


def decode_constrained(joint, compatible, ids, native_logits):
    mask = np.asarray(compatible, bool)
    q = joint.detach().cpu().numpy()
    if not mask.any():
        raise ValueError('No compatible interval')
    a, b = np.unravel_index(np.where(mask, q, -1.).argmax(), mask.shape)
    ids = np.asarray(ids); duration = ids[b]+1-ids[a]
    z = np.asarray(native_logits, dtype=np.float64).reshape(len(ids), 2)
    score = z[:, 0, None]+z[None, :, 1]
    same = mask & (ids[None, :]+1-ids[:, None] == duration)
    i, j = np.unravel_index(np.where(same, score, -np.inf).argmax(), mask.shape)
    assert same[i, j]
    return [int(i), int(j)], dict(adapted_MAP=[int(a), int(b)], physical_duration=int(duration),
        same_duration_candidates=int(same.sum()), compatible_preserved=True)


def fit_episode(source_head, head_inputs, offset_positions, ids, native_indices,
                native_logits, probability, high, low, lr, steps=5, keep_state=True):
    """Real private original head; frozen labels/features; selection sees loss only."""
    mask, audit = constraints(probability, high, low)
    source = cpu_state(source_head); source_hash = state_hash(source)
    if not mask.any():
        return dict(indices=list(native_indices), no_parameter_indices=list(native_indices),
            audit=audit, curve=[], best_step=0, parameter_l2=0., GT_online=False,
            source_unchanged=True, source_hash=source_hash, state=None)
    head = copy.deepcopy(source_head).float().eval().requires_grad_(True)
    device = next(head.parameters()).device
    inputs = [h.detach().to(device=device, dtype=torch.float32) for h in head_inputs]
    cm = torch.as_tensor(mask, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, eps=1e-4, weight_decay=0.)
    def forward():
        zs = [head(h)[-1] for h in inputs]
        q = envelope_joint(zs, offset_positions, len(ids))
        mass = q[cm].sum()
        if not bool(torch.isfinite(mass)) or float(mass.detach()) < 1e-280:
            raise FloatingPointError('Compatible probability underflow')
        return -mass.log(), q
    curve = []; best_loss = math.inf; best = None; best_step = 0; stop = None
    first_q = None; best_q = None
    for step in range(steps+1):
        try:
            loss, q = forward()
        except (ValueError, FloatingPointError) as error:
            stop = str(error); break
        value = float(loss.detach())
        row = dict(step=step, loss=value, mass=float(q[cm].sum().detach()))
        if step == 0: first_q = q.detach().cpu()
        if value < best_loss:
            best_loss, best, best_step, best_q = value, cpu_state(head), step, q.detach().cpu()
        curve.append(row)
        if step < steps:
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grads = [p.grad for p in head.parameters()]
            if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
                stop = 'nonfinite_gradient'; break
            row['gradient_norm'] = math.sqrt(sum(float(g.double().square().sum()) for g in grads))
            optimizer.step()
    if best is None:
        return dict(indices=list(native_indices), no_parameter_indices=list(native_indices),
            audit={**audit, 'reason':'numerical_abstain'}, curve=curve, best_step=0,
            parameter_l2=0., GT_online=False, source_unchanged=True, source_hash=source_hash,
            state=None, numerical_stop=stop)
    head.load_state_dict(best)
    with torch.no_grad(): restored_loss, restored_q = forward()
    assert torch.equal(best_q, restored_q.cpu()) and float(restored_loss) == best_loss
    indices, dec = decode_constrained(best_q, mask, ids, native_logits)
    no_param, dec0 = decode_constrained(first_q, mask, ids, native_logits)
    assert state_hash(cpu_state(source_head)) == source_hash
    d2 = sum(float((best[k].double()-source[k].double()).square().sum()) for k in source)
    return dict(indices=indices, no_parameter_indices=no_param, audit=audit, decode=dec,
        no_parameter_decode=dec0, curve=curve, best_step=best_step, parameter_l2=math.sqrt(d2),
        state=best if keep_state else None, state_sha256=state_hash(best), source_hash=source_hash,
        numerical_stop=stop, source_unchanged=True, best_restore_exact=True, GT_online=False)
