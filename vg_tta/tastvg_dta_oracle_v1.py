"""R1: episodic, GT-assisted distributional adaptation of native temporal heads.

No videos, specialists, spatial updates or cross-query temporal state here.
The frozen first MLP layer is required: decoder hidden is not head-last input.
"""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
STEPS = 3
BETA = 1.


def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def penultimate(hidden, head):
    # Official MLP is Linear -> ReLU -> eval-mode dropout -> Linear.
    return F.relu(F.linear(hidden.float(), head['0.weight'], head['0.bias'])).detach()


def offsets(frame_ids, records=None):
    ids = list(frame_ids)
    rec = records or [dict(offset=i, frame_ids=ids[i::2]) for i in [0, 1]]
    assert len(rec) == 2
    out = []
    for i, r in enumerate(rec):
        assert r['offset'] == i and r['frame_ids'] == ids[i::2]
        out.append([ids.index(f) for f in r['frame_ids']])
    assert sorted(out[0] + out[1]) == list(range(len(ids)))
    assert min(map(len, out)) >= 2
    return out


def joint(z):
    ij = torch.triu_indices(len(z), len(z), offset=1)
    v = z[ij[0], 0].double() + z[ij[1], 1].double()
    return v - v.logsumexp(0), ij


def gaussian(frame_ids, span):
    f = torch.as_tensor(frame_ids, dtype=torch.float64)
    assert (f[1:] > f[:-1]).all() and span[0] < span[1]
    sigma = float(torch.median(f[1:] - f[:-1]))
    # One observed cell per offset, in original physical-frame coordinates.
    # No clipping of GT boundaries to the observed support.
    ij = torch.triu_indices(len(f), len(f), offset=1)
    v = -((f[ij[0]] - span[0]).square() +
          (f[ij[1]] + 1 - span[1]).square()) / (2 * sigma ** 2)
    return v - v.logsumexp(0), sigma


def kl(logq, logp):
    return (logq.exp() * (logq - logp)).sum()


def native_decode(z, frame_ids, parts):
    # Match official FP32 log-softmax, strict upper triangle, first-max tie.
    raw = []
    for at in parts:
        zz = z[at]
        s = zz[:, 0].log_softmax(0); e = zz[:, 1].log_softmax(0)
        score = s[:, None] + e[None, :]
        score = score + (torch.ones(len(at), len(at)) * -1e32).tril(0)
        i, j = divmod(int(score.flatten().argmax()), len(at))
        assert i < j
        raw.append([at[i], at[j]])
    pair = [min(r[0] for r in raw), max(r[1] for r in raw)]
    return dict(indices=pair, physical_interval=[frame_ids[pair[0]], frame_ids[pair[1]] + 1],
                offset_indices=raw)


def fit_query(hidden, frame_ids, head, span, lr, records=None):
    x = penultimate(hidden, head); parts = offsets(frame_ids, records)
    w = head['1.weight'].clone().requires_grad_(True)
    b = head['1.bias'].clone().requires_grad_(True)
    initial_w = w.detach().clone(); initial_b = b.detach().clone()
    z0 = F.linear(x, w, b).detach()
    prior = [joint(z0[p])[0].detach() for p in parts]
    teachers, sigmas = zip(*(gaussian([frame_ids[i] for i in p], span) for p in parts))
    teacher_z = torch.empty_like(z0)
    for p, sigma in zip(parts, sigmas):
        f = torch.tensor([frame_ids[i] for i in p], dtype=torch.float64)
        teacher_z[p, 0] = (-((f - span[0]) / sigma).square() / 2).float()
        teacher_z[p, 1] = (-((f + 1 - span[1]) / sigma).square() / 2).float()
    states = [dict(weight=initial_w, bias=initial_b)]; trace = []; logits = [z0]

    def terms(z):
        pp = [joint(z[p])[0] for p in parts]
        q = torch.stack([kl(a, p) for a, p in zip(teachers, pp)]).mean()
        anchor = torch.stack([kl(a, p) for a, p in zip(prior, pp)]).mean()
        return q, anchor, q + BETA * anchor

    for step in range(STEPS):
        z = F.linear(x, w, b); q, anchor, loss = terms(z)
        gw, gb = torch.autograd.grad(loss, (w, b))
        assert torch.isfinite(loss) and torch.isfinite(gw).all() and torch.isfinite(gb).all()
        old_w = w.detach().clone(); old_b = b.detach().clone()
        with torch.no_grad():
            w.add_(gw, alpha=-lr); b.add_(gb, alpha=-lr)
            after = F.linear(x, w, b); aq, aa, al = terms(after)
        displacement = torch.cat(((w.detach() - initial_w).flatten(), b.detach() - initial_b))
        trace.append(dict(step=step + 1, GT_KL=float(q.detach()), prior_KL=float(anchor.detach()),
            loss=float(loss.detach()), after_GT_KL=float(aq), after_prior_KL=float(aa), after_loss=float(al),
            gradient_norm=float(torch.sqrt(gw.double().square().sum() + gb.double().square().sum())),
            bias_gradient_norm=float(gb.double().norm()),
            step_displacement=float(torch.sqrt((w.detach()-old_w).double().square().sum() + (b.detach()-old_b).double().square().sum())),
            arrival_displacement=float(displacement.double().norm()),
            parameters_sha256=tensor_hash(torch.cat((w.detach().flatten(), b.detach()))),
            prediction=native_decode(after, frame_ids, parts)))
        states.append(dict(weight=w.detach().clone(), bias=b.detach().clone(),
                           weight_gradient=gw.detach().clone(), bias_gradient=gb.detach().clone()))
        logits.append(after.detach().clone())
    return dict(before=native_decode(z0, frame_ids, parts), after=trace[-1]['prediction'],
        teacher_MAP=native_decode(teacher_z, frame_ids, parts), trace=trace,
        states=states, logits=logits, logq=list(teachers), logp0=prior, sigma_frames=list(sigmas),
        head_initial_sha256=tensor_hash(torch.cat((initial_w.flatten(), initial_b))),
        head_final_sha256=trace[-1]['parameters_sha256'], spatial_changed=False,
        episodic_reset=True, GT_supervised=True, lr=lr, steps=STEPS, beta=BETA)


def tiou(interval, span):
    a,b=interval; c,d=span
    inter=max(0,min(b,d)-max(a,c))
    return inter/(b-a+d-c-inter)


def paired_summary(rows, fields, seed=20261004):
    from collections import defaultdict
    if not rows:
        return dict(cells=0, sources=0, metrics={})
    source = defaultdict(list)
    for r in rows: source[r['source_id']].append([r[k] for k in fields])
    ids = sorted(source); x = np.array([np.mean(source[i],0) for i in ids])
    rng=np.random.default_rng(seed)
    boot=np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(100)])
    ci=np.quantile(boot,[.025,.975],axis=0)
    return dict(cells=len(rows),sources=len(ids),bootstrap_draws=10000,seed=seed,
        metrics={k:dict(mean=float(x[:,j].mean()),ci95=ci[:,j].tolist(),
            source_values={str(i):float(v) for i,v in zip(ids,x[:,j])},
            source_positive=int((x[:,j]>1e-12).sum()),source_negative=int((x[:,j]<-1e-12).sum()),
            cell_macro=float(np.mean([r[k] for r in rows]))) for j,k in enumerate(fields)})
