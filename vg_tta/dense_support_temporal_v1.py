"""F39: frozen final actionness -> legal interval inclusion, real temporal TTA.

No labels, query filtering, spatial fitting, interval relocation or registry I/O.
FP32 network, FP64 probability reductions; native s<e support is unchanged.
"""
import copy
import math
import time
import torch
from vg_tta.time_space_repair_v1 import legal_logp
from vg_tta.decota_tastvg_episode_v1 import fitted_merge, native_view_indices

EPSILON = 1e-12


def time_cells(frame_ids, *, device=None):
    """Exact F25 observed-clock convention [first_id,last_id+1), per offset.

    Midpoints use physical frame IDs, NOT index distance. No unsampled external
    context or GT-derived background is appended. Units cancel in normalization.
    """
    f = torch.as_tensor(frame_ids, dtype=torch.float64, device=device)
    assert len(f) >= 2 and bool((f[1:] > f[:-1]).all())
    edges = torch.cat((f[:1], (f[:-1]+f[1:])/2, f[-1:]+1))
    widths = edges[1:]-edges[:-1]
    assert bool((widths > 0).all())
    return edges, widths/widths.sum()


def teacher_from_logits(actions, records):
    assert len(actions) == len(records) == 2
    result = []
    for raw, rec in zip(actions, records):
        x = raw.detach().clone().reshape(-1).float()
        ids = rec['frame_ids']
        assert len(x) == len(ids) and bool(torch.isfinite(x).all())
        edges, omega = time_cells(ids, device=x.device)
        result.append(dict(raw_logits=x, a=x.sigmoid().detach().clone(), omega=omega,
                           cell_edges=edges, frame_ids=list(ids), valid_mask=torch.ones_like(x, dtype=torch.bool),
                           semantic_threshold=None, sigmoid_applications=1,
                           identity='final pred_actioness; pipeline.py raw action_embed(final time hidden state)',
                           forced_note='legal interval structural r=0/1 recorded separately'))
    return result


def log_marginals(lp, ij, n, eps=EPSILON):
    """Stable log inclusion AND log exclusion without 1-exp cancellation.

    Empty sets are structural constants (not logsumexp(-inf) gradient NaNs).
    Numerical clamp applies symmetrically to both event terms. No hard labels.
    """
    i = torch.arange(n, device=lp.device)
    inside = (ij[0, :, None] <= i) & (i <= ij[1, :, None])
    logs = []
    for mask in (inside, ~inside):
        possible = mask.any(0)
        # Provide one finite dummy only on impossible columns, then replace by
        # a constant. This keeps the unused autograd branch finite as well.
        safe = mask.clone()
        safe[0, ~possible] = True
        value = torch.logsumexp(lp[:, None].masked_fill(~safe, -torch.inf), dim=0)
        value = torch.where(possible, value, torch.full_like(value, math.log(eps)))
        logs.append(value.clamp(min=math.log(eps), max=math.log1p(-eps)))
    return logs[0], logs[1], (~inside.any(0)), inside.all(0)


def objective(logits, reference, teacher, beta):
    data = []; keep = []; audits = []
    assert len(logits) == len(reference) == len(teacher) == 2
    for z, p0, t in zip(logits, reference, teacher):
        lp, ij = legal_logp(z)
        n = z.reshape(-1, 2).shape[0]
        assert n == len(t['a']) and bool(t['valid_mask'].all())
        assert not t['a'].requires_grad and not p0.requires_grad
        li, lo, forced0, forced1 = log_marginals(lp, ij, n)
        a, w = t['a'].to(lp), t['omega'].to(lp)
        ev = -(w*(a*li+(1-a)*lo)).sum()
        kl = (p0.exp()*(p0-lp)).sum()  # KL(P0 || P_phi), NOT the reverse.
        data.append(ev); keep.append(kl)
        audits.append(dict(event=float(ev.detach()), keep=float(kl.detach()), positions=n,
            weight_sum=float(w.sum()), posterior_sum=float(lp.exp().sum().detach()),
            forced_zero=forced0.nonzero().flatten().cpu().tolist(),
            forced_one=forced1.nonzero().flatten().cpu().tolist(),
            inclusion=li.exp().detach().cpu().tolist()))
    ev, kl = torch.stack(data).mean(), torch.stack(keep).mean()
    return ev+beta*kl, dict(event=float(ev.detach()), keep=float(kl.detach()), offsets=audits)


def norm(xs):
    return math.sqrt(sum(float(x.detach().double().square().sum()) for x in xs if x is not None))


class OutputReplay:
    """T2: independent endpoint logits, no network weights, not an upper bound."""
    def __init__(self, zero):
        self.zero = zero
        self.named = [('output.'+str(i), torch.nn.Parameter(z.detach().clone())) for i,z in enumerate(zero['logits'])]
        self.initial = self.state()
    def state(self):
        return {n:p.detach().clone() for n,p in self.named}
    def restore(self, state):
        with torch.no_grad():
            for n,p in self.named: p.copy_(state[n])
    def values(self):
        return {**self.zero, 'logits':[p for n,p in self.named]}


def fit(it, records, ids, teacher, *, lr=1e-3, beta=.1, steps=5, output_control=False):
    it.restore(it.initial)
    params = [p for n,p in it.named]
    optimizer = torch.optim.Adam if output_control else torch.optim.AdamW
    opt = optimizer(params, lr=lr, betas=(.9,.999), eps=1e-4, weight_decay=0.)
    with torch.no_grad(): zero = it.values()
    refs = [legal_logp(z)[0].detach().clone() for z in zero['logits']]
    frozen_teacher = [t['a'].clone() for t in teacher]
    path = []; best = math.inf; saved = it.state(); selected = 0; start = time.perf_counter()
    for step in range(steps+1):
        opt.zero_grad(set_to_none=True)
        value = it.values(); loss, parts = objective(value['logits'], refs, teacher, beta)
        lv = float(loss.detach())
        assert math.isfinite(lv), 'nonfinite_current_loss'
        if lv < best: best, selected, saved = lv, step, it.state()
        assert torch.equal(value['boxes'], zero['boxes']), 'time changed spatial boxes'
        row = dict(step=step, loss=lv, parts=parts,
            logits=[z.detach().cpu().clone() for z in value['logits']],
            indices=list(fitted_merge(value['logits'], records, ids)),
            raw_offset_indices=[list(native_view_indices(z)) for z in value['logits']],
            state={n:p.detach().cpu().clone() for n,p in it.named},
            state_delta=norm([p-it.initial[n] for n,p in it.named]))
        row['physical_interval'] = [ids[row['indices'][0]], ids[row['indices'][1]]+1]
        row['raw_physical_intervals'] = [[r['frame_ids'][s],r['frame_ids'][e]+1]
                for r,(s,e) in zip(records,row['raw_offset_indices'])]
        path.append(row)
        if step == steps: break
        loss.backward()
        row['gradient_norm'] = norm([p.grad for p in params])
        row['unused_parameters'] = [n for n,p in it.named if p.grad is None]
        assert not row['unused_parameters']
        assert all(bool(torch.isfinite(p.grad).all()) for p in params), 'nonfinite_gradient'
        before = it.state(); opt_before = copy.deepcopy(opt.state_dict())
        opt.step(); after = it.state(); trials = []; accepted = False
        for alpha in (1., .5, .25, .125):
            it.restore({n:v+alpha*(after[n]-v) for n,v in before.items()})
            with torch.no_grad(): ll, _ = objective(it.values()['logits'], refs, teacher, beta)
            val = float(ll)
            trials.append(dict(alpha=alpha, loss=val if math.isfinite(val) else None))
            if math.isfinite(val) and val < lv-1e-9:
                accepted = True
                break
        if not accepted:
            it.restore(before); opt.load_state_dict(opt_before)
            assert all(torch.equal(p,before[n]) for n,p in it.named)
            # Check Adam moments/steps as well as weights after rejection.
            restored = opt.state_dict()
            assert restored['param_groups'] == opt_before['param_groups']
            for k,v in opt_before['state'].items():
                assert all(torch.equal(restored['state'][k][n], x) if torch.is_tensor(x)
                           else restored['state'][k][n] == x for n,x in v.items())
        row.update(accepted=accepted, trials=trials, optimizer_restored_on_reject=not accepted,
                   proposal_norm=norm([after[n]-before[n] for n in before]))
    it.restore(saved)
    with torch.no_grad(): final = it.values()
    assert all(torch.equal(z.cpu(),q) for z,q in zip(final['logits'],path[selected]['logits']))
    assert all(torch.equal(t['a'],a) for t,a in zip(teacher,frozen_teacher))
    out = dict(state={n:p.detach().cpu().clone() for n,p in it.named}, final=path[selected], path=path,
        best_step=selected, state_delta=path[selected]['state_delta'], parameters=sum(p.numel() for p in params),
        steps=steps, lr=lr, beta=beta, backwards=sum('gradient_norm' in r for r in path),
        optimizer=optimizer.__name__, seconds=time.perf_counter()-start, teacher_frozen=True,
        GT_online=False, failure=None, output_optimization=output_control,
        parameter_TTA=not output_control, spatial_invariant=True, actual_best_state_verified=True)
    it.restore(it.initial)
    assert all(torch.equal(p,it.initial[n]) for n,p in it.named)
    return out
