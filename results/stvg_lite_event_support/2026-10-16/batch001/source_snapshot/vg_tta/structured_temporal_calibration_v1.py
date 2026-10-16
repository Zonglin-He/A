"""F44: only normalization level and native-prior strength vary.

The official 66306-parameter head (P1), or explicitly labelled free output
logits (P2), minimizes the same NLL + mode margin + KL objective. No GT,
reliability weight, decision-state score, trust guard or spatial update.
"""
import copy
import math
import time

import torch
import torch.nn.functional as F

from vg_tta.dense_support_temporal_v1 import time_cells, norm
from vg_tta.dense_support_tuning_v1 import state_hash, move
from vg_tta.time_space_repair_v1 import legal_logp
from vg_tta.decota_tastvg_episode_v1 import native_view_indices, fitted_merge
from vg_tta.structured_temporal_v1 import objective as _f43_objective

FIXED = dict(epsilon=1e-6, temperature=1., margin=.2, gamma=1., beta=.1)
MENU = [dict(center_fraction=a, prior_weight=p)
        for a in (0., .5, 1.) for p in (.1, .3, 1.)]


def normalize(raw, center_fraction, epsilon=1e-6):
    x = raw.detach().clone().reshape(-1).double()
    assert 0 <= center_fraction <= 1 and epsilon > 0
    assert len(x) >= 2 and torch.isfinite(x).all()
    median = torch.quantile(x, .5)
    mad = torch.quantile((x-median).abs(), .5)
    z = (x-center_fraction*median)/(mad+epsilon)
    return dict(raw_logits=x, standardized_logits=z, a=z.sigmoid(),
                log_a=F.logsigmoid(z), log_not_a=F.logsigmoid(-z),
                median=float(median), mad=float(mad), mad_zero=bool(mad == 0),
                center_fraction=center_fraction)


def project(zero_logits, teacher, records, config):
    cfg = {**FIXED, **config}
    assert cfg['prior_weight'] > 0
    assert len(zero_logits) == len(teacher) == len(records) == 2
    offsets = []
    for z, t, record in zip(zero_logits, teacher, records):
        lp, ij = legal_logp(z)
        cal = normalize(t['raw_logits'], cfg['center_fraction'], cfg['epsilon'])
        ids = record['frame_ids']; edges, w = time_cells(ids, device=lp.device)
        assert len(cal['a']) == len(ids) == z.reshape(-1, 2).shape[0]
        assert t['valid_mask'].all() and torch.equal(t['omega'].to(w), w)
        q = w*(cal['log_a']-cal['log_not_a'])
        prefix = torch.cat((q.new_zeros(1), q.cumsum(0)))
        # BCE has a as the probability, and interval membership as its target.
        cost = -(w*cal['log_not_a']).sum()-(prefix[ij[1]+1]-prefix[ij[0]])
        score = -cost+cfg['prior_weight']*lp.detach()
        target = int(score.argmax())
        native_pair = native_view_indices(z)
        native = int(((ij[0] == native_pair[0]) & (ij[1] == native_pair[1])).nonzero()[0])
        s, e = [int(v) for v in ij[:, target]]
        offsets.append(dict(**cal, omega=w, cell_edges=edges, frame_ids=list(ids),
            logp0=lp.detach().clone(), ij=ij, cost=cost.detach(), score=score.detach(),
            target=target, target_indices=[s,e], interval=[ids[s],ids[e]+1],
            native=native, native_indices=list(native_pair),
            native_interval=[ids[native_pair[0]],ids[native_pair[1]]+1],
            valid_positions=len(ids)))
    return dict(offsets=offsets, config=cfg, teacher_frozen=True, GT_online=False)


def projected_indices(evidence, ids):
    intervals = [x['interval'] for x in evidence['offsets']]
    return [ids.index(min(x[0] for x in intervals)), ids.index(max(x[1] for x in intervals)-1)]


def objective(logits, evidence):
    # Reuse the sealed, tested F43 algebra, explicitly disabling its optional rho.
    return _f43_objective(logits, evidence, reliability=False)


def fit(it, records, ids, teacher, config, *, lr, steps=5, output_control=False):
    assert steps >= 0
    it.restore(it.initial); params = [p for _,p in it.named]
    optimizer = torch.optim.Adam if output_control else torch.optim.AdamW
    opt = optimizer(params, lr=lr, betas=(.9,.999), eps=1e-4, weight_decay=0.)
    with torch.no_grad(): zero = it.values()
    evidence = project(zero['logits'], teacher, records, config)
    teacher_hash = state_hash({str(i): t['raw_logits'] for i,t in enumerate(teacher)})
    path = []; best_loss = math.inf; best_step = 0; saved = it.state()
    started = time.perf_counter(); backwards = 0
    for step in range(steps+1):
        opt.zero_grad(set_to_none=True)
        value = it.values(); loss, parts = objective(value['logits'], evidence)
        lv = float(loss.detach()); assert math.isfinite(lv)
        assert torch.equal(value['boxes'], zero['boxes'])
        state = it.state()
        if lv < best_loss-1e-12:
            best_loss, best_step, saved = lv, step, state
        row = dict(step=step, loss=lv, parts=parts,
            logits=[z.detach().cpu().clone() for z in value['logits']],
            state_sha256=state_hash(state), state_delta=norm([p-it.initial[n] for n,p in it.named]),
            indices=list(fitted_merge(value['logits'], records, ids)),
            raw_offset_indices=[list(native_view_indices(z)) for z in value['logits']])
        row['physical_interval'] = [ids[row['indices'][0]],ids[row['indices'][1]]+1]
        row['raw_physical_intervals'] = [[r['frame_ids'][s],r['frame_ids'][e]+1]
                                       for r,(s,e) in zip(records,row['raw_offset_indices'])]
        path.append(row)
        if step == steps: break
        loss.backward(); backwards += 1
        row['gradient_norm'] = norm([p.grad for p in params])
        row['unused_parameters'] = [n for n,p in it.named if p.grad is None]
        assert not row['unused_parameters']
        assert all(torch.isfinite(p.grad).all() for p in params)
        before = it.state(); opt_before = copy.deepcopy(opt.state_dict()); opt.step(); after = it.state()
        trials = []; accepted = False
        for scale in (1., .5, .25, .125):
            it.restore({n:v+scale*(after[n]-v) for n,v in before.items()})
            with torch.no_grad(): pl,_ = objective(it.values()['logits'], evidence)
            pv = float(pl); trials.append(dict(alpha=scale, loss=pv if math.isfinite(pv) else None))
            if math.isfinite(pv) and pv < lv-1e-9:
                accepted = True; break
        row.update(accepted=accepted,trials=trials,optimizer_restored_on_reject=not accepted,
                   proposal_norm=norm([after[n]-before[n] for n in before]))
        if not accepted:
            it.restore(before); opt.load_state_dict(opt_before)
            actual = opt.state_dict()
            assert actual['param_groups'] == opt_before['param_groups']
            for key, group in opt_before['state'].items():
                assert all(torch.equal(actual['state'][key][n],x) if torch.is_tensor(x)
                           else actual['state'][key][n] == x for n,x in group.items())
    it.restore(saved)
    with torch.no_grad(): end = it.values()
    assert all(torch.equal(a.cpu(),b) for a,b in zip(end['logits'],path[best_step]['logits']))
    assert torch.equal(end['boxes'],zero['boxes'])
    assert teacher_hash == state_hash({str(i): t['raw_logits'] for i,t in enumerate(teacher)})
    it.restore(it.initial)
    assert all(torch.equal(p,it.initial[n]) for n,p in it.named)
    for p in params: p.grad = None
    return dict(path=path, state=move(saved,'cpu'), initial_state=move(it.initial,'cpu'),
        final={**path[best_step], 'boxes':zero['boxes'].detach().cpu().clone()},
        evidence=move(evidence,'cpu'), best_step=best_step, state_delta=path[best_step]['state_delta'],
        lr=lr, steps=steps, config=evidence['config'], backwards=backwards,
        seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in params),
        parameter_TTA=not output_control, output_control=output_control,
        teacher_frozen=True, GT_online=False, spatial_invariant=True, source_restored=True)
