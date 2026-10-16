"""F45: source-anchored path diagnosis; conditional soft structured teacher.

A4 is immutable. No GT, quality threshold, per-example eta, new signal,
warm start, or change to the original legal MAP decoder is used here.
"""
import copy
import math
import time

import torch

from vg_tta.dense_support_temporal_v1 import norm
from vg_tta.dense_support_tuning_v1 import move, state_hash
from vg_tta.decota_tastvg_episode_v1 import fitted_merge, native_view_indices
from vg_tta.time_space_repair_v1 import legal_logp
from vg_tta.structured_temporal_calibration_v1 import project, objective as hard_objective

ETAS = (0., .25, .5, .75, 1.)


def interpolate(a, b, eta):
    assert 0 <= eta <= 1
    assert a.shape == b.shape and a.dtype == b.dtype
    # Exact endpoint controls, avoiding subtraction/readdition rounding at eta=1.
    if eta == 0: return a.detach().clone()
    if eta == 1: return b.detach().clone()
    return (a.detach()+eta*(b.detach()-a.detach())).clone()


def interpolate_state(source, adapted, eta):
    assert source.keys() == adapted.keys()
    return {n: interpolate(v, adapted[n].to(v), eta) for n,v in source.items()}


def prediction(logits, boxes, records, ids):
    z = [x.detach().cpu().clone() for x in logits]
    ii = list(fitted_merge(z, records, ids))
    raw = [list(native_view_indices(x)) for x in z]
    return dict(logits=z, boxes=boxes.detach().cpu().clone(), indices=ii,
        physical_interval=[ids[ii[0]], ids[ii[1]]+1], raw_offset_indices=raw,
        raw_physical_intervals=[[r['frame_ids'][s],r['frame_ids'][e]+1]
                                for r,(s,e) in zip(records,raw)])


def path_point(head, fitted, records, ids, eta):
    assert all(torch.equal(head.initial[n].cpu(), fitted['initial_state'][n])
               for n in head.initial)
    state = interpolate_state(fitted['initial_state'], fitted['state'], eta)
    try:
        head.restore(state)
        with torch.no_grad(): values = head.values()
        assert torch.equal(values['boxes'], head.zero['boxes'])
        param = prediction(values['logits'],values['boxes'],records,ids)
        damped = [interpolate(a.cpu(),b.cpu(),eta)
                  for a,b in zip(head.zero['logits'],fitted['final']['logits'])]
        output = prediction(damped,values['boxes'],records,ids)
        if eta in (0.,1.):
            assert all(torch.equal(a,b) for a,b in zip(param['logits'],output['logits']))
        return dict(eta=eta, state=state, parameter=param, logit=output,
            state_sha256=state_hash(state), state_delta=norm([state[n]-v for n,v in fitted['initial_state'].items()]),
            nonlinear_logit_max=max(float((a-b).abs().max()) for a,b in zip(param['logits'],output['logits'])),
            parameter_TTA=eta>0 and fitted['state_delta']>0, new_backwards=0,
            source_restored=True, spatial_invariant=True, GT_online=False)
    finally:
        head.restore(head.initial)
        assert all(torch.equal(p,head.initial[n]) and p.grad is None for n,p in head.named)


def soft_evidence(zero_logits, teacher, records, config):
    evidence = project(zero_logits,teacher,records,config)
    evidence['soft_temperature'] = 1.
    for t in evidence['offsets']:
        logq = torch.log_softmax(t['score'].double(),dim=0).detach().clone()
        t['logq'] = logq
        t['q'] = logq.exp()
        assert not t['q'].requires_grad and torch.isfinite(logq).all()
        assert abs(float(t['q'].sum())-1.)<1e-12
        assert int(logq.argmax()) == t['target']
    return evidence


def soft_objective(logits, evidence):
    cfg = evidence['config']; rows=[]; components=[]
    for z,t in zip(logits,evidence['offsets']):
        lp,ij = legal_logp(z)
        assert torch.equal(ij,t['ij']) and not t['q'].requires_grad
        qkl = (t['q']*(t['logq']-lp)).sum()
        at = t['target']; other=lp.clone(); other[at]=-torch.inf
        competitor = int(other.detach().argmax()) if len(lp)>1 else None
        gap = lp[at]-lp[competitor] if competitor is not None else lp.new_zeros(())
        hinge = (cfg['margin']-gap).clamp_min(0) if competitor is not None else lp.new_zeros(())
        keep = (t['logp0'].exp()*(t['logp0']-lp)).sum()
        components.append(torch.stack([qkl,hinge,keep]))
        rows.append(dict(soft_kl=float(qkl.detach()),hinge=float(hinge.detach()),
            kl=float(keep.detach()),gap=float(gap.detach()),target=at,competitor=competitor,
            posterior_sum=float(lp.exp().sum().detach()),teacher_sum=float(t['q'].sum())))
    qkl,hinge,keep = torch.stack(components).mean(0)
    loss = qkl+cfg['gamma']*hinge+cfg['beta']*keep
    return loss,dict(soft_kl=float(qkl.detach()),hinge=float(hinge.detach()),
        kl=float(keep.detach()),gamma=cfg['gamma'],offsets=rows)


def fit_soft(head, records, ids, teacher, config, *, lr, steps=5, hard_equivalence=False):
    """F44 optimizer/state semantics; only NLL -> KL(Q||P) in Phase B.

    hard_equivalence is a test-only positive control, never a new method arm.
    """
    objective = hard_objective if hard_equivalence else soft_objective
    head.restore(head.initial); params=[p for _,p in head.named]
    opt=torch.optim.AdamW(params,lr=lr,betas=(.9,.999),eps=1e-4,weight_decay=0.)
    with torch.no_grad(): zero=head.values()
    evidence=soft_evidence(zero['logits'],teacher,records,config)
    teacher_hash=state_hash({str(i):t['raw_logits'] for i,t in enumerate(teacher)})
    path=[];best_loss=math.inf;best_step=0;saved=head.state();backwards=0;started=time.perf_counter()
    for step in range(steps+1):
        opt.zero_grad(set_to_none=True)
        value=head.values();loss,parts=objective(value['logits'],evidence)
        lv=float(loss.detach());assert math.isfinite(lv)
        assert torch.equal(value['boxes'],zero['boxes'])
        state=head.state()
        if lv<best_loss-1e-12:best_loss,best_step,saved=lv,step,state
        row=dict(step=step,loss=lv,parts=parts,state_sha256=state_hash(state),
            state_delta=norm([p-head.initial[n] for n,p in head.named]),
            **prediction(value['logits'],value['boxes'],records,ids))
        path.append(row)
        if step==steps:break
        loss.backward();backwards+=1
        row['gradient_norm']=norm([p.grad for p in params])
        row['unused_parameters']=[n for n,p in head.named if p.grad is None]
        assert not row['unused_parameters'] and all(torch.isfinite(p.grad).all() for p in params)
        before=head.state();opt_before=copy.deepcopy(opt.state_dict());opt.step();after=head.state()
        trials=[];accepted=False
        for scale in (1.,.5,.25,.125):
            head.restore({n:v+scale*(after[n]-v) for n,v in before.items()})
            with torch.no_grad(): pl,_=objective(head.values()['logits'],evidence)
            pv=float(pl);trials.append(dict(alpha=scale,loss=pv if math.isfinite(pv) else None))
            if math.isfinite(pv) and pv<lv-1e-9:accepted=True;break
        row.update(accepted=accepted,trials=trials,optimizer_restored_on_reject=not accepted,
                   proposal_norm=norm([after[n]-before[n] for n in before]))
        if not accepted:
            head.restore(before);opt.load_state_dict(opt_before)
            actual=opt.state_dict()
            assert actual['param_groups']==opt_before['param_groups']
            for key,group in opt_before['state'].items():
                assert all(torch.equal(actual['state'][key][n],x) if torch.is_tensor(x)
                           else actual['state'][key][n]==x for n,x in group.items())
    head.restore(saved)
    with torch.no_grad():end=head.values()
    assert all(torch.equal(a.cpu(),b) for a,b in zip(end['logits'],path[best_step]['logits']))
    assert torch.equal(end['boxes'],zero['boxes'])
    assert teacher_hash==state_hash({str(i):t['raw_logits'] for i,t in enumerate(teacher)})
    head.restore(head.initial)
    assert all(torch.equal(p,head.initial[n]) and p.grad is None for n,p in head.named)
    return dict(path=path,state=move(saved,'cpu'),initial_state=move(head.initial,'cpu'),
        final=path[best_step],evidence=move(evidence,'cpu'),best_step=best_step,
        state_delta=path[best_step]['state_delta'],lr=lr,steps=steps,config=evidence['config'],
        backwards=backwards,seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in params),
        hard_equivalence=hard_equivalence,parameter_TTA=True,teacher_frozen=True,
        source_restored=True,spatial_invariant=True,GT_online=False)
