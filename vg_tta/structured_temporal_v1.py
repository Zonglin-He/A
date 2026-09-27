"""F43: frozen dense evidence -> legal interval -> actual temporal-head TTA.

No GT, task metrics, spatial optimization or fallback by ground truth. The
single teacher is final pred_actioness, not TTS. All reductions are FP64;
the replayed official network is FP32. Each saved prediction is a real state.
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

DEFAULT = dict(temperature=1., epsilon=1e-6, prior_weight=1., margin=.2,
               gamma=1., beta=.1, native_delta=2.)
SELECTORS = ('loss', 'guard_loss', 'decision', 'guard_decision')


def calibrate(raw, temperature=1., epsilon=1e-6, enabled=True):
    x = raw.detach().clone().reshape(-1).double()
    assert torch.isfinite(x).all() and temperature > 0 and epsilon > 0
    center = torch.quantile(x, .5)
    mad = torch.quantile((x-center).abs(), .5)
    z = (x-center)/(mad+epsilon)/temperature if enabled else x
    # Stable log probabilities even when MAD=0 and some outliers are extreme.
    return dict(raw_logits=x, standardized_logits=z, log_a=F.logsigmoid(z),
                log_not_a=F.logsigmoid(-z), a=z.sigmoid(), median=float(center),
                mad=float(mad), mad_zero=bool(mad == 0), calibrated=enabled)


def project(zero_logits, teacher, records, *, calibrated=True, config=None):
    cfg = {**DEFAULT, **(config or {})}
    result = []
    for z, t, record in zip(zero_logits, teacher, records):
        lp, ij = legal_logp(z)
        cal = calibrate(t['raw_logits'], cfg['temperature'], cfg['epsilon'], calibrated)
        ids = record['frame_ids']; edges, w = time_cells(ids, device=lp.device)
        assert len(cal['a']) == len(ids) and len(ids) == z.shape[1]
        assert bool(t['valid_mask'].all()) and torch.equal(t['omega'].to(w), w)
        # Inclusive end *index*; the corresponding physical interval is [f_s,f_e+1).
        # D(I) = -sum w [m_I log(a) + (1-m_I) log(1-a)].
        q = w*(cal['log_a']-cal['log_not_a'])
        prefix = torch.cat((q.new_zeros(1), q.cumsum(0)))
        cost = -(w*cal['log_not_a']).sum()-(prefix[ij[1]+1]-prefix[ij[0]])
        score = -cost+cfg['prior_weight']*lp.detach()
        order = torch.argsort(score, descending=True, stable=True)
        top = int(order[0]); runner = int(order[1]) if len(order)>1 else top
        ni = native_view_indices(z)
        native = int(((ij[0] == ni[0]) & (ij[1] == ni[1])).nonzero()[0])
        s, e = (int(v) for v in ij[:, top])
        result.append(dict(**cal, omega=w, cell_edges=edges, frame_ids=list(ids),
            logp0=lp.detach().clone(), ij=ij, cost=cost.detach(), score=score.detach(),
            target=top, target_indices=[s,e], interval=[ids[s],ids[e]+1],
            native=native, native_indices=list(ni), native_interval=[ids[ni[0]],ids[ni[1]]+1],
            projection_margin=float(score[top]-score[runner]), single_candidate=len(order)==1,
            native_drop=float(lp[native]-lp[top]), valid_positions=len(ids)))
    assert len(result) == 2
    a,b = [r['interval'] for r in result]
    overlap = max(0,min(a[1],b[1])-max(a[0],b[0]))
    overlap /= max(a[1],b[1])-min(a[0],b[0])
    rho = overlap
    for r in result:rho *= 1/(1+math.exp(-r['projection_margin']))
    return dict(offsets=result, rho=rho, offset_iou=overlap, config=cfg,
                teacher_frozen=True, GT_online=False)


def projected_indices(evidence, ids):
    intervals=[r['interval'] for r in evidence['offsets']]
    return [ids.index(min(i[0] for i in intervals)), ids.index(max(i[1] for i in intervals)-1)]


def objective(logits, evidence, *, reliability=False, gamma=None):
    cfg=evidence['config'];gamma=cfg['gamma'] if gamma is None else gamma
    parts=[];nlls=[];hinges=[];kls=[]
    for z,t in zip(logits,evidence['offsets']):
        lp,ij=legal_logp(z);target=t['target']
        assert torch.equal(ij,t['ij']) and not t['logp0'].requires_grad
        nll=-lp[target]
        other=lp.clone();other[target]=-torch.inf
        if len(lp)>1:
            competitor=int(other.detach().argmax())
            gap=lp[target]-lp[competitor]
            hinge=(cfg['margin']-gap).clamp_min(0)
        else:
            competitor=None;gap=lp.new_zeros(());hinge=lp.new_zeros(())
        kl=(t['logp0'].exp()*(t['logp0']-lp)).sum()
        nlls.append(nll);hinges.append(hinge);kls.append(kl)
        parts.append(dict(nll=float(nll.detach()), hinge=float(hinge.detach()),
            kl=float(kl.detach()), competitor=competitor, target=target,
            gap=float(gap.detach()), posterior_sum=float(lp.exp().sum().detach())))
    nll=torch.stack(nlls).mean();hinge=torch.stack(hinges).mean();kl=torch.stack(kls).mean()
    weight=evidence['rho'] if reliability else 1.
    loss=weight*(nll+gamma*hinge)+cfg['beta']*kl
    return loss,dict(nll=float(nll.detach()),hinge=float(hinge.detach()),kl=float(kl.detach()),
                     reliability=weight,gamma=gamma,offsets=parts)


def decision(logits,evidence):
    scores=[];drops=[];raw=[]
    for z,t in zip(logits,evidence['offsets']):
        pair=native_view_indices(z);ij=t['ij']
        at=int(((ij[0]==pair[0]) & (ij[1]==pair[1])).nonzero()[0])
        scores.append(float(t['score'][at]))
        drops.append(float(t['logp0'][t['native']]-t['logp0'][at]))
        raw.append(list(pair))
    return dict(decision_score=sum(scores)/len(scores),native_drops=drops,
                guard=all(v<=evidence['config']['native_delta']+1e-12 for v in drops),
                raw_offset_indices=raw)


def prefer(row,previous,selector):
    if selector.startswith('guard_') and not row['guard']:return False
    if previous is None:return True
    # Fixed numerical tie tolerance; keep the earliest real state, including phi0.
    if selector.endswith('decision'):
        return row['decision_score']>previous['decision_score']+1e-12
    return row['loss']<previous['loss']-1e-12


def fit(it, records, ids, teacher, *, lr=.001, budgets=(5,), reliability=False,
        gamma=None, config=None):
    """One actual trajectory; four predeclared state selectors share it.

    Proposal acceptance only uses the training objective, with exact optimizer
    rollback on rejection. The native guard filters saved output states, not
    the teacher or the set of trainable proposals. No gradient clipping.
    """
    budgets=sorted(set(int(x) for x in budgets));assert budgets and budgets[0]>=0
    it.restore(it.initial);params=[p for _,p in it.named]
    opt=torch.optim.AdamW(params,lr=lr,betas=(.9,.999),eps=1e-4,weight_decay=0.)
    with torch.no_grad():zero=it.values()
    evidence=project(zero['logits'],teacher,records,config=config)
    teacher_hash=state_hash({str(i):t['raw_logits'] for i,t in enumerate(teacher)})
    best={k:None for k in SELECTORS};best_states={};states={};prefixes={};path=[]
    backwards=0;start=time.perf_counter()
    for step in range(budgets[-1]+1):
        opt.zero_grad(set_to_none=True)
        value=it.values();loss,parts=objective(value['logits'],evidence,reliability=reliability,gamma=gamma)
        lv=float(loss.detach());assert math.isfinite(lv)
        assert torch.equal(value['boxes'],zero['boxes'])
        st=it.state();h=state_hash(st)
        row=dict(step=step,loss=lv,parts=parts,**decision(value['logits'],evidence),
            logits=[z.detach().cpu().clone() for z in value['logits']],state_sha256=h,
            indices=list(fitted_merge(value['logits'],records,ids)),
            state_delta=norm([p-it.initial[n] for n,p in it.named]))
        row['physical_interval']=[ids[row['indices'][0]],ids[row['indices'][1]]+1]
        row['raw_physical_intervals']=[[r['frame_ids'][s],r['frame_ids'][e]+1]
                                    for r,(s,e) in zip(records,row['raw_offset_indices'])]
        path.append(row)
        for selector in SELECTORS:
            if prefer(row,best[selector],selector):
                best[selector]=row;best_states[selector]=st
        if step in budgets:
            choices={}
            for selector in SELECTORS:
                b=best[selector];sh=b['state_sha256']
                if sh not in states:states[sh]=move(best_states[selector],'cpu')
                choices[selector]=dict(best_step=b['step'],state_sha256=sh)
            prefixes[str(step)]=choices
        if step==budgets[-1]:break
        loss.backward();backwards+=1
        row['gradient_norm']=norm([p.grad for p in params])
        row['unused_parameters']=[n for n,p in it.named if p.grad is None]
        assert not row['unused_parameters']
        assert all(bool(torch.isfinite(p.grad).all()) for p in params)
        before=it.state();opt_before=copy.deepcopy(opt.state_dict());opt.step();after=it.state()
        trials=[];accepted=False
        for alpha in (1.,.5,.25,.125):
            it.restore({n:v+alpha*(after[n]-v) for n,v in before.items()})
            with torch.no_grad():pl,_=objective(it.values()['logits'],evidence,reliability=reliability,gamma=gamma)
            pv=float(pl);trials.append(dict(alpha=alpha,loss=pv if math.isfinite(pv) else None))
            if math.isfinite(pv) and pv<lv-1e-9:accepted=True;break
        row.update(accepted=accepted,trials=trials,optimizer_restored_on_reject=not accepted,
                   proposal_norm=norm([after[n]-before[n] for n in before]))
        if not accepted:
            it.restore(before);opt.load_state_dict(opt_before)
            restored=opt.state_dict();assert restored['param_groups']==opt_before['param_groups']
            for k,v in opt_before['state'].items():
                assert all(torch.equal(restored['state'][k][n],x) if torch.is_tensor(x)
                           else restored['state'][k][n]==x for n,x in v.items())
    # Every retained prefix/selector is replayed through the real head.
    for choices in prefixes.values():
        for choice in choices.values():
            it.restore(states[choice['state_sha256']])
            with torch.no_grad():end=it.values()
            r=path[choice['best_step']]
            assert all(torch.equal(x.cpu(),y) for x,y in zip(end['logits'],r['logits']))
            assert torch.equal(end['boxes'],zero['boxes'])
    assert teacher_hash==state_hash({str(i):t['raw_logits'] for i,t in enumerate(teacher)})
    it.restore(it.initial)
    assert all(torch.equal(p,it.initial[n]) for n,p in it.named)
    return dict(path=path,states=states,prefixes=prefixes,evidence=move(evidence,'cpu'),
        initial_state=move(it.initial,'cpu'),budgets=budgets,lr=lr,reliability=reliability,
        gamma=evidence['config']['gamma'] if gamma is None else gamma,backwards=backwards,
        parameters=sum(p.numel() for p in params),seconds=time.perf_counter()-start,
        GT_online=False,teacher_frozen=True,spatial_invariant=True,source_restored=True)


def selected_fit(trajectory,budget,boxes,selector='guard_decision'):
    choice=trajectory['prefixes'][str(budget)][selector]
    row=trajectory['path'][choice['best_step']]
    return dict(state=trajectory['states'][choice['state_sha256']],
        final={**row,'boxes':boxes},best_step=choice['best_step'],steps=budget,
        state_delta=row['state_delta'],parameters=trajectory['parameters'],selector=selector,
        parameter_TTA=True,GT_online=False)
