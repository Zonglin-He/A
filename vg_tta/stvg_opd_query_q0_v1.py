"""Frozen-state, finite Query/LN direction probes; no optimizer or GT."""
import hashlib, math
import numpy as np
import torch
from scripts.stvg_opd_query_q0_common_v1 import GROUPS, QUERY, canonical, spectrum, weights


def rms(a, b):
    return float(torch.sqrt(torch.mean((a.double()-b.double())**2)))


def complete_state(base):
    return {n:p.detach().cpu().clone() for n,p in base.named}


def same_state(base, expected):
    assert all(torch.equal(p.detach().cpu(), expected[n]) for n,p in base.named)


def jacobian(base, initial, positions):
    base.restore(initial)
    if not positions:
        return dict(Query=np.empty((0,256)), LN=np.empty((0,1536)), actual_scalar_VJPs=0)
    output=base.values()['boxes'][positions].flatten(); params=[p for n,p in base.named]
    rows=[]
    for i in range(len(output)):
        gs=torch.autograd.grad(output[i],params,retain_graph=i+1<len(output),allow_unused=False)
        rows.append(torch.cat([g.flatten() for g in gs]).detach().cpu().numpy().astype(float))
    j=np.stack(rows);assert j.shape==(4*len(positions),1792) and np.isfinite(j).all()
    same_state(base,initial)
    return dict(Query=j[:,:256],LN=j[:,256:],actual_scalar_VJPs=len(rows))


def episode(base, old, expert, dataset, key, budget):
    from vg_tta.decota_fixed_full_audit_v1 import top1_support
    from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
    initial=old['fit']['initial']; before=old['fit']['before']
    named=list(base.named);names=[n for n,p in named];assert names[0]==QUERY
    assert sum(p.numel() for n,p in named)==1792 and not old['fit']['GT_used']
    support=top1_support(expert);positions=[p for p,e in support]
    assert positions==old['fit']['positions'] and len(positions)<=4
    target=torch.tensor(np.asarray([e for p,e in support]),device=base.delta.device,dtype=torch.float32).reshape(-1,4)
    base.restore(initial)
    with torch.no_grad(): baseline=base.values()['boxes'].detach().cpu()
    assert torch.equal(baseline,before)
    j=jacobian(base,initial,positions)
    sq,sl=spectrum(j['Query']),spectrum(j['LN']);k=min(8,sq['rank'],sl['rank'])
    seed=int(hashlib.sha256(('q0-fixed-random|'+dataset+'|'+key).encode()).hexdigest()[:15],16)
    random=np.linalg.qr(np.random.default_rng(seed).standard_normal((256,max(1,k))),mode='reduced')[0].T[:k]
    directions={'Q_random':canonical(random),'Q_observable':sq['directions'][:k], 'LN_observable':sl['directions'][:k]}
    forwards=0

    def decode(delta):
        nonlocal forwards
        base.restore(initial)
        with torch.no_grad():
            offset=0
            for name,p in named:
                v=torch.as_tensor(delta[offset:offset+p.numel()],device=p.device,dtype=p.dtype).reshape(p.shape)
                p.copy_(initial[name].to(p)+v);offset+=p.numel()
            boxes=base.values()['boxes'].detach().cpu();forwards+=1
        assert torch.isfinite(boxes).all()
        base.restore(initial);same_state(base,initial)
        return boxes

    def reward(boxes):
        if not positions:return 0.
        with torch.no_grad():return float(aligned_iou(boxes[positions].to(target),target).mean())

    def calibrate(direction):
        trials=[]
        def probe(a):
            boxes=decode(direction*a);d=rms(boxes,baseline)
            trials.append(dict(amplitude=float(a),output_RMS=d,boxes=boxes))
            return d,boxes
        if budget==0:
            return dict(delta=np.zeros(1792,dtype=np.float32),boxes=baseline.clone(),amplitude=0.,output_RMS=0.,budget_attained=True,probes=[])
        d,b=probe(.001)
        estimate=min(1.,max(1e-8,budget/d*.001)) if d>0 else 1.
        d,b=probe(estimate);lo=0.;hi=estimate
        if d<budget:
            lo=estimate
            for _ in range(10):
                if lo>=1.:break
                hi=min(1.,lo*2);d,b=probe(hi)
                if d>=budget:break
                lo=hi
            if d<budget:
                hi=None
        if hi is not None:
            for _ in range(12):
                mid=(lo+hi)/2;d,b=probe(mid)
                if d<=budget:lo=mid
                else:hi=mid
        eligible=[dict(amplitude=0.,output_RMS=0.,boxes=baseline)]+[t for t in trials if t['output_RMS']<=budget]
        chosen=min(eligible,key=lambda t:(budget-t['output_RMS'],t['amplitude']))
        delta=(direction*chosen['amplitude']).astype(np.float32)
        boxes=decode(delta);d=rms(boxes,baseline)
        assert d<=budget+1e-12
        return dict(delta=delta,boxes=boxes,amplitude=chosen['amplitude'],output_RMS=d,
                    budget_attained=budget-d<=max(1e-7,.01*budget),probes=trials)

    groups={}
    try:
        for name in GROUPS:
            candidates=[dict(index=0,direction=None,sign=0,amplitude=0.,delta=np.zeros(1792,dtype=np.float32),
                boxes=baseline.clone(),output_RMS=0.,budget_attained=True,probes=[])]
            for index,v in enumerate(directions[name]):
                full=np.zeros(1792,dtype=np.float32)
                if name.startswith('Q_'):full[:256]=v
                else:full[256:]=v
                for sign in [1,-1]:
                    c=calibrate(full*sign);c.update(index=len(candidates),direction=index,sign=sign)
                    assert np.count_nonzero(c['delta'][256:] if name.startswith('Q_') else c['delta'][:256])==0
                    candidates.append(c)
            rewards=[reward(c['boxes']) for c in candidates]
            w=weights(rewards,old['config']['tau'])
            delta=sum(w[i]*c['delta'].astype(float) for i,c in enumerate(candidates)).astype(np.float32)
            mean_boxes=decode(delta)
            box_average=sum(float(w[i])*c['boxes'].double() for i,c in enumerate(candidates))
            top=int(np.argmax(rewards));ties=sum(r==rewards[top] for r in rewards)
            groups[name]=dict(candidates=candidates,rewards=rewards,weights=w,
                expert_top1_index=top,expert_top_tie_count=ties,all_rewards_equal=all(r==rewards[0] for r in rewards),
                mean_delta=delta,mean_boxes=mean_boxes,mean_reward=reward(mean_boxes),
                box_average_for_Jensen_diagnostic=box_average,
                Jensen_RMS=rms(mean_boxes,box_average),mean_output_RMS=rms(mean_boxes,baseline),
                inactive_scope_exact=True,all_readouts_restored=True)
        finite_difference=[]
        for label,jj,direction in [('Query',j['Query'],sq['directions'][:1]),('LN',j['LN'],sl['directions'][:1])]:
            if not len(direction):continue
            full=np.zeros(1792,dtype=np.float32)
            if label=='Query':full[:256]=direction[0]
            else:full[256:]=direction[0]
            plus=decode(full*.001);minus=decode(-full*.001)
            fd=(plus[positions].double()-minus[positions].double()).numpy().ravel()/.002
            analytic=jj@direction[0];err=float(np.linalg.norm(fd-analytic));den=float(np.linalg.norm(analytic))
            finite_difference.append(dict(scope=label,step=.001,absolute_L2_error=err,relative_L2_error=err/den if den else None,
                analytic_directional_derivative=analytic,finite_difference=fd,plus=plus,minus=minus,
                descriptive_not_a_complete_decoder_Jacobian_proof=True))
        base.restore(initial);same_state(base,initial)
        energy={}
        for label,s in [('Query',sq),('LN',sl)]:
            values=np.asarray(s['singular_values']);den=float(np.sum(values**2))
            energy[label]=float(np.sum(values[:k]**2)/den) if den else None
        return dict(before=baseline,initial={n:v.clone() for n,v in initial.items()},
            restored=complete_state(base),positions=positions,groups=groups,jacobian=j,
            SVD=dict(Query=sq,LN=sl),directions=directions,common_k=k,spectral_energy=energy,
            random_seed=seed,output_RMS_budget=budget,tau=old['config']['tau'],finite_difference=finite_difference,
            actual_decoder_forwards=forwards,actual_new_DINO_calls=0,optimizer_steps=0,
            Native_WHEN_fixed=True,GT_read=False,no_candidate_writeback=True)
    finally:base.restore(initial)
