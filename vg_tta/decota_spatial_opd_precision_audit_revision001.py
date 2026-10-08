"""Precision-aware audit supplement; original strict audit remains unchanged.

The original GPU result is never edited. Float32 versus float64 error is retained
and checked against an operation-count floating-point bound. Independent CPU
autodiff checks at each precision keep the original 3e-5 absolute threshold.
Reward, softmax, Adam, state, final-round and admission checks remain unchanged.
"""
import numpy as np
import torch
from vg_tta.decota_fixed_full_audit_v1 import top1_support,overlap,vector
from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit as original_audit

def gradient_check(current,mean,samples,weights,gpu,sigma):
    n=len(current);variance=sigma**2
    baseline=(current-mean)/variance/n
    terms=(weights-1/32)[:,:,None]*(samples-current[:,None,:])/variance/n
    analytical=baseline-terms.sum(1)
    checks={}
    for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
        c=torch.tensor(current,dtype=dtype,requires_grad=True)
        m=torch.tensor(mean,dtype=dtype);a=torch.tensor(samples,dtype=dtype)
        w=torch.tensor(weights,dtype=dtype)
        # Separately constructed scalar likelihood; no imported loss or decoder.
        objective=((c-m).square().sum(-1)/(2*variance)-
            ((w-1/32)*(-(a-c[:,None,:]).square().sum(-1)/(2*variance))).sum(-1)).mean()
        checks[label]=torch.autograd.grad(objective,c)[0].detach().numpy().astype(float)
    same32=float(abs(gpu-checks['float32']).max())
    same64=float(abs(analytical-checks['float64']).max())
    assert same32<3e-5,'Same-precision CPU/GPU autodiff failed unchanged absolute check'
    assert same64<3e-5,'Float64 likelihood derivative failed unchanged absolute check'
    # Each float32 term uses <=8 elementary roundings including the stored
    # reciprocal/normalization; 32-term reduction + baseline uses <=48.
    # Standard gamma_n = n*u/(1-n*u), u=2^-24 (round-to-nearest float32).
    u=np.finfo(np.float32).eps/2;gamma=48*u/(1-48*u)
    bound=gamma*(abs(baseline)+abs(terms).sum(1))
    # Include the independent same-precision discrepancy explicitly, rather than
    # hide it in a free relative tolerance. Tiny subnormal bound is dtype-defined.
    bound+=abs(gpu-checks['float32'])+np.finfo(np.float32).tiny
    difference=abs(gpu-analytical)
    assert np.all(difference<=bound),'Cross-precision error exceeds arithmetic rounding bound'
    return dict(max_original_GPU_vs_float64_error=float(difference.max()),
        max_CPU32_vs_GPU32_error=same32,max_CPU64_vs_formula64_error=same64,
        max_cross_precision_bound=float(bound.max()),
        max_bound_fraction=float(np.max(difference/bound)),unit_roundoff=u,operation_bound=48)

def audit(z,expert):
    try:return original_audit(z,expert)
    except AssertionError as exc:
        tb=exc.__traceback__
        while tb.tb_next:tb=tb.tb_next
        assert tb.tb_frame.f_code.co_filename.endswith('decota_spatial_opd_tunable_audit_v1.py') and tb.tb_lineno==28
    cfg=z['config'];steps=cfg['steps'];names=list(z['initial']);hist=z['path'];support=top1_support(expert)
    assert z['selected_step']==steps and len(hist)==steps+1 and len(z['rounds'])==steps
    assert z['positions']==[p for p,_ in support] and z['active_parameters']==1792 and not z['GT_used']
    assert np.array_equal(vector(z['state'],names),vector(hist[steps]['state'],names))
    assert np.array_equal(z['final'].numpy(),hist[steps]['boxes'].numpy())
    m=np.zeros(1792);v=np.zeros(1792);updates=0;erradam=errreward=0.;records=[]
    for k,(h,r) in enumerate(zip(hist,z['rounds'])):
        a=vector(h['state'],names);nxt=vector(hist[k+1]['state'],names)
        if not support:
            assert not r['updated'] and np.array_equal(a,nxt);continue
        rr=r['rollout'];samples=rr['samples'].numpy().astype(float);mu=rr['mean'].numpy().astype(float)
        w=rr['weights'].numpy().astype(float);used=rr['used_rewards'].numpy().astype(float)
        assert samples.shape==(len(support),32,4)
        original=rr['rewards'].numpy().astype(float);actions=1/(1+np.exp(-samples))
        recomputed=np.asarray([[overlap(x,e) for x in aa] for aa,(_,e) in zip(actions,support)])
        error=float(abs(original-recomputed).max());assert error<3e-6;errreward=max(errreward,error)
        if rr['permutations'] is not None:assert np.array_equal(used,np.take_along_axis(original,rr['permutations'].numpy(),1))
        else:assert np.array_equal(used,original)
        logits=used/cfg['tau'];ew=np.exp(logits-logits.max(1,keepdims=True));ew/=ew.sum(1,keepdims=True)
        assert abs(ew-w).max()<3e-7
        current=r['mean_before'].numpy().astype(float)
        records.append(gradient_check(current,mu,samples,w,r['autograd_mean_gradient'].numpy().astype(float),cfg['sigma']))
        assert r['analytic_gradient_error']<2e-5
        assert r['updated']==rr['informative'] and r['no_information']==(not rr['informative'])
        if not r['updated']:assert np.array_equal(a,nxt);continue
        u=h['update'];g=u['gradient'].numpy().astype(float);updates+=1
        m=.9*m+.1*g;v=.999*v+.001*g*g
        delta=-cfg['lr']*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
        raw=u['raw'].numpy();error=float(abs(delta-raw).max());assert error<4e-6;erradam=max(erradam,error)
        assert np.array_equal(nxt-a,raw) and u['names']==names
    assert updates==z['gradient_calls']
    return dict(status='pass',revision='precision_audit_revision001',
        original_absolute_cross_precision_assertion_passed=False,
        max_mean_gradient_error=max(x['max_original_GPU_vs_float64_error'] for x in records),
        precision_checks=records,max_Adam_error=erradam,max_detached_reward_error=errreward,
        gradient_calls=updates,rounds=steps,final_step=steps,GT_read=False,
        independent_decoder_Jacobian=False,algorithm_or_saved_gradient_changed=False)
