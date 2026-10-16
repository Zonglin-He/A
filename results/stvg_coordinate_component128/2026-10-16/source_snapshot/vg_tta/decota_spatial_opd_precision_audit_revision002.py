"""Audit CUDA float32 scalar-reciprocal softmax without changing the fitter.

PyTorch 2.7.0 commit 134179474539648ba7dee1317959529fbd0e7f89:
aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu:32-44 computes a*inv(b)
for a CPU scalar divisor, with both inv and multiplication in opmath float32.
The original mixed-precision 3e-7 assertion and its failure remain recorded.
Independent matched-operation checks retain 3e-7, and gamma_3 plus exact
softmax sensitivity intervals certify the discrepancy from exact float64.
"""
import numpy as np
import torch
from vg_tta.decota_fixed_full_audit_v1 import top1_support, overlap, vector
from vg_tta.decota_spatial_opd_precision_audit_revision001 import gradient_check, original_audit

def softmax_check(used, weights, tau):
    used=np.asarray(used,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    assert used.ndim==2 and used.shape[1]==32 and weights.shape==used.shape
    assert np.isfinite(used).all() and np.isfinite(weights).all() and tau>0
    assert np.all((weights>=0)&(weights<=1))
    assert np.array_equal(used,used.astype(np.float32).astype(np.float64))
    u=float(np.finfo(np.float32).eps/2);gamma3=3*u/(1-3*u)
    gamma32=32*u/(1-32*u)
    assert np.max(abs(weights.sum(1)-1))<=gamma32
    # Follow the pinned CUDA scalar-divisor kernel, separately from the fitter.
    tau32=np.float32(tau)
    inv32=np.float32(np.float32(1.0)/tau32)
    logits32=(used.astype(np.float32)*inv32).astype(np.float32)
    logits64=used/float(tau)
    difference=logits32.astype(np.float64)-logits64
    division_bound=gamma3*abs(logits64)+np.finfo(np.float32).tiny
    assert np.all(abs(difference)<=division_bound)
    exact=np.exp(logits64-logits64.max(1,keepdims=True));exact/=exact.sum(1,keepdims=True)
    rounded=logits32.astype(np.float64)
    reference=np.exp(rounded-rounded.max(1,keepdims=True));reference/=reference.sum(1,keepdims=True)
    cpu32=torch.softmax(torch.tensor(logits32,dtype=torch.float32),-1).numpy().astype(np.float64)
    same_error=float(abs(cpu32-weights).max())
    reference_error=float(abs(reference-weights).max())
    assert same_error<3e-7, 'Matched-kernel CPU32/GPU32 softmax fails unchanged 3e-7 check'
    assert reference_error<3e-7, 'Independent softmax of rounded CUDA logits fails unchanged 3e-7 check'
    # For delta=rounded-exact logits, the normalization ratio lies between
    # exp(min(delta)) and exp(max(delta)). These are exact componentwise bounds.
    lower=exact*np.exp(difference-difference.max(1,keepdims=True))
    upper=exact*np.exp(difference-difference.min(1,keepdims=True))
    assert np.all(reference>=lower-32*np.finfo(np.float64).eps)
    assert np.all(reference<=upper+32*np.finfo(np.float64).eps)
    sensitivity=np.maximum(exact-lower,upper-exact)
    bound=sensitivity+3e-7+32*np.finfo(np.float64).eps
    mixed=abs(weights-exact)
    assert np.all(mixed<=bound), 'Softmax cross-precision error exceeds derived sensitivity bound'
    return dict(max_original_GPU32_vs_float64_error=float(mixed.max()),
        original_absolute_3e_minus7_passed=bool(mixed.max()<3e-7),
        max_matched_kernel_CPU32_vs_GPU32_error=same_error,
        max_rounded_logits_float64_reference_error=reference_error,
        max_float32_logit_error=float(abs(difference).max()),
        max_float32_logit_gamma3_bound=float(division_bound.max()),
        max_cross_precision_sensitivity_bound=float(bound.max()),
        max_cross_precision_bound_fraction=float((mixed/bound).max()),
        unit_roundoff=u,division_operation_bound=3,
        matched_precision_absolute_threshold=3e-7,
        algorithm_or_stored_weights_changed=False)

def audit(z, expert):
    try:
        result=original_audit(z,expert)
        return dict(result,revision='precision_audit_revision002',
                    original_absolute_cross_precision_audit_passed=True,
                    algorithm_or_saved_gradient_changed=False)
    except AssertionError as exc:
        tb=exc.__traceback__
        while tb.tb_next:tb=tb.tb_next
        assert tb.tb_frame.f_code.co_filename.endswith('decota_spatial_opd_tunable_audit_v1.py')
        assert tb.tb_lineno in (25,28), 'Only original softmax/gradient cross-precision failures are supplemented'
        original_failed_line=tb.tb_lineno
    cfg=z['config'];steps=cfg['steps'];names=list(z['initial']);hist=z['path'];support=top1_support(expert)
    assert z['selected_step']==steps and len(hist)==steps+1 and len(z['rounds'])==steps
    assert z['positions']==[p for p,_ in support] and z['active_parameters']==1792 and not z['GT_used']
    assert np.array_equal(vector(z['state'],names),vector(hist[steps]['state'],names))
    assert np.array_equal(z['final'].numpy(),hist[steps]['boxes'].numpy())
    m=np.zeros(1792);v=np.zeros(1792);updates=0;erradam=errreward=0.;grad_records=[];weight_records=[]
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
        if rr['permutations'] is not None:
            assert np.array_equal(used,np.take_along_axis(original,rr['permutations'].numpy(),1))
        else:assert np.array_equal(used,original)
        weight_records.append(softmax_check(used,w,cfg['tau']))
        current=r['mean_before'].numpy().astype(float)
        grad_records.append(gradient_check(current,mu,samples,w,
                            r['autograd_mean_gradient'].numpy().astype(float),cfg['sigma']))
        assert r['analytic_gradient_error']<2e-5
        assert r['updated']==rr['informative'] and r['no_information']==(not rr['informative'])
        if not r['updated']:assert np.array_equal(a,nxt);continue
        update=h['update'];g=update['gradient'].numpy().astype(float);updates+=1
        m=.9*m+.1*g;v=.999*v+.001*g*g
        delta=-cfg['lr']*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
        raw=update['raw'].numpy();error=float(abs(delta-raw).max());assert error<4e-6;erradam=max(erradam,error)
        assert np.array_equal(nxt-a,raw) and update['names']==names
    assert updates==z['gradient_calls']
    return dict(status='pass',revision='precision_audit_revision002',
        original_absolute_cross_precision_audit_passed=False,original_failed_line=original_failed_line,
        max_mean_gradient_error=max(x['max_original_GPU_vs_float64_error'] for x in grad_records),
        precision_checks=grad_records,softmax_precision_checks=weight_records,
        max_Adam_error=erradam,max_detached_reward_error=errreward,
        gradient_calls=updates,rounds=steps,final_step=steps,GT_read=False,
        independent_decoder_Jacobian=False,algorithm_or_saved_gradient_changed=False,
        original_mixed_precision_softmax_failure_preserved=True)
