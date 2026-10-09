"""P2-only same-call GPU action readback; production actions are returned intact."""
import sys,time
import numpy as np
import torch

FIELD='actual_rollout_action_readback'
REVISION='P2_actual_action_reward_precision_revision005'
THRESHOLD=3e-6


class Installation:
    def __init__(self):
        import vg_tta.decota_spatial_opd_tunable_v1 as core
        from vg_tta.decota_spatial_opd_inline_gradient_precision004 import Installation as Inline
        self.core=core;self.Inline=Inline;self.old_action=core.action_boxes;self.old_fit=Inline.fit;self.contexts=[]
        owner=self
        def action(z):
            value=owner.old_action(z);frame=sys._getframe(1)
            if frame.f_code.co_name=='rollout' and frame.f_code.co_filename.endswith('decota_spatial_opd_tunable_v1.py'):
                assert owner.contexts and not value.requires_grad
                tick=time.perf_counter();owner.contexts[-1]['trace'].append(dict(round=frame.f_locals['round_index'],
                    samples=z.detach().cpu().clone(),actions=value.detach().cpu().clone(),
                    same_original_GPU_sigmoid_call=True,original_action_returned_without_modification=True))
                owner.contexts[-1]['CPU_seconds']+=time.perf_counter()-tick
            return value
        def fit(inst,*args,**kwargs):
            context=dict(trace=[],CPU_seconds=0.);owner.contexts.append(context)
            try:
                result=owner.old_fit(inst,*args,**kwargs)
                result[FIELD]=dict(revision=REVISION,trace=context['trace'],CPU_action_readback_seconds=context['CPU_seconds'],
                    original_actions_rewards_weights_loss_gradients_Adam_and_readout_unchanged=True)
                return result
            finally:owner.contexts.pop()
        core.action_boxes=action;Inline.fit=fit

    def uninstall(self):
        self.core.action_boxes=self.old_action;self.Inline.fit=self.old_fit


def geometry32(actions,evidence):
    a=np.asarray(actions,np.float32);e=np.asarray(evidence,np.float32)[:,None,:]
    pl,ph=a[...,:2]-a[...,2:]*np.float32(.5),a[...,:2]+a[...,2:]*np.float32(.5)
    el,eh=e[...,:2]-e[...,2:]*np.float32(.5),e[...,:2]+e[...,2:]*np.float32(.5)
    inter=np.maximum(np.minimum(ph,eh)-np.maximum(pl,el),np.float32(0)).prod(-1,dtype=np.float32)
    pa=np.maximum(ph-pl,np.float32(0)).prod(-1,dtype=np.float32)
    ea=np.maximum(eh-el,np.float32(0)).prod(-1,dtype=np.float32)
    return inter/np.maximum(pa+ea-inter,np.float32(1e-7))


def check(samples,evidence,saved,actions):
    from vg_tta.decota_fixed_full_audit_v1 import overlap
    from vg_tta.decota_spatial_opd_reward_audit_revision005 import interval,corners,product,positive,sub,minimum,maximum,add,rounded
    x,e,r,a=[np.asarray(z,float) for z in (samples,evidence,saved,actions)]
    assert x.ndim==3 and x.shape[-2:]==(32,4) and a.shape==x.shape
    assert e.shape==(len(x),4) and r.shape==x.shape[:-1]
    assert all(np.isfinite(v).all() and np.array_equal(v,v.astype(np.float32).astype(float)) for v in [x,e,r,a])
    assert ((a>=0)&(a<=1)).all() and (e[:,2:]>0).all()
    q=np.exp(-abs(x));a64=np.where(x>=0,1/(1+q),q/(1+q))
    readout=abs(a-a64)
    assert np.all(readout<=2*np.finfo(np.float32).eps),'Recorded GPU action failed independent sigmoid readout bound'
    same=abs(r-geometry32(a,e).astype(float))
    assert float(same.max())<THRESHOLD,'Independent CPU32 geometry on original GPU actions fails unchanged 3e-6'
    exact=np.asarray([[overlap(ai,ei) for ai in aa] for aa,ei in zip(a64,e)])
    aCPU=torch.sigmoid(torch.tensor(x,dtype=torch.float32)).numpy().astype(float)
    original_cpu=abs(r-geometry32(aCPU,e).astype(float))
    ai=interval(np.minimum(a,a64),np.maximum(a,a64));ec=e[:,None,:]
    p0,p1=corners(ai);e0,e1=corners(interval(ec))
    area_i=product(positive(sub(minimum(p1,e1),maximum(p0,e0))))
    area_p,area_e=product(positive(sub(p1,p0))),product(positive(sub(e1,e0)))
    den=maximum(sub(add(area_p,area_e),area_i),interval(float(np.float32(1e-7))))
    assert (den[0]>0).all()
    low,high=rounded(area_i[0]/den[1],area_i[1]/den[0])
    ref=geometry32(a,e).astype(float);assert np.all(ref>=low) and np.all(ref<=high)
    geometry_bound=np.maximum(abs(low-exact),abs(high-exact))
    bound=geometry_bound+same+32*np.finfo(float).eps
    mixed=abs(r-exact);assert np.all(mixed<=bound)
    return dict(revision=REVISION,max_original_GPU32_vs_float64_error=float(mixed.max()),
        original_absolute_3e_minus6_passed=bool(mixed.max()<THRESHOLD),
        max_original_CPU32_sigmoid_geometry_vs_GPU32_error=float(original_cpu.max()),
        original_CPU32_sigmoid_geometry_absolute_3e_minus6_passed=bool(original_cpu.max()<THRESHOLD),
        max_independent_CPU32_original_action_geometry_vs_GPU32_error=float(same.max()),
        matched_precision_absolute_threshold=THRESHOLD,
        max_actual_GPU_sigmoid_vs_float64_error=float(readout.max()),
        native_action_sigmoid_readout_absolute_bound=2*float(np.finfo(np.float32).eps),
        max_measured_CPU32_sigmoid_vs_actual_GPU_sigmoid_error=float(abs(aCPU-a).max()),
        max_float32_geometry_interval_bound=float(geometry_bound.max()),max_cross_precision_bound=float(bound.max()),
        max_cross_precision_bound_fraction=float((mixed/bound).max()),
        CUDA_transcendental_kernel_not_independently_proved=True,
        original_GPU_action_reward_weight_and_production_operations_unchanged=True)


def comparable(fit,keep_audit=False):
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import comparable as prior
    z=prior({k:v for k,v in fit.items() if k!=FIELD},keep_audit)
    if keep_audit and FIELD in fit:z[FIELD]={k:v for k,v in fit[FIELD].items() if k!='CPU_action_readback_seconds'}
    return z


def audit(fit,expert,original_component=None):
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import audit as previous
    import vg_tta.decota_spatial_opd_reward_audit_revision005 as reward
    assert fit[FIELD]['revision']==REVISION
    trace=fit[FIELD]['trace'];support=fit['positions'];steps=fit['config']['steps']
    expected=([0] if fit['arm']=='frozen_rollout' else list(range(steps))) if support else []
    assert [r['round'] for r in trace]==expected
    old=reward.reward_check;at=0
    def checked(samples,evidence,saved):
        nonlocal at
        r=trace[0 if fit['arm']=='frozen_rollout' else at]
        assert r['same_original_GPU_sigmoid_call'] and r['original_action_returned_without_modification']
        assert np.array_equal(r['samples'].numpy().astype(float),samples)
        result=check(samples,evidence,saved,r['actions'].numpy());at+=1;return result
    reward.reward_check=checked
    try:result=previous(fit,expert,original_component)
    finally:reward.reward_check=old
    assert at==(steps if support else 0)
    return dict(result,revision=REVISION,source_inline_gradient_revision=result['revision'],
        actual_rollout_action_readback_calls=len(trace),original_sigmoid_CPU32_reward_failures=sum(not r['original_CPU32_sigmoid_geometry_absolute_3e_minus6_passed'] for r in result['reward_precision_checks']),
        no_fitting_action_reward_or_scientific_configuration_changed=True)
