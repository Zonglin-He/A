"""P2-only precision audit for the original in-loop Gaussian derivative guard.

Only one assertion is supplemented in a process-local compiled copy. Original
likelihood, autograd gradient, optimizer, readout and source bytes stay intact.
The original 2e-5 failure is retained. Matched CPU32 autodiff keeps 2e-5, CPU64
autodiff checks the closed formula, and both float32 paths have gamma48 bounds.
"""
import ast,copy,time
import numpy as np
import torch
from vg_tta.stvg_opd_paper_component_audit_v1 import audit as _ORIGINAL_COMPONENT_AUDIT

REVISION='P2_inline_mean_gradient_precision_revision004'
FIELD='inline_mean_gradient_precision'
THRESHOLD=2e-5


@torch.enable_grad()
def check(current,mean,samples,weights,gpu,analytic32,sigma,original_error):
    c,m,s,w,g,a=[np.asarray(x,np.float64) for x in
                 (current,mean,samples,weights,gpu,analytic32)]
    n=len(c);v=float(sigma)**2
    assert n>0 and c.shape==m.shape==g.shape==a.shape==(n,4)
    assert s.shape==(n,32,4) and w.shape==(n,32) and v>0
    assert all(np.isfinite(x).all() for x in [c,m,s,w,g,a])
    assert all(np.array_equal(x,x.astype(np.float32).astype(float)) for x in [c,m,s,w,g,a])
    assert ((w>=0)&(w<=1)).all()
    u=float(np.finfo(np.float32).eps/2);gamma=48*u/(1-48*u)
    assert np.max(abs(w.sum(1)-1))<=32*u/(1-32*u)
    # CPU reconstruction uses the pinned CUDA CPU-scalar reciprocal-times-
    # multiplication path. It does not import the production likelihood.
    x=torch.tensor(c,dtype=torch.float32,requires_grad=True)
    mm=torch.tensor(m,dtype=torch.float32);ss=torch.tensor(s,dtype=torch.float32)
    ww=torch.tensor(w,dtype=torch.float32)
    inv2v=torch.tensor(np.float32(1)/np.float32(2*v))
    invn=torch.tensor(np.float32(1)/np.float32(n))
    kl=(x-mm).square().sum(-1)*inv2v
    logp=-(ss-x[:,None,:]).square().sum(-1)*inv2v
    objective=(kl-((ww-1/32)*logp).sum(-1)).sum()*invn
    cpu32=torch.autograd.grad(objective,x)[0].detach().numpy().astype(float)
    same=abs(g-cpu32)
    assert float(same.max())<THRESHOLD,'Matched CPU32/GPU32 gradient fails unchanged 2e-5'
    x64=torch.tensor(c,dtype=torch.float64,requires_grad=True)
    m64=torch.tensor(m,dtype=torch.float64);s64=torch.tensor(s,dtype=torch.float64)
    w64=torch.tensor(w,dtype=torch.float64)
    kl64=(x64-m64).square().sum(-1)/(2*v)
    logp64=-(s64-x64[:,None,:]).square().sum(-1)/(2*v)
    objective64=(kl64-((w64-1/32)*logp64).sum(-1)).mean()
    cpu64=torch.autograd.grad(objective64,x64)[0].detach().numpy()
    baseline=(c-m)/v/n
    terms=(w-1/32)[:,:,None]*(s-c[:,None,:])/v/n
    formula=baseline-terms.sum(1)
    same64=abs(cpu64-formula)
    assert float(same64.max())<THRESHOLD,'Independent CPU64 derivative/formula disagreement'
    # >=32 reductions plus <=16 elementary/subtraction/reciprocal/scale
    # roundings. Both branches are compared against the same float64 formula.
    scale=abs(baseline)+abs(terms).sum(1)
    arithmetic=gamma*scale+np.finfo(np.float32).tiny+64*np.finfo(float).eps*np.maximum(1,scale)
    bound_g=arithmetic+same
    assert np.all(abs(g-formula)<=bound_g),'Saved GPU derivative exceeds precision bound'
    assert np.all(abs(a-formula)<=arithmetic),'Original GPU analytic expression exceeds precision bound'
    error=float(abs(g-a).max())
    assert error==float(original_error),'Original guard discrepancy was rewritten'
    assert np.all(abs(g-a)<=bound_g+arithmetic)
    return dict(status='pass',original_inline_absolute_guard_passed=bool(error<THRESHOLD),
        original_inline_error=error,original_absolute_threshold=THRESHOLD,
        max_matched_CPU32_vs_GPU32_error=float(same.max()),matched_precision_absolute_threshold=THRESHOLD,
        max_CPU64_vs_formula64_error=float(same64.max()),
        max_GPU32_vs_formula64_error=float(abs(g-formula).max()),
        max_original_analytic32_vs_formula64_error=float(abs(a-formula).max()),
        max_GPU_precision_bound=float(bound_g.max()),max_analytic_precision_bound=float(arithmetic.max()),
        max_GPU_bound_fraction=float(np.max(abs(g-formula)/bound_g)),
        max_analytic_bound_fraction=float(np.max(abs(a-formula)/arithmetic)),
        unit_roundoff=u,operation_bound=48,production_loss_gradient_Adam_and_readout_unchanged=True)


class Installation:
    def __init__(self):
        import vg_tta.decota_spatial_opd_tunable_v1 as core
        self.core=core;self.old_fit=core.fit;self.contexts=[]
        from pathlib import Path
        path=Path(core.__file__);tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
        tests=[n for n in ast.walk(fn) if isinstance(n,ast.Assert) and n.lineno==100]
        assert len(tests)==1 and ast.unparse(tests[0].test)=='error < 2e-05'
        original_tree=copy.deepcopy(fn)
        tests[0].test=ast.Call(func=ast.Name(id='_inline004_guard',ctx=ast.Load()),
            args=[ast.Name(id=x,ctx=ast.Load()) for x in ['mu','rr','gmu','analytic','error','cfg','k']],keywords=[])
        self.original_AST=ast.dump(original_tree,include_attributes=True)
        restored=copy.deepcopy(fn)
        changed=next(n for n in ast.walk(restored) if isinstance(n,ast.Assert) and n.lineno==100)
        changed.test=copy.deepcopy(next(n for n in ast.walk(original_tree) if isinstance(n,ast.Assert) and n.lineno==100).test)
        assert ast.dump(restored,include_attributes=False)==ast.dump(original_tree,include_attributes=False)
        self.only_original_assertion_test_supplemented=True
        mod=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        assert '_inline004_guard' not in core.__dict__
        core.__dict__['_inline004_guard']=self.guard
        exec(compile(mod,str(path),'exec'),core.__dict__)
        self.compiled_fit=core.fit;core.fit=self.fit

    def guard(self,mu,rr,gmu,analytic,error,cfg,k):
        assert self.contexts
        tick=time.perf_counter()
        def array(t):return t.detach().cpu().numpy().astype(float)
        record=check(array(mu),array(rr['mean']),array(rr['samples']),array(rr['weights']),
                     array(gmu),array(analytic),cfg['sigma'],error)
        self.contexts[-1]['rounds'].append(dict(round=k,analytic32=analytic.detach().cpu().clone(),check=record))
        self.contexts[-1]['CPU_seconds']+=time.perf_counter()-tick
        return True

    def fit(self,*args,**kwargs):
        context=dict(rounds=[],CPU_seconds=0.);self.contexts.append(context)
        try:
            result=self.compiled_fit(*args,**kwargs)
            result[FIELD]=dict(revision=REVISION,rounds=context['rounds'],CPU_audit_seconds=context['CPU_seconds'],
                original_assertion_retained_as_evidence=True,no_production_computation_changed=True)
            return result
        finally:self.contexts.pop()

    def uninstall(self):
        self.core.fit=self.old_fit
        del self.core.__dict__['_inline004_guard']


def comparable(fit,include_new_checks=False):
    z={k:v for k,v in fit.items() if k!=FIELD}
    if include_new_checks and FIELD in fit:
        z[FIELD]={k:v for k,v in fit[FIELD].items() if k!='CPU_audit_seconds'}
    return z


def audit(fit,expert,original_component=None):
    from vg_tta.decota_fixed_full_audit_v1 import top1_support,vector
    from vg_tta.decota_spatial_opd_reward_audit_revision005 import reward_check
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import softmax_check
    from vg_tta.decota_spatial_opd_chart_revision003 import chart_audit
    if original_component is None:original_component=_ORIGINAL_COMPONENT_AUDIT
    assert not fit.get('direct_objective') and fit['active_parameters']==1792
    assert fit['arm'] in ['on_policy','frozen_rollout','shuffled_feedback']
    assert fit[FIELD]['revision']==REVISION
    passed=True;failed_line=None
    try:original_component(fit,expert)
    except AssertionError as exc:
        tb=exc.__traceback__
        while tb.tb_next:tb=tb.tb_next
        assert tb.tb_frame.f_code.co_filename.endswith('stvg_opd_paper_component_audit_v1.py')
        assert tb.tb_lineno in [36,40,43]
        passed=False;failed_line=tb.tb_lineno
    cfg=fit['config'];steps=cfg['steps'];names=list(fit['initial']);hist=fit['path'];support=top1_support(expert)
    assert fit['selected_step']==steps and len(hist)==steps+1 and len(fit['rounds'])==steps
    assert fit['positions']==[p for p,_ in support] and not fit['GT_used']
    assert np.array_equal(vector(fit['state'],names),vector(hist[steps]['state'],names))
    assert np.array_equal(fit['final'].numpy(),hist[steps]['boxes'].numpy())
    assert len(fit[FIELD]['rounds'])==(steps if support else 0)
    evidence=np.asarray([e for _,e in support],np.float32)
    m=np.zeros(1792);v=np.zeros(1792);updates=0;erradam=0.;gradients=[];rewards=[];weights=[]
    for k,(h,r) in enumerate(zip(hist,fit['rounds'])):
        a=vector(h['state'],names);nxt=vector(hist[k+1]['state'],names)
        if not support:
            assert not r['updated'] and np.array_equal(a,nxt);continue
        rr=r['rollout'];samples=rr['samples'].numpy().astype(float);mean=rr['mean'].numpy().astype(float)
        w=rr['weights'].numpy().astype(float);original=rr['rewards'].numpy().astype(float)
        used=rr['used_rewards'].numpy().astype(float)
        rewards.append(reward_check(samples,evidence,original))
        if rr['permutations'] is not None:
            assert np.array_equal(used,np.take_along_axis(original,rr['permutations'].numpy(),1))
        else:assert np.array_equal(used,original)
        weights.append(softmax_check(used,w,cfg['tau']))
        actual=fit[FIELD]['rounds'][k];assert actual['round']==k
        checked=check(r['mean_before'].numpy(),mean,samples,w,r['autograd_mean_gradient'].numpy(),
                      actual['analytic32'].numpy(),cfg['sigma'],r['analytic_gradient_error'])
        assert checked==actual['check'];gradients.append(checked)
        assert r['updated']==rr['informative'] and r['no_information']==(not rr['informative'])
        if not r['updated']:
            assert np.array_equal(a,nxt);continue
        update=h['update'];g=update['gradient'].numpy().astype(float);updates+=1
        m=.9*m+.1*g;v=.999*v+.001*g*g
        delta=-cfg['lr']*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
        raw=update['raw'].numpy();error=float(abs(delta-raw).max())
        assert error<4e-6;erradam=max(erradam,error)
        assert np.array_equal(nxt-a,raw) and update['names']==names
    assert updates==fit['gradient_calls']
    return dict(status='pass',revision=REVISION,original_component_absolute_audit_passed=passed,
        original_component_failed_line=failed_line,original_inline_guard_failures=sum(not x['original_inline_absolute_guard_passed'] for x in gradients),
        inline_gradient_precision_checks=gradients,reward_precision_checks=rewards,softmax_precision_checks=weights,
        max_mean_gradient_error=max((x['max_GPU32_vs_formula64_error'] for x in gradients),default=0.),
        max_detached_reward_error=max((x['max_original_GPU32_vs_float64_error'] for x in rewards),default=0.),
        max_Adam_error=erradam,gradient_calls=updates,rounds=steps,final_step=steps,
        authorized_chart_extension=chart_audit(fit),GT_read=False,independent_decoder_Jacobian=False,
        algorithm_or_saved_gradient_changed=False)
