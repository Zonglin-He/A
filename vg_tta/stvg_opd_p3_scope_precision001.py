"""Separate numerical recorder/audit for the locked P3 parameter scopes.

Science sources stay immutable. Each compiled function changes only the existing
2e-5 assertion test to the already qualified matched CPU reduction check.
Actual native logits/actions are recorded from their original calls; no clamp,
new objective, update scope, optimizer, output or parameter is introduced.
"""
import ast
import contextlib
import copy
import sys
import time
import numpy as np
import torch
from vg_tta.decota_spatial_opd_matched_gradient_precision006 import check
from vg_tta.decota_spatial_opd_inline_gradient_precision004 import FIELD
from vg_tta.decota_spatial_opd_action_readback_precision005 import FIELD as ACTION,REVISION as ACTION_REVISION
from vg_tta.decota_spatial_opd_chart_revision003 import REVISION as CHART
REVISION='P3_parameter_scope_precision_revision001'


class Installation:
    def __init__(self,qualification=False):
        import vg_tta.decota_spatial_opd_tunable_v1 as core
        import vg_tta.stvg_opd_paper_ablations_v1 as component
        from vg_tta.decota_spatial_opd_chart_revision003 import Installation as Chart
        self.core=core;self.component=component;self.contexts=[];self.qualification=qualification;self.last_vjp=[]
        self.old={};self.installed={}
        self.chart=Chart()  # Reuse exact original same-native-call head readback.
        self.chart.uninstall()
        for module,fn in [(core,'fit'),(component,'component_fit')]:
            self.old[module]=dict(fit=getattr(module,fn),mean=module.mean_coordinates,action=module.action_boxes)
            path=module.__file__;tree=ast.parse(open(path).read())
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn)
            assertions=[n for n in ast.walk(node) if isinstance(n,ast.Assert) and ast.unparse(n.test)=='error < 2e-05']
            assert len(assertions)==1;original=copy.deepcopy(node)
            assertions[0].test=ast.Call(func=ast.Name(id='_P3_scope_guard',ctx=ast.Load()),
                args=[ast.Name(id=x,ctx=ast.Load()) for x in ['mu','rr','gmu','analytic','error','cfg','k']],keywords=[])
            restored=copy.deepcopy(node);guard=next(n for n in ast.walk(restored) if isinstance(n,ast.Assert) and n.lineno==assertions[0].lineno)
            guard.test=copy.deepcopy(next(n for n in ast.walk(original) if isinstance(n,ast.Assert) and n.lineno==guard.lineno).test)
            assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
            namespace=dict(module.__dict__);namespace['_P3_scope_guard']=self.guard
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),path,'exec'),namespace)
            compiled=namespace[fn]
            namespace['mean_coordinates']=self.mean
            namespace['rollout']=module.rollout
            module.action_boxes=self.action(module)
            module.mean_coordinates=self.mean
            setattr(module,fn,self.wrap(compiled))
            self.installed[module]=dict(fit=getattr(module,fn),mean=module.mean_coordinates,action=module.action_boxes)
        core.TrickReplay.values=self.chart.values

    def action(self,module):
        def call(z):
            value=self.old[module]['action'](z);frame=sys._getframe(1)
            if frame.f_code.co_name=='rollout':
                tick=time.perf_counter()
                assert self.contexts and not value.requires_grad
                self.contexts[-1]['actions'].append(dict(round=frame.f_locals['round_index'],samples=z.detach().cpu().clone(),
                    actions=value.detach().cpu().clone(),same_original_GPU_sigmoid_call=True,original_action_returned_without_modification=True))
                self.contexts[-1]['action_seconds']+=time.perf_counter()-tick
            return value
        return call

    def guard(self,mu,rr,gmu,analytic,error,cfg,k):
        tick=time.perf_counter()
        def a(t):return t.detach().cpu().numpy().astype(float)
        checked=check(a(mu),a(rr['mean']),a(rr['samples']),a(rr['weights']),a(gmu),a(analytic),cfg['sigma'],error)
        self.contexts[-1]['gradient'].append(dict(round=k,analytic32=analytic.detach().cpu().clone(),check=checked))
        self.contexts[-1]['CPU_seconds']+=time.perf_counter()-tick;return True

    def mean(self,boxes):
        from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates
        frame=sys._getframe(1)
        if frame.f_code.co_name not in ['fit','component_fit']:return self.old[self.core]['mean'](boxes)
        loc=frame.f_locals;rp=loc['rp'];pos=loc['positions']
        assert torch.equal(boxes,rp._chart003_boxes[pos]);raw=rp._chart003_raw[pos];value,boundary=proposed_coordinates(boxes,raw)
        phase='before' if torch.is_grad_enabled() else 'after';context=self.contexts[-1]
        context['chart'].append(dict(round=loc['k'],phase=phase,boxes=boxes.detach().cpu(),raw_logits=raw.detach().cpu(),
            mean=value.detach().cpu(),boundary=boundary.detach().cpu(),same_native_call_verified=True))
        if self.qualification and phase=='before':
            heads=rp._chart003_heads;gs=torch.autograd.grad(value.sum(),heads,retain_graph=True);expected=[torch.zeros_like(h) for h in heads]
            for p in pos:expected[p%2][p//2,0]=1
            error=max(float((g-e).abs().max()) for g,e in zip(gs,expected))
            assert all(torch.isfinite(g).all() for g in gs) and error<2e-6
            context['vjp'].append(dict(round=loc['k'],native_head_VJP_max_error=error,independent_full_decoder_Jacobian=False))
        return value

    def wrap(self,compiled):
        def call(*args,**kwargs):
            c=dict(gradient=[],actions=[],chart=[],vjp=[],CPU_seconds=0.,action_seconds=0.);self.contexts.append(c)
            try:
                result=compiled(*args,**kwargs)
                result[FIELD]=dict(revision=REVISION,rounds=c['gradient'],CPU_audit_seconds=c['CPU_seconds'],
                    original_assertion_retained_as_evidence=True,no_production_computation_changed=True)
                result[ACTION]=dict(revision=ACTION_REVISION,trace=c['actions'],CPU_action_readback_seconds=c['action_seconds'],
                    original_actions_rewards_weights_loss_gradients_Adam_and_readout_unchanged=True)
                result['numerical_chart']=dict(revision=CHART,interior_formula='original_torch_logit',endpoint_formula='same_call_finite_native_logit',trace=c['chart'])
                self.last_vjp=c['vjp'];return result
            finally:self.contexts.pop()
        return call

    def uninstall(self):
        for module,n in [(self.core,'fit'),(self.component,'component_fit')]:
            old=self.old[module];setattr(module,n,old['fit']);module.mean_coordinates=old['mean'];module.action_boxes=old['action']
        self.core.TrickReplay.values=self.chart.old_values

    @contextlib.contextmanager
    def original(self):
        assert not self.contexts
        self.uninstall()
        try:yield
        finally:
            for module,fn in [(self.core,'fit'),(self.component,'component_fit')]:
                current=self.installed[module];setattr(module,fn,current['fit']);module.mean_coordinates=current['mean'];module.action_boxes=current['action']
            self.core.TrickReplay.values=self.chart.values


def inline_audit(fit,expert,original_component):
    from vg_tta.decota_fixed_full_audit_v1 import top1_support,vector
    from vg_tta.decota_spatial_opd_reward_audit_revision005 import reward_check
    from vg_tta.decota_spatial_opd_precision_audit_revision002 import softmax_check
    from vg_tta.decota_spatial_opd_chart_revision003 import chart_audit
    cfg=fit['config'];steps=cfg['steps'];allnames=list(fit['initial']);names=fit.get('optimizer_parameter_names',allnames)
    assert isinstance(steps,int) and steps>0 and cfg['samples']==32
    assert cfg['sigma']>0 and cfg['tau']>0 and cfg['lr']>0 and 0<=cfg['writeback']<=1
    assert len(vector(fit['initial'],allnames))==1792 and all(torch.isfinite(v).all() for v in fit['initial'].values())
    expected=[n for n in allnames if fit.get('scope','joint')=='joint' or
        (fit.get('scope')=='query_only')==(n=='spatial.query_residual')]
    assert names==expected and fit['active_parameters']==sum(fit['initial'][n].numel() for n in names)
    assert fit['active_parameters']=={'joint':1792,'query_only':256,'LN_only':1536}[fit.get('scope','joint')]
    assert not fit.get('direct_objective') and fit[FIELD]['revision']==REVISION and fit['arm']=='on_policy'
    passed=True;failed=None
    try:original_component(fit,expert)
    except AssertionError as exc:
        tb=exc.__traceback__
        while tb.tb_next:tb=tb.tb_next
        assert tb.tb_frame.f_code.co_filename.endswith('stvg_opd_paper_component_audit_v1.py') and tb.tb_lineno in [36,40,43]
        passed=False;failed=tb.tb_lineno
    hist=fit['path'];support=top1_support(expert)
    assert fit['selected_step']==steps and len(hist)==steps+1 and len(fit['rounds'])==steps and not fit['GT_used']
    assert fit['positions']==[p for p,_ in support] and len(fit[FIELD]['rounds'])==(steps if support else 0)
    assert fit['empty']==(not support)
    assert np.array_equal(vector(fit['initial'],allnames),vector(hist[0]['state'],allnames))
    assert torch.equal(fit['before'],hist[0]['boxes'])
    assert np.array_equal(vector(fit['state'],allnames),vector(hist[-1]['state'],allnames)) and np.array_equal(fit['final'].numpy(),hist[-1]['boxes'].numpy())
    evidence=np.asarray([e for _,e in support],np.float32);m=np.zeros(fit['active_parameters']);v=m.copy();updates=0;erradam=0.;gs=[];rs=[];ws=[]
    for k,(h,r) in enumerate(zip(hist,fit['rounds'])):
        for n in allnames:
            if n not in names:assert torch.equal(h['state'][n],hist[k+1]['state'][n])
        a=vector(h['state'],names);nxt=vector(hist[k+1]['state'],names)
        if not support:assert not r['updated'] and np.array_equal(a,nxt);continue
        rr=r['rollout'];samples=rr['samples'].numpy().astype(float);mean=rr['mean'].numpy().astype(float);w=rr['weights'].numpy().astype(float)
        original=rr['rewards'].numpy().astype(float);used=rr['used_rewards'].numpy().astype(float)
        assert rr['permutations'] is None and np.array_equal(original,used)
        rs.append(reward_check(samples,evidence,original));ws.append(softmax_check(used,w,cfg['tau']))
        recorded=fit[FIELD]['rounds'][k];assert recorded['round']==k
        g=check(r['mean_before'].numpy(),mean,samples,w,r['autograd_mean_gradient'].numpy(),recorded['analytic32'].numpy(),cfg['sigma'],r['analytic_gradient_error'])
        assert g==recorded['check'];gs.append(g)
        assert r['updated']==rr['informative'] and r['no_information']==(not rr['informative'])
        if not r['updated']:assert np.array_equal(a,nxt);continue
        u=h['update'];gradient=u['gradient'].numpy().astype(float);updates+=1
        assert len(gradient)==fit['active_parameters'] and u['names']==names
        m=.9*m+.1*gradient;v=.999*v+.001*gradient*gradient
        delta=-cfg['lr']*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
        raw=u['raw'].numpy();error=float(abs(delta-raw).max());assert error<4e-6;erradam=max(erradam,error)
        assert np.array_equal(nxt-a,raw)
    assert updates==fit['gradient_calls']
    return dict(status='pass',revision=REVISION,active_parameters=fit['active_parameters'],scope=fit.get('scope','joint'),
        inactive_parameters_bitwise_unchanged_each_round=True,original_component_absolute_audit_passed=passed,original_component_failed_line=failed,
        inline_gradient_precision_checks=gs,reward_precision_checks=rs,softmax_precision_checks=ws,
        max_Adam_error=erradam,gradient_calls=updates,rounds=steps,final_step=steps,authorized_chart_extension=chart_audit(fit),
        GT_read=False,independent_decoder_Jacobian=False,algorithm_or_saved_gradient_changed=False)


def audit(fit,expert,original_component):
    import vg_tta.decota_spatial_opd_inline_gradient_precision004 as inline
    from vg_tta.decota_spatial_opd_action_readback_precision005 import audit as action
    old=inline.audit;inline.audit=inline_audit
    try:result=action(fit,expert,original_component)
    finally:inline.audit=old
    return dict(result,revision=REVISION,production_fitting_scientific_scope_and_original_sources_unchanged=True)
