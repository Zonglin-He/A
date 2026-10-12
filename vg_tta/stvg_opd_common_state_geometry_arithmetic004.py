"""Diagnostic hooks: common-state gradients and a declared Direct budget control."""
import inspect
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached

def cos(a,b):
    a=a.detach().double().flatten();b=b.detach().double().flatten()
    n=a.norm()*b.norm();return float(torch.dot(a,b)/n) if float(n)>0 else None

class Geometry:
    def __init__(self,recorder,arm,reference):
        self.arm=arm;self.reference=reference;self.rows=[];self.extra=0;self.originals=[]
        for mod in [recorder.core,recorder.component]:
            fn=recorder.installed[mod]['fit'];compiled=None
            for c in fn.__closure__ or []:
                v=c.cell_contents
                if inspect.isfunction(v) and v.__name__ in ['fit','component_fit']:compiled=inspect.unwrap(v)
            assert compiled is not None
            ns=compiled.__globals__;old=ns['optimizer_step'];self.originals.append((ns,old));ns['optimizer_step']=self.wrapper(old)

    def wrapper(self,original):
        def step(opt,loss,params,informative):
            f=inspect.currentframe().f_back;l=f.f_locals;rp=l['rp'];k=l['k'];boxes=l['boxes'];pos=l['positions'];evidence=l['evidence']
            item=dict(round=k,parameter_names=[n for n,p in rp.named])
            if self.arm=='Full':
                from methods.decota_final_simplified_v1.objectives import generalized_iou
                p=boxes[pos];dl=(5*(p-evidence).abs().sum(-1)+2*(1-generalized_iou(p,evidence))).mean()
                a=torch.autograd.grad(loss,params,retain_graph=True);b=torch.autograd.grad(dl,params,retain_graph=True)
                ga=torch.cat([g.flatten() for g in a]);gb=torch.cat([g.flatten() for g in b])
                item.update(OPD_parameter_gradient=ga.detach().cpu(),Direct_parameter_gradient=gb.detach().cpu(),
                  joint_cosine=cos(ga,gb),query_cosine=cos(ga[:256],gb[:256]),LN_cosine=cos(ga[256:],gb[256:]),
                  OPD_norm=float(ga.double().norm()),Direct_norm=float(gb.double().norm()),same_parameter_state=True)
            prior=[p.detach().clone() for p in params]
            before=detached(opt.state_dict(),'cpu')
            update=original(opt,loss,params,informative)
            item['Adam_before']=before;item['Adam_after']=detached(opt.state_dict(),'cpu')
            if self.arm=='Direct_capped' and update is not None:
                proposed=[p.detach().clone() for p in params]
                ref=self.reference['path'];radius=float((ref[k+1]['boxes'].double()-ref[k]['boxes'].double()).norm())
                origin=boxes.detach().double()
                def apply(t):
                    with torch.no_grad():
                        for p,a,b in zip(params,prior,proposed):p.copy_(a+(b-a)*t)
                        value=rp.values()['boxes'];self.extra+=1
                        return float((value.double()-origin).norm())
                proposal=apply(1.);fraction=1.;actual=proposal;forwards=1
                if radius==0:
                    fraction=0.;actual=apply(0.);forwards+=1
                elif proposal>radius:
                    high=1.;low=.5;value=apply(low);forwards+=1
                    for _ in range(20):
                        if value<=radius:break
                        high=low;low/=2;value=apply(low);forwards+=1
                    else:low=0.;value=apply(0.);forwards+=1
                    for _ in range(8):
                        mid=(low+high)/2;v=apply(mid);forwards+=1
                        if v<=radius:low=mid;value=v
                        else:high=mid
                    fraction=low;actual=apply(low);forwards+=1
                assert actual<=radius+1e-12
                raw=torch.cat([(p.detach()-a).flatten() for p,a in zip(params,prior)]).cpu()
                update={**update,'proposal_raw':update['raw'],'raw':raw,
                  'output_cap':dict(reference_radius=radius,proposal_movement=proposal,applied_movement=actual,
                    fraction=fraction,extra_native_forwards=forwards,cap_satisfied=True,
                    budget_utilization=actual/radius if radius else None,Adam_moments_not_scaled=True)}
                item['cap']=update['output_cap']
            self.rows.append(item);return update
        return step

    def close(self):
        for ns,old in self.originals:ns['optimizer_step']=old

def bare(fit):
    from vg_tta.decota_spatial_opd_inline_gradient_precision004 import FIELD
    from vg_tta.decota_spatial_opd_action_readback_precision005 import FIELD as ACTION
    return {k:v for k,v in fit.items() if k not in [FIELD,ACTION,'numerical_chart','geometry']}

def mathematical_readback(fit):
    """Independent Adam proposal/applied displacement and recorded moment check."""
    cfg=fit['config'];names=list(fit['initial']);m=np.zeros(1792);v=m.copy();t=0;maximum=0.;scalars=0
    for k,h in enumerate(fit['path'][:-1]):
        nxt=fit['path'][k+1];a=np.concatenate([h['state'][n].numpy().ravel() for n in names]).astype(float)
        b=np.concatenate([nxt['state'][n].numpy().ravel() for n in names]).astype(float)
        u=h.get('update')
        if u is None:assert np.array_equal(a,b);continue
        g=u['gradient'].numpy().astype(float);m=.9*m+.1*g;v=.999*v+.001*g*g;t+=1
        expected=-cfg['lr']*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        proposed=u.get('proposal_raw',u['raw']).numpy().astype(float)
        err=float(np.max(np.abs(expected-proposed)));assert err<4e-6;maximum=max(maximum,err)
        raw=u['raw'].numpy().astype(float);assert np.array_equal((b.astype(np.float32)-a.astype(np.float32)).astype(float),raw)
        if 'output_cap' in u:
            c=u['output_cap'];assert 0<=c['fraction']<=1
            move=float((nxt['boxes'].double()-h['boxes'].double()).norm())
            assert abs(move-c['applied_movement'])<2e-10 and move<=c['reference_radius']+1e-12
            # Actual float32 application, including its rounding, is checked.
            applied=(a.astype(np.float32)+proposed.astype(np.float32)*np.float32(c['fraction'])).astype(np.float32)
            assert np.array_equal(applied,b.astype(np.float32))
            scalars+=4
        row=fit['geometry']['rounds'][k]
        after=row['Adam_after'];states=list(after['state'].values())
        ma=np.concatenate([s['exp_avg'].numpy().ravel() for s in states]).astype(float)
        va=np.concatenate([s['exp_avg_sq'].numpy().ravel() for s in states]).astype(float)
        assert np.max(np.abs(ma-m))<2e-5*(1+np.max(np.abs(m)))
        assert np.max(np.abs(va-v))<2e-5*(1+np.max(np.abs(v)))
        assert all(int(s['step'])==t for s in states)
        scalars+=1792*5
    assert fit['selected_step']==cfg['steps'] and len(fit['path'])==cfg['steps']+1
    assert not fit['GT_used'] and fit['active_parameters']==1792
    assert all(torch.equal(fit['initial'][n],fit['path'][0]['state'][n]) for n in names)
    assert all(torch.equal(fit['state'][n],fit['path'][-1]['state'][n]) for n in names)
    return dict(status='pass',scope='actual gradient/Adam moments/proposal/application/output cap, not independently replayed decoder Jacobian',
      scalar_checks=scalars,actual_Adam_steps=t,maximum_Adam_proposal_error=maximum,GT_read=False)
