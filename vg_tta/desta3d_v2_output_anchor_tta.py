"""Single added output-anchor factor, with fixed observed-input teacher support."""
import gc
import torch
from vg_tta.desta3d_v2_tta_pilot import configure, forward
from vg_tta.desta3d_v2_tta_objective import calibration_objective, CalibrationWeights, gradient_groups
from vg_tta.desta3d_v2_output_anchor import output_kl
from vg_tta.desta3d_v2_output_anchor_layout_v5 import replay_branch


def teacher_support(native,trace):
    enabled=[]
    # Failure is retained; no fabricated reference/time or invalid token support.
    if not native.get('event',{}).get('format_ok',False):return enabled
    for branch,index,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
        if index>=len(trace['branches']):continue
        value=trace['branches'][index]['logits'].get(kind)
        if value is None or not value.numel() or not torch.isfinite(value).all():continue
        if branch=='event' and value.ndim==2 and value.shape[0]==2:enabled.append(branch)
        if branch=='spatial' and value.ndim==3 and value.shape[1:]==(4,1001):enabled.append(branch)
    return enabled  # Legal token logits retained even if decoded box geometry is invalid.


def adapt_output_anchor(model,prompt,adapter,fields,view,trace,enabled,*,moments,coefficient,
                        save_step,steps=3,lr=1e-5,replay=replay_branch):
    assert coefficient>=0 and torch.isfinite(torch.tensor(coefficient))
    assert set(enabled)<= {'event','spatial'} and len(enabled)==len(set(enabled))
    assert torch.equal(fields['frame_times'],view['frame_times'])
    before={n:v.detach().clone() for n,v in adapter.state_dict().items()}
    initial,count=configure(adapter,'calibration');params=[p for p in adapter.parameters() if p.requires_grad]
    names=[n for n,p in adapter.named_parameters() if p.requires_grad]
    with torch.no_grad():teacher=forward(adapter,fields)
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.)
    history=[];weights=CalibrationWeights(alignment=.01)
    def vector():
        return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True)
        loss,terms=calibration_objective(adapter,forward(adapter,view),teacher,initial,weights=weights,source_moments=moments)
        loss.backward();base_loss=float(loss.detach());term_values={k:float(v.detach()) for k,v in terms.items()}
        del loss,terms;gc.collect()
        raw={'pre_gate':vector().cpu().clone()};components={};cache_audits={}
        for branch,index,kind in [('event',0,'time'),('spatial',1,'coordinate')]:
            if branch not in enabled or coefficient==0:continue
            previous=vector().clone()
            value,cache=replay(model,prompt,adapter,fields,trace,branch)
            ref=trace['branches'][index]['logits'][kind]
            kl=output_kl(value,ref)
            (coefficient*kl).backward()
            raw[branch+'_weighted']= (vector()-previous).cpu().clone()
            components[branch]=float(kl.detach());cache_audits[branch]=cache
            del value,kl,previous;gc.collect();torch.cuda.empty_cache()
        raw['total_before_clip']=vector().cpu().clone()
        assert all(torch.isfinite(v).all() and v.numel()==66816 for v in raw.values())
        gradients=gradient_groups(adapter)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(not p.requires_grad and p.grad is None for p in adapter.parameter_groups()['gates'])
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert torch.isfinite(torch.tensor(norm))
        opt.step();actual=sorted({int(s['step']) for s in opt.state.values()});assert actual==[step]
        assert all(torch.isfinite(p).all() for p in adapter.parameters())
        row={'step':step,'actual_Adam_steps':actual,'loss_before':base_loss,
             'terms_before':term_values,'output_KL_before':components,'output_coefficient':coefficient,
             'total_loss_before':base_loss+coefficient*sum(components.values()),
             'gradient_groups':gradients,'clip_norm_before':norm,
             'gradient_norms':{k:float(v.double().norm()) for k,v in raw.items()}}
        save_step(step,{'ordered_names':names,'raw_gradients':raw,'history':row,'cache_audits':cache_audits})
        history.append(row)
    changed={n:not torch.equal(v,before[n]) for n,v in adapter.state_dict().items()}
    assert all(not b or n in initial for n,b in changed.items())
    with torch.no_grad():
        final,terms=calibration_objective(adapter,forward(adapter,view),teacher,initial,weights=weights,source_moments=moments)
    report={'interface':'calibration','updated_groups':['branch_film','norm_affine'],'parameter_count':count,
        'steps':steps,'history':history,'changed_tensors':changed,'gates_unchanged':not changed['gate_event'] and not changed['gate_spatial'],
        'teacher':'fixed B1 same observed input; output teacher forcing on observed input; pre-gate student on mild view',
        'loss_after':float(final),'loss_after_scope':'pre-gate objective only; actual output KL separately read out',
        'terms_after':{k:float(v) for k,v in terms.items()},'joint':0.,'alignment':.01,'GT_read':False,
        'output_coefficient':coefficient,'output_support':enabled,'selection':'fixed terminal3, no loss/GT selection'}
    adapter.zero_grad(set_to_none=True);adapter.set_train_stage('frozen')
    return report
