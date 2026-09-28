"""Source-labelled diagnostic and label-free calibration share one update scope.

The task objective is the existing official GT teacher-forced PTD CE, not
native cached generation. Native free decoding must be evaluated separately.
"""
from contextlib import nullcontext
import torch
from vg_tta.desta3d_v2_tta_pilot import forward
from vg_tta.desta3d_v2_tta_objective import CalibrationWeights, calibration_objective


def active(adapter):
    names, params = zip(*[(n,p) for n,p in adapter.named_parameters() if p.requires_grad])
    assert sum(p.numel() for p in params)==66816
    return list(names), list(params)


def flat_gradient(loss, params, retain_graph=False):
    gs=torch.autograd.grad(loss,params,retain_graph=retain_graph,allow_unused=True)
    v=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                 for g,p in zip(gs,params)]).detach().float().cpu()
    assert v.shape==(66816,) and torch.isfinite(v).all()
    return v


def unlabeled_signals(adapter, view, teacher, initial, moments):
    """No label, supervised prefix, task gradient or selection input accepted."""
    _,params=active(adapter)
    weights=CalibrationWeights(alignment=.01)
    output=forward(adapter,view)
    total,terms=calibration_objective(adapter,output,teacher,initial,
                                    weights=weights,source_moments=moments)
    vectors={k:flat_gradient(getattr(weights,k)*v,params,True) for k,v in terms.items()}
    vectors['unlabeled_total']=flat_gradient(total,params)
    assert torch.allclose(sum(vectors[k] for k in terms),vectors['unlabeled_total'],atol=1e-7,rtol=2e-4)
    return vectors, {'total':float(total.detach()),'terms':{k:float(v.detach()) for k,v in terms.items()}}


def task_signals(model, processor, adapter, fields, data, *, gradients=True):
    from vg_tta.desta3d_v2_ptd import branch_injection
    from vg_tta.desta3d_v2_source import split_source_loss_masks
    from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
    _,params=active(adapter)
    masks=split_source_loss_masks(data,processor.tokenizer)
    assert int(masks['event'].sum())>0 and int(masks['spatial'].sum())>0
    model.train();model.model.visual.eval();adapter.eval()
    vectors={};result={}
    try:
        for branch in ('event','spatial'):
            d=dict(data);d['labels']=data['labels'].clone()
            d['labels'][~masks[branch].to(d['labels'].device)]=-100
            with (nullcontext() if gradients else torch.no_grad()):
                with branch_injection(model,adapter,data,fields,branch) as capture:
                    ce,stats=joint_loss(model,d)
                    assert torch.isfinite(ce)
                    if gradients:vectors['task_'+branch]=flat_gradient(ce,params)
                    result[branch]={'ce':float(ce.detach()),'tokens':int(masks[branch].sum()),
                        'relative_injection_norm':capture['relative_injection_norm'],
                        'cast_changed_elements':capture['changed_elements'],**stats}
                del ce,capture,d
        result['total']=result['event']['ce']+result['spatial']['ce']
        if gradients:vectors['task_total']=vectors['task_event']+vectors['task_spatial']
        assert all(p.grad is None for p in adapter.parameters())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        return vectors,result
    finally:
        model.eval()


def adam_step(adapter, opt, vector, *, step):
    names,params=active(adapter)
    assert vector.shape==(66816,) and torch.isfinite(vector).all()
    before=torch.cat([p.detach().reshape(-1).float().cpu() for p in params])
    opt.zero_grad(set_to_none=True)
    offset=0
    for p in params:
        p.grad=vector[offset:offset+p.numel()].reshape_as(p).to(p).clone();offset+=p.numel()
    norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
    assert torch.isfinite(torch.tensor(norm))
    opt.step()
    # Inspect live Parameter-bound optimizer state, not only serialized keys.
    assert len(opt.state)==len(params) and {id(k) for k in opt.state}=={id(p) for p in params}
    counters={n:int(opt.state[p]['step']) for n,p in zip(names,params)}
    assert set(counters.values())=={step}
    delta=torch.cat([p.detach().reshape(-1).float().cpu() for p in params])-before
    assert torch.isfinite(delta).all()
    opt.zero_grad(set_to_none=True)
    return delta,{'pre_clip_norm':norm,'clip_triggered':norm>1.,'actual_Adam_counters':counters,
        'live_parameter_binding':True,'delta_norm':float(delta.double().norm())}
