"""Fixed E supervision through the frozen spatial decoder; only QR256 learns.

Original per-frame Gaussian chart/antithetic sampling/likelihood retained.
No labels, state inheritance, LN update, teacher refresh, or output E overlay.
"""
import torch
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from methods.decota_final_simplified_v1.objectives import generalized_iou
from vg_tta.stvg_dense_e_opd64_v1 import NativeReplay,contract as original_contract
from vg_tta.decota_spatial_opd_tunable_v1 import rollout,likelihood_loss,optimizer_step
from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
QUERY='spatial.query_residual'
ARMS=['Sparse_OPD','Dense_OPD','Fixed_rollout','Dense_Direct']


def contract():
    c=original_contract()
    mu=torch.zeros(4,4,requires_grad=True);target=torch.tensor([[.4,.4,.2,.2]]).repeat(4,1)
    cfg=dict(sigma=.1,tau=.25,samples=32);rr=rollout(mu.detach(),target,'query-e-contract',0,config=cfg)
    l=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],cfg['sigma']);g=torch.autograd.grad(l,mu)[0]
    with torch.no_grad():mu.add_(.03)
    l=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],cfg['sigma']);g2=torch.autograd.grad(l,mu)[0]
    expected=((mu.detach()-rr['mean'])-((rr['weights']-1/32)[:,:,None]*(rr['samples']-mu.detach()[:,None])).sum(1))/(cfg['sigma']**2*4)
    assert (g2-expected).abs().max()<2e-5 and not torch.equal(g,g2)
    return dict(**c,fixed_rollout_shifted_policy_gradient=True,only_query_selected=256)


@torch.enable_grad()
def fit(base,ids,key,teacher,positions,arm,cfg,active):
    assert arm in ARMS and cfg['samples']==32
    base.restore(base.initial);rp=NativeReplay(base,ids,key,{})
    initial=detached(rp.state(),'cpu');named=list(rp.named);flags=[(p,p.requires_grad)for _,p in named]
    assert torch.count_nonzero(initial[QUERY])==0
    for n,p in named:p.requires_grad_(n==QUERY)
    params=[p for n,p in named if n==QUERY];assert sum(p.numel()for p in params)==256
    assert not any(p.requires_grad for p in base.model.parameters())
    assert not any(p.requires_grad for p in base.norm.parameters())
    assert not any(p.requires_grad for p in base.decoder.parameters())
    frozen_hash=state_hash(base.decoder.state_dict())
    target=teacher.detach().to(params[0]);assert target.shape==(len(ids),4)
    support=list(positions)if arm=='Sparse_OPD'else list(range(len(ids)))
    assert support and len(set(support))==len(support)
    opt=torch.optim.Adam(params,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    assert not opt.state
    path=[];rounds=[];frozen=None;backwards=0;endpoints=0
    def record_after(b):
        if not rounds:return
        r=rounds[-1];pre=path[-2]['boxes']
        r['output_coordinate_absolute_mean']=float((b-pre).abs().mean())
        r['output_displacement_l2']=float((b-pre).norm())
        r['reference_IoU_after']=float(aligned_iou(b,target.cpu()).mean())
    try:
        for k in range(cfg['steps']+1):
            value=rp.values()if k<cfg['steps']and active else None
            if value is None:
                with torch.no_grad():value=rp.values()
            b=value['boxes'];bc=b.detach().cpu();st=detached(rp.state(),'cpu')
            path.append(dict(step=k,boxes=bc,state=st))
            record_after(bc)
            if k==0:before=bc.clone()
            if k==cfg['steps']:break
            if not active:
                rounds.append(dict(round=k,updated=False,no_expert_support=True,reference_IoU_before=float(aligned_iou(b.detach(),target).mean())))
                continue
            rr=None;err=None
            if arm=='Dense_Direct':
                loss=(5*(b-target).abs().sum(-1)+2*(1-generalized_iou(b,target))).mean();informative=True
            else:
                mu,boundary=proposed_coordinates(b,value['native_raw_logits']);endpoints+=int(boundary.sum())
                use_mu=mu[support]
                if arm=='Fixed_rollout'and frozen is not None:rr=frozen
                else:
                    # Generate the same full-frame noise, then restrict only Sparse's loss.
                    fullrr=rollout(mu.detach(),target,key,k,config=cfg)
                    rr={n:(v[support]if torch.is_tensor(v)and v.ndim>0 and len(v)==len(ids)else v)for n,v in fullrr.items()}
                    rr['informative']=bool((rr['used_rewards']!=rr['used_rewards'][:,:1]).any())
                    if arm=='Fixed_rollout':frozen=rr
                loss=likelihood_loss(use_mu,rr['mean'],rr['samples'],rr['weights'],cfg['sigma'])
                grad=torch.autograd.grad(loss,use_mu,retain_graph=True)[0]
                expected=((use_mu.detach()-rr['mean'])-((rr['weights']-1/32)[:,:,None]*(rr['samples']-use_mu.detach()[:,None])).sum(1))/(cfg['sigma']**2*len(support))
                err=float((grad-expected).abs().max());assert err<3e-5
                path[-1]['chart']=dict(mean=use_mu.detach().cpu(),raw=value['native_raw_logits'][support].detach().cpu(),boundary=boundary[support].cpu(),autograd_gradient=grad.detach().cpu())
                informative=rr['informative']
            assert torch.isfinite(loss)
            update=optimizer_step(opt,loss,params,informative)
            if update is not None:
                backwards+=1;path[-1]['update']=update;path[-1]['Adam_after']=detached(opt.state_dict(),'cpu')
            rounds.append(dict(round=k,updated=update is not None,loss=float(loss.detach()),rollout=None if rr is None else detached(rr,'cpu'),analytic_gradient_error=err,
                parameter_step_norm=None if update is None else float(update['raw'].norm()),parameter_gradient_norm=None if update is None else float(update['gradient'].norm()),
                reference_IoU_before=float(aligned_iou(b.detach(),target).mean()),reward_variance=None if rr is None else float(rr['rewards'].var(1,unbiased=False).mean()),
                ESS=None if rr is None else float((1/rr['weights'].square().sum(1)).mean()),current_Student_resampled=arm in ['Sparse_OPD','Dense_OPD']))
            del value,b,loss,update
        state=path[-1]['state'];final=path[-1]['boxes']
        assert all(torch.equal(s['state'][n],initial[n])for s in path for n in initial if n!=QUERY)
        assert state_hash(base.decoder.state_dict())==frozen_hash
        assert all(p.grad is None for p in base.decoder.parameters())
        if not active:assert torch.equal(before,final)and torch.equal(state[QUERY],initial[QUERY])
        return dict(arm=arm,initial=initial,state=state,before=before,final=final,path=path,rounds=rounds,selected_step=cfg['steps'],config=cfg,
            active_parameters=256,parameter_names=[QUERY],gradient_calls=backwards,optimizer_initial_empty=True,feedback_positions=support,all_decoder_frames=list(range(len(ids))),
            teacher_frozen=True,loss_per_frame_mean=True,additional_sparse_loss=False,empty=not active,GT_used=False,pretrained_weights_frozen_exact=True,
            finite_native_endpoint_coordinates=endpoints,parameter_displacement_norm=float((state[QUERY]-initial[QUERY]).norm()),output_displacement_l2=float((final-before).norm()))
    finally:
        rp.restore(initial)
        for p,flag in flags:p.requires_grad_(flag)
