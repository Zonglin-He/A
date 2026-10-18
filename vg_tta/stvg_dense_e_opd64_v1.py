"""Fixed dense E teacher, matched Source-reset Direct and original Gaussian OPD.

No label reader. Every original decoder frame contributes once to the mean.
The same-call finite native logit extension preserves rounded sigmoid endpoints.
"""
import torch
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.objectives import generalized_iou
from vg_tta.c1_enabling_tricks_v1 import TrickReplay
from vg_tta.decota_optimizer_posterior_r1_v1 import flat
from vg_tta.decota_spatial_opd_tunable_v1 import rollout,likelihood_loss,optimizer_step
from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou


class NativeReplay(TrickReplay):
    def values(self):
        raw=[]
        hook=self.base.decoder.decoder.bbox_embed.register_forward_hook(lambda m,a,o:raw.append(o))
        try:out=super().values()
        finally:hook.remove()
        assert len(raw)==12
        out['native_raw_logits']=torch.stack([raw[5+(i%2)*6][i//2,0]for i in range(len(self.ids))]).float()
        assert torch.equal(out['native_raw_logits'].sigmoid(),out['boxes'])
        return out


def contract():
    import numpy as np
    mu=torch.tensor([[.1,.2,-.3,.4],[.7,-.1,.2,-.2]],requires_grad=True)
    cfg=dict(sigma=.1,tau=.25,samples=32)
    teacher=torch.tensor([[.3,.6,.2,.4],[.8,.1,.1,.2]])
    rr=rollout(mu.detach(),teacher,'dense-e-contract',0,config=cfg)
    loss=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],cfg['sigma'])
    grad=torch.autograd.grad(loss,mu)[0]
    x=mu.detach().numpy();z=rr['samples'].numpy();w=rr['weights'].numpy()
    expected=-np.sum((w-1/32)[:,:,None]*(z-x[:,None]),axis=1)/(.01*len(x))
    err=float(np.max(abs(expected-grad.numpy())));assert err<2e-5
    uniform=torch.full_like(rr['weights'],1/32)
    assert torch.count_nonzero(torch.autograd.grad(likelihood_loss(mu,rr['mean'],rr['samples'],uniform,.1),mu)[0])==0
    double=likelihood_loss(mu.repeat(2,1),rr['mean'].repeat(2,1),rr['samples'].repeat(2,1,1),rr['weights'].repeat(2,1),.1)
    assert abs(float(double-loss))<1e-6
    return dict(status='pass',count=4,independent_mean_gradient_max_error=err,uniform_weights_zero_gradient=True,duplicate_frames_mean_invariant=True,teacher_actions_weights_detached=True)


@torch.enable_grad()
def fit(base,ids,key,teacher,arm,cfg,active):
    assert arm in ['Dense_Direct','Dense_OPD'] and cfg['samples']==32
    base.restore(base.initial);rp=NativeReplay(base,ids,key,{})
    initial=detached(rp.state(),'cpu');params=[v for _,v in rp.named];names=[n for n,_ in rp.named]
    assert sum(v.numel()for v in params)==1792
    target=teacher.detach().to(params[0]);assert target.shape==(len(ids),4)and not target.requires_grad
    opt=torch.optim.Adam(params,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    assert len(opt.state)==0
    path=[];rounds=[];backwards=0;endpoints=0
    try:
        with torch.no_grad():before=rp.values()['boxes'].detach().clone()
        for k in range(cfg['steps']):
            value=rp.values();b=value['boxes'];entry=dict(step=k,boxes=detached(b,'cpu'),state=detached(rp.state(),'cpu'));path.append(entry)
            if not active:
                rounds.append(dict(round=k,updated=False,no_expert_support=True));continue
            if arm=='Dense_Direct':
                loss=(5*(b-target).abs().sum(-1)+2*(1-generalized_iou(b,target))).mean();rr=None;analytic_error=None
                informative=True
            else:
                mu,boundary=proposed_coordinates(b,value['native_raw_logits']);endpoints+=int(boundary.sum())
                rr=rollout(mu.detach(),target,key,k,config=cfg)
                loss=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],cfg['sigma'])
                grad=torch.autograd.grad(loss,mu,retain_graph=True)[0]
                expected=-( (rr['weights']-1/32)[:,:,None]*(rr['samples']-mu.detach()[:,None]) ).sum(1)/(cfg['sigma']**2*len(ids))
                analytic_error=float((grad-expected).abs().max());assert analytic_error<3e-5
                informative=rr['informative']
                entry['chart']=dict(mean=mu.detach().cpu(),raw=value['native_raw_logits'].detach().cpu(),boundary=boundary.cpu(),autograd_gradient=grad.detach().cpu())
            assert torch.isfinite(loss)
            update=optimizer_step(opt,loss,params,informative)
            if update is not None:
                backwards+=1;entry['update']=update
                entry['Adam_after']=detached(opt.state_dict(),'cpu')
            with torch.no_grad():post=rp.values()['boxes'];move=post-b.detach()
            rounds.append(dict(round=k,updated=update is not None,loss=float(loss.detach()),
                rollout=None if rr is None else detached(rr,'cpu'),analytic_gradient_error=analytic_error,
                parameter_step_norm=None if update is None else float(update['raw'].norm()),
                parameter_gradient_norm=None if update is None else float(update['gradient'].norm()),
                output_coordinate_absolute_mean=float(move.abs().mean()),output_displacement_l2=float(move.norm()),
                reference_IoU_before=float(aligned_iou(b.detach(),target).mean()),reference_IoU_after=float(aligned_iou(post,target).mean()),
                reward_variance=None if rr is None else float(rr['rewards'].var(1,unbiased=False).mean()),
                ESS=None if rr is None else float((1/rr['weights'].square().sum(1)).mean()),current_Student_resampled=arm=='Dense_OPD'))
        state=detached(rp.state(),'cpu')
        with torch.no_grad():final=rp.values()['boxes'].detach().cpu()
        path.append(dict(step=cfg['steps'],boxes=final,state=state))
        if not active:assert torch.equal(before.cpu(),final)and all(torch.equal(state[n],initial[n])for n in names)
        return dict(arm=arm,initial=initial,state=state,before=before.cpu(),final=final,path=path,rounds=rounds,
            selected_step=cfg['steps'],config=cfg,active_parameters=1792,parameter_names=names,gradient_calls=backwards,
            optimizer_initial_empty=True,all_decoder_frames=list(range(len(ids))),teacher_frozen=True,
            loss_per_frame_mean=True,additional_sparse_loss=False,empty=not active,GT_used=False,
            finite_native_endpoint_coordinates=endpoints,parameter_displacement_norm=float((flat([state[n]for n in names])-flat([initial[n]for n in names])).norm()),
            output_displacement_l2=float((final-before.cpu()).norm()))
    finally:rp.restore(initial)
