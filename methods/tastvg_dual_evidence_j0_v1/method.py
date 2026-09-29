"""Predict current tube, rerank current time, then update space for future arrivals."""
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from methods.decota_final_simplified_v1.objectives import prediction
from vg_tta.tastvg_corruption_c0c1_v1 import candidates
from vg_tta.tastvg_temporal_qualification_v1 import critic_scores
from vg_tta.tastvg_spatial_rank_s11_v1 import SpatialActor,reverse_kl,average_ranks,update_scale
from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
from vg_tta.tastvg_spatial_critic_s06_v1 import rewards


def combine_layers(boxes,logits,records,ids):
    assert len(boxes)==len(logits)==2 and all(b.shape[0]==z.shape[0]==6 for b,z in zip(boxes,logits))
    return [prediction([z[layer] for z in logits],torch.stack([boxes[i%2][layer,i//2] for i in range(len(ids))]),records,ids) for layer in range(6)]


@torch.no_grad()
def central_with_candidates(actor,data):
    """Observe native second-pass layer outputs; hooks do not change computation."""
    boxes=[];logits=[];calls=[0]
    def decoder_hook(mod,args,out):
        calls[0]+=1
        if calls[0]%2==0:boxes.append(out[0].flatten(1,2).detach().cpu())
    def temporal_hook(mod,args,out):logits.append(out.detach().cpu())
    hs=[actor.model.ground_decoder.register_forward_hook(decoder_hook),actor.model.temp_embed.register_forward_hook(temporal_hook)]
    try:ev,b,p=actor.values(data)
    finally:
        for h in hs:h.remove()
    assert calls[0]==4
    layers=combine_layers(boxes,logits,data['records'],data['frame_ids'])
    assert torch.equal(layers[-1]['boxes'],p['boxes']) and layers[-1]['indices']==p['indices']
    assert all(torch.equal(a,z) for a,z in zip(layers[-1]['logits'],p['logits']))
    return ev,b,p,layers,candidates(layers,data['records'],data['frame_ids'])['temporal']


def fast_rerank(pre,support,expert):
    assert support[0]['indices']==pre['indices']
    score=critic_scores([x['physical_interval'] for x in support],expert['proposals'],expert['proposal_confidence'])
    selected=int(np.argmax(score));c=support[selected]
    out={**pre,'indices':list(c['indices']),'physical_interval':list(c['physical_interval'])}
    return out,dict(scores=score.tolist(),selected=selected,candidates=support,teacher_interval_as_output=False)


class OnlineMethod:
    """One persistent actor per independent stream; evidence providers are lazy."""
    def __init__(self,model,deltas,*,fast=True,slow=True):
        self.actor=SpatialActor(model);self.model=model;self.deltas=deltas;self.fast=fast;self.slow=slow
        self.coeff=[model.cfg.SOLVER.BBOX_COEF,model.cfg.SOLVER.GIOU_COEF]
    def reset(self):self.actor.restore(self.actor.initial)
    def close(self):self.actor.close()
    def arrive(self,data,scheduled,temporal_provider=None,spatial_provider=None):
        actor=self.actor;before=actor.state()
        with torch.no_grad():
            if scheduled and self.fast:
                ev,boxes,pre,layers,tc=central_with_candidates(actor,data)
            else:ev,boxes,pre=actor.values(data);layers=tc=None
        out=pre;result=dict(prediction=detached(pre,'cpu'),pre_state=detached(before,'cpu'),pre_state_sha256=state_hash(before),updated=False,temporal_expert_read=False,spatial_expert_read=False,GT_read=False)
        if scheduled and self.fast:
            expert=temporal_provider();assert expert['pixel_sha256']==data['pixel_sha256']
            out,td=fast_rerank(pre,tc,expert);result.update(temporal=td,temporal_layers=layers,temporal_expert_read=True)
        # Seal current output in memory BEFORE any spatial write.
        result['output_prediction']=detached(out,'cpu');post_ev=ev;post=pre
        if scheduled and self.slow:
            cs=[]
            for k,delta in enumerate(self.deltas):
                _,_,cp=predict(self.model,data,{n:before[n]+delta[n] for n in before})
                if k==0:assert torch.equal(cp['boxes'],pre['boxes'])
                cs.append(dict(prediction=detached(cp,'cpu')))
            expert=spatial_provider();assert expert['pixel_sha256']==data['pixel_sha256']
            score=rewards([c['prediction']['boxes'].numpy() for c in cs],expert['boxes'],expert['valid'])
            result.update(candidates=cs,rewards=score.tolist() if score is not None else None,valid_expert_frames=int(expert['valid'].sum()),spatial_expert_read=True)
            if score is not None:
                target=torch.stack([c['prediction']['boxes'] for c in cs]).cuda().detach();rt=torch.tensor(score,device='cuda',dtype=torch.float64).detach()
                evg,bg,pg=actor.values(data);assert torch.equal(bg.detach().cpu(),pre['boxes'])
                loss,pi,qi,dist=reverse_kl(bg,target,rt,self.coeff);grad=torch.autograd.grad(loss,[v for _,v in actor.named]);assert all(torch.isfinite(g).all() for g in grad)
                scale,gn=update_scale(grad,'rank',.005,0.)
                result['update']=dict(rank=average_ranks(score).tolist(),update_scale=scale,global_gradient_norm=gn,loss_before=float(loss.detach()),p=pi.detach().cpu(),q=qi.detach().cpu(),distances=dist.detach().cpu(),gradients={n:g.detach().cpu() for (n,_),g in zip(actor.named,grad)},gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grad))),query_gradient_norm=float(grad[0].norm()),LN_gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in grad[1:]))),coefficients=self.coeff,lr=.005,candidate_targets_detached=not target.requires_grad,reward_detached=not rt.requires_grad)
                with torch.no_grad():
                    for (_,param),g in zip(actor.named,grad):param.add_(g,alpha=-scale)
                    post_ev,bpost,post=actor.values(data);after_loss,_,_,_=reverse_kl(bpost,target,rt,self.coeff)
                result['update']['loss_after']=float(after_loss);result['updated']=gn>0
        after=actor.state();result.update(post_prediction=detached(post,'cpu'),post_state=detached(after,'cpu'),post_state_sha256=state_hash(after),parameter_displacement=float(torch.sqrt(sum((after[n]-before[n]).square().sum() for n in before))),displacement_from_source=float(torch.sqrt(sum((after[n]-actor.initial[n]).square().sum() for n in before))))
        assert torch.equal(result['output_prediction']['boxes'],result['prediction']['boxes'])
        return result,post_ev
