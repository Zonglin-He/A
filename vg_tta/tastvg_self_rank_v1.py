"""Student self-rank control, with no spatial specialist reads."""
import hashlib
import numpy as np
import torch
from methods.tastvg_dual_evidence_j0_v1.method import OnlineMethod,central_with_candidates,fast_rerank
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks,update_scale
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry

ARMS=['self_rank']
def rank_loss(boxes,target,rank,coeff,arm):
    distances=geometry(boxes,target,*coeff);logp=(-distances).log_softmax(0)
    logq=(-torch.as_tensor(rank,device=boxes.device,dtype=boxes.dtype)).log_softmax(0)
    if arm=='pairwise_rank':
        r=torch.as_tensor(rank,device=boxes.device)
        better=r[:,None]<r[None,:]
        # Mean logistic loss over each strictly ordered pair once; ties omitted.
        differences=distances[:,None]-distances[None,:]
        loss=torch.nn.functional.softplus(differences[better]).mean() if better.any() else distances.sum()*0
    else:loss=(logp.exp()*(logp-logq)).sum()
    return loss,logp.exp(),logq.exp(),distances

class AblationMethod(OnlineMethod):
    def __init__(self,model,deltas,arm):
        super().__init__(model,deltas);assert arm in ARMS;self.arm=arm;self.assignment=None
    def arrive(self,data,scheduled,temporal_provider=None,spatial_provider=None):
        actor=self.actor;before=actor.state()
        with torch.no_grad():
            if scheduled:ev,boxes,pre,layers,tc=central_with_candidates(actor,data)
            else:ev,boxes,pre=actor.values(data);layers=tc=None
        result=dict(prediction=detached(pre,'cpu'),pre_state=detached(before,'cpu'),pre_state_sha256=state_hash(before),updated=False,temporal_expert_read=False,spatial_expert_read=False,GT_read=False,ablation=self.arm)
        out=pre
        if scheduled:
            teacher=temporal_provider();assert teacher['pixel_sha256']==data['pixel_sha256']
            out,td=fast_rerank(pre,tc,teacher);result.update(temporal=td,temporal_layers=layers,temporal_expert_read=True)
        result['output_prediction']=detached(out,'cpu');post_ev=ev;post=pre
        if scheduled:
            # No Sa2VA provider call, reward, validity mask or admission veto.
            center=before;cs=[]
            for k,delta in enumerate(self.deltas):
                _,_,cp=predict(self.model,data,{n:center[n]+delta[n] for n in center})
                if k==0:assert torch.equal(cp['boxes'].cpu(),pre['boxes'])
                cs.append(dict(prediction=detached(cp,'cpu')))
            assert state_hash(actor.state())==state_hash(before)
            target=torch.stack([c['prediction']['boxes'] for c in cs]).cuda().detach()
            evg,bg,pg=actor.values(data);assert torch.equal(bg.detach().cpu(),pre['boxes'])
            self_p=(-geometry(bg,target,*self.coeff)).softmax(0).detach()
            rank=average_ranks(self_p.cpu().numpy())
            result.update(candidates=cs,support_center_sha256=state_hash(center),
                          self_probabilities=self_p.cpu(),assigned_ranks=rank.tolist(),
                          correct_ranks=rank.tolist(),permutation=list(range(9)),
                          admission='scheduled; finite student geometry; no spatial expert veto')
            if scheduled:
                loss,pi,qi,dist=rank_loss(bg,target,rank,self.coeff,self.arm)
                extra=dict(coefficients=self.coeff,rank=rank.tolist(),p=pi.detach().cpu(),q=qi.detach().cpu(),distances=dist.detach().cpu())
                grad=torch.autograd.grad(loss,[v for _,v in actor.named]);assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grad)
                scale,gn=update_scale(grad,'rank',.005,0.)
                result['update']=dict(**extra,update_scale=scale,global_gradient_norm=gn,loss_before=float(loss.detach()),gradients={n:g.detach().cpu() for (n,_),g in zip(actor.named,grad)},lr=.005,target_detached=not target.requires_grad)
                with torch.no_grad():
                    for (_,param),g in zip(actor.named,grad):param.add_(g,alpha=-scale)
                    post_ev,bpost,post=actor.values(data)
                    after_loss=rank_loss(bpost,target,rank,self.coeff,self.arm)[0]
                result['update']['loss_after']=float(after_loss);result['updated']=gn>0
        after=actor.state();result.update(post_prediction=detached(post,'cpu'),post_state=detached(after,'cpu'),post_state_sha256=state_hash(after),parameter_displacement=float(torch.sqrt(sum((after[n]-before[n]).square().sum() for n in before))),displacement_from_source=float(torch.sqrt(sum((after[n]-actor.initial[n]).square().sum() for n in before))))
        assert torch.equal(result['output_prediction']['boxes'],result['prediction']['boxes'])
        return result,post_ev
