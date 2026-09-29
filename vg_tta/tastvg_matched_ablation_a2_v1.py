"""Raw-RKL and pairwise preference controls; frozen J0 and A1 unchanged."""
import hashlib
import numpy as np
import torch
from methods.tastvg_dual_evidence_j0_v1.method import OnlineMethod,central_with_candidates,fast_rerank
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks,update_scale
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
from vg_tta.tastvg_spatial_expansion_s0_v1 import alignment

ARMS=['raw_rkl','pairwise_rank']
def permutation(source,order):
    # Hash sorting is independent of Python/NumPy RNG versions and outcomes.
    return sorted(range(9),key=lambda k:hashlib.sha256(f'A1-rank-20260929|{source}|{order}|{k}'.encode()).hexdigest())

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
            expert=spatial_provider();assert expert['pixel_sha256']==data['pixel_sha256']
            valid=torch.as_tensor(expert['valid'],device='cuda',dtype=torch.bool)
            result.update(valid_expert_frames=int(valid.sum()),spatial_expert_read=True,rewards=None)
            cs=[];score=None;rank=None;target=None
            if self.arm!='direct_pl':
                center=actor.initial if self.arm=='off_policy' else before
                for k,delta in enumerate(self.deltas):
                    _,_,cp=predict(self.model,data,{n:center[n]+delta[n] for n in center})
                    if k==0:
                        reference=data['prediction']['boxes'].cpu() if self.arm=='off_policy' else pre['boxes']
                        assert torch.equal(cp['boxes'].cpu(),reference)
                    cs.append(dict(prediction=detached(cp,'cpu')))
                assert state_hash(actor.state())==state_hash(before)
                score=rewards([c['prediction']['boxes'].numpy() for c in cs],expert['boxes'],expert['valid'])
                result.update(candidates=cs,rewards=score.tolist() if score is not None else None,support_center_sha256=state_hash(center))
                if score is not None:
                    ranks=average_ranks(score);perm=self.assignment if self.arm=='random_rank' else list(range(9));assert sorted(perm)==list(range(9));rank=-score if self.arm=='raw_rkl' else ranks[perm]
                    target=torch.stack([c['prediction']['boxes'] for c in cs]).cuda().detach()
                    result.update(correct_ranks=ranks.tolist(),assigned_ranks=rank.tolist(),permutation=list(perm))
            if valid.any():
                evg,bg,pg=actor.values(data);assert torch.equal(bg.detach().cpu(),pre['boxes'])
                if self.arm=='direct_pl':
                    target=torch.as_tensor(expert['boxes'],device='cuda',dtype=bg.dtype).detach()
                    loss=alignment(bg,target,valid,1.,1.);extra=dict(coefficients=[1.,1.],supervision='sparse mask-box L1(sum coordinates)+GIoU; mean valid frames')
                else:
                    loss,pi,qi,dist=rank_loss(bg,target,rank,self.coeff,self.arm)
                    extra=dict(coefficients=self.coeff,rank=rank.tolist(),p=pi.detach().cpu(),q=qi.detach().cpu(),distances=dist.detach().cpu())
                grad=torch.autograd.grad(loss,[v for _,v in actor.named]);assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grad)
                scale,gn=update_scale(grad,'rank',.005,0.)
                result['update']=dict(**extra,update_scale=scale,global_gradient_norm=gn,loss_before=float(loss.detach()),gradients={n:g.detach().cpu() for (n,_),g in zip(actor.named,grad)},lr=.005,target_detached=not target.requires_grad)
                with torch.no_grad():
                    for (_,param),g in zip(actor.named,grad):param.add_(g,alpha=-scale)
                    post_ev,bpost,post=actor.values(data)
                    after_loss=alignment(bpost,target,valid,1.,1.) if self.arm=='direct_pl' else rank_loss(bpost,target,rank,self.coeff,self.arm)[0]
                result['update']['loss_after']=float(after_loss);result['updated']=gn>0
        after=actor.state();result.update(post_prediction=detached(post,'cpu'),post_state=detached(after,'cpu'),post_state_sha256=state_hash(after),parameter_displacement=float(torch.sqrt(sum((after[n]-before[n]).square().sum() for n in before))),displacement_from_source=float(torch.sqrt(sum((after[n]-actor.initial[n]).square().sum() for n in before))))
        assert torch.equal(result['output_prediction']['boxes'],result['prediction']['boxes'])
        return result,post_ev
