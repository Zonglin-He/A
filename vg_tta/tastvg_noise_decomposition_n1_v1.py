"""GT-audited mechanism oracle; never a deployable TTA method."""
import hashlib
import numpy as np
import torch
from methods.tastvg_dual_evidence_j0_v1.method import OnlineMethod,central_with_candidates,fast_rerank
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
from vg_tta.tastvg_spatial_rank_s11_v1 import update_scale
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
from vg_tta.box_stability_diagnostics_v1 import overlap

PRIMARY=['all','useful','noisy','useful_positive','useful_negative']
MATCHED=['useful_matched','noisy_matched']
ARMS=PRIMARY+MATCHED
TOL=1e-12

def categories(reward,quality):
    g=np.asarray(quality,dtype=np.float64);ys=np.sign(np.where(np.abs(g[1:]-g[0])>TOL,g[1:]-g[0],0)).astype(int)
    if reward is None:ye=np.zeros(8,dtype=int)
    else:
        e=np.asarray(reward,dtype=np.float64);ye=np.sign(np.where(np.abs(e[1:]-e[0])>TOL,e[1:]-e[0],0)).astype(int)
    return ye,ys

def eligible(arm,ye,yg):
    ye=np.asarray(ye);yg=np.asarray(yg)
    if arm=='all':mask=ye!=0  # GT ties retained: no GT-based filter in All.
    elif arm in ['useful','useful_matched']:mask=(ye*yg)==1
    elif arm in ['noisy','noisy_matched']:mask=(ye*yg)==-1
    elif arm=='useful_positive':mask=(ye==1)&(yg==1)
    elif arm=='useful_negative':mask=(ye==-1)&(yg==-1)
    else:raise ValueError(arm)
    return (np.flatnonzero(mask)+1).tolist()

def hash_subset(pool,n,key):
    assert 0<=n<=len(pool)
    return sorted(sorted(pool,key=lambda k:hashlib.sha256(f'N1-count-v1|{key}|{k}'.encode()).hexdigest())[:n])

def directional_loss(boxes,targets,labels,selected,coeff):
    d=geometry(boxes,targets,*coeff);z=-d
    if not selected:return d.sum()*0,d,torch.empty(0,device=d.device)
    ids=torch.as_tensor(selected,device=d.device,dtype=torch.long)
    y=torch.as_tensor(labels,device=d.device,dtype=d.dtype)[ids-1]
    terms=torch.nn.functional.softplus(-y*(z[ids]-z[0]))
    return terms.mean(),d,terms

class NoiseMethod(OnlineMethod):
    def prepare(self,data,scheduled,gt,temporal_provider=None,spatial_provider=None):
        actor=self.actor;before=actor.state()
        with torch.no_grad():
            if scheduled:ev,boxes,pre,layers,tc=central_with_candidates(actor,data)
            else:ev,boxes,pre=actor.values(data);layers=tc=None
        result=dict(prediction=detached(pre,'cpu'),pre_state=detached(before,'cpu'),pre_state_sha256=state_hash(before),updated=False,temporal_expert_read=False,spatial_expert_read=False,GT_read=False,oracle_signal_filter=True)
        out=pre
        if scheduled:
            teacher=temporal_provider();assert teacher['pixel_sha256']==data['pixel_sha256']
            out,td=fast_rerank(pre,tc,teacher);result.update(temporal=td,temporal_layers=layers,temporal_expert_read=True)
        result['output_prediction']=detached(out,'cpu')
        if scheduled:
            expert=spatial_provider();assert expert['pixel_sha256']==data['pixel_sha256'];cs=[]
            for k,delta in enumerate(self.deltas):
                _,_,cp=predict(self.model,data,{n:before[n]+delta[n] for n in before})
                if k==0:assert torch.equal(cp['boxes'].cpu(),pre['boxes'])
                cs.append(dict(prediction=detached(cp,'cpu')))
            assert state_hash(actor.state())==state_hash(before)
            score=rewards([c['prediction']['boxes'].numpy() for c in cs],expert['boxes'],expert['valid'])
            valid=np.asarray(gt['valid'],bool);truth=np.asarray(gt['boxes'],float);assert valid.any()
            quality=np.array([overlap(c['prediction']['boxes'].numpy()[valid],truth[valid]).mean() for c in cs])
            ye,yg=categories(score,quality)
            result.update(candidates=cs,rewards=None if score is None else score.tolist(),gt_sIoU=quality.tolist(),teacher_labels=ye.tolist(),gt_labels=yg.tolist(),GT_read=True,spatial_expert_read=True,valid_expert_frames=int(np.sum(expert['valid'])),support_center_sha256=state_hash(before))
        return result,ev
    def finish(self,data,prepared,arm,selected=None,match_count=None,hash_key=None):
        actor=self.actor;result,ev=prepared;before=actor.state();assert state_hash(before)==result['pre_state_sha256'];post_ev=ev;post=result['prediction'];result['arm']=arm
        if result['spatial_expert_read']:
            pool=eligible(arm,result['teacher_labels'],result['gt_labels']);chosen=pool if selected is None else selected
            assert set(chosen)<=set(pool)
            result.update(eligible_indices=pool,selected_indices=chosen,selected_count=len(chosen),match_count=match_count,hash_key=hash_key)
            if chosen:
                target=torch.stack([c['prediction']['boxes'] for c in result['candidates']]).cuda().detach()
                _,bg,_=actor.values(data);assert torch.equal(bg.detach().cpu(),result['prediction']['boxes'])
                loss,dist,terms=directional_loss(bg,target,result['teacher_labels'],chosen,self.coeff)
                grad=torch.autograd.grad(loss,[v for _,v in actor.named]);assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grad)
                scale,gn=update_scale(grad,'rank',.005,0.)
                result['update']=dict(coefficients=self.coeff,update_scale=scale,global_gradient_norm=gn,loss_before=float(loss.detach()),gradients={n:g.detach().cpu() for (n,_),g in zip(actor.named,grad)},lr=.005,target_detached=not target.requires_grad,distances=dist.detach().cpu(),terms=terms.detach().cpu())
                with torch.no_grad():
                    for (_,param),g in zip(actor.named,grad):param.add_(g,alpha=-scale)
                    post_ev,bpost,post=actor.values(data)
                    result['update']['loss_after']=float(directional_loss(bpost,target,result['teacher_labels'],chosen,self.coeff)[0])
                result['updated']=gn>0
        after=actor.state();result.update(post_prediction=detached(post,'cpu'),post_state=detached(after,'cpu'),post_state_sha256=state_hash(after),parameter_displacement=float(torch.sqrt(sum((after[n]-before[n]).square().sum() for n in before))),displacement_from_source=float(torch.sqrt(sum((after[n]-actor.initial[n]).square().sum() for n in before))))
        assert torch.equal(result['output_prediction']['boxes'],result['prediction']['boxes'])
        return result,post_ev
