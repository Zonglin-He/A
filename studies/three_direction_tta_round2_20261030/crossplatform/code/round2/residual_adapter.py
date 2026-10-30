"""Real episodic neural residual adaptation on frozen final query embeddings."""
import numpy as np
import torch
from torch import nn

CONFIG={'steps':[0,1,3],'learning_rate':.01,'center_bound_fraction':.15,
        'gradient_clip':1.,'rank_weight':1.,'pose_weight':1.,'trust_region_weight':.1}

class QueryResidual(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(288,5);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
    def forward(self,features,source_boxes,source_scores):
        features=features/torch.sqrt(features.square().mean(-1,keepdim=True)+1e-8)
        residual=self.linear(features)
        center=source_boxes[:,:3]+CONFIG['center_bound_fraction']*source_boxes[:,3:6]*residual[:,:3].tanh()
        yaw=(np.pi/2)*residual[:,3].tanh()
        boxes=torch.cat([center,source_boxes[:,3:6],yaw[:,None]],-1)
        logits=source_scores.clamp_min(1e-30).log()+residual[:,4]
        return boxes,logits,residual

def run(features,source_boxes,source_scores,teacher,control=False):
    """CPU default; no GT, no base model mutation, fresh zero adapter every call."""
    feat=torch.as_tensor(features,dtype=torch.float32);box=torch.as_tensor(source_boxes,dtype=torch.float32);score=torch.as_tensor(source_scores,dtype=torch.float32)
    idx=torch.as_tensor(teacher['query_indices'],dtype=torch.long)
    base_logscore=score[idx].clamp_min(1e-30).log()
    bonus=torch.zeros(len(idx)) if control else torch.as_tensor(teacher['logscore_bonus'],dtype=torch.float32)
    target_logp=(base_logscore+bonus).log_softmax(-1).detach()
    target=torch.as_tensor(np.c_[source_boxes[teacher['query_indices']],np.zeros(len(idx))] if control else teacher['target_boxes7'],dtype=torch.float32)
    accepted=torch.zeros(len(idx),dtype=torch.bool) if control else torch.as_tensor(teacher['accepted'],dtype=torch.bool)
    # Source dimensions are never made positive with abs/clipping. Invalid candidates
    # are excluded only from pose training, and are still counted/returned diagnostically.
    valid=(box[idx,3:6]>0).all(-1);accepted &= valid
    rng=torch.get_rng_state().clone()
    try:
        model=QueryResidual();assert sum(x.numel() for x in model.parameters())==1445
        opt=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate']);trace=[];outputs={}
        for step in range(4):
            pred,logits,residual=model(feat,box,score)
            if step in CONFIG['steps']:
                top=int(idx[logits[idx].argmax()]);outputs[step]={'box7':pred[top].detach().numpy().copy(),'query':top}
            if step==3:break
            # Exact zero bridge matters: tiny KL roundoff was amplified by Adam
            # even for unchanged source targets. Same loss in method/control.
            rank=torch.nn.functional.smooth_l1_loss(logits[idx].log_softmax(-1),target_logp)
            if accepted.any():
                p=pred[idx][accepted];t=target[accepted];d=box[idx,3:6][accepted]
                center=torch.nn.functional.smooth_l1_loss((p[:,:3]-t[:,:3])/d,torch.zeros_like(p[:,:3]))
                # Keep selected modes. No averaging of yaw angles or centers across modes.
                angle=torch.remainder(p[:,6]-t[:,6]+np.pi/2,np.pi)-np.pi/2
                pose=center+torch.nn.functional.smooth_l1_loss(angle,torch.zeros_like(angle))
            else:pose=residual.sum()*0.
            trust=residual.square().mean();loss=CONFIG['rank_weight']*rank+CONFIG['pose_weight']*pose+CONFIG['trust_region_weight']*trust
            opt.zero_grad(set_to_none=True);loss.backward();count=sum(int(p.grad is not None and torch.count_nonzero(p.grad)>0) for p in model.parameters());grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip']));opt.step()
            trace.append({'step':step+1,'loss':float(loss.detach()),'rank_logprob_huber':float(rank.detach()),'pose_loss':float(pose.detach()),'trust':float(trust.detach()),'nonzero_grad_tensors':count,'gradient_norm':grad,'parameter_delta_l2':float(sum(p.detach().square().sum() for p in model.parameters()).sqrt()),'accepted_geometry_candidates':int(accepted.sum())})
        return outputs,trace
    finally:torch.set_rng_state(rng)
