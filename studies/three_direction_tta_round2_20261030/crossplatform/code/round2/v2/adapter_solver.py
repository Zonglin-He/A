"""Fixed masked-Adam versus function-space constrained-Adam factor."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import residual_adapter as original

CONFIG={'learning_rate':.01,'steps':[0,1,3],'gradient_clip':1.,'trust_weight':.1,
        'center_step_m':.05,'center_total_m':.15,'yaw_step_radians':np.pi/36,
        'yaw_total_radians':np.pi/12,'logscore_step':.1,'logscore_total':.3,
        'maximum_backtrack_halvings':10,'objective_tolerance':1e-10,'bound_tolerance':1e-6}

class MaskedResidual(nn.Module):
    def __init__(self,mask):
        super().__init__();self.linear=nn.Linear(288,5);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias);self.register_buffer('mask',mask.float())
    def forward(self,features,boxes,scores):
        h=features/torch.sqrt(features.square().mean(-1,keepdim=True)+1e-8);res=self.linear(h)*self.mask[:,None]
        centers=boxes[:,:3]+.15*boxes[:,3:6]*res[:,:3].tanh();yaw=(np.pi/2)*res[:,3].tanh();base_log=scores.clamp_min(1e-30).log();logits=base_log+res[:,4]
        centers=torch.where(self.mask.bool()[:,None],centers,boxes[:,:3]);yaw=torch.where(self.mask.bool(),yaw,torch.zeros_like(yaw));logits=torch.where(self.mask.bool(),logits,base_log)
        pred=torch.cat([centers,boxes[:,3:6],yaw[:,None]],-1);return pred,logits,res

def wrapped(a):return torch.remainder(a+np.pi/2,np.pi)-np.pi/2
def run(features,source_boxes,source_scores,teacher,solver='constrained',control=False):
    assert solver in ['adam','constrained'];features=torch.as_tensor(features,dtype=torch.float32);boxes=torch.as_tensor(source_boxes,dtype=torch.float32);scores=torch.as_tensor(source_scores,dtype=torch.float32);ids=torch.as_tensor(teacher['query_indices'],dtype=torch.long);accept=torch.as_tensor(teacher['accepted'],dtype=torch.bool)&(boxes[ids,3:6]>0).all(-1)
    if control:accept=torch.zeros_like(accept)
    mask=torch.zeros(len(boxes),dtype=torch.bool);mask[ids[accept]]=True
    bonus=torch.zeros(len(ids)) if control else torch.as_tensor(teacher['logscore_bonus'],dtype=torch.float32);targetlog=(scores[ids].clamp_min(1e-30).log()+bonus).log_softmax(-1).detach();target=torch.as_tensor(teacher['target_boxes7'],dtype=torch.float32)
    rng=torch.get_rng_state().clone()
    try:
        model=MaskedResidual(mask);parameters=list(model.parameters());opt=torch.optim.Adam(parameters,lr=CONFIG['learning_rate']);outputs={};trace=[]
        def forwardloss():
            pred,logit,res=model(features,boxes,scores);rank=torch.nn.functional.smooth_l1_loss(logit[ids].log_softmax(-1),targetlog)
            if accept.any():
                bp=pred[ids][accept];bt=target[accept];dims=boxes[ids,3:6][accept];cent=torch.nn.functional.smooth_l1_loss((bp[:,:3]-bt[:,:3])/dims,torch.zeros_like(bp[:,:3]));ang=wrapped(bp[:,6]-bt[:,6]);pose=cent+torch.nn.functional.smooth_l1_loss(ang,torch.zeros_like(ang))
            else:pose=res.sum()*0.
            trust=res.square().mean();return rank+pose+CONFIG['trust_weight']*trust,pred,logit,res,rank,pose,trust
        for step in range(4):
            loss,pred,logit,res,rank,pose,trust=forwardloss();top=int(ids[logit[ids].argmax()]);
            if step in CONFIG['steps']:outputs[step]={'box7':pred[top].detach().numpy().copy(),'query':top,'chosen_accepted':bool(mask[top])}
            assert torch.equal(pred[~mask,:6],boxes[~mask]) and torch.count_nonzero(pred[~mask,6])==0
            assert torch.equal(logit[~mask],scores[~mask].clamp_min(1e-30).log())
            if step==3:break
            before=[p.detach().clone() for p in parameters];prevpred=pred.detach().clone();prevlog=logit.detach().clone();beforeloss=float(loss.detach());opt.zero_grad(set_to_none=True);loss.backward();nonzero=sum(int(p.grad is not None and torch.count_nonzero(p.grad)>0) for p in parameters);grad=float(torch.nn.utils.clip_grad_norm_(parameters,CONFIG['gradient_clip']));opt.step();delta=[p.detach()-b for p,b in zip(parameters,before)];alpha=1.;halvings=0;accepted_step=True
            if solver=='constrained':
                # tanh is1-Lipschitz, so this bounds physical output change
                # despite288feature fan-in and sourceextent scaling.
                h=features/torch.sqrt(features.square().mean(-1,keepdim=True)+1e-8);dr=(h@delta[0].T+delta[1])*mask[:,None];centerbound=float(torch.linalg.vector_norm(.15*boxes[:,3:6]*dr[:,:3],dim=-1).max());yawbound=float((np.pi/2*abs(dr[:,3])).max());scorebound=float(abs(dr[:,4]).max());alpha=min(1.,CONFIG['center_step_m']/max(centerbound,1e-30),CONFIG['yaw_step_radians']/max(yawbound,1e-30),CONFIG['logscore_step']/max(scorebound,1e-30));accepted_step=False
                for halvings in range(CONFIG['maximum_backtrack_halvings']+1):
                    with torch.no_grad():
                        for p,b,d in zip(parameters,before,delta):p.copy_(b+alpha*d)
                        candidate,cp,cl,cr,*_=forwardloss();tol=CONFIG['bound_tolerance'];constraints=(float(torch.linalg.vector_norm(cp[:,:3]-prevpred[:,:3],dim=-1).max())<=CONFIG['center_step_m']+tol and float(torch.linalg.vector_norm(cp[:,:3]-boxes[:,:3],dim=-1).max())<=CONFIG['center_total_m']+tol and float(abs(wrapped(cp[:,6]-prevpred[:,6])).max())<=CONFIG['yaw_step_radians']+tol and float(abs(wrapped(cp[:,6])).max())<=CONFIG['yaw_total_radians']+tol and float(abs(cl-prevlog).max())<=CONFIG['logscore_step']+tol and float(abs(cr[:,4]).max())<=CONFIG['logscore_total']+tol)
                    if constraints and float(candidate)<=beforeloss+CONFIG['objective_tolerance']:accepted_step=True;break
                    alpha*=.5
                if not accepted_step:
                    with torch.no_grad():
                        for p,b in zip(parameters,before):p.copy_(b)
                    alpha=0.
            with torch.no_grad():after,ap,al,ar,*_=forwardloss()
            trace.append({'step':step+1,'solver':solver,'loss_before':beforeloss,'loss_after':float(after),'rank_before':float(rank.detach()),'pose_before':float(pose.detach()),'trust_before':float(trust.detach()),'nonzero_grad_tensors':nonzero,'gradient_norm':grad,'parameter_delta_l2':float(sum(p.detach().square().sum() for p in parameters).sqrt()),'effective_proposal_scale':alpha,'backtrack_halvings':halvings,'step_accepted':accepted_step,'max_center_step_m':float(torch.linalg.vector_norm(ap[:,:3]-prevpred[:,:3],dim=-1).max()),'max_center_total_m':float(torch.linalg.vector_norm(ap[:,:3]-boxes[:,:3],dim=-1).max()),'max_yaw_step_radians':float(abs(wrapped(ap[:,6]-prevpred[:,6])).max()),'max_yaw_total_radians':float(abs(wrapped(ap[:,6])).max()),'max_logscore_step':float(abs(al-prevlog).max()),'max_logscore_total':float(abs(ar[:,4]).max()),'unaccepted_output_exact_source':True})
        return outputs,trace
    finally:torch.set_rng_state(rng)
