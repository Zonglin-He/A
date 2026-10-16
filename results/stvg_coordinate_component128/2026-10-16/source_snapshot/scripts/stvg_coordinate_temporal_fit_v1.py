"""512 shared zero-initialized endpoint weights; locked coordinate preference-KL study."""
import torch
from scripts.stvg_sparse_r1_student_v1 import _target_pair
from methods.decota_final_simplified_v1.objectives import project
from scripts.stvg_opd_p6_existing_common001 import config
CFG={'lr':.01,'steps':5,'tau_T':1.,'l2':.001}

def fit(case,hidden,candidates,rewards,fit_config):
 cfg=dict(fit_config)
 assert set(cfg)==set(CFG) and 0<=cfg["lr"]<=.03 and 0<cfg["tau_T"]<=4 and 1<=cfg["steps"]<=20 and 0<=cfg["l2"]<=.1
 assert len(hidden)==2 and all(h.ndim==2 and h.shape[-1]==256 for h in hidden)
 assert len(candidates)==len(rewards)>=1 and all(r is None or -1<=r<=1 for r in rewards)
 ev=project(case['zero']['logits'],case['zero']['actions'],case['records'],config(case['dataset']));hs=[h.detach().double().cpu() for h in hidden]
 valid=[j for j,r in enumerate(rewards) if r is not None];theta=torch.zeros(256,2,dtype=torch.float64,requires_grad=True)
 def decode():
  iv=[]
  for h,e,rec in zip(hs,ev['offsets'],case['records']):
   ij=e['ij'].cpu();score=e['score'].double().cpu()+h[ij[0]]@theta[:,0]+h[ij[1]]@theta[:,1];at=int(score.detach().argmax());s,t=map(int,ij[:,at]);iv.append([rec['frame_ids'][s],rec['frame_ids'][t]+1])
  return [min(x[0] for x in iv),max(x[1] for x in iv)]
 assert decode()==case['Projection_interval'], 'zero shared readout must preserve Projection prior exactly'
 features=[];base=[]
 for j in valid:
  vals=[];emb=[]
  for h,e,rec in zip(hs,ev['offsets'],case['records']):
   s,t,_=_target_pair(rec['frame_ids'],candidates[j]['interval_frames']);ij=e['ij'].cpu();at=int(torch.nonzero((ij[0]==s)&(ij[1]==t))[0]);vals.append(e['score'][at].double().cpu());emb.append(torch.stack([h[s],h[t]],1))
  base.append(torch.stack(vals).mean());features.append(torch.stack(emb).mean(0))
 base=torch.stack(base) if base else torch.empty(0,dtype=torch.float64);feat=torch.stack(features) if features else torch.empty(0,256,2,dtype=torch.float64);r=torch.tensor([rewards[j] for j in valid],dtype=torch.float64)
 p0=base.softmax(0) if valid else base;target=(base+r/cfg['tau_T']).softmax(0).detach() if valid else base
 no_information=len(valid)<2 or bool((r==r[0]).all())
 if no_information:target=p0.detach().clone()
 opt=torch.optim.Adam([theta],lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0);path=[];steps=0
 def loss():
  if not valid:return theta.sum()*0
  scores=base+(feat*theta).sum((1,2));lp=scores.log_softmax(0)
  return (target*(target.clamp_min(1e-300).log()-lp)).sum()+cfg['l2']*theta.square().sum()
 for k in range(cfg['steps']+1):
  L=loss();path.append(dict(step=k,loss=float(L.detach()),theta=theta.detach().clone(),physical_interval=decode(),parameter_norm=float(theta.detach().norm()),candidate_probabilities=(base+(feat*theta).sum((1,2))).softmax(0).detach() if valid else base))
  if k==cfg['steps'] or no_information or cfg['lr']==0:continue
  grad=torch.autograd.grad(L,theta)[0];assert torch.isfinite(grad).all();opt.zero_grad(set_to_none=True);theta.grad=grad;opt.step();steps+=1
 if no_information:assert torch.count_nonzero(theta)==0 and steps==0
 return dict(status='projection_only_zero_learning_rate' if cfg['lr']==0 else ('updated' if steps else 'no_information_exact_zero_parameter_noop'),path=path,final_interval=decode(),optimizer_steps=steps,active_parameters=512,original_frozen_features=True,query_reset=True,shared_across_frames=True,per_logit_residual=False,config=dict(cfg),valid_candidate_indices=valid,p0=p0,target=target,rewards=r,no_information=no_information,GT_read=False)

def contracts():
 # Analytic score-level KL gradient and exact equal-feedback target identity.
 b=torch.tensor([.1,.3,-.2],dtype=torch.float64);r=torch.tensor([.6,-.1,.2],dtype=torch.float64);s=b.clone().requires_grad_(True);t=(b+r).softmax(0).detach();L=(t*(t.log()-s.log_softmax(0))).sum();g=torch.autograd.grad(L,s)[0];assert torch.allclose(g,s.softmax(0)-t,atol=1e-15,rtol=0)
 equal=torch.zeros_like(r);assert torch.equal((b+equal).softmax(0),b.softmax(0))
 h=torch.randn(8,256,dtype=torch.float64,generator=torch.Generator().manual_seed(7));w=torch.zeros(256,2,dtype=torch.float64,requires_grad=True);base=torch.randn(8,2,dtype=torch.float64);assert torch.equal(base+h@w,base) and w.numel()==512
 return {'status':'pass','CPU_only':True,'synthetic_only':True,'checks':3,'GT_read':False}
