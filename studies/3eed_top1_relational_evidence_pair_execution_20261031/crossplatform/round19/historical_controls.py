"""Post-score fixed budget/objective checks on the historical q0-negative case.
No new method/GT selection, no source-model forwards, no newG performance claim.
"""
import sys,time
import numpy as np
from common import *
from features_v2 import feature
from fit_pair_v2 import optimize
sys.path.insert(0,str(ROOT/'crossplatform/round18'))
from geometry import iou_matrix
def main():
 import torch
 locked('HISTORICAL_CONTROL_LOCK.json');out=ROOT/'crossplatform/round19/historical_controls_attempt001';out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with np.load(ROOT/'crossplatform/round17/features96_attempt001/projection_source_PRIVATE.npz') as a:source={'weight':torch.tensor(a['4.weight'],device='cuda'),'bias':torch.tensor(a['4.bias'],device='cuda')}
 rr=[]
 for r in read(EXP.parent/'input/A64_PRIVATE.json'):
  z=feature(r);p=read(ROOT/'crossplatform/round17/G_selected64_attempt001/cases_PRIVATE'/(r['id']+'.json'))['preference'];qa=int(z['scores'].argmax())
  if p['status']!='usable' or qa not in p['negative_queries']:continue
  qb=max(p['positive_queries'],key=lambda q:(float(z['scores'][q]),-q));hist=ROOT/'crossplatform/round19/historical_pair_interface_attempt001'/(r['id']+'.npz')
  with np.load(hist) as a:hinge_new=a['after'].copy()
  spec={'margin':.001,'lr':1e-4,'max_steps':32};_,before,hinge_old,dold=optimize(z,qa,qb,spec,source,'cuda')
  layer=torch.nn.Linear(288,64,device='cuda');layer.load_state_dict(source);base={k:v.detach().clone() for k,v in layer.named_parameters()};x=torch.tensor(z['x'],device='cuda');t=torch.tensor(z['tokens'],device='cuda');mask=torch.tensor(z['token_mask'],device='cuda');ss0=torch.tensor(z['scores'],device='cuda');pos=p['positive_queries'];neg=p['negative_queries'];opt=torch.optim.AdamW(layer.parameters(),lr=.001,weight_decay=0.)
  def score():return ((torch.nn.functional.normalize(layer(x),dim=-1)@t.T/.07).softmax(-1)*mask).sum(-1)
  for _ in range(128):
   opt.zero_grad();s=score().clamp_min(1e-12);gap=s[pos].log().mean()-s[neg].log().mean();p0=(ss0/ss0.sum()).clamp_min(1e-12);pp=(s/s.sum()).clamp_min(1e-12);loss=torch.nn.functional.softplus(1-gap)+.1*(p0*(p0.log()-pp.log())).sum()+sum((v-base[k]).square().mean() for k,v in layer.named_parameters());loss.backward();opt.step()
  mean_new=score().detach().cpu().numpy()
  with np.load(EXP/'cache_audit_PRIVATE'/(r['id']+'.npz')) as a:mean_old=a['after'].copy()
  far=(iou_matrix(z['boxes'],z['boxes'][[qa,qb]]).max(1)<.1);results={}
  for name,s in [('source',z['scores']),('mean_original_budget',mean_old),('pair_original_budget',hinge_old),('mean_larger_budget',mean_new),('pair_larger_budget',hinge_new)]:
   prob=s/s.sum();results[name]={'sA':float(s[qa]),'sB':float(s[qb]),'gap_B_minus_A':float(s[qb]-s[qa]),'B_above_A':bool(s[qb]>s[qa]),'B_global_first':bool(s.argmax()==qb),'queries_score_above_half':int((s>.5).sum()),'far_box_queries':int(far.sum()),'far_box_queries_score_above_half':int(((s>.5)&far).sum()),'far_box_max_score':float(s[far].max()) if far.any() else None,'candidate_query_entropy':float(-(prob*np.log(np.maximum(prob,1e-30))).sum()),'query_entropy_not_instance_entropy':True}
  rr.append({'alias':r['alias'],'platform':r['platform'],'results':results,'pair_original_budget_steps':dold['steps'],'not_primary_or_newG_confirmation':True});np.savez_compressed(out/(r['id']+'.npz'),**{'source':z['scores'],'mean_original_budget':mean_old,'pair_original_budget':hinge_old,'mean_larger_budget':mean_new,'pair_larger_budget':hinge_new})
 dump(out/'RESULTS.json',rr);seal(out,'R19_postscore_historical_objective_budget_diagnostic',n=len(rr),no_new_GT_or_source_forward=True);print('HISTORY_CONTROLS_COMPLETE')
if __name__=='__main__':main()
