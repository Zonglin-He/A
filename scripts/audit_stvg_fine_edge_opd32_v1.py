"""Independent NumPy four-edge proposals, one-dimensional KL and metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_edge_opd32_v1 import D,P,FINE,PRE,CENTER,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.audit_stvg_roi_layout_opd32_v1 import sample,sm

def main():
 import numpy as np,torch
 from statistics import NormalDist
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maximum=dict(roi_reward=0.,teacher_prob=0.,state=0.,noise=0.)
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};groups=np.array([[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16]])
 def decode(E,a):
  d=.5*E[:,None,[2,3,2,3]]*np.tanh(.5*a);out=np.broadcast_to(E[:,None],(*a.shape[:2],4)).copy();out[:,:,:2]+=(d[:,:,:2]+d[:,:,2:])/2;out[:,:,2:]+=d[:,:,2:]-d[:,:,:2];return out
 for r in p['records']:
  alias=r['alias'];o=load(D/'predictions'/(alias+'.pt'));x=load(FINE/'features'/(alias+'.pt'));co=load(PRE/'features'/(alias+'.pt'));old=load(CENTER/'predictions'/(alias+'.pt'));E=x['E'].numpy().astype(np.float64);n=len(E);allowed=o['allowed'].numpy();assert torch.equal(o['allowed'],old['allowed']);assert np.array_equal(o['outputs']['Center_live'],old['outputs']['Fine_live']);checks['exact_Center_reuse_and_scope']+=1
  seed=int.from_bytes(hashlib.sha256(('fine-edge32-v1|'+alias).encode()).digest()[:8],'little');rng=np.random.default_rng(seed);noise=[]
  for st in range(2):
   u=(rng.integers(1,2**53,size=(n,4,2)).astype(np.float64)/2**53+np.arange(2)[None,None])/4;v=np.concatenate([u,1-u[...,::-1]],-1);noise.append(np.vectorize(NormalDist().inv_cdf)(v))
  eps=np.stack(noise);err=float(abs(eps-o['noise'].numpy()).max());maximum['noise']=max(maximum['noise'],err);assert err<1e-12 and abs(eps.sum(-1)).max()<1e-12;checks['independent_stratified_antithetic_noise']+=eps.size
  pos=x['positions'];boxes=np.asarray([f['boxes'][k].tolist()for f,k in zip(co['frames'],co['selected'])if k>=0]).reshape(-1,1,4)
  if pos:tt,tw=sample(x['maps'][pos],x['valid'][pos],boxes);tt=tt[:,0];tw=tw[:,0]
  else:tt=np.empty((0,9,1024));tw=np.empty((0,9))
  for mode,fit in o['fits'].items():
   mu=np.zeros((n,4));perm=o['permutation']if mode=='Edge_shuffle'else None;maps=x['maps'][perm]if perm is not None else x['maps'];vv=x['valid'][perm]if perm is not None else x['valid']
   for st,t in enumerate(fit['rounds']):
    center=np.zeros_like(mu)if mode=='Edge_fixed'else mu.copy();aa=np.broadcast_to(center[:,None],(n,17,4)).copy()
    for e in range(4):aa[:,1+4*e:5+4*e,e]+=eps[st,:,e]
    bb=decode(E,aa);assert abs(bb-t['candidates'].numpy()).max()<1e-12;assert abs(aa-t['latent_actions'].numpy()).max()<1e-12;legal=np.isfinite(bb).all(-1)&(bb[:,:,2:]>0).all(-1);assert np.array_equal(legal,t['legal']);ph,w=sample(maps,vv,bb);ok=w.sum(-1)>1e-6;raw=t['visual_feedback'];assert np.array_equal(ok,raw['valid'])
    if pos:
     mass=w[None]*tw[:,None,None];per=(np.einsum('nkjc,ajc->ankj',ph,tt)*mass).sum(-1)/np.maximum(mass.sum(-1),1e-12);err=float(abs(per-raw['per_anchor_rewards'].numpy()).max());maximum['roi_reward']=max(maximum['roi_reward'],err);assert err<3e-5;checks['independent_ROI_reward_scalars']+=per.size
    ef=t['edge_feedback'];rr=raw['rewards'].numpy()[:,groups];valid=ok[:,groups]&legal[:,groups];safe=valid.copy();safe[~safe.any(-1)]=True;ee=np.concatenate((np.zeros((n,4,1)),eps[st]),-1);prior=sm(np.where(safe,-.5*ee**2,-np.inf));qq=sm(np.where(safe,-.5*ee**2+rr/.1,-np.inf));info=valid.any(-1)&((np.where(valid,rr,-np.inf).max(-1)-np.where(valid,rr,np.inf).min(-1))>1e-6)&bool(pos);qq[~info]=prior[~info];err=float(abs(qq-ef['prob'].numpy()).max());maximum['teacher_prob']=max(maximum['teacher_prob'],err);assert err<1e-12 and np.array_equal(info,ef['informative']);acts=np.stack([aa[:,groups[e],e]for e in range(4)],1);sp=sm(np.where(safe,-.5*(acts-mu[:,:,None])**2,-np.inf));g=((sp-qq)*acts).sum(-1);g[~(allowed[:,None]&info)]=0;new=mu-.5*g;err=float(abs(new-t['after'].numpy()).max());maximum['state']=max(maximum['state'],err);assert err<1e-12 and abs(g-t['gradient'].numpy()).max()<1e-12;out=decode(E,new[:,None])[:,0];out[~allowed]=E[~allowed];assert abs(out-fit['path'][st+1]['boxes'].numpy()).max()<1e-12;assert not new[~allowed].any();mu=new;checks['four_conditional_KL_actual_rounds']+=1;checks['per_edge_conditional_distributions']+=4*n
   assert np.array_equal(o['outputs'][mode],fit['final']);checks['final2_output']+=1
  assert torch.equal(o['fits']['Edge_fixed']['path'][1]['boxes'],o['fits']['Edge_live']['path'][1]['boxes']);checks['first_Fixed_Live_exact']+=1
  for b in o['outputs'].values():
   assert np.array_equal(b[~allowed],E[~allowed])and(b[:,2:]>0).all();checks['outside_planned_exact_and_legal_sizes']+=1
   if not pos:assert np.array_equal(b,E);checks['no_reference_noop']+=1
  independent_metrics(o,plans[r['dataset']][r['query_ordinal']],gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']],rows[alias],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,ep in g['paired'].items():
   a,b=key.split('_minus_')
   for e,want in ep.items():assert paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rr])==want;checks['parent_bootstrap']+=1
 assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),maximum_errors=maximum,no_GPU_model_rerun=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maximum,flush=True)
if __name__=='__main__':main()
