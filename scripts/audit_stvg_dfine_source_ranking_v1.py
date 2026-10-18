"""Independent NumPy expected-IoU and scalar good/bad source-ranking audit."""
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_dfine_source_capture_v1 import D,P
from scripts.run_stvg_dfine_refinement32_v1 import verify
from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
p=verify();torch.set_num_threads(4);scores=read(P/'SOURCE_FEEDBACK_RANKING_PUBLIC.json');lookup={r['alias']:r for r in scores['rows']};count=collections.Counter();err=0.
for s in read(P/'SOURCE_TRAINING.json')['rows']:
 ds=s['source_domain'];rec={r['alias']:r for r in p['source']if r['dataset']==ds};v=load(D/'source_validation'/ds/f"epoch{s['selected']['epoch']:02}.pt")
 for o in v['packets']:
  a=o['alias'];r=rec[a];x=load(D/'source_features'/(a+'.pt'));label=load(D/'source_supervision'/(a+'.pt'));tt=o['paths']['Fixed_distill'];b=torch.cat([t['candidates']for t in tt],1);reward=torch.cat([t['rewards']for t in tt],1).numpy();lp=torch.cat([t['prior']for t in tt],1);post=(lp+torch.tensor(reward)/.1).softmax(-1).numpy();teacher=tt[0]['teacher'];B=b.numpy();T=teacher['boxes'].numpy();X=np.concatenate((B[...,:2]-B[...,2:]/2,B[...,:2]+B[...,2:]/2),-1)[:,:,None];Y=np.concatenate((T[...,:2]-T[...,2:]/2,T[...,:2]+T[...,2:]/2),-1)[:,None];inter=np.maximum(0,np.minimum(X[...,2:],Y[...,2:])-np.maximum(X[...,:2],Y[...,:2])).prod(-1);union=np.maximum(0,X[...,2:]-X[...,:2]).prod(-1)+np.maximum(0,Y[...,2:]-Y[...,:2]).prod(-1)-inter;ii=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);mask=teacher['valid'].numpy()[:,None];rr=(ii*mask).sum(-1)/np.maximum(mask.sum(-1),1);error=float(abs(rr-reward).max());err=max(err,error);assert error<1e-12;count['independent_expected_IoU_rewards']+=rr.size
  valid=((cell_overlap(b,*x['valid'].shape[-2:])*x['valid'][:,None]).sum((-1,-2))>1e-12).numpy();valid[:,0]=True;results=collections.defaultdict(list)
  for t,fid in enumerate(x['frame_ids']):
   if t in x['planned_positions']or int(fid)not in label['gt']:continue
   vals=np.array([iou_gt(bb,label['gt'][int(fid)],dict(input=r['input']),ds)for bb in B[t]]);gain=vals-vals[0];G=[k for k in range(len(vals))if valid[t,k]and gain[k]>1e-10];H=[k for k in range(len(vals))if valid[t,k]and gain[k]<-1e-10];strong=any(valid[t,k]and gain[k]>=.05 for k in range(len(vals)))
   for name,score in [('raw',reward[t]),('posterior',post[t])]:
    if o['active']and G and H:results[name+'_auc'].append(sum(1 if score[i]-score[j]>1e-12 else .5 if abs(score[i]-score[j])<=1e-12 else 0 for i in G for j in H)/(len(G)*len(H)))
    if o['active']and strong:results['strong_pool_no_good_outranks_E_'+name].append(float(not any(score[k]>score[0]+1e-12 for k in G)))
  for key in ['raw_auc','posterior_auc','strong_pool_no_good_outranks_E_raw','strong_pool_no_good_outranks_E_posterior']:
   z=lookup[a]['means'][key];arr=results[key]
   if arr:assert abs(float(np.mean(arr))-z)<1e-12
   else:assert z is None
   count['independent_source_E_relative_statistics']+=1
verify();put(P/'ROOT_SOURCE_RANKING_AUDIT.json',dict(status='pass',source_queries=32,checks=dict(count),max_expected_reward_error=err,no_new_forward_or_update=True));print(dict(count),err,flush=True)
a=read(D/'PUBLIC_FILES_FINAL.json');a['files']+=['scripts/audit_stvg_dfine_source_ranking_v1.py',str((P/'ROOT_SOURCE_RANKING_AUDIT.json').relative_to(ROOT))];put(D/'PUBLIC_FILES_FINAL.json',a,True)
# Stage was local-only; preserve it and restage the complete additional audit.
import shutil
r=D/'publication/preranking_audit_stage002';r.mkdir()
for n in ['COMMIT_PLAN.json','private-export.index']:shutil.move(str(D/'publication'/n),str(r/n))
