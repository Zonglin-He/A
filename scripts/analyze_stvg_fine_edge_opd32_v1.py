"""Postseal fixed-size geometric ceiling, actual edge-pool quality and joint reward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_edge_opd32_v1 import D,P,FINE,PRE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def fixedsize(w,h,g,clip):
 import numpy as np
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import iou
 g=np.asarray(g,dtype=float);gw,gh=g[2:]-g[:2];ii=min(w,gw)*min(h,gh);analytic=ii/(w*h+gw*gh-ii)
 if not clip:return float(analytic),float(analytic)
 def intervals(s,t,d):
  start=max(0.,min(s,t-d));out=[(start,start+d)]
  out.extend((0.,v)for v in np.unique(np.clip([d,s,t],0,d))if v>0)
  return out
 boxes=np.array([[a,c,b,d]for a,b in intervals(g[0],g[2],w)for c,d in intervals(g[1],g[3],h)])
 return float(analytic),float(iou(boxes,np.asarray(g)).max())

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import xy,iou,summarize_frame
 from vg_tta.stvg_fine_edge_opd32_v1 import GROUPS,build,feedback
 torch.set_num_threads(4);p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32;gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[];extra=0
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));x=load(FINE/'features'/(a+'.pt'));co=load(PRE/'features'/(a+'.pt'));E=o['outputs']['E_MBR'];ids=np.asarray(o['frame_ids']);known=np.array([int(i)in gt[ds][0][q]for i in ids]);unobs=~np.isin(np.arange(len(ids)),o['planned_positions']);m=known&unobs;support=m&o['allowed'].numpy();row=plans[ds][q];gtxy=np.array([gt[ds][0][q].get(int(i),[np.nan]*4)for i in ids]);Qe=np.full(len(E),np.nan);Qe[known]=iou(xy(E,row,ds)[known],gtxy[known]);ceil=np.full(len(E),np.nan);plain=ceil.copy()
  for t in np.flatnonzero(known):plain[t],ceil[t]=fixedsize(E[t,2]*row['input']['width'],E[t,3]*row['input']['height'],gtxy[t],ds=='hc2');assert ceil[t]+1e-10>=Qe[t]
  geo={label:dict(known_nodes=int(mask.sum()),E_IoU=mean(Qe[mask]),unclipped_fixed_size_free_center_IoU=mean(plain[mask]),official_clip_aware_free_center_IoU=mean(ceil[mask]),recoverable_by_center_pp=mean(100*(ceil[mask]-Qe[mask])),unrecoverable_with_fixed_size_pp=mean(100*(1-ceil[mask])))for label,mask in [('all_known_unobserved',m),('predicted_scope_known_unobserved',support)]}
  pos=x['positions'];bb=[f['boxes'][k].tolist()for f,k in zip(co['frames'],co['selected'])if k>=0];ctx=build(x['maps'],x['valid'],pos,bb);ranks={};joint={}
  for mode,fit in o['fits'].items():
   ranks[mode]={};joint[mode]=[];perm=o['permutation']if mode=='Edge_shuffle'else None;composed=feedback(ctx,torch.stack([fit['path'][j]['boxes']for j in [1,2]],1),torch.zeros(len(E),2,2),.1,perm);extra+=2*len(E)
   for st,t in enumerate(fit['rounds']):
    b=t['candidates'].numpy();quality=np.full(b.shape[:2],np.nan)
    for j in np.flatnonzero(known):quality[j]=iou(xy(b[j],row,ds),gtxy[j])
    frames=[];massgain=[];expected=[];missing_visual=0;headroom=[]
    for j in np.flatnonzero(support):
     for e,idx in enumerate(GROUPS.numpy()):
      ef=t['edge_feedback'];valid=ef['valid'][j,e].numpy();qq=quality[j,idx];legal=t['legal'][j,idx].numpy();headroom.append(100*max(0,float((qq[legal]-Qe[j]).max())))
      if not valid.any():missing_visual+=1;continue
      rr=ef['rewards'][j,e].numpy();prob=ef['prob'][j,e].numpy();pr=ef['prior'][j,e].numpy();z=summarize_frame(qq,rr,prob,valid,Qe[j],bool(pos),Qe[j],Qe[j]);[z.pop(k)for k in ['on_gain_pp','step1_gain_pp','posterior_good_but_On_nonpositive']];z={k.replace('above_E_','above_shared_base_').replace('outranks_E_','outranks_shared_base_'):v for k,v in z.items()};frames.append(z)
      if pos:
       good=(qq>Qe[j]+1e-10)&valid;massgain.append(float(prob[good].sum()-pr[good].sum()));expected.append(float(100*((prob-pr)*qq).sum()))
    ranks[mode][str(st+1)]=dict(known_scope_nodes=int(support.sum()),known_scope_edge_slots=int(support.sum())*4,defined_visual_edge_slots=len(frames),missing_visual_edge_slots=missing_visual,legal_candidate_best_GT_gain_pp=mean(headroom),frame_edge_means={k:mean([v[k]for v in frames])for k in frames[0]}if frames else {},teacher_minus_prior_good_mass=mean(massgain),teacher_minus_prior_expected_IoU_pp=mean(expected))
    # Diagnostic only, after all deployed outputs already sealed; never accepts a step.
    new=fit['path'][st+1]['boxes'];old=fit['path'][st]['boxes'];delta=(composed['rewards'][:,st]-(t['visual_feedback']['rewards'][:,0]if st==0 else composed['rewards'][:,st-1])).numpy();joint[mode].append(dict(step=st+1,teacher_complete_box_reward_delta=mean(delta[o['allowed'].numpy()])if pos else None,changed_size_nodes=int((new[:,2:]!=torch.as_tensor(E)[:,2:]).any(-1).sum()),minimum_width=float(new[:,2].min()),minimum_height=float(new[:,3].min())))
  rows.append(dict(alias=a,dataset=ds,active=bool(pos),geometry=geo,edge_ranking=ranks,complete_box_diagnostics=joint))
 groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(geometry={scope:{k:paired([r['geometry'][scope][k]for r in rr])for k in rr[0]['geometry'][scope]if k!='known_nodes'}for scope in rr[0]['geometry']},edge_ranking={mode:{st:{k:paired([r['edge_ranking'][mode][st]['frame_edge_means'].get(k)for r in rr])for k in sorted(set(k for r in rr for k in r['edge_ranking'][mode][st]['frame_edge_means']))}for st in ['1','2']}for mode in ['Edge_fixed','Edge_live','Edge_shuffle']})
 put(P/'POST_SEAL_DIAGNOSTICS.json',dict(status='complete',count=32,groups=groups,rows=rows,GT_offline_after_all_outputs=True,free_center_size_ceiling_is_per_frame_not_tube_or_online=True,HC_lower_coordinate_clipping_separately_handled=True,second_round_candidates_GT_evaluated=True,extra_composed_box_ROI_evaluations=extra,extra_diagnostic_evaluations_not_used_for_updates=True,geometry_not_limited_by_teacher_or_exploration=True));print('postseal diagnostics complete',extra,flush=True)
if __name__=='__main__':main()
