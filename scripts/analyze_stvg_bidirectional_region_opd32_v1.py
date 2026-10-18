"""Saved candidate quality, E-relative ranks, reward scale and real step response."""
from pathlib import Path
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_bidirectional_region_opd32_v1 import D,P,OLD,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import summarize_frame
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();rows=[]
 for r in p['records']:
  a=r['alias'];o=load(D/'predictions'/(a+'.pt'));old=load(OLD/'predictions'/(a+'.pt'));qq=load(ROOT/'artifacts/stvg_cosine_candidate_headroom32_v1/diagnostics'/(a+'.pt'));Q=qq['quality'];mask=qq['mask'];rd=o['direct'];valid=rd['feedback']['valid'].numpy().copy();valid[:,0]=True;raw=rd['feedback']['rewards'].numpy();pt=rd['feedback']['prob'].numpy();frames=[]
  assert torch.equal(rd['candidates'],old['direct']['candidates'])
  if o['active']:assert torch.equal(rd['feedback']['forward_rewards'],old['direct']['feedback']['per_anchor_rewards'])
  for t in np.flatnonzero(mask):
   f=summarize_frame(Q[t],raw[t],pt[t],valid[t],Q[t,0],o['active'],Q[t,0],Q[t,0])
   for k in ['on_gain_pp','step1_gain_pp','posterior_good_but_On_nonpositive']:f.pop(k)
   frames.append(f)
  means={k:mean([f[k]for f in frames])for k in frames[0]}if frames else {};allowed=torch.ones(len(o['frame_ids']),dtype=torch.bool);allowed[o['planned_positions']]=False;scale={}
  for mode,fit in o['fits'].items():
   scale[mode]=[]
   for t in fit['rounds']:
    fb=t['feedback'];reward=fb['rewards'][allowed];forward=fb['forward_rewards'][:,allowed];reverse=fb['reverse_rewards'][:,allowed]
    scale[mode].append(dict(reward_mean=float(reward.mean()),candidate_reward_std_mean=float(reward.std(-1,unbiased=False).mean()),forward_mean=float(forward.mean())if len(forward)else None,
     reverse_mean=float(reverse.mean())if len(reverse)else None,forward_reverse_mean_abs_difference=float((forward-reverse).abs().mean())if len(forward)else None,
     teacher_ESS=float((1/fb['prob'][allowed].square().sum(-1)).mean()),mean_absolute_center_step=float(t['output_step_L2']),eta_step_L2=float((t['eta_after']-t['eta_before']).norm()),
     changed_frames=t['nonzero_gradient_frames'],same_affinity_but_current_candidate_region_recomputed=True))
  rows.append(dict(alias=a,dataset=r['dataset'],active=o['active'],known_unobserved=int(mask.sum()),frame_means=means,scale=scale))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];keys=next(r['frame_means']for r in rr if r['frame_means']).keys()
  groups[ds]=dict(count=16,active=sum(r['active']for r in rr),E_relative_ranking={k:paired([r['frame_means'].get(k)for r in rr])for k in keys},
   actual_step_scale={mode:[{k:mean([r['scale'][mode][step][k]for r in rr])for k in ['reward_mean','candidate_reward_std_mean','forward_mean','reverse_mean','forward_reverse_mean_abs_difference','teacher_ESS','mean_absolute_center_step','eta_step_L2','changed_frames']}for step in range(2)]for mode in ['Fixed_dense','On_policy_dense','Shuffled_OPD']})
 put(P/'FEEDBACK_AND_RESPONSE_PUBLIC.json',dict(status='complete',groups=groups,rows=rows,GT_quality_from_original_matching_candidate_cache=True,no_new_predictions=True,only_post_seal_analysis=True,unknown_GT_excluded_not_zero=True))
 z=read(P/'RESULTS_PUBLIC.json');g=z['groups'];passing=all(g[ds]['paired']['On_policy_dense_minus_E_MBR'][k]['mean']>0 for ds in g for k in ['v','unobserved_actual_s'])
 put(P/'DECISION.json',dict(status='evaluated',passed=passing,rule=p['decision'],scope='Only coarse bidirectional feedback + existing two-step external-center policy',all96_fits192_rounds_actually_ran=True,
  new_feedback_or_onpolicy_advantage_established=False if not passing else None,next='stop this interface; retain all negative results; single Sol review must justify any materially different next experiment'))
 print('decision',passing,flush=True)

if __name__=='__main__':main()
