"""Postseal E-relative candidate diagnostics, visibility, fixed eligibility."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_bijection_layout_opd32_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import summarize_frame
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();z=read(P/'RESULTS_PUBLIC.json');gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));cand=o['direct']['candidates'].numpy();n,k=cand.shape[:2];quality=np.full((n,k),np.nan);known=[]
  for t,f in enumerate(o['frame_ids']):
   if int(f)in gt[ds][0][q]and t not in o['planned_positions']:
    quality[t]=[iou_gt(b,gt[ds][0][q][int(f)],plans[ds][q],ds)for b in cand[t]];known.append(t)
  rank={}
  for key,item in [('Bijection',o['direct']),('Diagonal',o['diagonal'])]:
   fb=item['feedback'];valid=fb['valid'].numpy().copy();valid[:,0]=True;frames=[]
   for t in known:
    v=summarize_frame(quality[t],fb['rewards'][t].numpy(),fb['prob'][t].numpy(),valid[t],quality[t,0],o['active'],quality[t,0],quality[t,0])
    for field in ['on_gain_pp','step1_gain_pp','posterior_good_but_On_nonpositive']:v.pop(field)
    frames.append(v)
   rank[key]={field:mean([v[field]for v in frames])for field in frames[0]}if frames else {}
  permutation=o['direct']['feedback']['assignment'];diagdelta=None
  if permutation is not None:
   diagdelta=float((o['direct']['feedback']['per_anchor_rewards']-o['diagonal']['feedback']['per_anchor_rewards']).mean());nonidentity=float((permutation!=torch.arange(9)).double().mean())
  else:nonidentity=None
  fits={}
  for mode,f in o['fits'].items():
   out=f['final'].numpy();d=(out[:,:2]-o['outputs']['E_MBR'][:,:2])/(.25*o['outputs']['E_MBR'][:,2:]);fits[mode]=dict(output_center_L2=float(np.linalg.norm(out[:,:2]-o['outputs']['E_MBR'][:,:2])),moved_nodes=int((out[:,:2]!=o['outputs']['E_MBR'][:,:2]).any(1).sum()),state_L2=float(f['path'][-1]['state'].norm()),residual_first_difference=float(np.linalg.norm(np.diff(d,axis=0))),steps=[dict(prior_ESS=float((1/t['feedback']['prior'].square().sum(-1)).mean()),teacher_ESS=float((1/t['feedback']['prob'].square().sum(-1)).mean()),zero_noise_prior_mass=float(t['feedback']['prior'][:,0].mean()),zero_noise_teacher_mass=float(t['feedback']['prob'][:,0].mean()),gradient_L2=float(t['gradient'].norm()),sampling_center_L2=float(t['sampling_center'].norm()))for t in f['rounds']])
  rows.append(dict(alias=a,dataset=ds,active=o['active'],known_unobserved_nodes=len(known),eligible_nodes=int((o['amplitude']>0).sum()),ranking=rank,mean_assignment_minus_diagonal_reward=diagdelta,nonidentity_slot_fraction=nonidentity,visible_bin_mass_mean=float(o['direct']['feedback']['visible_bin_mass'].mean()),partial_bin_candidate_fraction=float((o['direct']['feedback']['visible_bin_mass']<9-1e-5).double().mean()),fits=fits))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,ranking={})
  for key in ['Bijection','Diagonal']:
   keys=next(r['ranking'][key]for r in rr if r['ranking'][key]).keys();groups[ds]['ranking'][key]={k:paired([r['ranking'][key].get(k)for r in rr])for k in keys}
  for k in ['mean_assignment_minus_diagonal_reward','nonidentity_slot_fraction','partial_bin_candidate_fraction']:groups[ds][k]=paired([r[k]for r in rr])
 needed=['Bijection_live_minus_'+a for a in ['E_MBR','Bijection_fixed','Bijection_shuffle','Bijection_readout','Diagonal_live']];gate=all(z['groups'][ds]['paired'][c]['v']['mean']>0 for ds in groups for c in needed)and all(z['groups'][ds]['paired'][needed[0]]['unobserved_actual_s']['mean']>0 for ds in groups)
 put(P/'FEEDBACK_DIAGNOSTICS_PUBLIC.json',dict(status='complete',groups=groups,rows=rows,GT_only_after_all32_seal=True,no_new_prediction_or_update=True,units='generic paired per_parent_pp retains native AUC/proportion/reward units; explicit pp fields are percentage points'))
 put(P/'DECISION.json',dict(status='evaluated',passed=gate,rule=p['decision'],all128fits256rounds_actually_ran=True,decision_scope='nine-patch bijective visual feedback with fixed-scoped independent center categorical OPD',independent_confirmation=False))
 print('gate',gate,flush=True)
 for ds,g in z['groups'].items():print(ds,{c:{k:round(t[k]['mean'],7)for k in ['v','unobserved_actual_s']}for c,t in g['paired'].items()if c.startswith('Bijection_live_')},flush=True)
if __name__=='__main__':main()
