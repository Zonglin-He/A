"""Postseal E-relative likelihood/posterior ranking and actual state diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_ca_query_opd32_v1 import D,P,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import summarize_frame
 from vg_tta.stvg_ca_query_opd32_v1 import feedback
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32;gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));cand=o['direct']['DINO']['candidates'].numpy();n,k=cand.shape[:2];quality=np.full((n,k),np.nan);known=[]
  for t,fid in enumerate(o['frame_ids']):
   if int(fid)in gt[ds][0][q]and t not in o['planned_positions']:quality[t]=[iou_gt(b,gt[ds][0][q][int(fid)],plans[ds][q],ds)for b in cand[t]];known.append(t)
  rank={};eps=torch.cat(tuple(o['noise']),1)
  for key in ['DINO','E','cosine']:
   fb=feedback(o['fields'][key],o['valid'],torch.from_numpy(cand),eps);valid=fb['valid'].numpy().copy();valid[:,0]=True;frames=[]
   for t in known:
    item=summarize_frame(quality[t],fb['likelihood'][t].numpy(),fb['prob'][t].numpy(),valid[t],quality[t,0],o['active'],quality[t,0],quality[t,0])
    for fld in ['on_gain_pp','step1_gain_pp','posterior_good_but_On_nonpositive']:item.pop(fld)
    frames.append(item)
   rank[key]={fld:mean([v[fld]for v in frames])for fld in frames[0]}if frames else {}
  fits={}
  for mode,f in o['fits'].items():
   fits[mode]=dict(output_center_L2=float(np.linalg.norm(f['final'].numpy()[:,:2]-o['outputs']['E_MBR'][:,:2])),final_state_L2=float(f['path'][-1]['state'].norm()),steps=[dict(prior_ESS=float((1/t['feedback']['prior'].square().sum(1)).mean()),teacher_ESS=float((1/t['feedback']['prob'].square().sum(1)).mean()),zero_noise_prior_mass=float(t['feedback']['prior'][:,0].mean()),zero_noise_teacher_mass=float(t['feedback']['prob'][:,0].mean()),gradient_L2=float(t['gradient'].norm()),sampling_center_L2=float(t['sampling_center'].norm()),visible_fraction_mean=float(t['feedback']['visible_fraction'].mean()),partial_visible_candidates_fraction=float((t['feedback']['visible_fraction']<1-1e-8).double().mean()))for t in f['rounds']])
  rows.append(dict(alias=a,dataset=ds,active=o['active'],known_unobserved_nodes=len(known),ranking=rank,fits=fits))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,ranking={})
  for source in ['DINO','E','cosine']:
   keys=next(r['ranking'][source]for r in rr if r['ranking'][source]).keys();groups[ds]['ranking'][source]={k:paired([r['ranking'][source].get(k)for r in rr])for k in keys}
 put(P/'FEEDBACK_DIAGNOSTICS_PUBLIC.json',dict(status='complete',groups=groups,rows=rows,GT_only_after_all32_seal=True,no_new_predictions_or_updates=True,units='AUC/proportions/likelihood retain native units; explicitly pp fields are percentage points',candidate_quality_only_initial34_not_moved_second_Live=True));print({ds:{s:groups[ds]['ranking'][s].get('raw_good_bad_AUC',{}).get('mean')for s in ['DINO','E','cosine']}for ds in groups},flush=True)
if __name__=='__main__':main()
