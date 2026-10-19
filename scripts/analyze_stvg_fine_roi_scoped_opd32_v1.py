"""Post-seal E-relative candidate feedback and unchanged-denominator scope audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_roi_scoped_opd32_v1 import D,P,ROI,OLD,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import numpy as np,torch
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import summarize_frame
 torch.set_num_threads(4);p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32;z=read(P/'RESULTS_PUBLIC.json');rows=[]
 for r in p['records']:
  a=r['alias'];o=load(D/'predictions'/(a+'.pt'));co=load(ROI/'predictions'/(a+'.pt'));qq=load(ROOT/'artifacts/stvg_cosine_candidate_headroom32_v1/diagnostics'/(a+'.pt'));old=load(OLD/'predictions'/(a+'.pt'));assert torch.equal(o['direct']['candidates'],old['direct']['candidates']);rank={}
  for key,rd in [('Fine',o['direct']),('Coarse',co['direct'])]:
   fb=rd['feedback'];valid=fb['valid'].numpy().copy();valid[:,0]=True;rank[key]={}
   for label,mask in [('all_known_unobserved',qq['mask']),('predicted_scope_known_unobserved',qq['mask']&o['allowed'].numpy())]:
    frames=[]
    for t in np.flatnonzero(mask):
     v=summarize_frame(qq['quality'][t],fb['rewards'][t].numpy(),fb['prob'][t].numpy(),valid[t],qq['quality'][t,0],o['active'],qq['quality'][t,0],qq['quality'][t,0]);[v.pop(k)for k in ['on_gain_pp','step1_gain_pp','posterior_good_but_On_nonpositive']];frames.append(v)
    rank[key][label]={k:mean([v[k]for v in frames])for k in frames[0]}if frames else {}
  rows.append(dict(alias=a,dataset=r['dataset'],active=o['active'],allowed_nodes=int(o['allowed'].sum()),all_actual_nodes=len(o['allowed']),ranking=rank))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]={}
  for kind in ['Fine','Coarse']:
   groups[ds][kind]={}
   for label in ['all_known_unobserved','predicted_scope_known_unobserved']:
    keys=next((r['ranking'][kind][label].keys()for r in rr if r['ranking'][kind][label]),[]);groups[ds][kind][label]={k:paired([r['ranking'][kind][label].get(k)for r in rr])for k in keys}
 put(P/'POST_SEAL_FEEDBACK.json',dict(status='complete',count=32,groups=groups,rows=rows,GT_offline_only=True,initial34_candidate_quality_reused=True,moved_second_round_candidate_GT_quality='unknown',conditional_scope_rank_not_main_denominator=True,ratios_and_AUC_native_units=True,scope_not_selected_from_GT=True));print('postseal feedback complete',flush=True)
if __name__=='__main__':main()
