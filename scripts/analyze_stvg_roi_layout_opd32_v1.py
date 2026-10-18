"""Post-seal E-relative ranking and predicted-time support attribution only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_roi_layout_opd32_v1 import D,P,PRE,OLD,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import summarize_frame
 torch.set_num_threads(4);p=verify();z=read(P/'RESULTS_PUBLIC.json');gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));qq=load(ROOT/'artifacts/stvg_cosine_candidate_headroom32_v1/diagnostics'/(a+'.pt'));old=load(OLD/'predictions'/(a+'.pt'));assert torch.equal(o['direct']['candidates'],old['direct']['candidates']);rank={}
  for key,rd in [('Layout',o['direct']),('Unordered',o['pooled'])]:
   f=rd['feedback'];valid=f['valid'].numpy().copy();valid[:,0]=True;frames=[]
   for t in np.flatnonzero(qq['mask']):
    v=summarize_frame(qq['quality'][t],f['rewards'][t].numpy(),f['prob'][t].numpy(),valid[t],qq['quality'][t,0],o['active'],qq['quality'][t,0],qq['quality'][t,0])
    for k in ['on_gain_pp','step1_gain_pp','posterior_good_but_On_nonpositive']:v.pop(k)
    frames.append(v)
   rank[key]={k:mean([v[k]for v in frames])for k in frames[0]}if frames else {}
  ids=np.asarray(o['frame_ids']);iv=o['interval'];known=np.array([int(f)in gt[ds][0][q]for f in ids]);unobs=~np.isin(np.arange(len(ids)),o['planned_positions']);inside=(ids>=iv[0])&(ids<iv[1]);left=np.r_[ids[0],ids[:-1]];right=np.r_[ids[1:],ids[-1]];relevant=(right>=iv[0])&(left<iv[1]);delta={}
  base=np.array([iou_gt(b,gt[ds][0][q][int(f)],plans[ds][q],ds)if k else np.nan for b,f,k in zip(o['outputs']['E_MBR'],ids,known)])
  for mode in ['On_policy_dense','Fixed_dense','Unordered_OPD']:
   si=np.array([iou_gt(b,gt[ds][0][q][int(f)],plans[ds][q],ds)if k else np.nan for b,f,k in zip(o['outputs'][mode],ids,known)]);d=si-base;den=int((known&unobs).sum());delta[mode]={}
   for label,mask in [('inside_MBR',inside),('outside_MBR',~inside),('interpolation_relevant',relevant),('interpolation_irrelevant',~relevant)]:
    m=mask&known&unobs;delta[mode][label]=dict(known_actual_nodes=int(m.sum()),mean_delta_pp=mean(100*d[m]),contribution_to_all_unobserved_s_pp=100*float(d[m].sum())/den if den else None)
  mass=o['direct']['feedback']['common_valid_bin_mass'];massstats=dict(mean=float(mass.mean())if mass.numel()else None,min=float(mass.min())if mass.numel()else None,fraction_below3=float((mass<3).double().mean())if mass.numel()else None)
  rows.append(dict(alias=a,dataset=ds,active=o['active'],ranking=rank,temporal_partition=delta,common_bin_mass=massstats,predicted_interval_only_defines_partition=True))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,ranking={},temporal_partition={})
  for kind in ['Layout','Unordered']:
   keys=next(r['ranking'][kind]for r in rr if r['ranking'][kind]).keys();groups[ds]['ranking'][kind]={k:paired([r['ranking'][kind].get(k)for r in rr])for k in keys}
  for mode in ['On_policy_dense','Fixed_dense','Unordered_OPD']:groups[ds]['temporal_partition'][mode]={label:dict(parents_with_known=sum(r['temporal_partition'][mode][label]['known_actual_nodes']>0 for r in rr),known_actual_nodes=sum(r['temporal_partition'][mode][label]['known_actual_nodes']for r in rr),mean_delta_pp=paired([r['temporal_partition'][mode][label]['mean_delta_pp']for r in rr]),contribution_to_all_unobserved_s_pp=paired([r['temporal_partition'][mode][label]['contribution_to_all_unobserved_s_pp']for r in rr]))for label in ['inside_MBR','outside_MBR','interpolation_relevant','interpolation_irrelevant']}
 put(P/'POST_SEAL_ANALYSIS.json',dict(status='complete',groups=groups,rows=rows,new_model_or_candidate_generation=False,no_new_deployed_output=True,initial34_candidate_quality_reuse_exact=True,new_Live_second_candidates_GT_quality='unknown',units='ranking proportions/AUC in native units despite generic paired per_parent_pp name; explicit pp keys are percentage points',scope='GT post-seal only; no mask/teacher/parameter selected or changed'))
 needed=['On_policy_dense_minus_E_MBR','On_policy_dense_minus_Fixed_dense','On_policy_dense_minus_Shuffled_OPD','On_policy_dense_minus_Unordered_OPD','On_policy_dense_minus_Layout_readout'];gate=all(z['groups'][ds]['paired'][c]['v']['mean']>0 for ds in groups for c in needed)and all(z['groups'][ds]['paired'][needed[0]]['unobserved_actual_s']['mean']>0 for ds in groups)
 put(P/'DECISION.json',dict(status='evaluated',passed=gate,rule=p['decision'],all128fits256rounds_actually_ran=True,decision_scope='this coarse ordered3x3 ROI scorer plus two-step independent-center update',new_domain_means_not_confirmation=True));print('decision',gate,{ds:groups[ds]['temporal_partition']['On_policy_dense']for ds in groups},flush=True)
if __name__=='__main__':main()
