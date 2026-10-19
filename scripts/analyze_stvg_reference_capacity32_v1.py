"""Postseal label/support/capacity decomposition from saved scalar outputs only."""
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_reference_capacity32_v1 import D,P,verify,read,put,load
from scripts.score_stvg_joint_robust_correction32_v1 import paired

def main():
 p=verify(True);x=read(P/'RESULTS_PUBLIC.json');rows=[]
 for r in p['records']:
  a=r['alias'];z=load(D/'inputs'/(a+'.pt'));s=load(ROOT/f'artifacts/stvg_dino_endpoint_confirm32_v1/source/{a}.pt');q=next(q for q in x['rows']if q['alias']==a);v=q['values'];noobs=not s['expert']['observations'];rows.append(dict(alias=a,dataset=r['dataset'],no_original_DINO_observations=noobs,known_GT_refs=q['known_planned_GT'],matched_refs=q['matched_GT_DINO_support'],GT_estimator_selected=q['GT_E_estimator_support']['known4'],GT_E_known_minus_matched_v_pp=100*(v['GT_E_known4']['v']-v['GT_E_matched']['v']),GT_E_gain_v_pp=100*(v['GT_E_known4']['v']-v['E_MBR']['v']),candidate_oracle_gain_v_pp=100*(v['Candidate_combo_oracle']['v']-v['E_MBR']['v']),GT_state_gain_v_pp=100*(v['GT_state_known4']['v']-v['E_MBR']['v']),GT_state_anchor_loss_before=q['fits']['GT_state_known4']['loss_before'],GT_state_anchor_loss_after=q['fits']['GT_state_known4']['loss_after'],GT_state_output_L2=q['fits']['GT_state_known4']['output_L2_from_E']))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(GT_E_known_minus_matched_v_pp=paired([r['GT_E_known_minus_matched_v_pp']for r in rr]),GT_refs_available=sum(r['known_GT_refs']for r in rr),GT_refs_selected_by_existing_E=sum(r['GT_estimator_selected']for r in rr),parents_no_original_DINO=sum(r['no_original_DINO_observations']for r in rr),no_original_DINO_contribution_to_all16_GT_E_v_pp=sum(r['GT_E_gain_v_pp']for r in rr if r['no_original_DINO_observations'])/16,nonempty_GT_state_conditional_gain_v_pp=paired([r['GT_state_gain_v_pp']for r in rr if r['known_GT_refs']]),GT_state_mean_anchor_loss_before=float(np.mean([r['GT_state_anchor_loss_before']for r in rr])),GT_state_mean_anchor_loss_after=float(np.mean([r['GT_state_anchor_loss_after']for r in rr])),GT_state_mean_trajectory_L2=float(np.mean([r['GT_state_output_L2']for r in rr])))
 put(P/'SUPPORT_AND_CAPACITY_PUBLIC.json',dict(status='complete_postseal_mechanism',groups=groups,rows=rows,no_new_predictions_or_GT_queries=True,coverage_is_not_pure_label_quality=True,GT_subset_family_does_not_contain_arbitrary_current_E=True));print(groups,flush=True)
if __name__=='__main__':main()
