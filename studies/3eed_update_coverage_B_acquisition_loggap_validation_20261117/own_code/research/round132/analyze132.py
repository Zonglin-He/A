"""Full-cohort reason/decision/coverage summaries and every task-changing row."""
from common132 import *
import numpy as np,collections
def main():
 assert rd(H132/'SCALAR_AUDIT.json')['n']==1798;O=H132/'scores_attempt001';rows=[r for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];summaries=[];subsets=[];cases=[]
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)];den=sum(r['sampling_weight'] for r in rr)
  for kind in ['B3','LOG']:
   dd=[r['diagnostics'][kind] for r in rr];active=[d for d in dd if d['active']];recover=[d for d in dd if d['old_zero_recoverable']];summary={'source':src,'target':tgt,'kind':kind,'n':len(rr),'B_usable':sum(d['B_available'] for d in dd),'active':len(active),'parameter_changed':sum(d['parameter_changed'] for d in dd),'newly_active_after_old_C_equals_S':sum(r['old_skip_reason']=='C_equals_S' and r['diagnostics'][kind]['active'] for r in rr),'newly_active_after_old_no_B':sum(r['old_skip_reason']=='no_usable_B' and r['diagnostics'][kind]['active'] for r in rr),'effective_wrong':sum(d['effective_wrong'] for d in dd),'query_changed_vs_ownzero':sum(d['query_changed'] for d in dd),'group_changed_vs_ownzero':sum(d['geometry_group_changed'] for d in dd),'IoU_changed_vs_ownzero':sum(d['iou_changed'] for d in dd),'better_vs_ownzero':sum(d['IoU_gain']>1e-12 for d in dd),'worse_vs_ownzero':sum(d['IoU_gain']< -1e-12 for d in dd),'accepted_steps':sum(d['accepted_steps'] for d in dd),'shrunk_steps':sum(d['shrunk_steps'] for d in dd),'active_margin_shrinks':sum(d['margin_change']< -1e-8 for d in active),'active_margin_widens':sum(d['margin_change']>1e-8 for d in active),'oldzero_recoverable':len(recover),'recoverable_S_good':sum(d['S_has_Acc25'] for d in recover),'recoverable_C_good':sum(d['C_has_Acc25'] for d in recover),'recoverable_good_removed_C':sum(d['all_good_removed_by_C_when_S_has_good'] for d in recover),'active_q0_supported':sum(d['own_q0_in_S'] for d in active),'active_q0_excluded_C':sum(d['own_q0_in_S'] and not d['own_q0_in_C'] for d in active),'mean_initial_mass_correct':float(np.mean([d['correct_retained_initial_mass'] for d in active])) if active else None,'mean_initial_mass_wrong':float(np.mean([d['wrong_retained_initial_mass'] for d in active])) if active else None,'joint_two_factor_execution_seconds_mean':float(np.mean([r['seconds_new'] for r in rr]))};summaries.append(summary)
   predicates={'all':lambda r:True,'old_no_B':lambda r:r['old_skip_reason']=='no_usable_B','old_C_equals_S':lambda r:r['old_skip_reason']=='C_equals_S','old_active':lambda r:r['old_skip_reason']=='actual_update','new_B_rank2or3':lambda r:r['diagnostics']['B3']['selected_B_rank'] in [2,3],'active_this_factor':lambda r:r['diagnostics'][kind]['active'],'effective_wrong':lambda r:r['diagnostics'][kind]['effective_wrong']}
   for sub,predicate in predicates.items():
    ss=[r for r in rr if predicate(r)]
    if not ss:continue
    for control in [kind+'_zero',kind+'_direct',kind+'_wrong','partial_F']:
     delta=np.array([r['IoUs'][kind+'_partial']-r['IoUs'][control] for r in ss]);w=np.array([r['sampling_weight'] for r in ss]);subsets.append({'source':src,'target':tgt,'kind':kind,'subset':sub,'n':len(ss),'control':control,'raw_gain_pp':float(delta.mean()*100),'weighted_subset_gain_pp':float(np.average(delta,weights=w)*100),'contribution_to_full_primary_gain_pp':float((delta*w).sum()*100/den/6),'better':int((delta>1e-12).sum()),'worse':int((delta< -1e-12).sum()),'same':int((abs(delta)<=1e-12).sum())})
   for r in rr:
    for control in [kind+'_zero','partial_F']:
     delta=r['IoUs'][kind+'_partial']-r['IoUs'][control]
     if abs(delta)<=1e-12:continue
     d=r['diagnostics'][kind];cases.append({k:r[k] for k in ['source','target','id_PRIVATE','frame_PRIVATE','content_group_PRIVATE','scene_PRIVATE','old_skip_reason']}|{'kind':kind,'control':control,'change':'better' if delta>0 else 'worse','control_IoU':r['IoUs'][control],'zero_IoU':r['IoUs'][kind+'_zero'],'direct_IoU':r['IoUs'][kind+'_direct'],'partial_IoU':r['IoUs'][kind+'_partial'],'wrong_IoU':r['IoUs'][kind+'_wrong'],'gain_IoU':delta,'weighted_six_direction_contribution_pp':delta*r['sampling_weight']*100/den/6,**d})
 summary=[]
 for kind in ['B3','LOG']:
  for control in [kind+'_zero','partial_F']:
   cc=[c for c in cases if c['kind']==kind and c['control']==control];summary.append({'kind':kind,'control':control,'improved':sum(c['change']=='better' for c in cc),'worsened':sum(c['change']=='worse' for c in cc),'same':1798-len(cc),'changed_content_groups':len({c['content_group_PRIVATE'] for c in cc}),'changed_recordings':len({(c['target'],c['scene_PRIVATE']) for c in cc}),'largest_positive_contribution_pp':max([0.]+[c['weighted_six_direction_contribution_pp'] for c in cc]),'largest_negative_contribution_pp':min([0.]+[c['weighted_six_direction_contribution_pp'] for c in cc])})
 dump(H132/'PROCESS_SUMMARY.json',summaries);dump(H132/'SUBSET_COMPARISONS.json',subsets);dump(H132/'CHANGED_CASES_PRIVATE.json',cases);dump(H132/'CASE_SUMMARY.json',summary);print(summary)
if __name__=='__main__':main()
