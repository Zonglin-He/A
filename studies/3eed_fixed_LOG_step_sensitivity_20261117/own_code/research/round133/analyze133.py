"""All endpoint changes plus complete fixed R13249good/27bad structural audit."""
from common133 import *
import numpy as np,collections
def category(ass,S,C,q0,q):
 if ass[q0]==ass[q]:return 'same_geometric_group_query_switch'
 if not S[ass[q0]] or not S[ass[q]]:return 'unsupported_group_competition'
 if not C[ass[q0]]:return 'previous_choice_excluded_by_C'
 if not C[ass[q]]:return 'update_selected_excluded_group'
 return 'both_retained_groups_reordered'
def main():
 assert rd(H133/'SCALAR_AUDIT.json')['n']==1798;O=H133/'scores_attempt001';rows=[r for p in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(p)];process=rd(H133/'PROCESS_ROWS_PRIVATE.json');oldcases={(r['source'],r['target'],r['id_PRIVATE']):r for r in rd(R133/'research/round132/CHANGED_CASES_PRIVATE.json') if r['kind']=='LOG' and r['control']=='LOG_zero'};assert len(oldcases)==76;cases=[];prior=[];summaries=[];cost=[];tailsummary=[]
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if r['source']==src and r['target']==tgt];den=sum(r['sampling_weight'] for r in rr)
  for step in [1,3,5]:
   d=[r['diagnostics'][str(step)] for r in rr];summaries.append({'source':src,'target':tgt,'steps':step,'n':len(rr),'active':sum(r['active'] for r in rr),'query_changed':sum(x['query_changed'] for x in d),'group_changed':sum(x['geometric_group_changed'] for x in d),'same_group_query_switch':sum(x['same_group_query_switch'] for x in d),'IoU_changed':sum(x['IoU_changed'] for x in d),'better_vs_zero':sum(x['IoU_gain']>1e-12 for x in d),'worse_vs_zero':sum(x['IoU_gain']< -1e-12 for x in d),'margin_shrinks':sum(r['active'] and r['diagnostics'][str(step)]['original_top_vs_best_other_group_margin_change']< -1e-8 for r in rr),'margin_widens':sum(r['active'] and r['diagnostics'][str(step)]['original_top_vs_best_other_group_margin_change']>1e-8 for r in rr)})
   for label in ['partial','wrong']:
    pp=[p for p in process if p['source']==src and p['target']==tgt and p['steps']==step and p['label']==label];cost.append({'source':src,'target':tgt,'label':label,'steps':step,'all_n':len(rr),'active_n':len(pp),'nonzero_parameter_states':sum(p['parameter_L2']>0 for p in pp),'mean_active_parameter_L2':float(np.mean([p['parameter_L2'] for p in pp])),'mean_active_path_length':float(np.mean([p['path_length'] for p in pp])),'mean_active_accepted_radius_sum':float(np.mean([p['accepted_radius_sum'] for p in pp])),'mean_nominal_radius_budget':float(np.mean([p['nominal_radius_budget'] for p in pp])),'accepted_steps':sum(p['accepted_steps'] for p in pp),'reader_evaluations_if_run_independently':sum(p['full_reader_evaluations_if_independently_run'] for p in pp),'backward_evaluations_if_run_independently':sum(p['backward_evaluations_if_independently_run'] for p in pp),'mean_active_objective_after':float(np.mean([p['objective_after'] for p in pp])),'mean_active_task_after':float(np.mean([p['task_after'] for p in pp])),'mean_active_feature_keep_after':float(np.mean([p['feature_keep_after'] for p in pp]))})
  for r in rr:
   name=src+'_to_'+tgt;sid=r['id_PRIVATE'];a=np.load(B133/'formal'/name/'samples_PRIVATE'/(sid+'.npz'));qq=dict(zip(a['arm_names'],a['queries']));f=np.load(R133/'crossplatform/round132/log_scopes_PRIVATE'/name/(sid+'.npz'));ass=f['assignment'];S=f['S'];C=f['C'];q0=int(qq['zero_F']);q3=int(qq['partial3_F']);y=np.load(R133/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];shared={k:r[k] for k in ['source','target','id_PRIVATE','frame_PRIVATE','content_group_PRIVATE','scene_PRIVATE','active']};shared.update(zero_IoU=r['IoUs']['zero_F'],direct_IoU=r['IoUs']['direct_F'],step1_IoU=r['IoUs']['partial1_F'],step3_IoU=r['IoUs']['partial3_F'],step5_IoU=r['IoUs']['partial5_F'],S_has_good=r['S_has_good'],C_has_good=r['C_has_good'],zero_gt_Acc25=r['q0_gt_Acc25'],zero_already_IoU_oracle=r['zero_already_IoU_oracle'],best_S_IoU=float(y[S[ass]].max()) if S.any() else None,best_C_IoU=float(y[C[ass]].max()) if C.any() else None,identity_not_certified=True)
   if (src,tgt,sid) in oldcases:
    prior.append(shared|{'old_change':oldcases[(src,tgt,sid)]['change'],'category_at3':category(ass,S,C,q0,q3),'zero_group_in_S':bool(S[ass[q0]]),'zero_group_in_C':bool(C[ass[q0]]),'step3_group_in_S':bool(S[ass[q3]]),'step3_group_in_C':bool(C[ass[q3]]),'step3_same_group_as_zero':bool(ass[q0]==ass[q3])})
   for step in [1,3,5]:
    for control in ['zero_F']+(['partial3_F'] if step!=3 else []):
     delta=r['IoUs']['partial'+str(step)+'_F']-r['IoUs'][control]
     if abs(delta)<=1e-12:continue
     q=int(qq['partial'+str(step)+'_F']);cases.append(shared|{'steps':step,'control':control,'change':'better' if delta>0 else 'worse','gain_IoU':delta,'weighted_six_direction_contribution_pp':delta*r['sampling_weight']*100/den/6,'category_vs_zero':category(ass,S,C,q0,q)})
 for step in [1,3,5]:
  for control in ['zero_F']+(['partial3_F'] if step!=3 else []):
   cc=[c for c in cases if c['steps']==step and c['control']==control];tailsummary.append({'steps':step,'control':control,'better':sum(c['change']=='better' for c in cc),'worse':sum(c['change']=='worse' for c in cc),'same':1798-len(cc),'changed_content_groups':len({c['content_group_PRIVATE'] for c in cc}),'changed_recordings':len({(c['target'],c['scene_PRIVATE']) for c in cc}),'largest_positive_contribution_pp':max([0.]+[c['weighted_six_direction_contribution_pp'] for c in cc]),'largest_negative_contribution_pp':min([0.]+[c['weighted_six_direction_contribution_pp'] for c in cc])})
 oldsummary={'n':len(prior),'better':sum(r['old_change']=='better' for r in prior),'worse':sum(r['old_change']=='worse' for r in prior),'categories_by_old_change':{change:dict(collections.Counter(r['category_at3'] for r in prior if r['old_change']==change)) for change in ['better','worse']},'bad_cases_zero_Acc25':sum(r['old_change']=='worse' and r['zero_gt_Acc25'] for r in prior),'bad_cases_zero_excluded_C':sum(r['old_change']=='worse' and r['zero_group_in_S'] and not r['zero_group_in_C'] for r in prior),'GT_used_only_postseal':True,'actual_instance_identity_unknown':True}
 times=[]
 for r in rows:
  rec=rd(B133/'formal'/(r['source']+'_to_'+r['target'])/'records_PRIVATE'/(r['id_PRIVATE']+'.json'))
  if not r['active']:continue
  t=rec['timing'];times.append({'source':r['source'],'target':r['target'],'id_PRIVATE':r['id_PRIVATE'],'B_provider':rec['B_provider'],'prefix_and_B_seconds':t['prefix_and_B_seconds'],'correct_decode1_seconds':t['partial']['decode1_seconds'],'correct_extra_steps4and5_seconds':t['partial']['append4and5_seconds'],'correct_decode5_seconds':t['partial']['decode5_seconds'],'joint_new_seconds':rec['seconds_new_joint_work']})
 timing={'active':len(times),'B_provider_counts':dict(collections.Counter(r['B_provider'] for r in times)),'mean_measured_incremental_seconds':{k:float(np.mean([r[k] for r in times])) for k in ['prefix_and_B_seconds','correct_decode1_seconds','correct_extra_steps4and5_seconds','correct_decode5_seconds','joint_new_seconds']},'not_standalone_end_to_end_method_latency':True,'no_timing_claim_that_cached3_is_cheaper_than1':True}
 assert len(prior)==76 and oldsummary['better']==49 and oldsummary['worse']==27
 for name,x in [('STEP_SUMMARY',summaries),('COST_SUMMARY',cost),('CHANGED_CASES_PRIVATE',cases),('R132_ALL76_CASES_PRIVATE',prior),('R132_CASE_SUMMARY',oldsummary),('CASE_SUMMARY',tailsummary),('TIMING_ROWS_PRIVATE',times),('TIMING_SUMMARY',timing)]:dump(H133/(name+'.json'),x)
 print(oldsummary);print(tailsummary)
if __name__=='__main__':main()
