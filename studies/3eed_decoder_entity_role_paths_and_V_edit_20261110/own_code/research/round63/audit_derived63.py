"""Independent saved-only validity, overall values and finite-arm reach audit."""
from common63 import *
H=ROOT/'research/round63';mx=0.;checks=0
for mode,analysis in [('screen','ANALYSIS_SCREEN_VALIDATED.json'),('heads','ANALYSIS_HEADS.json'),('edit','ANALYSIS_EDIT.json')]:
 root=H/('score_'+mode+'336_attempt001');ss=read(root/'SEALED.json')
 for name,digest in ss['files'].items():assert sha(root/name)==digest
 tasks=read(root/'ALL_TASK_ROWS_PRIVATE.json');a=read(H/analysis)
 if mode=='screen':assert a['strength'] is None and not a['original_screen_strength_valid'] and a['native_traces_independently_verified']
 for m in a['metrics']:
  vals=np.array([t['IoU'][m['arm']] for t in tasks]);base=np.array([t['IoU']['frozen_ABC'] for t in tasks]);e=abs(vals.mean()*100-m['balanced_mIoU_percent']);mx=max(mx,e);assert e<1e-10;assert m['IoU_improved_vs_ABC']==int((vals>base+1e-10).sum()) and m['IoU_worsened_vs_ABC']==int((vals<base-1e-10).sum());assert m['query_changes_vs_ABC']==sum(t['queries'][m['arm']]!=t['queries']['frozen_ABC'] for t in tasks);assert m['rescued25']==int(((base<=.25)&(vals>.25)).sum());assert m['broken25']==int(((base>.25)&(vals<=.25)).sum());checks+=1
 if mode=='screen':
  values=np.asarray([[t['IoU'][k] for k in t['IoU'] if k not in ['source','r59_direct_all']] for t in tasks]);ref=np.array([t['IoU']['frozen_ABC'] for t in tasks]);r=read(H/'CAUSAL_REACH_PUBLIC_SAFE.json');assert abs(values.max(1).mean()*100-r['screen_finite_armenvelope_mIoU_percent'])<1e-10;assert int(((ref<=.25)&(values.max(1)>.25)).sum())==r['screen_union_rescued_ABC_errors25']
notice=read(H/'validated_native_diagnostics_attempt001/VALIDITY_NOTICE.json');assert notice['original_prediction_seal_sha256']==sha(ROOT/'crossplatform/round63/screen336_attempt001/SEALED.json');assert notice['original_task_score_seal_sha256']==sha(H/'score_screen336_attempt001/SEALED.json');eq=read(H/'functional_equivalence_audit_attempt001/AUDIT_RESULT.json');assert eq['max_hidden_error']==0 and eq['paired_sample_views']==1008
rows=read(H/'ALL_EDIT_PROCESS_SCALARS_PUBLIC_SAFE.json');stats=read(H/'ANALYSIS_EDIT.json')['strength']
for s in stats:
 rr=[x for x in rows if x['arm']==s['arm'] and x['valid_roles']];assert len(rr)==s['valid_input_view_n'];assert sum(x['actual_changed_parameter_n']>0 for x in rr)==s['changed_matrix_view_n'];assert max(x['protected_value_max_abs_change'] for x in rr)==s['max_protected_value_change'];assert max(x['selected_attention_max_abs_change'] for x in rr)==s['max_selected_attention_change'];assert float(np.median([x['median_relative_branch_change'] for x in rr]))==s['median_relative_branch_change']
control=read(H/'control_direction_diagnostic_attempt001/ROWS_PUBLIC_SAFE.json');s=read(H/'control_direction_diagnostic_attempt001/SUMMARY_PUBLIC_SAFE.json');known=[r for r in control if r['valid_roles']];assert len(control)==1008 and len(known)==900
for key,v in s['median_statistics'].items():assert float(np.median([r[key] for r in known]))==v
out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R63_INDEPENDENT_OVERALL_REACH_STRENGTH_AND_METADATA_VALIDITY_AUDIT','metric_arms_checked':checks,'max_balanced_mean_error_pp':mx,'original_invalid_strength_quarantined':True,'original_predictions_and_task_scores_not_changed':True,'all_new_value_process_summary_fields_match_saved_rows':True,'all_posthoc_control_medians_match_saved_rows':True,'functional_all1008_representations_bitexact_receipt_verified':True,'no_GT_model_media_or_scoring_calls':True,'program_sha256':sha(__file__)};dump(out/'AUDIT_RESULT.json',result);print(result)
