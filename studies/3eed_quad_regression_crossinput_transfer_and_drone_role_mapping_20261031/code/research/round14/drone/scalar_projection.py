"""Explicit finite Drone scalar projection; no generic private JSON dump.
Functions only: caller must check actual scoring/release seals before export.
No model, GT container, image, sourcefeature or taskIoU access.
"""
import csv,gzip,io,json,math,re
ARM_FIELDS=('case_alias','cohort','scene_alias','arm','delta_iou_vs_source','iou','Acc25','Acc50','query_changed','actual_changed','changed_axis_bitmask','center_L2_error_m','size_L2_error_m','center_shift_L2_m','size_shift_L2_m','raw_nonpositive_sizes','status')
BEHAVIOR_FIELDS=('text_role_status','text_target_quote_resolved','two_numbering_target_agreement_defined','two_numbering_target_agreement','distinct_reference_collision_observed','defined_reference_order_conflict','semantic_status','semantic_mask_defined','predicted_family_conflict_rejected','subject_family_unsupported_unknown','mask_choice_comparison_defined','mask_choice_changed_vs_old','old_controls_strictly_reused','ignored_unresolved_reference_ID_answers','unknown_reference_slots_defined_call_n','unknown_reference_slots_sum','mapping_only_mapping_status','mapping_only_visible_support_n','mapping_only_feasible_group_n','mapping_only_q0_positive_feasible','mapping_only_q0_preserved_by_positive_support','mapping_only_unique_group_readout','mapping_only_mapping_NA','combined_mapping_status','combined_visible_support_n','combined_feasible_group_n','combined_q0_positive_feasible','combined_q0_preserved_by_positive_support','combined_unique_group_readout','combined_mapping_NA')
ARMS=('source','old_full_order0','old_full_consensus','old_noun_consensus','semantic_only','mapping_only','combined')
COHORTS=('old_R13_Drone32_development','recent_content_disjoint32_historically_exposed')
def scalar(value):
 if value is None or isinstance(value,(bool,int)):return value
 if isinstance(value,float):
  if not math.isfinite(value):raise ValueError('Nonfinite public scalar')
  return value
 if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_:.-]+',value):return value
 raise ValueError('Only finite scalar flags/status tokens permitted')
def project(result):
 if result['state']!='R14_Drone64_seven_direct_arms_scored_after_bothbranch_complete_seal' or result['n']!=64:raise ValueError('Complete actual Drone64 scored payload required')
 rows=result['rows_PRIVATE']
 if len(rows)!=64 or len({r['id'] for r in rows})!=64 or len({r['case_alias'] for r in rows})!=64:raise ValueError('Unique64 rows/aliases required')
 if {c:sum(r['case_role']==c for r in rows) for c in COHORTS}!={c:32 for c in COHORTS}:raise ValueError('Separate32/32 cohorts required')
 scenes={s:f'scene_{i+1:02d}' for i,s in enumerate(sorted({r['scene'] for r in rows}))};ids={r['id'] for r in rows};rawscenes=set(scenes)
 def writer(fields):
  b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=fields);w.writeheader();return b,w
 ab,aw=writer(ARM_FIELDS);bb,bw=writer(('case_alias','cohort','scene_alias')+BEHAVIOR_FIELDS);n=0
 for r in rows:
  alias=r['case_alias']
  if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,79}',alias) or any(x in alias for x in ids|rawscenes):raise ValueError('Opaque public case alias required')
  base={'case_alias':alias,'cohort':r['case_role'],'scene_alias':scenes[r['scene']]}
  if set(r['arms'])!=set(ARMS):raise ValueError('Actual7-arm names required')
  for arm in ARMS:
   v=r['arms'][arm]
   e=v['coordinate_errors'];record={**base,'arm':arm,'delta_iou_vs_source':v['iou']-r['arms']['source']['iou'],'iou':v['iou'],'Acc25':v['Acc25'],'Acc50':v['Acc50'],'query_changed':v['query_changed'],'actual_changed':v['actual_changed'],'changed_axis_bitmask':sum(int(x)<<j for j,x in enumerate(v['actual_changed_axes'])),'center_L2_error_m':e['center_L2_m'],'size_L2_error_m':e['size_L2_m'],'center_shift_L2_m':v['center_shift_L2_m'],'size_shift_L2_m':v['size_shift_L2_m'],'raw_nonpositive_sizes':v['raw_nonpositive_sizes'],'status':v['status']};aw.writerow({k:scalar(z) for k,z in record.items()});n+=1
  bw.writerow({k:scalar(v) for k,v in {**base,**{k:r['behavior'].get(k) for k in BEHAVIOR_FIELDS}}.items()})
 if n!=448:raise ValueError('All448 casearm outputs retained')
 outputs={'DRONE_ARM_SCALARS.csv.gz':gzip.compress(ab.getvalue().encode(),mtime=0),'DRONE_CASE_BEHAVIOR.csv.gz':gzip.compress(bb.getvalue().encode(),mtime=0)}
 for data in outputs.values():
  text=gzip.decompress(data).decode()
  if any(x in text for x in ids|rawscenes):raise ValueError('Original identifier/scene leak')
 schema={'arm_rows':448,'behavior_rows':64,'unique_cases':64,'related_rows_not_independent_samples':True,'arm_fields':list(ARM_FIELDS),'case_behavior_fields':['case_alias','cohort','scene_alias',*BEHAVIOR_FIELDS],'gzip_mtime':0,'missing_semantics':'Empty CSV value is NA/notdefined; False/0 remain definedFalse/defined0. Allunknown/sourcefallbacks stay in32/32/64 denominators.','scope':'Two-numbering and role rates are modelbehaviors, not main-subject accuracy. Maskchoicechanged is an independent detector-hypothesis comparison, not physicalidentity proof. No qindex/maskindex/headword/caption/prompt/rawanswer/GTcoords/sourcecoords/features/weights.'}
 return outputs,schema
