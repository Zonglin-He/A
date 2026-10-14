"""Explicit anonymous sealed-scalar export. No captions, RGB, GT or raw facts."""
import json,time,shutil,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/stvg_hierarchical_evidence_guided_v1'
D=BASE/'public_stage/results/stvg_query_aligned_sparse_r1/2026-10-14'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def anon(s):return ('D1' if s.startswith('hc2_') else 'D2')+'_'+f"{int(s.split('_')[1])+1:02d}"
def main():
 r=read(BASE/'P0_B_POSTSEAL_REVIEW.json');decision=read(BASE/'TECHNICAL_DECISION.json')
 assert r['GT_after_all_responses_and_selection_seal'] is True
 assert read(BASE/'WORKER_EXIT_RECEIPT.json')['exit_code']==0
 assert not (BASE/'RESOURCE_GUARD_STOP001.json').exists() and not (BASE/'FAILURE_CAPTURE.json').exists()
 pairrows=[dict(row,case_id=anon(row['case_id'])) for row in r['pair_rows']]
 choices=read(BASE/'OUTPUT_ONLY_SELECTION_BARRIER.json')['records'];rows=[]
 for case,v in r['scores'].items():
  rows.append(dict(case=anon(case),dataset=v['dataset'],metrics=v,selections={f'{branch}_{protocol}':choices[f'{case}_{branch}_{protocol}'] for branch in ['temporal','spatial'] for protocol in ['B','C']}))
 costs=[]
 for f in read(BASE/'P0_B_GLOBAL_READOUT_BARRIER.json')['files']:
  z=read(ROOT/f['path'])
  if 'raw_output' not in z:continue
  meta=z['request_metadata'];costs.append(dict(case=anon(meta['case_id']),dataset=meta['dataset'],branch=meta['branch'],pair_index=meta['pair_index'],control=meta['control'],kind=z['kind'],
   output_valid=z['parsed'].get('status')!='invalid_output',truncated=z['output_truncated_at_cap'],**{k:z[k] for k in ['input_token_count','output_token_count','preprocess_seconds','generate_seconds','total_seconds','GPU_peak_allocated_bytes']},
   response_sha256=h(ROOT/f['path']),marked_RGB_sha256=meta.get('marked_RGB_sha256'),source_case_sha256=meta['source_case_sha256']))
 geometry=[]
 for row in read(BASE/'P0_A_ACTUAL_INTEGRATION.json')['records']:geometry.append(dict(row,case_id=anon(row['case_id'])))
 sources=[dict(case=anon(v['case_id']),dataset=v['dataset'],source_case_sha256=v['source_case_sha256']) for v in read(BASE/'ROSTER_LOCK.json')['records']]
 summary={k:r[k] for k in ['status','scope','groups','output_validity','fact_dimension_discrimination','conditional_feedback_triggers','calls','maximum_calls','official_independent_max_abs_error','candidate_metric_count','actual_cost','P1_student_fits_executed','online_carry_executed','historically_exposed','fresh_unseen_claim']}
 summary.update(technical_decision=decision['status'],current_finite_scope='P0A actual geometry plus P0B fixed8 BvsC; downstream conditional stages only',
   candidate_metric_readouts_are_not_fits=True,teacher_feedback_is_student_candidate_pseudoreference_not_independent_geometry=True,
   B_C_visual_budget_matched=True,C_extra_text_alignment_calls_counted=True,A_is_cached_old_unmapped_geometry_not_fresh_identical_protocol=True,
   DINO_calls=0,YOLO_calls=0,SAM2_calls=0,CURRENT_promoted=False,source_report_sha256=h(BASE/'P0_B_POSTSEAL_REVIEW.json'),
   GPU_OOM_observed=False,resource_guard_stop=False,teacher_model_load_seconds=read(BASE/'MODEL_LOAD_RECEIPT.json')['seconds'],
   verifier_cost_not_end_to_end_TTA=True,GT_read_after_seal=True,reviewer_GT_exposure_incident='restricted source reviewer searched annotation JSON; interrupted; exact split/count unknown; no fresh-unseen claim; not used for candidates')
 summary['cost_by_kind']={kind:dict(calls=len(a),seconds=sum(v['total_seconds'] for v in a),input_tokens=sum(v['input_token_count'] for v in a),output_tokens=sum(v['output_token_count'] for v in a)) for kind in ['direct','facts','alignment'] for a in [[v for v in costs if v['kind']==kind]]}
 supplement=read(BASE/'ROOT_SUPPLEMENTAL_SCALAR_READBACK.json')
 summary['raw_B_pair_groups']=supplement['B_groups']
 summary['C_missing_schema_only_count']=supplement['C_missing_schema_only']
 summary['C_raw_preference_counts']=supplement['C_raw_preference_counts']
 summary['C_panel_position_order_value_count']=supplement['C_fact_order_left_right_value_count']
 summary['C_order_value_total']=supplement['C_fact_order_total']
 summary['C_renderer_caption_relation_copy_count']=supplement['C_relation_exact_renderer_caption_count']
 diagnostic=read(BASE/'ROOT_UNQUALIFIED_OUTPUT_DIAGNOSTIC.json')
 write(D/'anonymous_unqualified_C_categorical_evidence.json',dict(status='unqualified_raw_diagnostic_not_schema_repair_or_adaptation_authority',
  records=[dict(case=anon(v['case_id']),branch=v['branch'],missing_schema_only=v['missing_schema_only'],raw_preferences=v['raw_preferences'],canonical_ids=v['canonical_ids'],response_sha256=v['response_sha256']) for v in diagnostic['records']]))
 write(D/'anonymous_raw_B_pair_controls.json',[dict(v,case_id=anon(v['case_id'])) for v in supplement['B_pair_rows']])
 for name,obj in [('summary.json',summary),('anonymous_pair_preferences.json',pairrows),('anonymous_all_case_readouts.json',rows),('anonymous_all_cost_rows.json',costs),('anonymous_geometry_audit.json',geometry),('source_case_hash_bindings.json',sources)]:write(D/name,obj)
 # Code / protocol only; raw reviewer output, original queries and media stay private.
 names=['scripts/stvg_hierarchical_geometry_v1.py','scripts/stvg_query_alignment_inputs_v1.py','scripts/stvg_query_alignment_schema_v1.py','scripts/run_stvg_query_alignment_v1.py','scripts/launch_stvg_query_alignment_v1.py',
  'scripts/stvg_query_alignment_resource_guard_v1.py','scripts/review_stvg_query_alignment_v1.py','scripts/export_stvg_query_alignment_v1.py','scripts/report_stvg_query_alignment_v1.py','protocols/stvg_query_aligned_sparse_r1_v1.md']
 for n in names:
  target=BASE/'public_stage'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,target)
 for n in ['CPU_CONTRACTS.json','SCHEMA_CPU_CONTRACTS.json','ROOT_SELECTION_CPU_CONTRACTS.json','TECHNICAL_DECISION.json']:
  write(D/n,read(BASE/n))
 write(D/'instance_interface_scope.json',{k:v for k,v in read(BASE/'source_audit/INSTANCE_CANDIDATE_CAPABILITY_READBACK.json').items() if k!='source_hashes'})
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
