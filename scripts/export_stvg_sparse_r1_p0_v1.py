"""Explicit anonymous export of a sealed, finite P0 screen. No inference/scoring."""
import json, hashlib, statistics, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_sparse_stvg_r1_semantic_opd_v1'
STAGE=BASE/'public_stage'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def anon(k):
 d,i=k.split('_');return ('D1' if d=='hc2' else 'D2')+f'_{int(i)+1:02d}'
def metrics(x):return {k:x[k] for k in ['m_sIoU_dense_GT','m_tIoU','m_vIoU']}
def main():
 r=json.loads((BASE/'P0_POSTSEAL_REVIEW.json').read_text())
 assert digest(BASE/'P0_POSTSEAL_REVIEW.json')=='5f2ad912ffd37d5d3a0d63e3bee55f97c66c177c1dba2bb7e21d36ec4492c8ba'
 dest=STAGE/'results/stvg_sparse_stvg_r1_semantic_opd/2026-10-14'
 rows=[]
 for row in r['metrics']:
  raw=json.loads((BASE/'P0/outputs'/f"{row['case_id']}_{row['branch']}_{row['control']}.json").read_text())
  rd=row['output_readback']
  # Exact JSON fence removal only; no repair or private free-form evidence exported.
  s=raw['raw_output'].strip();s=s[7:-3].strip() if s.startswith('```json') and s.endswith('```') else s
  try:obj=json.loads(s)
  except Exception:obj={}
  e={}
  for v in obj.get('candidate_evaluations',obj.get('evaluations',[])):
   if not isinstance(v,dict):continue
   cid=v.get('candidate_id');sem=v.get('semantic',{});bd=v.get('boundary',{})
   if cid in row['candidate_grid_metrics']:
    e[cid]={'referent_match':sem.get('referent_match','unknown'),'event_relation_match':sem.get('event_relation_match','unknown'),'start_boundary':bd.get('start_boundary','unknown'),'end_boundary':bd.get('end_boundary','unknown'),'extent':bd.get('extent','unknown'),'target_support':v.get('target_support','unknown')}
  # selected evaluation remains available even when other candidate states are omitted.
  se=rd.get('selected_evaluation')
  if isinstance(se,dict) and se.get('candidate_id') in row['candidate_grid_metrics']:
   cid=se['candidate_id'];sem=se.get('semantic',{});bd=se.get('boundary',{})
   e[cid]={'referent_match':sem.get('referent_match','unknown'),'event_relation_match':sem.get('event_relation_match','unknown'),'start_boundary':bd.get('start_boundary','unknown'),'end_boundary':bd.get('end_boundary','unknown'),'extent':bd.get('extent','unknown'),'target_support':se.get('target_support','unknown')}
  out={'case':anon(row['case_id']),'dataset':row['dataset'],'branch':row['branch'],'control':row['control'],
       'output_readback':{k:rd.get(k) for k in ['strict_bare_valid','structural_wrapper_valid','minimal_selected_id_valid','selected_id','abstain','rank_order_complete','selected_rank_length','selected_evaluation_six_states_valid','selected_evaluation_status','selected_evidence_frame_unit_status','truncated_at_cap','wrapper_kind']},
       'Native':metrics(row['Native']['metrics']),'Projection':metrics(row['Projection']['metrics']),
       'candidate_grid':{cid:{'geometry_ordinal':v['geometry_ordinal'],'geometry_sha256':v['geometry_sha256'],'metrics':metrics(v['metrics']),'model_evaluation_supplied':cid in e,'categorical_feedback':e.get(cid,{k:'unknown' for k in ['referent_match','event_relation_match','start_boundary','end_boundary','extent','target_support']})} for cid,v in row['candidate_grid_metrics'].items()},
       'candidate_oracle_diagnostic_only':{'geometry_ordinal':row['candidate_oracle']['geometry_ordinal'],'metrics':metrics(row['candidate_oracle']['metrics'])},
       'raw_rank_selection':row['raw_rank_selection'],'pseudo_reference':row['pseudo_reference'],
       'cost':{k:raw[k] for k in ['input_token_count','output_token_count','preprocess_seconds','generate_seconds','total_seconds','GPU_peak_allocated_bytes']},
       'custody':{'response_sha256':digest(BASE/'P0/outputs'/f"{row['case_id']}_{row['branch']}_{row['control']}.json"),'source_case_sha256':raw['source_case_sha256'],'runtime_sha256':raw['runtime_sha256']}}
  # Explicit subobject selection; do not export any future/free-form private fields.
  out['raw_rank_selection']={k:v for k,v in (out['raw_rank_selection'] or {}).items() if k in ['status','actual_model_choice','geometry_ordinal','metrics','rank_order_complete','ranked_ids_count','usable_as_pseudo_reference','full_rank_correlation_computed']}
  out['pseudo_reference']={k:v for k,v in (out['pseudo_reference'] or {}).items() if k in ['status','fallback','metrics','selected_geometry_ordinal','source_fallback']}
  rows.append(out)
 pairs=[]
 for v in r['selected_id_equivariance']:
  pairs.append({'case':anon(v['case_id']),**{k:v.get(k) for k in ['branch','status','base_abstained','id_swap_abstained','both_abstained','base_selected_geometry_ordinal','id_swap_selected_geometry_ordinal','selected_id_equivariant','full_rank_order_comparison_available']}})
 # Aggregates use base controls only; invalid choices do not become zero outcomes.
 groups=[]
 for d in ['hc2','vidstg']:
  for branch in ['temporal','spatial']:
   z=[v for v in rows if v['dataset']==d and v['branch']==branch and v['control']=='base']
   ok=[v for v in z if v['raw_rank_selection'].get('actual_model_choice') and 'metrics' in v['raw_rank_selection']]
   mean=lambda a:statistics.mean(a)
   p=[v for v in pairs if v['branch']==branch and v['case'].startswith('D1' if d=='hc2' else 'D2')]
   groups.append({'dataset':d,'branch':branch,'total_cases':len(z),'accepted_choices':len(ok),'delta_vIoU_pp_matched_Projection':100*mean([v['raw_rank_selection']['metrics']['m_vIoU']-v['Projection']['m_vIoU'] for v in ok]) if ok else None,'Native_vIoU_percent':100*mean([v['Native']['m_vIoU'] for v in z]),'Projection_vIoU_percent':100*mean([v['Projection']['m_vIoU'] for v in z]),'pool_oracle_vIoU_percent':100*mean([v['candidate_oracle_diagnostic_only']['metrics']['m_vIoU'] for v in z]),'pool_oracle_headroom_pp':100*mean([v['candidate_oracle_diagnostic_only']['metrics']['m_vIoU']-v['Projection']['m_vIoU'] for v in z]),'ID_same':sum(v['selected_id_equivariant'] is True for v in p),'ID_changed':sum(v['selected_id_equivariant'] is False for v in p),'ID_unavailable':sum(v['selected_id_equivariant'] is None for v in p)})
 costs=[v['cost'] for v in rows]
 summary={'status':'complete_finite_P0_screen_not_qualified','scope':'one_fixed_8_case_48_request_protocol','decision':'stop_current_sparse_ID_protocol_before_P1_P2','case_count':8,'requests':48,'candidate_GT_diagnostic_count':192,'model_candidate_supplied_evaluation_count':sum(v['model_evaluation_supplied'] for row in rows for v in row['candidate_grid'].values()),'strict_bare_valid':sum(v['output_readback']['strict_bare_valid'] is True for v in rows),'full_wrapper_schema_valid':sum(v['output_readback']['structural_wrapper_valid'] is True for v in rows),'minimal_choice_valid':sum(v['output_readback']['minimal_selected_id_valid'] is True for v in rows),'ID_pairs':16,'ID_comparable':sum(v['selected_id_equivariant'] is not None for v in pairs),'ID_same':sum(v['selected_id_equivariant'] is True for v in pairs),'groups':groups,'cost':{'preprocess_plus_generate_seconds_total':sum(v['total_seconds'] for v in costs),'generate_seconds_total':sum(v['generate_seconds'] for v in costs),'median_request_seconds':statistics.median(v['total_seconds'] for v in costs),'min_request_seconds':min(v['total_seconds'] for v in costs),'max_request_seconds':max(v['total_seconds'] for v in costs),'input_tokens_total':sum(v['input_token_count'] for v in costs),'output_tokens_total':sum(v['output_token_count'] for v in costs),'max_GPU_allocated_bytes':max(v['GPU_peak_allocated_bytes'] for v in costs),'model_load_seconds':json.loads((BASE/'MODEL_LOAD_RECEIPT.json').read_text())['model_load_seconds'],'scope':'cached_student_input_frozen_verifier_preprocess_and_generation_not_end_to_end_TTA'},'P1_executed':False,'P2_executed':False,'P3_executed':False,'student_new_fit_count':0,'student_parameter_update_count':0,'DINO_calls':0,'YOLO_calls':0,'SAM2_calls':0,'GT_read_only_after_seal':True,'historically_exposed':True,'fresh_unseen_claim':False,'CURRENT_promoted':False,'official_independent_max_abs_error':r['metrics'][0]['Native']['official_independent_max_abs_error'],'maximum_all_metric_error':json.loads((BASE/'ROOT_READBACK.json').read_text())['official_independent_max_abs_error'],'source_report_sha256':digest(BASE/'P0_POSTSEAL_REVIEW.json')}
 write(dest/'anonymous_responses.json',rows);write(dest/'ID_swap_controls.json',pairs);write(dest/'summary.json',summary)
 # Official categorical wrong-query evidence only; no private rationale/coordinates/caption.
 control=[]
 query_to_case={}
 for f in (BASE/'P0/outputs').glob('*wrong_query.json'):
  z=json.loads(f.read_text());query_to_case[(z['branch'],z['request_metadata']['query_sha256'])]=anon(z['case_id'])
 for v in r['wrong_query_candidate_specific_visual_evidence']:
  ev=v['model_evaluation'];control.append({'case':query_to_case[(v['branch'],v['query_sha256'])],'branch':v['branch'],'control':v['control'],'geometry_ordinal':v['geometry_ordinal'],'visible_candidate_id':v['visible_candidate_id'],'evidence_frame_slot_unit_status':v['evidence_frame_slot_unit_status'],'semantic':ev['semantic'],'boundary':ev['boundary'],'target_support':ev['target_support'],'human_adjudication_required':True,'machine_contradiction_claim':None,'postseal_reader_evidence_eligible':True})
 present={(v['case'],v['branch'],v['visible_candidate_id']) for v in control}
 for row in rows:
  if row['control']!='wrong_query':continue
  for cid,v in row['candidate_grid'].items():
   if not v['model_evaluation_supplied'] or (row['case'],row['branch'],cid) in present:continue
   a=v['categorical_feedback']
   control.append({'case':row['case'],'branch':row['branch'],'control':'wrong_query','geometry_ordinal':v['geometry_ordinal'],'visible_candidate_id':cid,'evidence_frame_slot_unit_status':'unqualified_raw_response_not_evidence_unit_verified','semantic':{k:a[k] for k in ['referent_match','event_relation_match']},'boundary':{k:a[k] for k in ['start_boundary','end_boundary','extent']},'target_support':a['target_support'],'human_adjudication_required':True,'machine_contradiction_claim':None,'postseal_reader_evidence_eligible':False,'raw_categorical_feedback_only':True})
 summary['wrong_query_raw_supplied_evaluation_count']=len(control)
 summary['wrong_query_postseal_reader_evidence_count']=sum(v['postseal_reader_evidence_eligible'] for v in control)
 summary['wrong_query_unqualified_raw_supplement_count']=sum(not v['postseal_reader_evidence_eligible'] for v in control)
 write(dest/'summary.json',summary)
 write(dest/'wrong_query_categorical_evidence.json',control)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
