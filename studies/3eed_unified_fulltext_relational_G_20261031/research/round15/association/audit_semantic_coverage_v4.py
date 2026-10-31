"""V4 decode-budget actual sealed semantic process/coverage replay; no GT/sensors/model/media IO.
Counts are parser/constraint behaviors, never target identity or task quality.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
B=Path(__file__).resolve().parents[3];sys.path[:0]=[str(B/'crossplatform/round15')]
from semantic_model_runner_v4 import read,sha,canonical_sha,derive,parse_complete_text,parse_complete_visual
from semantic_runner_v3 import text_messages,visual_messages
import semantic_graph_v4 as _sg
from semantic_runner import json_safe

def dump(p,v):Path(p).write_text(json.dumps(json_safe(v),indent=2,allow_nan=False)+'\n')
def clean_statuses(items):return dict(collections.Counter(str(x) for x in items))
def raw_topkeys(result):
 raw=result['raw_answer_PRIVATE'].strip()
 if raw.startswith('```'):
  lines=raw.splitlines();raw='\n'.join(lines[1:-1])
 try:d=json.loads(raw)
 except (ValueError,TypeError):return None
 return sorted(d) if isinstance(d,dict) else None

def main(a):
 started=time.perf_counter();checks=0
 def ck(condition,label):
  nonlocal checks
  if not condition:raise ValueError(label)
  checks+=1
 ck(sha(a.plan)==a.plan_sha256,'root plan bytehash');ck(sha(a.predictions/'SEALED.json')==a.seal_sha256,'actual complete seal bytehash');p=read(a.plan);s=read(a.predictions/'SEALED.json')
 expected_state='sealed_R15_semantic_engineering4_before_GT' if a.expected_n==4 else 'sealed_R15_semantic204_three_calls_before_GT';ck(s['state']==expected_state and s['n_cases']==a.expected_n and s['n_model_calls']==a.expected_n*3,'complete actual4or204 only');ck(s['plan_sha256']==a.plan_sha256 and s['program_sha256']==p['program_sha256'],'plan/program seal');ck(s['model_frozen_parameters_buffers_unchanged'] and s['no_generated_code_GT_model_update_or_sourceROI'],'executed frozen check proof')
 for rel,h in p['program_sha256'].items():ck(sha(B/rel)==h,'program source '+rel)
 for rel,h in s['files'].items():
  path=(a.predictions/rel).resolve();ck(path.is_relative_to(a.predictions.resolve()) and sha(path)==h,'member '+rel)
 for key in ('manifest','case_index'):ck(sha(p[key])==p[key+'_sha256'],'input file '+key)
 rows=read(p['manifest']);lookup={r['id']:r for r in rows};cases={c['id']:c for c in read(p['case_index'])};ids=p['engineering_case_ids'] if a.expected_n==4 else [r['id'] for r in rows]
 ck(len(ids)==a.expected_n and len(set(ids))==a.expected_n,'selected complete inventory')
 aliases={cases[i]['case_alias'] for i in ids};expected_calls={alias+'__'+x+'.json' for alias in aliases for x in ('text','visual_order0','visual_order1')};ck({x.name for x in (a.predictions/'calls_PRIVATE').glob('*.json')}==expected_calls,'all3calls each no omissions')
 load=read(a.predictions/'MODEL_LOAD_RECEIPT.json');ck(load['strict_local_safetensors_eval'] and not any(load['loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'strict model loading receipt')
 details=[];calls=[];expectedroles=collections.Counter();allrelation=[]
 for sid in ids:
  row=lookup[sid];case=cases[sid];alias=case['case_alias'];expectedroles[case['role']]+=1
  meta=read(a.predictions/'scene_metadata_PRIVATE'/(alias+'.json'));payload=read(a.predictions/'cases_PRIVATE'/(alias+'.json'));meta_hash=canonical_sha(meta);ck(payload['sample_id_PRIVATE']==sid and payload['case_alias']==alias and payload['role']==case['role'] and payload['platform']==row['platform'],'case binding');ck(payload['scene_metadata_sha256']==meta_hash,'compact scene canonical hash')
  obs=meta['objects'];mids={o['mask_id'] for o in obs};ck(len(mids)==len(obs)<=100,'all mask inventory <=100');ck(meta['platform']==row['platform'] and meta['image_sha256']==payload['bindings']['image_sha256'],'observation RGB/platform binding');ck(meta['diagnostics']['all_masks_before_any_q0_or_semantic_selection'] and not meta['diagnostics']['q0_ROI_filter_applied'],'whole scene no q0ROI')
  for o in obs:
   ck(o['depth_camera']['n']==o['n_exclusive'],'exclusive depth count');ck(o['depth_camera']['defined']==(o['n_exclusive']>=8),'depth defined threshold')
   if o['mask_geometry_defined']:
    ck(np.asarray(o['mask_centroid_norm2']).shape==(2,) and np.asarray(o['mask_bbox_norm4']).shape==(4,) and np.asarray(o['mask_extent_norm2']).shape==(2,),'actual geometry shapes')
  tc=read(a.predictions/'calls_PRIVATE'/(alias+'__text.json'));ck(tc['scene_metadata_sha256']==meta_hash and tc['bindings']==payload['bindings'],'text observation/cache bridge');graph=parse_complete_text(tc['result'],row['caption']);ck(json_safe(graph)==tc['parsed'],'exact text reparse')
  case_calls=[('text',tc)];visual=[];expected_messages=[text_messages(row['caption'])]
  for order in range(2):
   vc=read(a.predictions/'calls_PRIVATE'/(alias+'__visual_order'+str(order)+'.json'));ck(vc['scene_metadata_sha256']==meta_hash and vc['bindings']==payload['bindings'],'visual observation/cache bridge');cand=vc['candidate_mapping_PRIVATE'];ck(len(cand)==len(obs) and {c['mask_id'] for c in cand}==mids and len({c['candidate_id'] for c in cand})==len(cand),'all masks numbered, no source crop/beam');pv=parse_complete_visual(vc['result'],cand,graph);ck(json_safe(pv)==vc['parsed'],'exact visual reparse');visual.append(pv);case_calls.append(('visual',vc));expected_messages.append(visual_messages(row['caption'],graph,cand))
  expected=derive(graph,visual,meta);ck(json_safe(expected)==payload['semantics'],'exact allmask graph/proposal/disabled/flip replay')
  for call_i,(kind,call) in enumerate(case_calls):
   rr=call['result'];budget=p['semantic_config']['text_max_new_tokens'] if kind=='text' else p['semantic_config']['visual_max_new_tokens'];ck(rr['max_new_tokens']==budget and 0<=rr['generated_tokens']<=budget and rr['input_tokens']<=8192,'actual token budget');ck(rr['greedy_no_retry_no_tools'] is True,'deterministic/no tool receipt');ck(rr['prompt_payload_version']=='schema_fewshot_v2_once','actual V2 prompt version');ck(rr['messages_sha256']==canonical_sha(expected_messages[call_i]),'expected exact messages delivered to processor');ck(hashlib.sha256(rr['serialized_prompt_PRIVATE'].encode()).hexdigest()==rr['serialized_prompt_sha256'],'actual formatted prompt private bytes hash');ck(len(rr['input_ids_sha256'])==64 and all(c in '0123456789abcdef' for c in rr['input_ids_sha256']),'actual model input tensor SHA receipt');ck(len(rr['image_grid_thw'])==(0 if kind=='text' else 2) and all(g[0]==1 and g[1]*g[2]//4<=1024 for g in rr['image_grid_thw']),'complete wholeimage visual grid');calls.append({'kind':kind,'ended_eos':rr['ended_eos'],'generated_tokens':rr['generated_tokens'],'cap_nonEOS':not rr['ended_eos'] and rr['generated_tokens']==budget,'parse_status':call['parsed']['status']})
  literal=expected['graph_match'];disabled=expected['graph_relation_disabled'];flip=expected['graph_relation_flip'];relation_evidence=[]
  # Count every actual primitive execution, including contradicted assignments
  # which intentionally are not retained in successful-assignment witnesses.
  # Wrapper returns identical values; no scene/model/GT recomputation occurs.
  original_relation=_sg.relation_evidence
  def spy(rel,left,right,config=_sg.DEFAULT_CONFIG):
   result=original_relation(rel,left,right,config);relation_evidence.append({'id':rel['id'],'predicate':rel['predicate'],**result});return result
  _sg.relation_evidence=spy
  try:literal_spy=_sg.match_scene_graph(graph,meta,expected['combined_visual'])
  finally:_sg.relation_evidence=original_relation
  ck(json_safe(literal_spy)==json_safe(literal),'observation spy exact result no method change')
  allrelation.extend(x['status'] for x in relation_evidence)
  detail={'case_alias':alias,'role':case['role'],'platform':row['platform'],'mask_inventory_n':len(obs),'exclusive_depth_defined_masks_n':sum(o['depth_camera']['defined'] for o in obs),'text_EOS':tc['result']['ended_eos'],'text_generated_tokens':tc['result']['generated_tokens'],'text_raw_topkeys_schema':raw_topkeys(tc['result']),'text_parse_status':graph['status'],'text_parse_reason':graph.get('reason'),'entities_n':len(graph.get('entities',[])),'attributes_n':len(graph.get('attributes',[])),'relations_n':len(graph.get('relations',[])),'predicate_counts':clean_statuses(r['predicate'] for r in graph.get('relations',[])),'literal_relation_status_counts':clean_statuses(x['status'] for x in relation_evidence),'literal_relation_measurement_evaluations_n':len(relation_evidence),'literal_status':literal['status'],'literal_search_complete':literal['search_complete'],'literal_assignments_evaluated':literal['assignments_evaluated'],'literal_possible_targets_n':len(literal.get('possible_target_mask_ids',[])),'literal_supported_targets_n':len(literal.get('supported_target_mask_ids',[])),'literal_unique_supported':literal['status']=='unique_supported','proposal_status':expected['proposal_consensus']['status'],'disabled_status':disabled['status'],'flip_status':flip['status'],'flip_applied':flip.get('flip_applied',False),'visual_parse_statuses':[v['status'] for v in visual],'visual_EOS':[c[1]['result']['ended_eos'] for c in case_calls[1:]],'visual_generated_tokens':[c[1]['result']['generated_tokens'] for c in case_calls[1:]],'source_fallback_required_literal':literal['status']!='unique_supported','not_main_identity_or_task_correctness':True};details.append(detail)
 ck(dict(expectedroles)==s['role_counts'],'complete role denominator')
 aggregate={'n_cases':a.expected_n,'n_calls':len(calls),'role_counts':dict(expectedroles),'text_parse_counts':clean_statuses(d['text_parse_status'] for d in details),'visual_parse_counts':clean_statuses(v for d in details for v in d['visual_parse_statuses']),'literal_graph_status_counts':clean_statuses(d['literal_status'] for d in details),'proposal_status_counts':clean_statuses(d['proposal_status'] for d in details),'disabled_status_counts':clean_statuses(d['disabled_status'] for d in details),'flip_applied_n':sum(d['flip_applied'] for d in details),'literal_unique_supported_n':sum(d['literal_unique_supported'] for d in details),'literal_fallback_n':sum(d['source_fallback_required_literal'] for d in details),'literal_relation_measurement_evaluations_n':len(allrelation),'literal_relation_measurement_status_counts':clean_statuses(allrelation),'relation_count_scope':'actual interpreter primitive calls, may repeat one registered relation over all-mask/assignment search; not independent observations','EOS_counts_by_kind':{kind:{'n':sum(c['kind']==kind for c in calls),'EOS_n':sum(c['kind']==kind and c['ended_eos'] for c in calls),'cap_nonEOS_n':sum(c['kind']==kind and c['cap_nonEOS'] for c in calls)} for kind in ('text','visual')},'complete_text_two_visual_contract_cases_n':sum(d['text_parse_status']=='PARSED_text_graph' and all(v=='PARSED_visual_graph' for v in d['visual_parse_statuses']) for d in details),'actual_usable_literal_semantic_supervision_present':any(d['literal_unique_supported'] for d in details),'usable_hypothesis_not_identity_truth':True}
 out={'state':'PASS_R15_ACTUAL_SEALED_SEMANTIC_PROCESS_AND_COVERAGE_REPLAY_NOT_TASK_VALIDATION','checks':checks,'aggregate':aggregate,'frozen_proof_scope':'Producer strictlocal load, locked verify_unchanged code and final actual unchanged flag checked. No second model reload or independent parameter inference.','transport_proof_scope':'Exact expected messages reconstructed and canonicalSHA checked, actual private serializedprompt bytes rehashed, actual input-tensor SHA recorded by locked producer. No independent tokenizer/model reload.','prediction_seal_sha256':a.seal_sha256,'plan_sha256':a.plan_sha256,'auditor_sha256':sha(__file__),'seconds':time.perf_counter()-started,'new_model_GPU_GT_raw_RGB_P_decode_or_taskIoU':False,'actual_scope':'EOS/parser/constraint behavior, not correctness. A run may PASS this audit while all semantic paths are NA.'}
 if a.output.exists():raise ValueError('New audit output required')
 a.output.mkdir(parents=True);dump(a.output/'SUMMARY.json',out);dump(a.output/'CASES_PRIVATE.json',details);dump(a.output/'SEALED.json',{'state':'sealed_R15_noGT_semantic_process_coverage_audit','n_cases':a.expected_n,'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'prediction_seal_sha256':a.seal_sha256,'plan_sha256':a.plan_sha256,'auditor_sha256':sha(__file__)})
 print(json.dumps(out['aggregate']))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('predictions','plan','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--plan-sha256',required=True);p.add_argument('--seal-sha256',required=True);p.add_argument('--expected-n',type=int,choices=(4,204),required=True);main(p.parse_args())
