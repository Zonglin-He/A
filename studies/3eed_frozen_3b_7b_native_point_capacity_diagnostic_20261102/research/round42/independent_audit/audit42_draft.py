"""DRAFT: independent post-seal CPU audit, never model/GPU/GT scoring.

Not executed during preparation. Uses only A208 and fixed synthetic fixtures.
"""
import sys,json,re,math,hashlib,time,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
P=ROOT/'crossplatform/round42/pointA208_attempt001';EXP=ROOT/'research/round42/experiment'
PRE=ROOT/'crossplatform/round37/prepared208_attempt001';P3=ROOT/'crossplatform/round37/inference208_attempt003';M3=ROOT/'crossplatform/round38/predictions208_attempt001';SYN=ROOT/'crossplatform/round37/unit_diagnostic_attempt003'
ARMS=['source','R29_direct','legacy_G','3B_point_compatible_fullmap','3B_point_compatible_oldsubset','3B_som_compatible_fullmap','7B_point_compatible_fullmap','7B_point_compatible_oldsubset']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify(folder):
 s=read(folder/'SEALED.json')
 for n,h in s['files'].items():assert sha(folder/n)==h,n
 return s
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def parse(raw):
 t=raw.strip()
 for p in [r'```(?:json)?\s*(.*?)\s*```',r'<answer>\s*(.*?)\s*</answer>']:
  m=re.fullmatch(p,t,re.S|re.I)
  if m:t=m.group(1).strip()
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d:raise ValueError('duplicate')
   d[k]=v
  return d
 try:o=json.loads(t,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
 except Exception:return 'parse_failure',None
 if isinstance(o,list):
  if len(o)!=1 or not isinstance(o[0],dict):return 'parse_failure',None
  o=o[0]
 if not isinstance(o,dict):return 'parse_failure',None
 k=set(o)&{'point','point_2d','point2d'}
 if len(k)!=1 or set(o)-k-{'label'} or ('label' in o and not isinstance(o['label'],str)):return 'parse_failure',None
 v=o[next(iter(k))]
 if v is None:return 'explicit_null',None
 if not isinstance(v,list) or len(v)!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v):return 'parse_failure',None
 return 'parsed',v
def readout(st,v,grid,case,z,subset):
 point=m=None;q0=int(z['source_scores'].argmax());status=st
 if st=='parsed':
  x,y=v;w,h=grid[2]*14,grid[1]*14;W,H=case['image_size_wh']
  if 0<=x<w and 0<=y<h:point=[x*W/w,y*H/h]
  if point is None:status='coordinate_outside_image'
  else:
   hits=np.flatnonzero(z['masks_bool'][:,math.floor(point[1]),math.floor(point[0])])
   if len(hits)==1:m=int(hits[0]);status='single_independent_mask'
   else:status='no_mask' if not len(hits) else 'overlapping_multiple_masks'
 table=case['old_subset_groups'] if subset else case['full_groups'];g=table.get(str(m),[]) if m is not None else [];q=max(g,key=lambda j:(float(z['source_scores'][j]),-j)) if g else q0
 status='mapped_original_query' if g else status if m is None else 'absent_from_old_subset' if subset and str(m) not in table else 'mask_without_3D_group'
 return {'query':int(q),'mapped':bool(g),'status':status,'mask_index':m,'original_point':point,'group_size':len(g),'query_changed':q!=q0,'old_subset_contains_selected_mask':m is not None and str(m) in case['old_subset_groups'],'full_group_nonempty':m is not None and bool(case['full_groups'].get(str(m),[]))}
def cpu_processor(plan):
 # Processor/tokenizer only; no conditional-generation model constructor.
 from transformers import AutoProcessor,AutoTokenizer
 d=plan['reference3B_processor_dir']
 for f,h in plan['reference3B_processor_files_sha256'].items():assert sha(Path(d)/f)==h
 pr=AutoProcessor.from_pretrained(d,local_files_only=True,trust_remote_code=False,use_fast=False,min_pixels=200704,max_pixels=802816);pr.tokenizer=AutoTokenizer.from_pretrained(d,local_files_only=True,trust_remote_code=False,use_fast=True);return pr
def input_replay(processor,r,old,new):
 from PIL import Image
 sys.path.insert(0,str(ROOT/'crossplatform/round37'));from run37_attempt003 import messages
 im=Image.open(r['imagepath']).convert('RGB');grid=new['image_grid_thw'];assert grid==old['image_grid_thw'] and new['images_array_sha256']==old['images_array_sha256']==[ah(np.asarray(im))]
 wh=[grid[0][2]*14,grid[0][1]*14];prompt=processor.apply_chat_template(messages('point',r['caption'],[],wh),tokenize=False,add_generation_prompt=True);p=processor(text=[prompt],images=[im],return_tensors='pt',padding=False,truncation=False)
 assert hashlib.sha256(prompt.encode()).hexdigest()==new['serialized_prompt_sha256']==old['serialized_prompt_sha256'];assert hashlib.sha256(r['caption'].encode()).hexdigest()==new['query_sha256']==old['query_sha256'];assert p['image_grid_thw'].tolist()==grid
 assert p['input_ids'].shape[1]==new['input_tokens']==old['input_tokens'];assert ah(p['input_ids'].numpy())==new['input_ids_array_sha256'];assert ah(p['pixel_values'].numpy())==new['pixel_values_array_sha256']
def call(key,old,processor,r):
 rawp=P/'raw_answers_PRIVATE'/(key+'.json');tokp=P/'token_ids_PRIVATE'/(key+'.json');bp=P/'input_bindings_PRIVATE'/(key+'.json');cp=P/'calls_PRIVATE'/(key+'.json');raw=read(rawp)['raw_answer_PRIVATE'];tokens=read(tokp)['generated_token_ids'];b=read(bp);c=read(cp)
 assert c['raw_answer_PRIVATE']==raw and sha(rawp)==c['raw_answer_file_sha256'] and sha(tokp)==c['token_ids_file_sha256'];assert tokp.stat().st_mtime_ns<=rawp.stat().st_mtime_ns<=cp.stat().st_mtime_ns
 for k,v in b.items():assert c[k]==v,(key,k)
 st,v=parse(raw);assert c['parsed']['status']==st and c['parsed']['value']==v
 assert c['generated_tokens']==len(tokens) and c['ended_with_eos']==any(t in [151645,151643] for t in tokens);assert c['max_new_tokens']==96 and c['actual_repetition_penalty']==1.05
 assert c['raw_persisted_before_parse_and_derived_hashes'] and c['3B7B_reconstructed_CPU_input_tensors_exact'];input_replay(processor,r,old,c);return c,st,v
def main():
 started=time.perf_counter();seal=verify(P);evaluation_auth=read(EXP/'POSTSEAL_EVALUATION_AUTHORIZATION.json');assert evaluation_auth['independent_CPU_audit_authorized'] is True and evaluation_auth['prediction_seal_sha256']==sha(P/'SEALED.json');plan=read(EXP/'PLAN_LOCK.json');impl=read(EXP/'IMPLEMENTATION_LOCK.json');auth=read(EXP/'EXECUTION_AUTHORIZATION.json');assert seal['plan_sha256']==sha(EXP/'PLAN_LOCK.json')==auth['plan_sha256'];assert seal['implementation_sha256']==sha(EXP/'IMPLEMENTATION_LOCK.json')==auth['implementation_sha256']
 for n,h in impl['program_sha256'].items():assert sha(ROOT/n)==h
 for n,h in plan['input_sha256'].items():assert sha(ROOT/n)==h
 verify(PRE);verify(P3);verify(M3);verify(SYN)
 processor=cpu_processor(plan);rows=read(ROOT/'research/round27/input/ALL528_PRIVATE.json')[:208];assert len(rows)==208 and collections.Counter(r['platform'] for r in rows)=={'drone':104,'quad':104};expectedkeys={r['id']+'_point' for r in rows};syn=[]
 for W,H in [(672,448),(1008,672)]:
  for x,y in [(25,25),(25,75),(75,25),(75,75)]:syn.append(f'{W}x{H}_{x}_{y}')
 expectedkeys|={'UNIT_'+n for n in syn}
 for folder in ['raw_answers_PRIVATE','token_ids_PRIVATE','input_bindings_PRIVATE','calls_PRIVATE']:assert {f.stem for f in (P/folder).glob('*.json')}==expectedkeys
 ledger=[json.loads(x) for x in (P/'GENERATION_BUDGET.jsonl').read_text().splitlines()];assert len(ledger)==216 and {x['key'] for x in ledger}==expectedkeys and len({x['key'] for x in ledger})==216
 preids=plan['preflight2_ids_PRIVATE'];assert [x['key'] for x in ledger[8:10]]==[x+'_point' for x in preids];assert all(x['phase']=='fixed8_synthetic' for x in ledger[:8]);assert all(x['phase']=='preflight_reused' for x in ledger[8:10]);units=[]
 for n in syn:
  old=read(SYN/(n+'.json'));r={'imagepath':str(SYN/(n+'.png')),'caption':'the blue square'};c,st,v=call('UNIT_'+n,old['result'],processor,r);assert c['model_resized_wh']==old['canvas_wh']==c['original_wh'];W,H=c['model_resized_wh'];hit=False
  if st=='parsed' and 0<=v[0]<W and 0<=v[1]<H:
   from PIL import Image
   hit=Image.open(SYN/(n+'.png')).convert('RGB').getpixel((math.floor(v[0]),math.floor(v[1])))==(0,0,255)
  units.append({'fixture':n,'hit':hit,'parse_status':st,'absolute_L2':float(np.linalg.norm(np.asarray(v)/[W,H]-old['center_norm'])) if st=='parsed' else None,'counterfactual1000_L2':float(np.linalg.norm(np.asarray(v)/1000-old['center_norm'])) if st=='parsed' else None})
 u=read(P/'COORDINATE_DIAGNOSTIC.json');assert u['n']==8 and u['blue_hits7B']==sum(x['hit'] for x in units) and u['blue_hits3B_cached']==1
 for a,b in zip(units,u['records']):
  assert a['fixture']==b['fixture'] and a['hit']==b['hit_actual_blue_square']
  for k in ['absolute_L2','counterfactual1000_L2']:assert (a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and abs(a[k]-b[k])<1e-12)
 for pair in u['resize_pairs']:
  cx,cy=pair['center_norm'];names=[f'{W}x{H}_{int(cx*100)}_{int(cy*100)}' for W,H in [(672,448),(1008,672)]];calls=[read(P/'calls_PRIVATE'/('UNIT_'+n+'.json')) for n in names]
  if all(c['parsed']['status']=='parsed' for c in calls):
   vals=[np.array(c['parsed']['value'],float) for c in calls];absolute=np.linalg.norm(vals[0]/[672,448]-vals[1]/[1008,672]);counter=np.linalg.norm(vals[0]/1000-vals[1]/1000)
   assert abs(pair['absolute_normalized_resize_difference']-absolute)<1e-12 and abs(pair['counterfactual1000_resize_difference']-counter)<1e-12
  else:assert pair['absolute_normalized_resize_difference'] is None and pair['counterfactual1000_resize_difference'] is None
 # Same immutable identity records, no media review or task-IoU-based labels.
 ident=ROOT/'research/round37/independent_identity';aseal=read(ident/'ANNOTATIONS_SEALED.json');annotation_sha=sha(ident/'IDENTITY_RECORDS_PRIVATE.json');assert annotation_sha==aseal['identity_records_sha256']=='6e382dc7cd7cdc4257a8d4811c1b640ca0d466eaab6b2b72fb1dde916c099097'
 anns={r['review_alias']:r for r in read(ident/'IDENTITY_RECORDS_PRIVATE.json')};co={r['row_PRIVATE']['id']:r for r in read(ident/'COHORT_PRIVATE.json')};assert len(co)==24 and sum(a['status']=='clear_single_mask' for a in anns.values())==18;identity=[]
 output=[]
 for r in rows:
  case=read(PRE/'cases_PRIVATE'/(r['id']+'.json'));z=npz(PRE/'arrays_PRIVATE'/(r['id']+'.npz'));old=read(P3/'calls_PRIVATE'/(r['id']+'_point.json'))['result'];c,st,v=call(r['id']+'_point',old,processor,r);pred=read(P/'predictions_PRIVATE'/(r['id']+'.json'));q=npz(P/'predictions_PRIVATE'/(r['id']+'.npz'));base=npz(M3/'predictions_PRIVATE'/(r['id']+'.npz'));bq=dict(zip(base['arm_names'].tolist(),map(int,base['queries'])))
  valid=(z['support_count']>=3)&(z['support_coverage']>=.15)&(z['projection_iou']>=.1)&(z['source_boxes'][:,3:]>0).all(1)[:,None];unique=valid.sum(1)==1
  for m in range(z['masks_bool'].shape[0]):assert case['full_groups'][str(m)]==np.flatnonzero(valid[:,m]&unique).tolist()
  for m,g in case['old_subset_groups'].items():assert g==case['full_groups'][m]
  reads={k:readout(st,v,c['image_grid_thw'][0],case,z,k=='oldsubset') for k in ['fullmap','oldsubset']};assert reads==pred['readouts'];qq=[bq['source'],bq['R29_direct'],bq['legacy_G'],bq['point_compatible_fullmap'],bq['point_compatible_oldsubset'],bq['som_compatible_fullmap'],reads['fullmap']['query'],reads['oldsubset']['query']]
  assert q['arm_names'].tolist()==ARMS and np.array_equal(q['queries'],qq) and np.array_equal(q['boxes'],z['source_boxes'][qq]);output.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':dict(zip(ARMS,map(int,qq))),'readouts':reads,'parse_status':st})
  if r['id'] in co:
   cohort=co[r['id']];a=anns[cohort['review_alias']];clear=a['status']=='clear_single_mask';target=next((x['mask_index'] for x in case['candidates'] if clear and x['target_id']==a['target_display_ids'][0]),None);claim=reads['fullmap']['mask_index'];targetg=case['full_groups'].get(str(target),[]) if clear else [];hits=[]
   if st=='parsed':
    W,H=case['image_size_wh'];pw,ph=c['image_grid_thw'][0][2]*14,c['image_grid_thw'][0][1]*14
    if 0<=v[0]<pw and 0<=v[1]<ph:hits=np.flatnonzero(z['masks_bool'][:,math.floor(v[1]*H/ph),math.floor(v[0]*W/pw)]).tolist()
   for mapping in ['fullmap','oldsubset']:
    rd=reads[mapping];identity.append({'review_alias':cohort['review_alias'],'original_alias_PRIVATE':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'arm':'7B_point_compatible_'+mapping,'mapping':mapping,'annotation_status':a['status'],'identity_determinate':clear,'review_target_display_ids':a['target_display_ids'],'claimed_mask_index_PRIVATE':claim,'claimed_display_id':next((x['target_id'] for x in case['candidates'] if x['mask_index']==claim),None),'claimed_mask_matches_review':bool(claim==target) if clear and claim is not None else None,'point_hits_review_mask_including_overlap':target in hits if clear and st=='parsed' else None,'hit_count':len(hits) if st=='parsed' else None,'mapped':rd['mapped'],'mapping_status':rd['status'],'selected_query_PRIVATE':rd['query'],'selected_query_in_review_group':rd['query'] in targetg if targetg else None,'prior_seen_artifact_access':a['prior_seen_artifact_access'],'case_specific_prior_seen':a['case_specific_prior_seen'],'ambiguous_not_scored_correct':not clear,'no_new_image_review_nonexpert_not_gold':True})

 load=read(P/'MODEL_LOAD_RECEIPT.json');assert load['actual_parameter_count']==8292166656 and all(not load['loading_info'].get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);assert load['model_generation_config']['repetition_penalty']==1.05
 resource=read(P/'FINAL_RESOURCE_RECEIPT.json');assert resource['GPU_wall_seconds_including_load_and_synthetic']<=2700 and resource['frozen_unchanged'];verifycost=read(P/'INPUT_WEIGHT_VERIFY_COST.json')
 assert len(identity)==48 and sha(ident/'IDENTITY_RECORDS_PRIVATE.json')==annotation_sha
 identity_summary=[]
 for mapping in ['fullmap','oldsubset']:
  for pl in ['drone','quad','both']:
   group=[r for r in identity if r['mapping']==mapping and (pl=='both' or r['platform']==pl)];clear=[r for r in group if r['identity_determinate']];claims=[r for r in clear if r['claimed_mask_index_PRIVATE'] is not None];match=[r for r in claims if r['claimed_mask_matches_review']];wrong=[r for r in claims if not r['claimed_mask_matches_review']]
   identity_summary.append({'mapping':mapping,'platform':pl,'n':len(group),'clear':len(clear),'ambiguous':len(group)-len(clear),'clear_matches':len(match),'clear_contradicts':len(wrong),'clear_no_unique_mask':len(clear)-len(claims),'correct_claim_mapped3D':sum(r['mapped'] for r in match),'wrong_claim_mapped3D':sum(r['mapped'] for r in wrong),'identity_not_gold_no_subset_method_selection':True})
 dump(HERE/'IDENTITY24_JOIN_PRIVATE.json',identity);dump(HERE/'IDENTITY24_SUMMARY.json',identity_summary)
 result={'state':'PASS_R42_INDEPENDENT_CPU_CONTRACT','observations':208,'raw_token_binding_calls':216,'CPU_processor_tensor_replays_without_model':216,'selected_query_readouts_reconstructed':416,'eight_arm_fixed_boxes':1664,'all_cached3B_controls_unchanged':True,'parse_schema_and_raw_position_unchanged':True,'atomic_raw_before_derived_files_mtime_contract':True,'matched_source_fullgroups_thresholds_unique_replayed':True,'synthetic7B_blue_hits':sum(x['hit'] for x in units),'synthetic3B_cached_blue_hits':1,'perfect_localization_not_a_gate':True,'GPUwall_seconds':resource['GPU_wall_seconds_including_load_and_synthetic'],'verification_cost_outsideGPUwall':verifycost,'no_GPU_model_generation_GT_scoring_or_FINAL_in_this_audit':True,'weight_hash_proof':'reuse sealed download/loader binding, not independent reread of16.6GB','prediction_seal_sha256':sha(P/'SEALED.json'),'immutable_identity24_clear18_ambiguous6_sha256':annotation_sha,'no_new_image_review_or_annotation_changes':True,'seconds':time.perf_counter()-started}
 dump(HERE/'AUDIT_RESULT.json',result);dump(HERE/'AUDITED_SELECTED_QUERIES_PRIVATE.json',output);dump(HERE/'AUDITED_SYNTHETIC_ONLY.json',units);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
