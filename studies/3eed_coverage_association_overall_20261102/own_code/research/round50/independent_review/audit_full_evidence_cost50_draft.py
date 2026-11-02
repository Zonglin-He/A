"""UNEXECUTED post-seal DINO/numeric provenance, arithmetic and cost audit.

Independent stdlib/NumPy replay; no producer import, processor/model, RGB,
raw XYZ, original annotation container, task score, or protected FINAL read.
Physical durable call records are distinct from all logical evidence uses.
"""
from pathlib import Path
import collections,hashlib,json,re,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/round50/experiment'
PLAN=EXP/'FULL_EVIDENCE_COST_AUDIT_PLAN_LOCK.json'
AUTH=EXP/'ROOT_INDEPENDENT_FULL_EVIDENCE_COST_AUDIT_AUTHORIZATION.json'
OUT=Path(__file__).resolve().parent/'full_evidence_cost_audit_attempt001'
HASHES={};SEALS={};HISTORICAL={}
EXPECTED_RUNTIME={'transformers':'4.57.1','torch':'2.7.0+cu128','numpy':'2.3.5',
 'Pillow':'12.0.0','safetensors':'0.8.0','accelerate':'1.15.0'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 p=Path(p)
 if p not in HASHES:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  HASHES[p]=h.hexdigest()
 return HASHES[p]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def dump(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert 'FINAL384' not in p.as_posix() and 'round46/final' not in p.as_posix()
 return ROOT/p
def exact(a,b,label):
 a=np.asarray(a);b=np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
def outer_seal(folder):
 folder=Path(folder)
 if folder not in SEALS:SEALS[folder]=read(folder/'SEALED.json')
 return SEALS[folder]
def member(folder,rel):
 folder=Path(folder);p=folder/rel
 assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
 assert outer_seal(folder)['files'][rel]==sha(p),('sealed_member',rel)
 return p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy(order='K') for k in z.files}
def historical(provenance):
 file=provenance.get('file_PRIVATE',provenance.get('file'))
 p=local(file);expected=provenance['sha256'];assert sha(p)==expected
 key=(str(p),expected)
 if key not in HISTORICAL:HISTORICAL[key]=read(p)
 value=HISTORICAL[key]
 for part in provenance.get('value_path',[]):value=value[part]
 return value
def scalar32(x):
 v=np.float32(x);assert np.isfinite(v) and float(v)==x
 return v
def rect_iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
 inter=np.maximum(hi-lo,0).prod()
 return float(inter/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter,1e-12))
def nms(boxes,scores):
 keep=[]
 for i in np.argsort(-scores,kind='stable'):
  if all(rect_iou(boxes[i],boxes[j])<.5 for j in keep):keep.append(int(i))
 return keep
def dino_value(raw,binding):
 b=raw['pred_boxes_normalized_cxcywh'];s=raw['scores']
 assert b.shape==(900,4) and s.shape==(900,) and b.dtype==s.dtype==np.float32
 assert np.isfinite(b).all() and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
 H,W,C=binding['image_shape'];assert C==3 and H>0 and W>0
 # Preserve original float32 half-size/subtraction before multiplying by
 # int64 dimensions, which promotes the resulting pixel boxes to float64.
 boxes=np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),axis=1)*np.array([W,H,W,H])
 selected=np.flatnonzero(s>=.25);selected=selected[nms(boxes[selected],s[selected])[:10]]
 return dict(boxes_PRIVATE=boxes[selected].tolist(),scores=s[selected].tolist(),query_sha256=binding['query_sha256'])
def unflip(value,width):
 result=dict(value);b=np.asarray(value['boxes_PRIVATE']).reshape(-1,4)
 result['boxes_PRIVATE']=np.c_[width-b[:,2],b[:,1],width-b[:,0],b[:,3]].tolist()
 return result
def stable_indices(original,flip):
 return [j for j,b in enumerate(original['boxes_PRIVATE']) if flip is None or
  max([rect_iou(b,x) for x in flip['boxes_PRIVATE']],default=0)>=.5]
def query_hash(query):return hashlib.sha256((query.casefold().strip(' .')+'.').encode()).hexdigest()
def numeric_value(value):
 assert value['generated_tokens']==0 and type(value['input_tokens']) is int and 0<value['input_tokens']<=8192
 assert np.isfinite(value['log_odds_yes_no']) and np.isfinite(value['yes_no_probability'])
 assert 0<=value['yes_no_probability']<=1
 scalar32(value['log_odds_yes_no']);scalar32(value['yes_no_probability'])
def raw_numeric(call,global_ids):
 raw=call['raw'];value=call['value'];numeric_value(value)
 assert raw['logits_to_keep']==1
 for kind in ('yes','no'):
  ids=raw[kind+'_token_ids'];logits=np.asarray(raw[kind+'_logits'],np.float32)
  assert ids==sorted(set(ids)) and len(ids)==len(logits)>0
  assert all(type(i) is int and i>=0 for i in ids) and np.isfinite(logits).all()
  for x in raw[kind+'_logits']:scalar32(x)
  recorded=float(scalar32(raw[kind+'_logsumexp']))
  # Independent float64 mathematical LSE is a rounding diagnostic, not the
  # stored readout. It is checked against an a-priori float32 error bound.
  ld=logits.astype(np.float64);m=float(ld.max());math_lse=m+float(np.log(np.exp(ld-m).sum()))
  assert abs(recorded-math_lse)<=8*np.finfo(np.float32).eps*(1+abs(math_lse)),('LSE_rounding',kind)
 assert not set(raw['yes_token_ids'])&set(raw['no_token_ids'])
 ids=(tuple(raw['yes_token_ids']),tuple(raw['no_token_ids']))
 if global_ids[0] is None:global_ids[0]=ids
 assert ids==global_ids[0]
 # Torch originally subtracts two float32 LSE tensors. Subtracting their
 # JSON floats in Python float64 would be a different measurement.
 expected=float(np.float32(raw['yes_logsumexp'])-np.float32(raw['no_logsumexp']))
 assert expected==value['log_odds_yes_no'],'stored_float32_yes_minus_no_not_exact'
 binding=call['binding']
 assert value['image_sha256']==binding['image_sha256'] and value['prompt_sha256']==binding['prompt_sha256']
 return 1
def case_numeric_expected(case,logical):
 if logical['kind']=='target':return case['targets_PRIVATE'][str(logical['mask_index_PRIVATE'])][logical['view']][logical['mode']]
 assert logical['kind']=='reference'
 observations=[o for o in case['references'][logical['entity']] if o['proposal_index']==logical['proposal_index']]
 assert len(observations)==1
 return observations[0][logical['mode']]

def main():
 start=time.perf_counter();auth=read(AUTH);plan=read(PLAN)
 assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_FULL_EVIDENCE_COST_AUDIT_NO_GT'
 assert auth['program_sha256']==sha(__file__) and auth['audit_plan_sha256']==sha(PLAN)
 assert plan['n']==5912 and plan['no_GT_rawmedia_models_or_optimization'] is True
 assert plan['expected_target_pool_counts']=={'old':9459,'wide':44093}
 assert plan['expected_DINO_whole_case_reuse']=={'cases':224,'logical_uses':857}
 pins={**plan['programs_sha256'],**plan['inputs_sha256']}
 for rel,h in pins.items():assert sha(local(rel))==h,('root_pin',rel)
 run=local(plan['runroot']);assert run.is_relative_to(ROOT/'crossplatform/round50')
 for folder,h in plan['sealed_roots_sha256'].items():assert sha(local(folder)/'SEALED.json')==h
 for folder in (run/'prepared',run/'text',run/'frames'):
  assert folder.relative_to(ROOT).as_posix() in plan['sealed_roots_sha256']
 assert plan['source_capture_plan'] in plan['inputs_sha256']
 source_plan=read(local(plan['source_capture_plan']))
 assert source_plan['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 assert not OUT.exists();OUT.mkdir(parents=True)
 stages={};chunks=[];dino_cases={};case_rows=[];global_ids=[None];physical_keys=collections.Counter();current_stage=None;current_chunk=None
 try:
  for stage in ('dino','numeric'):
   current_stage=stage;cfg=plan['stages'][stage];ep=local(cfg['evidence_plan_file'])
   assert cfg['evidence_plan_file'] in plan['inputs_sha256']
   producer=read(ep);assert producer['selected_configuration']=='wideentry_oldW' and producer['output_root']==plan['runroot'] and producer['no_GT_or_parameter_updates'] is True
   recipe=producer['model_recipes'][stage];recipe_sha=digest(recipe);assets=producer['model_assets'][stage]
   assert recipe==cfg['model_recipe'] and assets==cfg['model_assets'], 'exact_asset_manifest_and_recipe_binding'
   # Model weight/tokenizer hashes remain explicit in the root-pinned asset
   # manifest. This audit does not load a model or copy its asset files.
   assert assets['files_sha256'] and all(isinstance(h,str) and len(h)==64 for h in assets['files_sha256'].values())
   work=local(cfg['worklist_root']);assert sha(work/'SEALED.json')==cfg['worklist_seal_sha256']
   wc=read(member(work,'COUNTS.json'));budgets=read(member(work,'CASE_BUDGETS_PRIVATE.json'));budget_by={b['id_PRIVATE']:b for b in budgets};assert len(budget_by)==5912
   aggregate=collections.Counter();seconds=[];physical_seconds=[];all_ids=set();whole_cases=set();oldpool_total=widepool_total=0
   for name,job in producer['chunks'].items():
    current_chunk=name;assert job['stage']==stage and job['model_recipe']==recipe_sha
    folder=run/job['output_subdir'];relative=folder.relative_to(ROOT).as_posix()
    assert sha(folder/'SEALED.json')==cfg['chunk_seals_sha256'][relative]
    s=outer_seal(folder);assert s['state']=='R50_FULL_FROZEN_'+stage.upper()+'_EVIDENCE_NO_GT' and s['plan_sha256']==sha(ep)
    for f,h in s['files'].items():assert sha(folder/f)==h
    counts=read(member(folder,'COUNTS.json'));assert counts['model_unchanged'] is True and counts['runtime']['packages']==EXPECTED_RUNTIME
    assert counts['generated_tokens']==counts['optimizer_steps']==0 and int(counts['peak_bytes'])>=0
    for field in ('seconds_this_process','load_seconds'):assert np.isfinite(counts[field]) and counts[field]>=0
    assert counts['load_seconds']<=counts['seconds_this_process']
    rowsfile=local(job['rows_file']);assert sha(rowsfile)==job['rows_sha256'];rows=read(rowsfile)
    ids={r['id'] for r in rows};assert len(ids)==len(rows) and not ids&all_ids;all_ids|=ids
    assert counts['n']==s['n']==len(rows)
    bindings=read(member(folder,'RUN_BINDINGS.json'));assert bindings=={'plan_sha256':sha(ep),'stage':stage,'max_unique_new_calls':job['max_new_calls']}
    reuse={}
    if job.get('reuse_index_file'):
     rp=local(job['reuse_index_file']);assert sha(rp)==job['reuse_index_sha256']
     for x in read(rp):
      assert x['kind']==stage and x['model_recipe']==recipe_sha
      reuse.setdefault(x['input_key'],[]).append(x)
    whole={}
    if job.get('case_reuse_index_file'):
     cp=local(job['case_reuse_index_file']);assert sha(cp)==job['case_reuse_index_sha256']
     whole={digest(x['binding']):x for x in read(cp)}
    callfiles=sorted((folder/'calls_PRIVATE').glob('*.json'));calls={}
    for p in callfiles:
     rec=read(member(folder,p.relative_to(folder).as_posix()));key=p.stem
     assert rec['state']=='COMPLETED' and rec['model_forward_entered'] is True and rec['generated_tokens']==0
     b=rec['binding'];assert key==digest(b) and b['kind']==stage and b['model_recipe']==recipe_sha
     assert np.isfinite(rec['seconds']) and rec['seconds']>=0
     assert rec['started_utc']<=rec['completed_utc']
     if stage=='numeric':aggregate['raw_float32_numeric_readouts_exact']+=raw_numeric(rec,global_ids)
     else:
      rp=member(folder,'calls_PRIVATE/'+rec['raw']['file_PRIVATE']);assert sha(rp)==rec['raw']['sha256']
      assert 0<rec['raw']['input_tokens']<=256
      value=dino_value(load(rp),b);assert value==rec['value'],'independent_DINO_NMS_value_not_exact'
      aggregate['DINO_raw_900_and_NMS_exact']+=1
     calls[key]=rec;physical_keys[(stage,key)]+=1;physical_seconds.append(float(rec['seconds']))
    assert len(calls)==counts['new_input_forwards_all_processes_of_this_chunk']<=job['max_new_calls']
    assert 0<=counts['unique_new_forwards_this_process']<=len(calls)
    cases={};expected_uses={};logical_types=collections.Counter()
    for row in rows:
     sid=row['id'];case=read(member(folder,'cases_PRIVATE/'+sid+'.json'));cases[sid]=case
     g=read(member(run/'text','cases_PRIVATE/'+sid+'.json'));p=read(member(run/'prepared','cases_PRIVATE/'+sid+'.json'))
     assert case['actual_subject_query_sha256']==p['actual_subject_query_sha256']==hashlib.sha256(g['subject_query'].encode()).hexdigest()
     assert case['frame_bindings']==p['bindings'] and case['plan_sha256']==sha(ep) and case['fixed_original_recipe'] is True
     oldpool=p['mask_pool_PRIVATE'];wide=p['wide_pool_PRIVATE'];assert set(oldpool)<=set(wide)
     oldpool_total+=len(oldpool);widepool_total+=len(wide)
     refs=g['references'] if oldpool else [];entities={r['entity'] for r in refs}
     assert case['original_oldpool_reference_gate'] is True  # recipe declaration, not a per-case bool(oldpool) result
     b=budget_by[sid];assert b['chunk']==name and b['old_pool']==len(oldpool) and b['wide_pool']==len(wide) and b['sample_retained_even_zero_calls'] is True
     target_n=reference_n=0
     if stage=='dino':
      assert set(case['objects'])==entities
      if 'historical_whole_case_reuse' in case:whole_cases.add(sid)
      for ref in refs:
       obj=case['objects'][ref['entity']];allowed=not bool(re.search(r'\b(left|right|clockwise)\b',ref['own_query'],re.I))
       assert (obj['flip'] is not None)==allowed
       assert obj['original']['query_sha256']==query_hash(ref['own_query'])
       assert obj['stable_indices']==stable_indices(obj['original'],obj['flip'])
       for view in (('original','flip') if allowed else ('original',)):
        logical={'id_PRIVATE':sid,'reference':ref['entity'],'view':view};expected_uses[digest(logical)]=logical;reference_n+=1
      dino_cases[sid]=case
     else:
      assert case['id_PRIVATE']==sid and set(case['targets_PRIVATE'])==set(map(str,wide)) and set(case['references'])==entities
      for m in wide:
       vv=case['targets_PRIVATE'][str(m)];assert len(vv)==2
       for view,v in enumerate(vv):
        assert v['visual_evidence']==v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no']
        assert v['full']['prompt_sha256']==v['gray']['prompt_sha256']
        for mode in ('full','gray'):
         numeric_value(v[mode]);logical={'id_PRIVATE':sid,'kind':'target','mask_index_PRIVATE':m,'view':view,'mode':mode};expected_uses[digest(logical)]=logical;target_n+=1
      for ref in refs:
       obj=dino_cases[sid]['objects'][ref['entity']];ob=case['references'][ref['entity']]
       assert [x['proposal_index'] for x in ob]==obj['stable_indices'][:3]
       for v in ob:
        j=v['proposal_index'];assert v['box_PRIVATE']==obj['original']['boxes_PRIVATE'][j] and v['detector_score']==obj['original']['scores'][j]
        assert v['visual_evidence']==v['full']['log_odds_yes_no']-v['gray']['log_odds_yes_no'] and v['full']['prompt_sha256']==v['gray']['prompt_sha256']
        for mode in ('full','gray'):
         numeric_value(v[mode]);logical={'id_PRIVATE':sid,'kind':'reference','entity':ref['entity'],'proposal_index':j,'mode':mode};expected_uses[digest(logical)]=logical;reference_n+=1
     assert b['target_numeric_logical_calls']==target_n and b['reference_logical_calls']==reference_n and b['logical_call_upper_bound']==target_n+reference_n
     case_rows.append({'stage':stage,'row':f'C{len(case_rows)+1:05d}','platform':row['platform'],
      'old_pool_count':len(oldpool),'wide_pool_count':len(wide),'target_logical_uses':target_n,'reference_logical_uses':reference_n,
      'whole_case_historical_reuse':sid in whole_cases,'no_generated_tokens_or_optimizer':True})
    usefiles=sorted((folder/'uses_PRIVATE').glob('*.json'));assert {p.stem for p in usefiles}==set(expected_uses)
    used_physical=set()
    for f in usefiles:
     u=read(member(folder,f.relative_to(folder).as_posix()));logical=u['logical_binding'];assert logical==expected_uses[f.stem] and u['plan_sha256']==sha(ep)
     key=u['input_key'];kind=u['reuse_kind'];logical_types[kind]+=1;sid=logical['id_PRIVATE'];case=cases[sid]
     if kind=='new_or_resumed_call':assert key in calls;value=calls[key]['value'];binding=calls[key]['binding'];used_physical.add(key)
     elif kind=='historical_exact_cache':
      assert key in reuse and any(x['file']==u['provenance']['file_PRIVATE'] and x['sha256']==u['provenance']['sha256'] and x['value_path']==u['provenance']['value_path'] for x in reuse[key])
      value=historical(u['provenance']);binding=None
      assert all(historical(x)==value for x in reuse[key])
      if stage=='dino':assert all(x['value_path'][-1]=='original' for x in reuse[key]), 'whole_already_unflipped_values_are_not_raw_forward_cache'
     else:
      assert stage=='dino' and kind=='historical_whole_case_cache' and key in whole
      entry=whole[key];assert entry['binding']['id_PRIVATE']==sid and entry['binding']['graph_sha256']==digest(read(member(run/'text','cases_PRIVATE/'+sid+'.json')))
      assert entry['binding']['old_pool']==read(member(run/'prepared','cases_PRIVATE/'+sid+'.json'))['mask_pool_PRIVATE'] and entry['binding']['frame_binding']==case['frame_bindings']
      assert u['provenance']['file']==entry['file'] and u['provenance']['sha256']==entry['sha256']
      prior_root=local(entry['legacy_dino_root']);assert sha(prior_root/'SEALED.json')==entry['legacy_dino_seal_sha256']
      assert member(prior_root,local(entry['file']).relative_to(prior_root).as_posix())==local(entry['file'])
      for rel,h in entry['legacy_recipe_programs_sha256'].items():assert sha(local(rel))==h
      assert sha(local(entry['legacy_assets_receipt_file']))==entry['legacy_assets_receipt_sha256']
      prior=historical(u['provenance']);assert digest(prior['objects'])==entry['objects_sha256'] and case['objects']==prior['objects']
      continue
     if stage=='numeric':assert value==case_numeric_expected(case,logical)
     else:
      expected=case['objects'][logical['reference']][logical['view']]
      if logical['view']=='flip':
       # A raw flip call is flipped back once in the case, unlike the
       # historical whole-case cache whose boxes were already unflipped.
       framecase=read(member(run/'frames','cases_PRIVATE/'+sid+'.json'));width=framecase['original_hw'][1]
       if binding is not None:assert width==binding['image_shape'][1]
       # An exactly symmetric image may legally reuse the historical raw
       # original-view value for a flip. The input-key proof, not the label
       # "original", determines reuse; it still gets exactly one unflip.
       value=unflip(value,width)
      assert value==expected
    assert used_physical==set(calls), 'physical_completed_call_without_logical_use'
    assert counts['logical_requests']==len(usefiles)
    assert counts['historical_exact_cache_logical_uses']==logical_types['historical_exact_cache']
    assert counts['historical_whole_case_logical_uses']==logical_types['historical_whole_case_cache']
    assert counts['nonhistorical_logical_uses']==logical_types['new_or_resumed_call']
    for k in ('new_input_forwards_all_processes_of_this_chunk','logical_requests','historical_exact_cache_logical_uses','historical_whole_case_logical_uses','nonhistorical_logical_uses'):aggregate[k]+=counts[k]
    aggregate['chunks']+=1;aggregate['cases']+=len(rows);aggregate['physical_forward_safe_upper_bound']+=job['max_new_calls']
    seconds.append(float(counts['seconds_this_process']))
    chunks.append({'stage':stage,'chunk':name,'cases':len(rows),'physical_completed_forwards':len(calls),
     'logical_uses':len(usefiles),'historical_exact_uses':logical_types['historical_exact_cache'],
     'historical_whole_case_uses':logical_types['historical_whole_case_cache'],
     'new_or_samechunk_completed_input_uses':logical_types['new_or_resumed_call'],
     'physical_call_cap':job['max_new_calls'],'ledger_wall_seconds_this_process':counts['seconds_this_process'],
     'model_load_seconds_overlaps_wall':counts['load_seconds'],'process_peak_bytes':counts['peak_bytes'],
     'model_unchanged':True,'generated_tokens':0,'optimizer_steps':0})
   assert len(all_ids)==5912 and aggregate['cases']==5912
   assert oldpool_total==9459 and widepool_total==44093
   assert aggregate['logical_requests']==wc['logical_requests_fixed_before_model_execution']
   assert aggregate['physical_forward_safe_upper_bound']==wc['physical_call_safe_upper_bound']
   assert aggregate['chunks']==wc['chunks']==len(cfg['chunk_seals_sha256'])
   if stage=='dino':
    assert len(whole_cases)==224 and aggregate['historical_whole_case_logical_uses']==857
    assert len(whole_cases)==wc['DINO_whole_case_exact_reuse_cases'] and aggregate['historical_whole_case_logical_uses']==wc['DINO_whole_case_logical_uses']
   else:
    assert sum(x['target_logical_uses'] for x in case_rows if x['stage']==stage)==176372
   stages[stage]={**dict(aggregate),'sum_chunk_ledger_wall_seconds':float(sum(seconds)),
    'sum_physical_call_record_seconds_overlaps_wall':float(sum(physical_seconds)),
    'max_process_peak_bytes':max(x['process_peak_bytes'] for x in chunks if x['stage']==stage),
    'old_subject_logical_budget':37836,'wide_subject_logical_budget':176372,
    'model_recipe_sha256':recipe_sha,'model_assets_manifest_sha256':digest(assets),
    'all_assets_manifest_hashes_bound_not_model_reloaded_by_auditor':True}
 except BaseException as exc:
  dump(OUT/'PARTIAL_CHUNKS.json',chunks);dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),
   'stage':current_stage,'chunk':current_chunk,'no_GT_or_model_or_automatic_retry':True});raise
 dump(OUT/'ALL_CHUNK_COSTS.json',chunks);dump(OUT/'ANONYMOUS_CASE_LOGICAL_USES.json',case_rows)
 dump(OUT/'SUMMARY.json',stages)
 result={'state':'PASS_R50_INDEPENDENT_FULL_FROZEN_EVIDENCE_CALL_COST_AND_RAW_ARITHMETIC_AUDIT_NO_GT',
  'n':5912,'stages':stages,'physical_forward_records':sum(s['new_input_forwards_all_processes_of_this_chunk'] for s in stages.values()),
  'global_distinct_physical_input_keys_not_a_forward_count':len(physical_keys),
  'same_input_key_executed_in_multiple_chunks':sum(n>1 for n in physical_keys.values()),
  'all_logical_uses_separate_from_physical_forward_count':True,
  'numeric_stored_yes_minus_no_replayed_in_float32_fullgray_difference_in_original_Python_float':True,
  'DINO_all_new900raw_outputs_threshold_stableNMS_top10_and_caseflip_stability_replayed':True,
  'new_grounder_forwards_in_evidence_stages':0,
  'cached_source_zero_new_forwards_is_not_zero_end_to_end_source_cost':True,
  'three_arms_share_one_cache_collection_not_three_independently_timed_deployments':True,
  'chunk_wall_load_and_call_seconds_overlap_do_not_sum_them_as_disjoint_components':True,
  'runtime_frozen_flags_and_hash_provenance_checked_not_semantic_identity_truth':True,
  'no_rawRGB_rawXYZ_GT_taskscore_model_or_protected_FINAL_specific_artifacts_read':True,
  'program_sha256':sha(__file__),'audit_plan_sha256':sha(PLAN),'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result)
 dump(OUT/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps(result))

if __name__=='__main__':main()
