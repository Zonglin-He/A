"""R85 simple local saved-scalar public exporter. Saved scalar exporter.

No score/model/GT/network operations; no imports of experiment producers.
A root-pinned report, decision, scalar runs, registry and explicit own-code
allowlist are required. Missing conditional full/TTA results are never invented.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,math,re,ahocorasick
ROOT=Path(__file__).resolve().parents[2]
PREFIX='studies/3eed_structured_observation_candidate_residual_TTA_20261112'
DENIED={'id','id_PRIVATE','alias','scene_PRIVATE','caption','subject_query','queries','query_indices','query_index','source_boxes','selected_original_boxes','source_scores','coordinates','features','tokens','weights','raw_answer','imagepath','pointpath','metapath','mask_indices','mask_index','point_indices','char_span','positive_mask'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(s):
 p=Path(s);assert not p.is_absolute() and '..' not in p.parts and 'round46/final' not in s and 'FINAL384' not in s;return ROOT/p
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(p,s):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def csvfile(p,rr,fields):
 st=io.StringIO();w=csv.DictWriter(st,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rr);write(p,st.getvalue())
def safe_code(s):return s.replace(str(ROOT),'WORKSPACE_ROOT').replace(str(ROOT.parent),'LOCAL_HOME')
def check(x):
 if isinstance(x,dict):
  for k,v in x.items():assert k not in DENIED,('private_data_field',k);check(v)
 elif isinstance(x,list):
  for v in x:check(v)
 elif isinstance(x,float):assert math.isfinite(x)
def scalar_projection(x):return {k:v for k,v in x.items() if v is None or type(v) in (bool,int,float) or (isinstance(v,str) and (k=='state' or k.endswith('_sha256')))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);ap.add_argument('--attempt',type=int,required=True);a=ap.parse_args()
 plan=read(a.plan);auth=read(a.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R85_LOCAL_SAVED_SCALAR_WHITELIST_EXPORT' and auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(a.plan)
 assert plan['prefix']==PREFIX and plan['no_network_models_GT_or_media'] is True and a.attempt>=1
 pins=plan['inputs_sha256']
 def get(rel):assert rel in pins and sha(local(rel))==pins[rel];return read(local(rel))
 registry=get(plan['private_registry']);assert auth['private_registry_sha256']==sha(local(plan['private_registry']))
 base=get(plan['publication_base_snapshot']);assert auth['publication_base_snapshot_sha256']==sha(local(plan['publication_base_snapshot']))
 out=ROOT/'research/round85/publication'/f'public_export_attempt{a.attempt:03d}';assert not out.exists();out.mkdir(parents=True)
 counts={}
 try:
  for run in plan['scalar_runs']:
   label=run['label'];assert re.fullmatch('[A-Z0-9_]{1,32}',label) and label not in counts
   folder=local(run['score_root']);seal=get(run['score_root']+'/SEALED.json')
   for f,h in seal['files'].items():assert sha(folder/f)==h
   tasks=get(run['score_root']+'/ALL_TASK_ROWS_PRIVATE.json');summary=get(run['score_root']+'/TASK_SUMMARY.json');scenes=get(run['score_root']+'/ALL_SCENE_SUMMARIES_PRIVATE.json');neg=get(run['score_root']+'/ALL_ORDERED_NEGATIVE_COMPARISONS.json')
   receipt=get(run['score_root']+'/SCORING_RECEIPT.json');arms=receipt['arm_names'];n=receipt['n'];assert n==len(tasks) and receipt['task_values']==n*len(arms) and receipt['ordered_negative_comparisons']==len(neg)
   assert n in (336,1280,1024) and receipt['single_saved_IoU_join'] is True and receipt['new_GT_scored_task_values']==0
   # Existing source/external baseline values may be reused; preserve separate
   # new-GT and reused counts rather than claiming every table value was rescored.
   assert receipt['new_GT_scored_task_values']+receipt['reused_saved_task_values']==receipt['task_values']
   rowmap={r['row']:f'{label}_{i+1:05d}' for i,r in enumerate(tasks)};assert len(rowmap)==n
   scenevalues={pl:sorted({r['scene_PRIVATE'] for r in tasks if r['platform']==pl}) for pl in ('drone','quad')}
   def scene(pl,s):return ('D' if pl=='drone' else 'Q')+'_S'+str(scenevalues[pl].index(s)+1).zfill(2)
   scalar=[]
   for r in tasks:
    assert set(r['IoU'])==set(r['statuses'])==set(arms)
    for arm in arms:scalar.append({'row':rowmap[r['row']],'platform':r['platform'],'scene_alias':scene(r['platform'],r['scene_PRIVATE']),'part':r['part'],'split':r['split'],'arm':arm,'IoU':r['IoU'][arm],**r['statuses'][arm]})
   fields=['row','platform','scene_alias','part','split','arm','IoU','unknown','fallback','no_parameter_update']
   for pl in ('drone','quad'):
    rr=[r for r in scalar if r['platform']==pl];csvfile(out/label/(pl.upper()+'_ALL_TASK_SCALARS.csv'),rr,fields)
    nn=[{'row':rowmap[r['row']],**{k:r[k] for k in ('platform','part','split','arm','reference','delta_pp')}} for r in neg if r['platform']==pl]
    csvfile(out/label/(pl.upper()+'_ALL_ORDERED_NEGATIVES.csv'),nn,['row','platform','part','split','arm','reference','delta_pp'])
   sc=[]
   for r in scenes:
    for arm in arms:sc.append({'scope':r['scope'],'platform':r['platform'],'scene_alias':scene(r['platform'],r['scene_PRIVATE']),'n':r['n'],'arm':arm,'mIoU_percent':r['mIoU_percent'][arm]})
   csvfile(out/label/'ALL_SCENE_ARM_SCALARS.csv',sc,['scope','platform','scene_alias','n','arm','mIoU_percent'])
   dump(out/label/'TASK_SUMMARY.json',summary);dump(out/label/'SCORE_RECEIPT.json',scalar_projection(receipt))
   dump(out/label/'SHARED_CALIBRATION_SELECTION.json',get(run['score_root']+'/SHARED_CALIBRATION_SELECTION.json'))
   for key,name in [('independent_audit','CONTRACT_NUMERIC_AUDIT.json'),('saved_scalar_audit','SAVED_SCALAR_AUDIT.json')]:
    evidence=get(run[key]);assert evidence['state'].startswith('PASS_');dump(out/label/name,scalar_projection(evidence))
   counts[label]={'n':n,'arms':len(arms),'task_values':len(scalar),'all_ordered_negatives':len(neg),'scene_arm_rows':len(sc),'new_GT_scored_task_values':receipt['new_GT_scored_task_values'],'reused_saved_task_values':receipt['reused_saved_task_values']}
  for key,name in [('report','REPORT.md'),('decision','DECISION.json'),('public_fixed_configuration','protocol/FIXED_CONFIGURATION.json'),('normalized_costs','costs/COLLECTION_COUNTS_TIMES.json'),('normalized_history','history/FAILURES_AND_REPAIRS.json')]:
   rel=plan[key];assert rel in pins and sha(local(rel))==pins[rel]
   if rel.endswith('.json'):
    value=get(rel);check(value)
    if key=='normalized_costs':
     assert set(value)=={'counts','timing_seconds','peak_bytes','flags'}
     assert all(isinstance(g,dict) and all(v is None or type(v) in (bool,int,float) for v in g.values()) for g in value.values())
    if key=='normalized_history':
     assert set(value)=={'events'}
     allowed={'stage','attempt','status','failure_type','repair_type','valid_outputs','attempted_calls','entered_forwards','completed_calls','seconds','before_GT','producer_predictions_changed','tolerance_changed','prior_attempt_preserved'}
     for event in value['events']:
      assert set(event)<=allowed
      for v in event.values():assert (re.fullmatch('[A-Za-z0-9_]{1,96}',v) is not None) if isinstance(v,str) else (v is None or type(v) in (bool,int,float))
    dump(out/name,value)
   else:write(out/name,safe_code(local(rel).read_text()))
  for rel in plan['own_code_allowlist']:
   assert rel in pins and sha(local(rel))==pins[rel] and rel.endswith('.py')
   assert rel.startswith(('crossplatform/round85/','research/round85/')) or rel=='tools/manual_r83_guard.py'
   assert not any(x in rel for x in ('vendor','overlay','upstream','annotate'))
   write(out/'own_code'/rel,safe_code(local(rel).read_text()))
  dump(out/'SCHEMA_COUNTS.json',counts)
  for item in plan.get('auxiliary_saved_scalar_reports',[]):
   rel=item['source'];assert rel in pins and sha(local(rel))==pins[rel];value=get(rel);check(value);dest=Path(item['public_path']);assert not dest.is_absolute() and '..' not in dest.parts
   if item.get('table_csv'):
    assert dest.suffix=='.csv' and isinstance(value,list) and value;fields=sorted(set().union(*(set(row) for row in value)));csvfile(out/dest,[{k:row.get(k) for k in fields} for row in value],fields)
   else:assert dest.suffix=='.json';dump(out/dest,value)
  for item in plan.get('auxiliary_text_reports',[]):
   rel=item['source'];assert rel in pins and sha(local(rel))==pins[rel];dest=Path(item['public_path']);assert dest.suffix=='.md' and '..' not in dest.parts and not dest.is_absolute();write(out/dest,safe_code(local(rel).read_text()))
  write(out/'README.md','# R85 saved results\n\nAll registered task values, UNKNOWN/fallback/no-update rows and ordered negatives are retained. Raw identifiers, original captions, media, point/mask/query indices, coordinates, feature vectors, model responses and weights are excluded. Historical target exposure and dependency/resource differences are recorded in REPORT.md. This local exporter does not publish or verify the remote old tree.\n')
  files=sorted(p for p in out.rglob('*') if p.is_file());findings=[]
  tokens=[]
  for r in registry['tokens']:
   v=r['value'];assert set(r)=={'kind','value'} and isinstance(v,str) and v
   pat=re.compile(r'(?<![A-Za-z0-9_])'+re.escape(v)+r'(?![A-Za-z0-9_])') if r['kind'] in {'id','alias','scene'} else None
   tokens.append((r['kind'],v,pat))
  automaton=ahocorasick.Automaton();buckets={}
  for kind,value,pattern in tokens:buckets.setdefault(value,[]).append((kind,pattern))
  for value,entries in buckets.items():automaton.add_word(value,(value,entries))
  automaton.make_automaton()
  for p in files:
   assert not p.is_symlink() and p.suffix in {'.py','.json','.csv','.md'}
   text=p.read_text();assert '\x00' not in text
   if p.suffix=='.json':check(json.loads(text))
   if p.suffix=='.csv':assert not (set(next(csv.reader(io.StringIO(text))))&DENIED)
   for end,(value,entries) in automaton.iter(text):
    start=end-len(value)+1
    for kind,pattern in entries:
     if pattern is not None and ((start>0 and re.match(r'[A-Za-z0-9_]',text[start-1])) or (end+1<len(text) and re.match(r'[A-Za-z0-9_]',text[end+1]))):continue
     findings.append({'file':p.relative_to(out).as_posix(),'kind':kind,'token_sha256':hashlib.sha256(value.encode()).hexdigest()})
  assert not findings,('privacy_findings_hashes_only',findings)
  dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS_R85_LOCAL_EXPLICIT_SCALAR_WHITELIST_AND_TOKEN_SCAN','findings':0,'registry_sha256':sha(local(plan['private_registry'])),'tokens_checked_per_file':len(tokens),'no_code_CSV_report_or_JSON_excluded':True})
  files=sorted(p for p in out.rglob('*') if p.is_file());dump(out/'MANIFEST.json',{'prefix':PREFIX,'manifest_excludes_itself':True,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
  files=sorted(p for p in out.rglob('*') if p.is_file());payload=out.parent/f'TREE_PAYLOAD_LOCAL_attempt{a.attempt:03d}.json';assert not payload.exists()
  dump(payload,[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
  dump(out.parent/f'LOCAL_EXPORT_ATTEMPT{a.attempt:03d}_AUDIT.json',{'state':'R85_LOCAL_EXPORT_READY_FOR_INDEPENDENT_REVIEW_NOT_PUBLISHED','files':len(files),'bytes':sum(p.stat().st_size for p in files),'prefix':PREFIX,'counts':counts,'manifest_sha256':sha(out/'MANIFEST.json'),'payload_sha256':sha(payload),'base_snapshot_sha256':sha(local(plan['publication_base_snapshot'])),'old_study_count_not_hardcoded':True,'remote_immutable_bytes_preservation_CAS_requires_root_receipt':True,'network_calls':0})
 except BaseException as e:dump(out/'EXPORT_FAILED.json',{'error_type':type(e).__name__,'error':str(e),'no_network_GT_or_method_mutation':True});raise
if __name__=='__main__':main()
