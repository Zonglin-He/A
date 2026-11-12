"""R86 independent public package audit, saved scalars only. Saved-only.

No exporter import, task GT, models, geometry, raw media or network. A root
plan pins the frozen package/payload, private saved tables, code and registry.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,math,re,ahocorasick
ROOT=Path(__file__).resolve().parents[2]
PREFIX='studies/3eed_tristate_owned_relations_discrimination_priority_TTA_20261112'
DENIED={'id','id_PRIVATE','alias','aliases','scene_PRIVATE','caption','subject_query','actual_head_word','queries','query_indices','query_index','point_indices','mask_indices','mask_index','token_mask','positive_mask','source_boxes','selected_original_boxes','source_scores','boxes','bbox','coordinates','features','tokens','h','weights','raw_answer','yes_logits','no_logits','imagepath','pointpath','metapath','raw','input_key','logical_binding','private_key','api_key','access_token'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def local(s):
 p=Path(s);assert not p.is_absolute() and '..' not in p.parts and 'round46/final' not in s and 'FINAL384' not in s;return ROOT/p
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def equal(x,y,p='$'):
 if isinstance(y,dict):
  assert isinstance(x,dict) and set(x)==set(y),('keys',p)
  for k in y:equal(x[k],y[k],p+'.'+k)
 elif isinstance(y,list):
  assert isinstance(x,list) and len(x)==len(y),('length',p)
  for i,(a,b) in enumerate(zip(x,y)):equal(a,b,p+'['+str(i)+']')
 elif type(y) is bool:assert type(x) is bool and x==y,('bool',p)
 elif type(y) in (int,float):assert type(x) in (int,float) and math.isfinite(x) and x==y,('number',p)
 else:assert type(x) is type(y) and x==y,('value',p)
def public_json(x,p='$'):
 if isinstance(x,dict):
  for k,v in x.items():assert k not in DENIED,('private_key',p,k);public_json(v,p+'.'+k)
 elif isinstance(x,list):
  for v in x:public_json(v,p+'[]')
 elif isinstance(x,float):assert math.isfinite(x)
def csv_equal(p,rows,fields):
 with p.open(newline='') as f:r=csv.DictReader(f);assert r.fieldnames==fields;actual=list(r)
 assert len(actual)==len(rows),(p.name,'row_count')
 for i,(a,b) in enumerate(zip(actual,rows)):
  assert set(b)==set(fields)
  for k,v in b.items():
   if type(v) is bool:assert a[k]==str(v),(p.name,i,k)
   elif type(v) is int:assert a[k]==str(v),(p.name,i,k)
   elif type(v) is float:assert math.isfinite(float(a[k])) and float(a[k])==v,(p.name,i,k)
   elif v is None:assert a[k]=='',(p.name,i,k)
   else:assert a[k]==str(v),(p.name,i,k)
 return len(actual)
def receipt(x):return {k:v for k,v in x.items() if v is None or type(v) in (bool,int,float) or (isinstance(v,str) and (k=='state' or k.endswith('_sha256')))}
def sanitized(s):return s.replace(str(ROOT),'WORKSPACE_ROOT').replace(str(ROOT.parent),'LOCAL_HOME')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R86_INDEPENDENT_LOCAL_PUBLIC_EXPORT_AUDIT_SAVED_ONLY'
 assert auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(args.plan)
 assert plan['no_GT_models_geometry_media_network_or_original_manifest_read'] is True
 pins=plan['inputs_sha256']
 def get(rel):assert rel in pins and sha(local(rel))==pins[rel],('pin',rel);return read(local(rel))
 pub=local(plan['public_package']);manifest=get(plan['manifest']);payload=get(plan['payload']);authorplan=get(plan['publication_plan'])
 assert auth['manifest_sha256']==pins[plan['manifest']] and auth['payload_sha256']==pins[plan['payload']]
 assert local(plan['manifest'])==pub/'MANIFEST.json' and manifest['prefix']==PREFIX and authorplan['prefix']==PREFIX
 out=local(plan['output_directory']);assert not out.exists();out.mkdir(parents=True)
 try:
  files=sorted(p for p in pub.rglob('*') if p.is_file());rels={p.relative_to(pub).as_posix() for p in files}
  assert not any(p.is_symlink() for p in pub.rglob('*')) and len(files)==len(rels)
  assert manifest['manifest_excludes_itself'] is True and len(manifest['files'])==len(files)-1
  assert {r['path'] for r in manifest['files']}==rels-{'MANIFEST.json'}
  for r in manifest['files']:
   assert set(r)=={'path','bytes','sha256'} and not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts
   pp=pub/r['path'];assert pp.stat().st_size==r['bytes'] and sha(pp)==r['sha256']
  assert len(payload)==len(files) and {r['path'] for r in payload}=={PREFIX+'/'+r for r in rels}
  for r in payload:
   assert set(r)=={'path','mode','type','content'} and r['mode']=='100644' and r['type']=='blob'
   assert (pub/r['path'][len(PREFIX)+1:]).read_bytes()==r['content'].encode('utf-8')
  codes=authorplan['own_code_allowlist'];assert len(set(codes))==len(codes)
  assert {r for r in rels if r.endswith('.py')}=={'own_code/'+r for r in codes}
  for rel in codes:
   assert rel in pins and sha(local(rel))==pins[rel] and rel.endswith('.py')
   assert not any(t in rel for t in ('vendor','overlay','upstream','annotate'))
   assert (pub/'own_code'/rel).read_text()==sanitized(local(rel).read_text()),('sanitized_code_bytes',rel)
  counts={}
  for run in authorplan['scalar_runs']:
   label=run['label'];assert re.fullmatch('[A-Z0-9_]{1,32}',label) and label not in counts
   root=run['score_root'].rstrip('/')+'/';seal=get(root+'SEALED.json')
   for name,h in seal['files'].items():assert sha(local(root+name))==h
   tasks=get(root+'ALL_TASK_ROWS_PRIVATE.json');neg=get(root+'ALL_ORDERED_NEGATIVE_COMPARISONS.json');scenes=get(root+'ALL_SCENE_SUMMARIES_PRIVATE.json');summary=get(root+'TASK_SUMMARY.json');scoring=get(root+'SCORING_RECEIPT.json')
   n=scoring['n'];arms=scoring['arm_names'];assert len(tasks)==n and n in (336,1280,1024) and n*len(arms)==scoring['task_values']
   assert scoring['ordered_negative_comparisons']==len(neg)
   assert scoring['new_GT_scored_task_values']+scoring['reused_saved_task_values']==scoring['task_values']
   mapping={r['row']:f'{label}_{i+1:05d}' for i,r in enumerate(tasks)};assert len(mapping)==n
   scene_values={pl:sorted({r['scene_PRIVATE'] for r in tasks if r['platform']==pl}) for pl in ('drone','quad')}
   def scene(pl,s):return ('D' if pl=='drone' else 'Q')+'_S'+str(scene_values[pl].index(s)+1).zfill(2)
   rows=[]
   for r in tasks:
    assert set(r['IoU'])==set(r['statuses'])==set(arms)
    for arm in arms:rows.append({'row':mapping[r['row']],'platform':r['platform'],'scene_alias':scene(r['platform'],r['scene_PRIVATE']),'part':r['part'],'split':r['split'],'arm':arm,'IoU':r['IoU'][arm],**r['statuses'][arm]})
   assert len(rows)==n*len(arms)
   for pl in ('drone','quad'):
    csv_equal(pub/label/(pl.upper()+'_ALL_TASK_SCALARS.csv'),[r for r in rows if r['platform']==pl],['row','platform','scene_alias','part','split','arm','IoU','unknown','fallback','no_parameter_update'])
    nn=[{'row':mapping[r['row']],**{k:r[k] for k in ('platform','part','split','arm','reference','delta_pp')}} for r in neg if r['platform']==pl]
    csv_equal(pub/label/(pl.upper()+'_ALL_ORDERED_NEGATIVES.csv'),nn,['row','platform','part','split','arm','reference','delta_pp'])
   sr=[]
   for r in scenes:
    for arm in arms:sr.append({'scope':r['scope'],'platform':r['platform'],'scene_alias':scene(r['platform'],r['scene_PRIVATE']),'n':r['n'],'arm':arm,'mIoU_percent':r['mIoU_percent'][arm]})
   csv_equal(pub/label/'ALL_SCENE_ARM_SCALARS.csv',sr,['scope','platform','scene_alias','n','arm','mIoU_percent'])
   equal(read(pub/label/'TASK_SUMMARY.json'),summary);equal(read(pub/label/'SCORE_RECEIPT.json'),receipt(scoring))
   equal(read(pub/label/'SHARED_CALIBRATION_SELECTION.json'),get(root+'SHARED_CALIBRATION_SELECTION.json'))
   for key,name in [('independent_audit','CONTRACT_NUMERIC_AUDIT.json'),('saved_scalar_audit','SAVED_SCALAR_AUDIT.json')]:
    x=get(run[key]);assert x['state'].startswith('PASS_');equal(read(pub/label/name),receipt(x))
   counts[label]={'n':n,'arms':len(arms),'task_values':len(rows),'all_ordered_negatives':len(neg),'scene_arm_rows':len(sr),'new_GT_scored_task_values':scoring['new_GT_scored_task_values'],'reused_saved_task_values':scoring['reused_saved_task_values']}
  equal(read(pub/'SCHEMA_COUNTS.json'),counts)
  for key,name in [('report','REPORT.md'),('decision','DECISION.json'),('public_fixed_configuration','protocol/FIXED_CONFIGURATION.json'),('normalized_costs','costs/COLLECTION_COUNTS_TIMES.json'),('normalized_history','history/FAILURES_AND_REPAIRS.json')]:
   rel=authorplan[key];assert rel in pins and sha(local(rel))==pins[rel]
   if rel.endswith('.json'):equal(read(pub/name),get(rel))
   else:assert (pub/name).read_text()==sanitized(local(rel).read_text())
  for item in authorplan.get('auxiliary_saved_scalar_reports',[]):
   rel=item['source'];assert rel in pins and sha(local(rel))==pins[rel];value=get(rel)
   if item.get('table_csv'):
    fields=sorted({key for row in value for key in row});canonical=[dict((key,row[key] if key in row else None) for key in fields) for row in value];csv_equal(pub/item['public_path'],canonical,fields)
   else:equal(read(pub/item['public_path']),value)
  for item in authorplan.get('auxiliary_text_reports',[]):
   rel=item['source'];assert rel in pins and sha(local(rel))==pins[rel];assert (pub/item['public_path']).read_text()==sanitized(local(rel).read_text())
  registry=get(plan['private_registry']);assert plan['private_registry']==authorplan['private_registry']
  assert sha(local(plan['private_registry']))==auth['private_registry_sha256']
  assert len(registry['tokens'])==plan['expected_registry_tokens']
  assert registry['coverage']['full5912_ids']==5912 and registry['coverage']['dev336_ids']==336
  assert registry['coverage']['full_caption_records']==5912 and registry['coverage']['full_media_path_records']==17736
  assert registry['coverage']['registry_builder_opens_only_saved_registry_and_selected_nonFINAL_manifest'] is True
  assert registry['coverage']['R69_selected_input_ids']==2304 and registry['coverage']['R69_train_input_ids']==1024
  scans=[]
  for t in registry['tokens']:
   assert set(t)=={'kind','value'} and t['kind'] in {'id','alias','scene','caption','media_path'} and isinstance(t['value'],str) and t['value']
   v=t['value'];pattern=re.compile(r'(?<![A-Za-z0-9_])'+re.escape(v)+r'(?![A-Za-z0-9_])') if t['kind'] in {'id','alias','scene'} else None;scans.append((t['kind'],v,pattern))
  automaton=ahocorasick.Automaton();buckets={}
  for kind,value,pattern in scans:buckets.setdefault(value,[]).append((kind,pattern))
  for value,entries in buckets.items():automaton.add_word(value,(value,entries))
  automaton.make_automaton()
  findings=[]
  for p in files:
   rel=p.relative_to(pub).as_posix();assert p.suffix in {'.py','.json','.csv','.md'} and not any(t in Path(rel).parts for t in ('vendor','runtime_overlay','upstream','.git','node_modules'))
   text=p.read_text();assert '\x00' not in text
   if p.suffix=='.json':public_json(json.loads(text))
   if p.suffix=='.csv':assert not (set(next(csv.reader(io.StringIO(text))))&DENIED)
   if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?<!\w)(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{12,}',text):findings.append({'file':rel,'kind':'secret_pattern'})
   for end,(value,entries) in automaton.iter(text):
    start=end-len(value)+1
    for kind,pattern in entries:
     if pattern is not None and ((start>0 and re.match(r'[A-Za-z0-9_]',text[start-1])) or (end+1<len(text) and re.match(r'[A-Za-z0-9_]',text[end+1]))):continue
     findings.append({'file':rel,'kind':kind,'token_sha256':hashlib.sha256(value.encode()).hexdigest()})
  assert not findings,('privacy_findings_hashes_only',findings)
  base=get(authorplan['publication_base_snapshot']);assert auth['base_snapshot_sha256']==sha(local(authorplan['publication_base_snapshot']))
  result={'state':'PASS_R86_INDEPENDENT_PUBLIC_EXPORT_SAVED_SCALAR_BYTES_AND_PRIVACY_AUDIT','prefix':PREFIX,'files':len(files),'bytes':sum(p.stat().st_size for p in files),'counts':counts,'privacy_findings':0,'tokens_scanned_in_every_public_file':len(scans),'no_file_excluded_from_token_scan':True,'all_own_code_paths_and_sanitized_bytes_verified':True,'all_task_status_scene_negative_summary_selection_receipt_values_verified':True,'manifest_sha256':sha(pub/'MANIFEST.json'),'payload_sha256':sha(local(plan['payload'])),'base_snapshot_sha256':sha(local(authorplan['publication_base_snapshot'])),'only_single_new_study_payload':True,'remote_old_tree_preservation_immutable_verify_and_CAS_not_claimed':True,'no_GT_models_geometry_media_or_network':True,'program_sha256':sha(__file__),'plan_sha256':sha(args.plan)}
  dump(out/'AUDIT_RESULT.json',result);dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}});print(json.dumps(result),flush=True)
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'no_GT_network_or_package_mutation':True});raise
if __name__=='__main__':main()
