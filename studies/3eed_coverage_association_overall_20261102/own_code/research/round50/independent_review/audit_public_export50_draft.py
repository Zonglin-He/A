"""UNEXECUTED independent saved-scalar/package/privacy audit; no publisher import.

Requires a root grant pinning this program, an audit plan, the complete local
public package, payload, publication plan, saved scalars and private registry.
No model, geometry, labels, media, original manifests, or network operations.
"""
from pathlib import Path
import argparse, collections, csv, hashlib, io, json, math, re

ROOT=Path(__file__).resolve().parents[3]
PREFIX='studies/3eed_coverage_association_overall_20261102'
DEV_ARMS=['source','old_G','wideentry_oldW','oldentry_newW_k3','wideentry_newW_k3','oldentry_newW_k10','wideentry_newW_k10']
FULL_ARMS=['source','old_G','wideentry_oldW']
TTA_ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']
DENIED_KEYS={'id','id_PRIVATE','alias','aliases','scene_PRIVATE','caption','subject_query','actual_head_word','class_name',
 'token_mask','positive_mask','positive_masks','offsets','char_span','queries','query','query_indices','query_index',
 'point_indices','mask_indices','mask_index','mask_id','source_boxes','source_scores','selected_original_boxes',
 'boxes','bbox','boxes_PRIVATE','box_PRIVATE','coordinates','features','tokens','h','weights','yes_logits','no_logits',
 'raw_answer','imagepath','pointpath','metapath','raw','input_key','logical_binding','value_path','api_key','access_token','private_key'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert 'round46/final' not in rel and 'FINAL384' not in rel
 return ROOT/p

def check_struct(x,path='$'):
 if isinstance(x,dict):
  for k,v in x.items():assert k not in DENIED_KEYS,('forbidden_data_key',path,k);check_struct(v,path+'.'+k)
 elif isinstance(x,list):
  for i,v in enumerate(x):check_struct(v,path+'[]')
 elif isinstance(x,float):assert math.isfinite(x),('nonfinite',path)

def equal(x,y,path='$'):
 if isinstance(y,dict):
  assert isinstance(x,dict) and set(x)==set(y),('keys',path)
  for k in y:equal(x[k],y[k],path+'.'+k)
 elif isinstance(y,list):
  assert isinstance(x,list) and len(x)==len(y),('list_length',path)
  for i,(a,b) in enumerate(zip(x,y)):equal(a,b,path+'['+str(i)+']')
 elif type(y) is bool:assert type(x) is bool and x==y,('bool',path)
 elif type(y) in (int,float):assert type(x) in (int,float) and x==y,('numeric',path)
 else:assert type(x) is type(y) and x==y,('value',path)

def csv_match(p,expected,fields):
 with p.open(newline='') as f:
  rr=csv.DictReader(f);assert rr.fieldnames==fields,('csv_fields',p.name)
  rows=list(rr)
 assert len(rows)==len(expected),('csv_count',p.name)
 for i,(r,e) in enumerate(zip(rows,expected)):
  assert set(e)==set(fields)
  for k in fields:
   v=e[k];actual=r[k]
   if type(v) is bool:assert actual==str(v),(p.name,i,k)
   elif type(v) is int:assert int(actual)==v and str(int(actual))==actual,(p.name,i,k)
   elif type(v) is float:assert math.isfinite(float(actual)) and float(actual)==v,(p.name,i,k)
   elif v is None:assert actual=='',(p.name,i,k)
   else:assert actual==str(v),(p.name,i,k)
 return len(rows)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for k in ('plan','authorization'):ap.add_argument('--'+k,required=True)
 a=ap.parse_args();plan=read(a.plan);auth=read(a.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_PUBLIC_EXPORT_AUDIT_SAVED_ONLY'
 assert auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(a.plan)
 assert plan['no_GT_media_models_original_manifests_or_network'] is True
 pins=plan['inputs_sha256']
 def get(rel):
  assert rel in pins and sha(local(rel))==pins[rel],('input_pin',rel)
  return read(local(rel))
 out=local(plan['audit_output']);assert not out.exists();out.mkdir(parents=True)
 consumed={}
 try:
  pub=local(plan['public_package']);assert pub.is_dir() and not pub.is_symlink()
  manifest=get(plan['manifest']);payload=get(plan['payload'])
  assert auth['manifest_sha256']==pins[plan['manifest']] and auth['payload_sha256']==pins[plan['payload']]
  assert local(plan['manifest'])==pub/'MANIFEST.json'
  assert manifest['prefix']==PREFIX and manifest['manifest_excludes_itself'] is True
  actual=sorted(p for p in pub.rglob('*') if p.is_file())
  assert not any(p.is_symlink() for p in pub.rglob('*'))
  rels=[p.relative_to(pub).as_posix() for p in actual];assert len(set(rels))==len(rels)
  for p in actual:
   rel=p.relative_to(pub).as_posix();assert p.suffix in {'.py','.json','.csv','.md'} and '..' not in Path(rel).parts
   assert not any(q in {'vendor','runtime_overlay','runtime_accelerate','runtime_deps','upstream','node_modules','.git'} for q in Path(rel).parts)
   p.read_text();assert '\x00' not in p.read_text()
  claimed=manifest['files'];assert len(claimed)==len(actual)-1
  assert len({r['path'] for r in claimed})==len(claimed)
  assert {r['path'] for r in claimed}==set(rels)-{'MANIFEST.json'}
  for r in claimed:
   assert set(r)=={'path','bytes','sha256'};p=pub/r['path']
   assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],('manifest_bytes',r['path'])
  assert isinstance(payload,list) and len(payload)==len(actual)
  assert len({r['path'] for r in payload})==len(payload)
  assert {r['path'] for r in payload}=={PREFIX+'/'+r for r in rels}
  for r in payload:
   assert set(r)=={'path','mode','type','content'} and r['mode']=='100644' and r['type']=='blob'
   rel=r['path'][len(PREFIX)+1:];assert (pub/rel).read_bytes()==r['content'].encode(),('payload_bytes',rel)
  # Every existing root/study lies outside this payload. Remote preservation,
  # immutable-byte verification and branch CAS remain a separate root receipt.
  pubplan=get(plan['publication_plan']);pubauth=get(plan['publication_authorization'])
  assert pubauth['state']=='ROOT_AUTHORIZED_R50_LOCAL_WHITELIST_PUBLIC_EXPORT_ONLY'
  assert pubauth['plan_sha256']==pins[plan['publication_plan']]
  assert pubauth['private_privacy_token_registry_sha256']==pins[plan['private_registry']]
  assert pubauth.get('publication_base_snapshot_sha256')==plan['publication_base_snapshot_sha256']
  own=plan['own_public_code_allowlist'];assert isinstance(own,list) and len(set(own))==len(own)
  assert {r for r in rels if r.endswith('.py')}==set(own),'own-code whitelist differs'
  for rel,h in plan['own_public_code_sha256'].items():assert rel in own and sha(pub/rel)==h
  assert set(plan['own_public_code_sha256'])==set(own)
  # DEV saved scalars, negatives, scenes and summary. Only saved IoUs are read.
  dev=get(plan['dev_rows']);assert len(dev)==336
  dn=get(plan['dev_negatives']);assert len(dn)==1758
  ds=get(plan['dev_scenes']);dm={r['row']:f'DEV{i+1:04d}' for i,r in enumerate(dev)}
  scene_names={pl:sorted({r['scene_PRIVATE'] for r in dev if r['platform']==pl}) for pl in ('drone','quad')}
  def dscene(pl,s):return pl+'_scene'+str(scene_names[pl].index(s)+1).zfill(2)
  expected=[]
  for r in dev:
   assert set(r['IoU'])==set(DEV_ARMS)
   for arm in DEV_ARMS:expected.append({'row':dm[r['row']],'split':r['split'],'platform':r['platform'],'scene_alias':dscene(r['platform'],r['scene_PRIVATE']),'arm':arm,'IoU':r['IoU'][arm],'changed_vs_source':r['queries'][arm]!=r['queries']['source'],'changed_vs_old_G':r['queries'][arm]!=r['queries']['old_G']})
  assert len(expected)==2352;csv_match(pub/'dev/results/ALL2352_TASK_SCALARS.csv',expected,list(expected[0]))
  expected=[{**{k:v for k,v in r.items() if k!='row'},'row':dm[r['row']]} for r in dn]
  csv_match(pub/'dev/results/ALL1758_ORDERED_NEGATIVE_COMPARISONS.csv',expected,list(expected[0]))
  expected=[]
  for r in ds:
   for arm in DEV_ARMS:expected.append({'split':r['part'],'platform':r['platform'],'scene_alias':dscene(r['platform'],r['scene_PRIVATE']),'n':r['n'],'arm':arm,'mIoU_percent':r['mIoU_percent'][arm]})
  assert len(expected)==140;csv_match(pub/'dev/results/ALL_SCENE_ARM_SCALARS.csv',expected,list(expected[0]))
  for name,key in [('TASK_SUMMARY.json','dev_summary'),('FACTORIAL_CONTRASTS.json','dev_contrasts'),('DEV_SELECTION_AND_FULL_GATE.json','dev_gate')]:equal(read(pub/'dev/results'/name),get(plan[key]))
  # Full task tables, all ordered negatives and source/old-G negative subsets.
  full=get(plan['full_rows']);fn=get(plan['full_negatives']);fscene=get(plan['full_scenes']);assert len(full)==5912
  fm={r['row']:f'F{i+1:05d}' for i,r in enumerate(full)}
  fsn={pl:sorted({r['scene_PRIVATE'] for r in full if r['platform']==pl}) for pl in ('drone','quad')}
  assert {p:len(v) for p,v in fsn.items()}=={'drone':2,'quad':3}
  def fsc(pl,s):return ('D' if pl=='drone' else 'Q')+'_S'+str(fsn[pl].index(s)+1).zfill(2)
  scalar=[]
  for r in full:
   assert set(r['IoU'])==set(FULL_ARMS)
   for arm in FULL_ARMS:scalar.append({'row':fm[r['row']],'part':r['part'],'platform':r['platform'],'scene_alias':fsc(r['platform'],r['scene_PRIVATE']),'arm':arm,'IoU':r['IoU'][arm],'changed_vs_source':r['queries'][arm]!=r['queries']['source'],'changed_vs_old_G':r['queries'][arm]!=r['queries']['old_G']})
  assert len(scalar)==17736
  for pl,n in [('drone',8952),('quad',8784)]:csv_match(pub/f'full/results/{pl.upper()}_{n}_TASK_SCALARS.csv',[r for r in scalar if r['platform']==pl],list(scalar[0]))
  publicneg=[{k:r[k] for k in ('part','platform','arm','reference','delta_pp')}|{'row':fm[r['row']]} for r in fn]
  assert len(publicneg)==plan['expected_full_negative_count']
  nf=['row','part','platform','arm','reference','delta_pp']
  for pl in ('drone','quad'):
   group=[r for r in publicneg if r['platform']==pl]
   csv_match(pub/f'full/results/{pl.upper()}_ALL_ORDERED_NEGATIVE_COMPARISONS.csv',group,nf)
   for ref in ('source','old_G'):csv_match(pub/f'full/results/{pl.upper()}_ALL_NEGATIVES_VS_{ref.upper()}.csv',[r for r in group if r['reference']==ref],nf)
  expected=[]
  for r in fscene:
   for arm in FULL_ARMS:expected.append({'part':r['part'],'platform':r['platform'],'scene_alias':fsc(r['platform'],r['scene_PRIVATE']),'n':r['n'],'arm':arm,'mIoU_percent':r['mIoU_percent'][arm]})
  csv_match(pub/'full/results/ALL_SCENE_ARM_SCALARS.csv',expected,list(expected[0]))
  fullsummary=get(plan['full_summary']);metrics=['n','mIoU_percent','Acc25_percent','Acc50_percent','gain_vs_source_pp','gain_vs_old_G_pp','improved_vs_source','worsened_vs_source','improved_vs_old_G','worsened_vs_old_G','query_changes_vs_source','query_changes_vs_old_G','scene_clusters']
  projected={}
  for part in ('ALL5912','DEV336','NONDEV5576'):
   projected[part]={pl:{arm:{k:fullsummary[part][pl][arm][k] for k in metrics} for arm in FULL_ARMS} for pl in ('drone','quad')}
   for kind in ('balanced_mIoU_percent','micro_mIoU_percent'):projected[part][kind]={arm:fullsummary[part][kind][arm] for arm in FULL_ARMS}
  equal(read(pub/'full/results/TASK_SUMMARY.json'),projected)
  nt=get(plan['normalized_TTA']);assert nt['n']==336 and nt['arm_names']==TTA_ARMS and len(nt['task_rows'])==2016
  equal(read(pub/'tta/results/ALL2016_ANONYMOUS_TASK_SCALARS.json'),nt['task_rows'])
  for key in ('negative_comparisons','scene_rows','summary','update_diagnostics','selection_and_gate'):equal(read(pub/'tta/results'/(key.upper()+'.json')),nt[key])
  assert len(nt['negative_comparisons'])==plan['expected_TTA_negative_count']
  schemacounts=read(pub/'SCHEMA_COUNTS.json')
  assert schemacounts['dev']['task_values']==2352 and schemacounts['dev']['negative_comparisons']==1758
  assert schemacounts['full']['task_values']==17736 and schemacounts['full']['negative_comparisons']==len(fn)
  assert schemacounts['tta']['task_values']==2016 and schemacounts['tta']['negative_comparisons']==len(nt['negative_comparisons'])
  for rel,key in [('full/costs/COLLECTION_COMPONENT_COUNTS_AND_TIMES.json','normalized_costs'),('history/TECHNICAL_FAILURES_AND_REPAIRS.json','normalized_history')]:
   source=get(plan[key])
   if key=='normalized_history':source={'events':source['events'],'raw_logs_call_payloads_and_vendor_overlay_excluded':True}
   equal(read(pub/rel),source)
  # Anonymous collection-cost rows are checked from already audited saved scalars.
  cost=get(plan['cost_rows']);cf=['stage','chunk','cases','physical_completed_forwards','logical_uses','historical_exact_uses','historical_whole_case_uses','new_or_samechunk_completed_input_uses','physical_call_cap','ledger_wall_seconds_this_process','model_load_seconds_overlaps_wall','process_peak_bytes','model_unchanged','generated_tokens','optimizer_steps']
  csv_match(pub/'full/costs/ALL_CHUNK_COSTS.csv',[{k:r[k] for k in cf} for r in cost],cf)
  lc=get(plan['cost_logical_rows']);lf=['stage','row','platform','old_pool_count','wide_pool_count','target_logical_uses','reference_logical_uses','whole_case_historical_reuse','no_generated_tokens_or_optimizer']
  assert len(lc)==11824;csv_match(pub/'full/costs/ALL11824_CASE_STAGE_LOGICAL_USE_COUNTS.csv',[{k:r[k] for k in lf} for r in lc],lf)
  # Every byte-bearing public file is scanned, including source, CSV and manifest.
  privacy=get(plan['private_registry']);assert len(privacy['tokens'])==plan['expected_private_registry_tokens']==29937
  assert privacy['coverage']['full5912_ids']==5912 and privacy['coverage']['dev336_ids']==336
  assert privacy['coverage']['full_caption_records']==5912 and privacy['coverage']['full_media_path_records']==17736
  assert privacy['coverage']['no_protected_FINAL_specific_artifact_read'] is True
  findings=[];secret_patterns=[r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'(?<![A-Za-z0-9])(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{12,}']
  scans=[]
  for item in privacy['tokens']:
   assert set(item)=={'kind','value'} and item['kind'] in {'id','alias','scene','caption','media_path'}
   value=item['value'];assert isinstance(value,str) and value
   pattern=re.compile(r'(?<![A-Za-z0-9_])'+re.escape(value)+r'(?![A-Za-z0-9_])') if item['kind'] in {'id','alias','scene'} else None
   scans.append((item['kind'],value,pattern))
  for p in actual:
   rel=p.relative_to(pub).as_posix();text=p.read_text()
   if p.suffix=='.json':check_struct(json.loads(text))
   if p.suffix=='.csv':assert not (set(next(csv.reader(io.StringIO(text)))) & DENIED_KEYS),('forbidden_csv_column',rel)
   for pattern in secret_patterns:
    if re.search(pattern,text):findings.append({'path':rel,'kind':'secret_pattern'})
   for kind,value,pattern in scans:
    hit=pattern.search(text) is not None if pattern is not None else value in text
    if hit:findings.append({'path':rel,'kind':kind,'token_sha256':hashlib.sha256(value.encode()).hexdigest()})
  assert not findings,('privacy_findings_hashes_only',findings)
  # The scientific result is not generalized to parameter-adaptation success.
  decision=read(pub/'DECISION.json');assert decision['TTA_gate_PASS'] is False and decision['TTA_adoption_decision']=='STOP_NOT_ADOPTED' and decision['TTA_stop_without_full_TTA'] is True
  assert decision['full_gate_PASS'] is True
  report=(pub/'REPORT.md').read_text();assert '历史暴露' in report and '不是' in report and '不采纳' in report and '不运行全量 TTA' in report
  result={'state':'PASS_R50_INDEPENDENT_PUBLIC_EXPORT_SAVED_SCALAR_BYTES_AND_PRIVACY_AUDIT','no_GT_media_original_manifests_models_or_network_read':True,
   'audit_plan_sha256':sha(a.plan),'program_sha256':sha(__file__),'prefix':PREFIX,'files':len(actual),'bytes':sum(p.stat().st_size for p in actual),
   'manifest_sha256':sha(pub/'MANIFEST.json'),'payload_sha256':sha(local(plan['payload'])),'DEV_task_scalars':2352,'DEV_ordered_negatives':1758,
   'full_task_scalars':17736,'full_ordered_negatives':len(fn),'TTA_task_scalars':2016,'TTA_ordered_negatives':len(nt['negative_comparisons']),
   'private_registry_tokens_checked_per_public_file':len(privacy['tokens']),'privacy_findings':0,'no_public_file_excluded_from_text_scan':True,
   'only_one_new_study_payload_verified':True,'publication_base_snapshot_sha256':plan['publication_base_snapshot_sha256'],
   'remote_old_tree_preservation_and_CAS_not_claimed_by_local_audit':True}
  dump(out/'AUDIT_RESULT.json',result)
  dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
  print(json.dumps(result),flush=True)
 except BaseException as e:
  dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'no_GT_model_network_or_rerun':True});raise

if __name__=='__main__':main()
