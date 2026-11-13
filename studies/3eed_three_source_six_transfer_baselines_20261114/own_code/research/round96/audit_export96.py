"""Independent byte/privacy/all-task scalar audit of the publication package; no models or scoring."""
from pathlib import Path
import json,hashlib,csv,re,math,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round96';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
DENIED={'id','id_PRIVATE','caption','scene_PRIVATE','metapath','imagepath','pointpath','original_meta_path','source_boxes','selected_original_boxes','query_indices','query_index','queries','coordinates','features','weights','raw_answer','api_key','access_token','private_key','positive_map'}
def check(v):
 if isinstance(v,dict):
  for k,x in v.items():assert k not in DENIED,k;check(x)
 elif isinstance(v,list):
  for x in v:check(x)
 elif isinstance(v,float):assert math.isfinite(v)
def main():
 plan=rd(H/'experiment/PUBLIC_EXPORT_LOCK.json')
 for f,s in plan['inputs_sha256'].items():assert sha(R/f)==s,f
 m=rd(O/'MANIFEST.json');idx=rd(P/'EXPORT_INDEX_PRIVATE.json');assert m['prefix']==idx['prefix'];files=sorted(p for p in O.rglob('*') if p.is_file());assert {p.relative_to(O).as_posix() for p in files}=={v['path'] for v in m['files']}|{'MANIFEST.json'}
 for x in m['files']:assert sha(O/x['path'])==x['sha256'] and (O/x['path']).stat().st_size==x['bytes']
 for f in idx['own_code_files']:assert (O/'own_code'/f).read_text()==(R/f).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 count=values=0;S=H/'frozen_scores_attempt003';matrix=rd(R/'crossplatform/round96/frozen_matrix_attempt003/MATRIX_SEALED.json')
 for d in matrix['directions']:
  src,tgt=d['source'],d['target'];rows=rd(S/(src+'_to_'+tgt+'_ALL_ROWS_PRIVATE.json'));csvrows=[]
  for f in sorted((O/'tasks').glob(src+'_to_'+tgt+'_part*.csv')):
   with f.open(newline='') as stream:csvrows+=list(csv.DictReader(stream))
  assert len(csvrows)==len(rows)==d['n'];sc=sorted({r['scene_PRIVATE'] for r in rows})
  for i,(a,b) in enumerate(zip(csvrows,rows)):
   assert a['row']=='ROW'+str(i+1).zfill(6) and a['source']==src and a['target']==tgt and a['scene_cluster']=='CLUSTER'+str(sc.index(b['scene_PRIVATE'])+1).zfill(3)
   for arm,z in b['arms'].items():
    for key,v in z.items():assert float(a[arm+'__'+key])==v
    values+=1
   count+=1
 assert count==18620 and values==186200
 with (O/'results/ALL_METHOD_ROWS.csv').open(newline='') as f:actual=list(csv.DictReader(f))
 expect=rd(S/'ALL_METHOD_ROWS.json');assert len(actual)==len(expect)==60
 for a,b in zip(actual,expect):
  for k,v in b.items():assert (a[k]==str(v)) if isinstance(v,(str,bool)) else (float(a[k])==v)
 with (O/'training/ALL_EPOCH_ROWS.csv').open(newline='') as f:curves=list(csv.DictReader(f))
 assert len(curves)==200
 for pl in ['drone','quad']:
  steps={}
  for f in sorted((R/'crossplatform/round96'/(pl+'_source100')/'attempts').glob('*/steps.jsonl')):
   for line in f.read_text().splitlines():
    t=json.loads(line);steps[t['epoch'],t['batch']]=t
  for c in [x for x in curves if x['source']==pl]:
   ep=int(c['epoch']);rr=[v for (e,b),v in steps.items() if e==ep];assert int(c['optimizer_updates'])==len(rr);assert abs(float(c['mean_loss'])-sum(r['loss'] for r in rr)/len(rr))<1e-12;assert float(c['min_loss'])==min(r['loss'] for r in rr) and float(c['max_loss'])==max(r['loss'] for r in rr)
 metadata=rd(O/'training/SOURCE100_METADATA.json');assert {x['source'] for x in metadata}=={'waymo','drone','quad'}
 for x in metadata:
  if x['source']=='waymo':r=rd(R/'crossplatform/round6/SOURCE_CHECKPOINT.json');assert x['checkpoint_sha256']==r['checkpoint_sha256'] and x['epoch']==100
  else:r=rd(R/'crossplatform/round96'/(x['source']+'_source100')/'TRAINING_COMPLETE.json');assert x['checkpoint_sha256']==r['primary_checkpoint_sha256'] and x['updates']==r['completed_optimizer_steps'] and x['epoch']==100
 registry=rd(P/'PRIVATE_PRIVACY_TOKEN_REGISTRY.json');buckets={}
 for r in registry['tokens']:buckets.setdefault(r['value'],[]).append(r['kind'])
 auto=ahocorasick.Automaton()
 for t,k in buckets.items():auto.add_word(t,(t,k))
 auto.make_automaton();findings=[]
 for f in files:
  assert not f.is_symlink() and f.suffix in {'.json','.md','.py','.csv'};s=f.read_text();assert '\x00' not in s
  if f.suffix=='.json':check(json.loads(s))
  if f.suffix=='.csv':assert not(set(next(csv.reader(s.splitlines())))&DENIED)
  for end,(t,kinds) in auto.iter(s):
   start=end-len(t)+1
   for kind in kinds:
    if kind in {'scene','id','alias'} and ((start>0 and re.match(r'[A-Za-z0-9_]',s[start-1])) or (end+1<len(s) and re.match(r'[A-Za-z0-9_]',s[end+1]))):continue
    findings.append({'file':f.relative_to(O).as_posix(),'kind':kind,'token_sha256':hashlib.sha256(t.encode()).hexdigest()})
 assert not findings,findings
 out=H/'independent_public_audit_attempt001';out.mkdir();a={'state':'PASS_R96_INDEPENDENT_PUBLIC_BYTES_ALL_TASK_READOUT_VALUES_AND_PRIVACY','files':len(files),'bytes':sum(f.stat().st_size for f in files),'direction_task_rows':count,'readout_task_values':values,'privacy_findings':0,'tokens_scanned_per_file':len(registry['tokens']),'own_code_files_verified':len(idx['own_code_files']),'private_registry_sha256':sha(P/'PRIVATE_PRIVACY_TOKEN_REGISTRY.json'),'manifest_sha256':sha(O/'MANIFEST.json'),'raw_models_media_GT_or_network_read':False};(out/'AUDIT_RESULT.json').write_text(json.dumps(a,indent=2)+'\n');print(a)
if __name__=='__main__':main()
