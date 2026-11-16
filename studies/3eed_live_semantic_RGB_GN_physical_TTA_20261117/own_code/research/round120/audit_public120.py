"""Independent full task, physical update/context and privacy export verification."""
from pathlib import Path
import json,csv,hashlib,math,re,ahocorasick
R=Path(__file__).resolve().parents[2];H=R/'research/round120';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cr(p):
 with p.open() as f:return list(csv.DictReader(f))
def parts(p):return [r for f in sorted(p.glob('part*.csv')) for r in cr(f)]
def eq(a,v):return a=='' if v is None else a==str(v) if isinstance(v,(str,bool)) else float(a)==v
def main():
 mf=rd(O/'MANIFEST.json');ff=sorted(f for f in O.rglob('*') if f.is_file());assert len(ff)==len(mf['files'])+1
 for f in mf['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 for f in rd(P/'EXPORT_INDEX_PRIVATE.json')['sources']:assert (O/f['public']).read_text()==(R/f['source']).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
 n=values=aliases=0;mapping={};rowmap={};physical=[]
 for f in sorted((H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json')):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f);got=cr(O/'tasks'/(name+'.csv'));assert len(rr)==len(got);sc=sorted({r['scene_PRIVATE'] for r in rr});mapping[name]=sc;rowmap[name]={}
  for i,(a,r) in enumerate(zip(got,rr)):
   tag=f'ROW{i+1:06d}';rowmap[name][r['id_PRIVATE']]=tag;assert a['row']==tag and a['scene_cluster']==f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}" and float(a['all_arm_shared_cache_seconds'])==r['seconds']
   for k in ['cohort','source','target','protected_alias_fallback','C_available']:assert eq(a[k],r[k])
   for k,x in r['arms'].items():
    for m,v in x.items():
     if m!='query_PRIVATE':assert float(a[k+'__'+m])==v
    values+=1
   n+=1;aliases+=r['protected_alias_fallback']
   if not r['protected_alias_fallback']:
    rec=rd(R/'crossplatform/round120/formal'/r['cohort']/(r['source']+'_to_'+r['target'])/'records_PRIVATE'/(r['id_PRIVATE']+'.json'));norms={x['setting']:x for x in rec['evaluation']['writeback']}
    for st,v in rec['evaluation']['physical_losses'].items():physical.append({'row':tag,'cohort':r['cohort'],'source':r['source'],'target':r['target'],'setting':st,'scope_parameters':224,'train_cells':rec['fit']['train_cells'],'holdout_cells':rec['fit']['holdout_cells'],'source_GN_L2':rec['fit']['source_GN_L2'],'trust_radius':rec['fit']['trust_radius'],'fit_depth':v['fit_depth'],'heldout_depth':v['heldout_depth'],'changed_tensors':norms[st]['changed_tensors'],'actual_writeback_L2':norms[st]['actual_writeback_L2'],'proposal_L2':norms[st]['proposal_L2']})
 for co in ['DEV','EXP']:
  for src in ['waymo','drone','quad']:
   want=[r for r in physical if r['cohort']==co and r['source']==src];got=parts(O/'physical'/co/src);assert len(want)==len(got)
   for a,b in zip(got,want):assert all(eq(a[k],v) for k,v in b.items())
 for fn in ['CONTEXT_EFFECT_ROWS_PRIVATE','GOOD_FAILURE_CASES_PRIVATE']:
  got=parts(O/'diagnostics'/fn.replace('_PRIVATE',''));rr=rd(H/(fn+'.json'));assert len(got)==len(rr)
  for a,r in zip(got,rr):assert a['row']==rowmap[r['cohort']+'_'+r['source']+'_to_'+r['target']][r['id_PRIVATE']] and all(eq(a[k],v) for k,v in r.items() if k!='id_PRIVATE')
 for fn in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS','POSTSTRATIFIED_METHOD_ROWS','POSTSTRATIFIED_AGGREGATE_ROWS']:
  a=cr(O/'results'/(fn+'.csv'));b=rd(H/'scores_attempt001'/(fn+'.json'));assert len(a)==len(b)
  for x,y in zip(a,b):assert all(eq(x[k],v) for k,v in y.items())
 allsc=rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json')
 for name,sc in mapping.items():
  co,src,_,tgt=name.split('_');want=[{k:v for k,v in r.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}"} for r in allsc if r['cohort']==co and r['source']==src and r['target']==tgt];got=parts(O/'scenes'/name);assert len(got)==len(want)
  for a,b in zip(got,want):assert all(eq(a[k],v) for k,v in b.items())
 got=parts(O/'diagnostics/ZERO_NUMERICS');want=rd(H/'zero_numerics_attempt001/ALL_ROWS_PRIVATE.json');assert len(got)==len(want)==2150
 for a,r in zip(got,want):assert a['row']==rowmap[r['cohort']+'_'+r['source']+'_to_'+r['target']][r['id_PRIVATE']] and all(eq(a[k],v) for k,v in r.items() if not k.endswith('_PRIVATE'))
 tokens={(x['kind'],x['value']) for x in rd(R/'research/round96/publication/PRIVATE_PRIVACY_TOKEN_REGISTRY.json')['tokens']}
 for f in (R/'research/round97/input').glob('*.json'):
  for r in rd(f):
   for kind,k in [('id','id'),('scene','scene'),('caption','caption')]:tokens.add((kind,r[k]))
 b={}
 for kind,t in tokens:b.setdefault(t,set()).add(kind)
 auto=ahocorasick.Automaton()
 for t,k in b.items():auto.add_word(t,(t,k))
 auto.make_automaton();denied={'id','id_PRIVATE','row_PRIVATE','scene_PRIVATE','caption','metapath','imagepath','pointpath','source_boxes','query_PRIVATE','queries','coordinates','features','weights','api_key','access_token','private_key'}
 def check(x):
  if isinstance(x,dict):
   assert not set(x)&denied
   for v in x.values():check(v)
  elif isinstance(x,list):
   for v in x:check(v)
  elif isinstance(x,float):assert math.isfinite(x)
 findings=[]
 for f in ff:
  assert not f.is_symlink() and f.suffix in ['.py','.md','.json','.csv'];s=f.read_text()
  if f.suffix=='.json':check(json.loads(s))
  if f.suffix=='.csv':assert not set(next(csv.reader(s.splitlines())))&denied
  for end,(t,kk) in auto.iter(s):
   start=end-len(t)+1
   for k in kk:
    if k in {'id','scene','alias'} and ((start>0 and re.match('[A-Za-z0-9_]',s[start-1])) or (end+1<len(s) and re.match('[A-Za-z0-9_]',s[end+1]))):continue
    findings.append((str(f.relative_to(O)),k,hashlib.sha256(t.encode()).hexdigest()))
 assert not findings,findings;assert n==2304 and values==82944 and aliases==154 and len(physical)==10750
 out={'state':'PASS_FULL_TASK_PHYSICAL_GN_STATE_CONTEXT_SCENE_NEGATIVE_AND_PRIVACY','files':len(ff),'bytes':sum(f.stat().st_size for f in ff),'n':n,'task_values':values,'GN_physical_states':len(physical),'aliases':aliases,'source_files_checked':len(rd(P/'EXPORT_INDEX_PRIVATE.json')['sources']),'privacy_findings':0,'deny_tokens':len(tokens)};d=H/'independent_public_audit_attempt001';d.mkdir();(d/'AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
