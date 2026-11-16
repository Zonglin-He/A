"""Full fixed-expansion evidence: all anonymous tasks and fit endpoints, no raw arrays."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round103';B=R/'crossplatform/round103';P=H/'publication';O=P/'public_export_attempt001';S=H/'scores_attempt001';PREFIX='studies/3eed_consensus_binding_matching_parameter_TTA_20261116';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rows):
 assert rows;p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;assert not O.exists();O.mkdir(parents=True);scmaps={}
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scmaps[name]=sc;tasks=[];fits=[]
  for i,r in enumerate(rr):
   alias='ROW'+str(i+1).zfill(6);a={'row':alias,'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3),'unknown_roles':r['status']['unknown_roles'],'visible_seeds':r['status']['visible_seeds']}
   for arm,x in r['arms'].items():
    for k,v in x.items():
     if k!='query_PRIVATE':a[arm+'__'+k]=v
   tasks.append(a)
   for fit,x in r['fits'].items():fits.append({'row':alias,'source':r['source'],'target':r['target'],'fit':fit,**x})
  for label,z in [('tasks',tasks),('fit_endpoints',fits)]:
   for i in range(0,len(z),1000):table(O/label/(name+'_part'+str(i//1000+1).zfill(3)+'.csv'),z[i:i+1000])
 for key in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(key+'.csv'),rd(S/(key+'.json')))
 scenes=[]
 for r in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):scenes.append({k:v for k,v in r.items() if k!='scene_PRIVATE'}|{'scene_cluster':'CLUSTER'+str(scmaps[r['source']+'_to_'+r['target']].index(r['scene_PRIVATE'])+1).zfill(3)})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for key in ['REPORT.md','DECISION.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json','ENGINEERING.json','FIT_SUMMARY.json']:(O/key).write_text((H/key).read_text().replace(str(R),'WORKSPACE_ROOT'))
 dump(O/'FIXED_CONFIGURATION.json',rd(H/'experiment/DESIGN_LOCK.json'));dump(O/'DECISION_RULE_LOCK.json',rd(H/'experiment/DECISION_RULE_LOCK.json'));dump(O/'COUNTS.json',{'directed_inputs':768,'unique_target_expressions':384,'arms':34,'task_readouts':26112,'fit_endpoint_rows':2304,'all_trial_scalars_retained_locally':True,'raw_media_weights_features_or_captions_published':False,'FINAL_access':False})
 files=set(H.glob('*.py'));files.update((H/'attempts').glob('*.py'));files.add(R/'tools/exploration_guard_round103.py');files.update(R/f for f in rd(H/'experiment/CODE_LOCK.json')['code_sha256'])
 for f in sorted(files):
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 allfiles=sorted(p for p in O.rglob('*') if p.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in allfiles]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(p.relative_to(R)) for p in sorted(files)]});print({'files':len(allfiles)+1,'bytes':sum(p.stat().st_size for p in O.rglob('*') if p.is_file())})
if __name__=='__main__':main()
