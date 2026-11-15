"""Own code and anonymous complete evidence; private atlas/arrays never exported."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round99';B=R/'crossplatform/round99';P=H/'publication';O=P/'public_export_attempt001';S=H/'scores_attempt001';PREFIX='studies/3eed_fixed_evidence_multiview_matching_fit_20261115';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['state'].startswith('PASS_');assert not O.exists();O.mkdir(parents=True);scmaps={}
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rows=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rows});scmaps[name]=sc;out=[];trials=[];fits=[]
  for i,r in enumerate(rows):
   alias='ROW'+str(i+1).zfill(4);a={'row':alias,'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3)}
   for arm,u in r['arms'].items():
    for k,v in u.items():
     if k!='query_PRIVATE':a[arm+'__'+k]=v
   out.append(a);rec=rd(B/name/'records_PRIVATE'/(r['id_PRIVATE']+'.json'))
   for fit,u in rec.items():
    fits.append({'row':alias,'source':r['source'],'target':r['target'],'fit':fit,**{k:v for k,v in u.items() if k!='trials'}})
    for j,t in enumerate(u['trials']):trials.append({'row':alias,'source':r['source'],'target':r['target'],'fit':fit,'trial_index':j,**t})
  table(O/'tasks'/(name+'.csv'),out);table(O/'fits'/(name+'.csv'),fits)
  for i in range(0,len(trials),1000):table(O/'trials'/(name+'_part'+str(i//1000+1).zfill(3)+'.csv'),trials[i:i+1000])
 for key in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(key+'.csv'),rd(S/(key+'.json')))
 table(O/'results/COMPONENT_ROWS.csv',rd(H/'COMPONENT_ROWS.json'));table(O/'results/COMPONENT_MACRO.csv',rd(H/'COMPONENT_MACRO.json'))
 scene=[]
 for r in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):scene.append({k:v for k,v in r.items() if k!='scene_PRIVATE'}|{'scene_cluster':'CLUSTER'+str(scmaps[r['source']+'_to_'+r['target']].index(r['scene_PRIVATE'])+1).zfill(3)})
 table(O/'results/ALL_SCENE_ROWS.csv',scene)
 cases=[{k:r[k] for k in ['case','source','target','contrast','kind','method','control','method_IoU','control_IoU','change_IoU','oracle_IoU_diagnostic','selection']} for r in rd(H/'GOOD_FAILURE_CASES_PRIVATE.json')];table(O/'diagnostics/GOOD_FAILURE_SCALARS.csv',cases)
 for key in ['REPORT.md','PART_A_REPORT.md','ITERATION_RECORD.md','LITERATURE.md','DECISION.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json']:(O/key).write_text((H/key).read_text().replace(str(R),'WORKSPACE_ROOT'))
 config=rd(H/'experiment/DESIGN_LOCK.json');dump(O/'FIXED_CONFIGURATION.json',config);dump(O/'COHORT_POLICY.json',rd(H/'experiment/COHORT_LOCK.json'));dump(O/'COUNTS.json',{'directed_inputs':768,'unique_target_expressions':384,'arms':20,'task_readouts':15360,'old_R98_decomposition_inputs':18620,'private_atlas_pages':12,'actually_visually_reviewed_pages':2,'no_identity_gold_created':True,'FINAL_access':False,'raw_media_features_or_weights_published':False})
 files=set(H.glob('*.py'));files.update((H/'attempts').glob('*.py'));files.add(R/'tools/exploration_guard_20261115.py');files.update(R/f for f in rd(H/'experiment/CODE_LOCK.json')['code_sha256'])
 for f in sorted(files):
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 allfiles=sorted(p for p in O.rglob('*') if p.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in allfiles]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(files)]});print({'files':len(allfiles)+1,'bytes':sum(p.stat().st_size for p in O.rglob('*') if p.is_file())})
if __name__=='__main__':main()
