"""Anonymous complete task/update/trial export; excludes raw data and learned arrays."""
from pathlib import Path
import json,csv,io,hashlib,argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args();tag=f'{a.attempt:03d}';R=Path(__file__).resolve().parents[2];H=R/'research/round113';B=R/'crossplatform/round113';P=H/'publication';O=P/('public_export_attempt'+tag);S=H/'scores_attempt001';PREFIX='studies/3eed_native_geometry_anchored_RGB_bridge_TTA_20261117';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(exist_ok=True,parents=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;O.mkdir(exist_ok=False,parents=True);scmap={};idmap={}
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scmap[name]=sc;tasks=[];fits=[];trials=[];grads=[]
  for i,r in enumerate(rr):
   alias=f'ROW{i+1:06d}';idmap[(name,r['id_PRIVATE'])]=alias;row={'row':alias,'source':r['source'],'target':r['target'],'scene_cluster':f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}",'protected_alias_fallback':r['protected_alias_fallback'],'C_available':r['C_available'],'seconds':r['seconds']}
   for k,x in r['arms'].items():
    for metric,val in x.items():
     if metric!='query_PRIVATE':row[k+'__'+metric]=val
   tasks.append(row);rec=rd(B/name/'records_PRIVATE'/(r['id_PRIVATE']+'.json'))
   for setting,x in r['fits'].items():
    fits.append({'row':alias,'source':r['source'],'target':r['target'],'fit':setting,**{k:v for k,v in x.items() if k!='trials'}})
    for j,t in enumerate(x['trials']):trials.append({'row':alias,'source':r['source'],'target':r['target'],'fit':setting,'trial':j+1,**t})
   for parent,x in rec.get('gradients',{}).items():grads.append({'row':alias,'source':r['source'],'target':r['target'],'parent':parent,**{k:v for k,v in x.items() if k!='group_gradient_L2'},**{'grad_'+k:v for k,v in x['group_gradient_L2'].items()}})
  table(O/'tasks'/(name+'.csv'),tasks);table(O/'fit_endpoints'/(name+'.csv'),fits);table(O/'trials'/(name+'.csv'),trials);table(O/'gradients'/(name+'.csv'),grads)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(n+'.csv'),rd(S/(n+'.json')))
 scenes=[]
 for x in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];scenes.append({k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{scmap[name].index(x['scene_PRIVATE'])+1:03d}"})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for n in ['DECISION','CONCENTRATION_SUMMARY','SUBSET_RESULTS','UPDATE_SUMMARY','NATIVE_AUDIT','SCALAR_AUDIT','ENGINEERING','ENGINEERING_AMENDMENT_002','ENGINEERING_AMENDMENT_004','QUERY_RGB_ASSOCIATION_SUMMARY','ATLAS_REVIEW_SUMMARY','AUXILIARY_PATH_GRADIENT_SUMMARY','GATE_ATTRIBUTION_SUMMARY','SUPPLEMENT_DECISION','SUPPLEMENT_NATIVE_AUDIT','SUPPLEMENT_SCALAR_AUDIT']:
  if (H/(n+'.json')).exists():dump(O/(n+'.json'),rd(H/(n+'.json')))
 for n in ['REPORT.md','LITERATURE.md','BASELINE_SCOPE_AUDIT.md','ITERATION_RECORD.md']:
  if (H/n).exists():(O/n).write_text((H/n).read_text())
 dump(O/'DESIGN.json',rd(H/'experiment/DESIGN_LOCK.json'))
 activity=[]
 for x in rd(H/'UPDATE_ACTIVITY_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];activity.append({'row':idmap[(name,x['id_PRIVATE'])],**{k:v for k,v in x.items() if k!='id_PRIVATE'}})
 table(O/'UPDATE_ACTIVITY.csv',activity)
 sup=H/'supplement_scores_attempt001'
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(sup/(name+'_ALL_ROWS_PRIVATE.json'));tt=[]
  for r in rr:
   z={'row':idmap[(name,r['id_PRIVATE'])],'source':r['source'],'target':r['target']}
   for k,x in r['arms'].items():
    if not k.startswith('wrong_natural_'):continue
    for metric,val in x.items():
     if metric!='query_PRIVATE':z[k+'__'+metric]=val
   tt.append(z)
  table(O/'supplement/tasks'/(name+'.csv'),tt)
 for n in ['ALL_METHOD_ROWS','AGGREGATE_ROWS']:table(O/'supplement/results'/(n+'.csv'),rd(sup/(n+'.json')))
 dump(O/'SUPPLEMENT_DESIGN.json',rd(H/'experiment/SUPPLEMENT_LOCK.json'))
 files={R/p for p in rd(H/'experiment/CODE_LOCK.json')['files']};files.update(H.glob('*.py'));files.update((H/'attempts').glob('*.py'))
 for f in sorted(files):
  assert '/upstream/' not in str(f);p=O/'own_code'/f.relative_to(R);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 ff=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in ff]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(files)]});print('EXPORT113',len(ff)+1,flush=True)
if __name__=='__main__':main()
