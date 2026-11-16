"""Full anonymous public evidence, no media/identity/coordinate/feature/weight output."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round118';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rr):
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;O.mkdir(parents=True,exist_ok=False);mapping={}
 for f in sorted((H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json')):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f);sc=sorted({r['scene_PRIVATE'] for r in rr});mapping[name]=sc;tasks=[]
  for i,r in enumerate(rr):
   row={'row':f'ROW{i+1:06d}','cohort':r['cohort'],'source':r['source'],'target':r['target'],'scene_cluster':f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}",'protected_alias_fallback':r['protected_alias_fallback'],'C_available':r['C_available'],'cached_CPU_seconds':r['seconds']}
   for k,x in r['arms'].items():
    for m,v in x.items():
     if m!='query_PRIVATE':row[k+'__'+m]=v
   tasks.append(row)
  table(O/'tasks'/(name+'.csv'),tasks)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(n+'.csv'),rd(H/'scores_attempt001'/(n+'.json')))
 scenes=[]
 for x in rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json'):
  name=x['cohort']+'_'+x['source']+'_to_'+x['target'];scenes.append({k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{mapping[name].index(x['scene_PRIVATE'])+1:03d}"})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for n in ['DECISION','CONCENTRATION_SUMMARY','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT','ENGINEERING']:dump(O/(n+'.json'),rd(H/(n+'.json')))
 (O/'REPORT.md').write_text((H/'REPORT.md').read_text());dump(O/'DESIGN.json',rd(H/'experiment/DESIGN_LOCK.json'));files=set(H.glob('*.py'))|{R/'tools/exploration_guard_round118.py',R/'research/round6/source_validation_scoring/batch_yaw_iou.py',R/'tools/obb_iou.py'};sources=[]
 for f in sorted(files):
  dst=O/'own_code'/f.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(f.relative_to(R)),'public':str(dst.relative_to(O))})
 ff=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':'studies/3eed_source_compatible_context_provider_factorial_20261117','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in ff]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'sources':sources});print('EXPORT118',len(ff)+1,flush=True)
if __name__=='__main__':main()
