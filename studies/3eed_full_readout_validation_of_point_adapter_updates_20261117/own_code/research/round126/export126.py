"""Public anonymous results/code/config, all negatives; never private vectors/media/IDs."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round126';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rr for k in r)),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;O.mkdir(parents=True,exist_ok=False);rowmap={};scmap={};procs=[]
 for f in sorted((H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json')):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rows=rd(f);scmap[name]=sorted({r['scene_PRIVATE'] for r in rows});rowmap[name]={};tasks=[]
  for i,r in enumerate(rows):
   tag=f'ROW{i+1:05d}';rowmap[name][r['id_PRIVATE']]=tag;out={'row':tag,'cohort':r['cohort'],'source':r['source'],'target':r['target'],'scene_cluster':f"CLUSTER{scmap[name].index(r['scene_PRIVATE'])+1:03d}",'protected_alias_fallback':r['protected_alias_fallback'],'B_active':r['B_active'],'C_available':r['C_available'],'extra_CPU_validator_shared_seconds':r['seconds']}
   for k,v in r['arms'].items():
    for kk,vv in v.items():
     if kk!='query_PRIVATE':out[k+'__'+kk]=vv
   tasks.append(out)
  table(O/'tasks'/(name+'.csv'),tasks)
 for r in rd(H/'PROCESS_ROWS_PRIVATE.json'):
  name=r['cohort']+'_'+r['source']+'_to_'+r['target'];procs.append({k:v for k,v in r.items() if k!='id_PRIVATE'}|{'row':rowmap[name][r['id_PRIVATE']]})
 for co in ['DEV','EXP']:
  for src in ['waymo','drone','quad']:table(O/'correspondence'/co/(src+'.csv'),[r for r in procs if r['cohort']==co and r['source']==src])
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS','POSTSTRATIFIED_METHOD_ROWS','POSTSTRATIFIED_AGGREGATE_ROWS']:table(O/'results'/(name+'.csv'),rd(H/'scores_attempt001'/(name+'.json')))
 allsc=rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json')
 for name,scs in scmap.items():
  co,src,_,tgt=name.split('_');rows=[{k:v for k,v in r.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{scs.index(r['scene_PRIVATE'])+1:03d}"} for r in allsc if r['cohort']==co and r['source']==src and r['target']==tgt];table(O/'scenes'/(name+'.csv'),rows)
 cases=[]
 for r in rd(H/'GOOD_FAILURE_CASES_PRIVATE.json'):
  name=r['cohort']+'_'+r['source']+'_to_'+r['target'];cases.append({k:v for k,v in r.items() if k!='id_PRIVATE'}|{'row':rowmap[name][r['id_PRIVATE']]})
 table(O/'diagnostics/GOOD_FAILURE_CASES.csv',cases)
 diagnostic=[]
 for r in rd(H/'FIT_TRANSFER_ROWS_PRIVATE.json'):
  name=r['cohort']+'_'+r['source']+'_to_'+r['target'];diagnostic.append({k:v for k,v in r.items() if k not in ['id_PRIVATE','values']}|{'row':rowmap[name][r['id_PRIVATE']]}|{k+'__IoU':v for k,v in r['values'].items()})
 table(O/'diagnostics/FIT_VERSUS_TRANSFER.csv',diagnostic)
 for name in ['DECISION','CONCENTRATION','VALIDATOR_COMPONENT_SUMMARY','HELDOUT_C_TASK_ROWS','FIT_TRANSFER_DIAGNOSTIC','RGB_READOUT_SPLIT_DIAGNOSTIC','DIAGNOSTIC_SCALAR_AUDIT','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT']:dump(O/(name+'.json'),rd(H/(name+'.json')))
 for f in (H/'experiment').glob('*.json'):dump(O/f.name,rd(f))
 for name in ['REPORT.md']:(O/name).write_text((H/name).read_text())
 sources=[];files=set(H.glob('*.py'))|{R/p for p in rd(H/'experiment/DESIGN_LOCK.json')['files'] if p.endswith('.py')};files.update({R/'research/round116/core116.py',R/'research/round91/core91.py',R/'research/round91/targets91.py',R/'research/round79/core79.py',R/'research/round76/model76.py',R/'research/round6/source_validation_scoring/batch_yaw_iou.py',R/'tools/obb_iou.py'})
 for f in sorted(files):
  rel=f.relative_to(R);dst=O/'own_code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(rel),'public':str(dst.relative_to(O))})
 for f in sorted((H/'attempts').glob('*')):
  if f.suffix not in ['.py','.json','.log']:continue
  dst=O/'attempts'/f.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(f.relative_to(R)),'public':str(dst.relative_to(O))})
 ff=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':'studies/3eed_full_readout_validation_of_point_adapter_updates_20261117','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in ff]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'sources':sources});print('EXPORT126',len(ff)+1,flush=True)
if __name__=='__main__':main()
