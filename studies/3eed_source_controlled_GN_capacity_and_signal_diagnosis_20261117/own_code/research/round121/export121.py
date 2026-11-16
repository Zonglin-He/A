"""Anonymous source diagnostic evidence only; no caches/GT identities/weights/media."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round121';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==384;O.mkdir(parents=True,exist_ok=False);rr=rd(H/'scores_attempt001/ALL_ROWS_PRIVATE.json');rowmap={};scmap={};steps=[]
 for src in ['waymo','drone','quad']:
  rows=[r for r in rr if r['source']==src];scmap[src]=sorted({r['scene_PRIVATE'] for r in rows});tasks=[]
  for i,r in enumerate(rows):
   tag=f'ROW{i+1:05d}';rowmap[(src,r['id_PRIVATE'])]=tag;out={'row':tag,'source':src,'scene_cluster':f"CLUSTER{scmap[src].index(r['scene_PRIVATE'])+1:03d}",'frame_group':r['frame_group'],'GT_eligible':r['GT_eligible'],'oracle_IoU':r['oracle_IoU'],'all_arm_shared_cache_seconds':r['seconds']}
   for k,v in r['arms'].items():
    for kk,vv in v.items():
     if kk!='query_PRIVATE':out[k+'__'+kk]=vv
   tasks.append(out);rec=rd(R/f"crossplatform/round121/formal/{src}/records_PRIVATE/{r['id_PRIVATE']}.json")
   for t,v in enumerate(rec['oracle10_steps']):
    for j,trial in enumerate(v['trials']):steps.append({'row':tag,'source':src,'step':t+1,'trial':j+1,'before':v['before'],'gradient_L2':v['gradient_L2'],'step_accepted':v['accepted'],'total_patch_L2':v['total_patch_L2'],**trial})
  table(O/'tasks'/f'{src}.csv',tasks)
 for name in ['PHYSICAL_ROWS_PRIVATE','ANGLE_ROWS_PRIVATE','GOOD_FAILURE_CASES_PRIVATE']:
  rows=[{k:v for k,v in r.items() if k!='id_PRIVATE'}|{'row':rowmap[(r['source'],r['id_PRIVATE'])]} for r in rd(H/(name+'.json'))]
  for src in ['waymo','drone','quad']:table(O/'diagnostics'/name.replace('_PRIVATE','')/(src+'.csv'),[r for r in rows if r['source']==src])
 for src in ['waymo','drone','quad']:table(O/'oracle_diagnostic_steps'/f'{src}.csv',[r for r in steps if r['source']==src])
 for name in ['AGGREGATE_ROWS','ALL_PAIRED_ROWS']:table(O/'results'/(name+'.csv'),rd(H/'scores_attempt001'/(name+'.json')))
 table(O/'results/SCENE_ROWS.csv',[{k:v for k,v in r.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{scmap[r['source']].index(r['scene_PRIVATE'])+1:03d}"} for r in rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json')])
 for name in ['PREFLIGHT','ENGINEERING','NATIVE_AUDIT','SCALAR_AUDIT','QUALIFICATION','CONCENTRATION','DECISION','SIGNAL_SUMMARY','ASSERTION_AMENDMENT','NATIVE_QUALIFICATION_DIAGNOSIS','DECISION_EQUIVALENCE']:dump(O/(name+'.json'),rd(H/(name+'.json')))
 for f in (H/'experiment').glob('*.json'):dump(O/f.name,rd(f))
 (O/'REPORT.md').write_text((H/'REPORT.md').read_text());sources=[];files=set(H.glob('*.py'))|{R/p for p in rd(H/'experiment/DESIGN_LOCK.json')['files'] if p.endswith('.py')};files.update({R/'research/round116/audit116.py',R/'research/round120/audit120.py'})
 for f in sorted(files):
  rel=f.relative_to(R);dst=O/'own_code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(rel),'public':str(dst.relative_to(O))})
 for f in sorted((H/'attempts').glob('*')):
  if f.suffix not in ['.py','.log']:continue
  dst=O/'failed_attempts'/f.name;dst.parent.mkdir(exist_ok=True,parents=True);dst.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(f.relative_to(R)),'public':str(dst.relative_to(O))})
 ff=sorted(p for p in O.rglob('*') if p.is_file());dump(O/'MANIFEST.json',{'prefix':'studies/3eed_source_controlled_GN_capacity_and_signal_diagnosis_20261117','files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in ff]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'sources':sources});print('EXPORT121',len(ff)+1)
if __name__=='__main__':main()
