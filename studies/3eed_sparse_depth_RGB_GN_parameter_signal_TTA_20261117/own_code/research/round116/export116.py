"""Full anonymous task and adaptation evidence; no media/weights/features/identity exports."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round116';P=H/'publication';O=P/'public_export_attempt002';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rr):
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;O.mkdir(parents=True,exist_ok=False);mapping={};fits=[];activity=[];phys=[]
 for f in sorted((H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json')):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f);sc=sorted({r['scene_PRIVATE'] for r in rr});mapping[name]=sc;tasks=[]
  for i,r in enumerate(rr):
   row={'row':f'ROW{i+1:06d}','source':r['source'],'target':r['target'],'scene_cluster':f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}",'protected_alias_fallback':r['protected_alias_fallback'],'C_available':r['C_available'],'seconds':r['seconds']}
   for k,x in r['arms'].items():
    for m,v in x.items():
     if m!='query_PRIVATE':row[k+'__'+m]=v
   tasks.append(row);base={k:row[k] for k in ['row','source','target']};fits.append(dict(base,settings=r['fits']))
   for k,v in r['physical_holdout'].items():phys.append(dict(base,state=k,depth_loss=v))
   if not r['protected_alias_fallback']:
    rec=rd(R/'crossplatform/round116/formal'/name/'records_PRIVATE'/(r['id_PRIVATE']+'.json'))
    for v in rec['evaluation']['writeback']:activity.append(dict(base,**v))
  table(O/'tasks'/(name+'.csv'),tasks)
 
 for name in sorted(mapping):
  source,_,target=name.split('_');dump(O/'FIT_RECORDS'/(name+'.json'),[x for x in fits if x['source']==source and x['target']==target])
 table(O/'UPDATE_ACTIVITY.csv',activity);table(O/'PHYSICAL_HOLDOUT.csv',phys)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(n+'.csv'),rd(H/'scores_attempt001'/(n+'.json')))
 scenes=[]
 for x in rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];scenes.append({k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{mapping[name].index(x['scene_PRIVATE'])+1:03d}"})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for n in ['DECISION','CONCENTRATION_SUMMARY','RECOVERY_SUMMARY','DEPTH_SIGNAL_INFORMATION_SUMMARY','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT','ENGINEERING']:dump(O/(n+'.json'),rd(H/(n+'.json')))
 for n in ['REPORT.md','LITERATURE.md']:(O/n).write_text((H/n).read_text())
 dump(O/'DESIGN.json',rd(H/'experiment/DESIGN_LOCK.json'));files={R/p for p in rd(H/'experiment/DESIGN_LOCK.json')['files']};files.update(H.glob('*.py'));sources=[]
 for f in sorted(files):
  assert '/upstream/' not in str(f);p=O/'own_code'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(f.relative_to(R)),'public':str(p.relative_to(O))})
 ff=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':'studies/3eed_sparse_depth_RGB_GN_parameter_signal_TTA_20261117','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in ff]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'sources':sources});print('EXPORT116',len(ff)+1,flush=True)
if __name__=='__main__':main()
