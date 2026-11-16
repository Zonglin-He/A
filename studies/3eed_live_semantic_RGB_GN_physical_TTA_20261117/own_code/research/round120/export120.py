"""Complete anonymous task/physical/context/writeback evidence, no learned arrays."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round120';P=H/'publication';O=P/'public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(p,rr):
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;O.mkdir(parents=True,exist_ok=False);mapping={};rowmap={};phys=[]
 for f in sorted((H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json')):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f);sc=sorted({r['scene_PRIVATE'] for r in rr});mapping[name]=sc;rowmap[name]={};tasks=[]
  for i,r in enumerate(rr):
   tag=f'ROW{i+1:06d}';rowmap[name][r['id_PRIVATE']]=tag;row={'row':tag,'cohort':r['cohort'],'source':r['source'],'target':r['target'],'scene_cluster':f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}",'protected_alias_fallback':r['protected_alias_fallback'],'C_available':r['C_available'],'all_arm_shared_cache_seconds':r['seconds']}
   for k,a in r['arms'].items():
    for n,v in a.items():
     if n!='query_PRIVATE':row[k+'__'+n]=v
   tasks.append(row)
   if not r['protected_alias_fallback']:
    rec=rd(R/'crossplatform/round120/formal'/r['cohort']/(r['source']+'_to_'+r['target'])/'records_PRIVATE'/(r['id_PRIVATE']+'.json'));norms={x['setting']:x for x in rec['evaluation']['writeback']}
    for st,v in rec['evaluation']['physical_losses'].items():phys.append({'row':tag,'cohort':r['cohort'],'source':r['source'],'target':r['target'],'setting':st,'scope_parameters':224,'train_cells':rec['fit']['train_cells'],'holdout_cells':rec['fit']['holdout_cells'],'source_GN_L2':rec['fit']['source_GN_L2'],'trust_radius':rec['fit']['trust_radius'],'fit_depth':v['fit_depth'],'heldout_depth':v['heldout_depth'],'changed_tensors':norms[st]['changed_tensors'],'actual_writeback_L2':norms[st]['actual_writeback_L2'],'proposal_L2':norms[st]['proposal_L2']})
  table(O/'tasks'/(name+'.csv'),tasks)
 for co in ['DEV','EXP']:
  for source in ['waymo','drone','quad']:
   rows=[r for r in phys if r['cohort']==co and r['source']==source]
   for j in range(0,len(rows),1024):table(O/'physical'/co/source/(f'part{j//1024+1:02d}.csv'),rows[j:j+1024])
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS','POSTSTRATIFIED_METHOD_ROWS','POSTSTRATIFIED_AGGREGATE_ROWS']:table(O/'results'/(n+'.csv'),rd(H/'scores_attempt001'/(n+'.json')))
 allsc=rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json')
 for name,sc in mapping.items():
  co,source,_,target=name.split('_');rows=[{k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':f"CLUSTER{sc.index(x['scene_PRIVATE'])+1:03d}"} for x in allsc if x['cohort']==co and x['source']==source and x['target']==target]
  for j in range(0,len(rows),1024):table(O/'scenes'/name/(f'part{j//1024+1:02d}.csv'),rows[j:j+1024])
 for fn in ['CONTEXT_EFFECT_ROWS_PRIVATE','GOOD_FAILURE_CASES_PRIVATE']:
  result=[]
  for r in rd(H/(fn+'.json')):
   name=r['cohort']+'_'+r['source']+'_to_'+r['target'];result.append({k:v for k,v in r.items() if k!='id_PRIVATE'}|{'row':rowmap[name][r['id_PRIVATE']]})
  for j in range(0,len(result),1024):table(O/'diagnostics'/fn.replace('_PRIVATE','')/(f'part{j//1024+1:02d}.csv'),result[j:j+1024])
 numeric=[]
 for r in rd(H/'zero_numerics_attempt001/ALL_ROWS_PRIVATE.json'):
  name=r['cohort']+'_'+r['source']+'_to_'+r['target'];numeric.append({k:v for k,v in r.items() if not k.endswith('_PRIVATE')}|{'row':rowmap[name][r['id_PRIVATE']]})
 for j in range(0,len(numeric),1024):table(O/'diagnostics/ZERO_NUMERICS'/(f'part{j//1024+1:02d}.csv'),numeric[j:j+1024])
 for n in ['DECISION','CONCENTRATION_SUMMARY','PHYSICAL_SUMMARY','CONTEXT_EFFECT_SUMMARY','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT','ENGINEERING','ENGINEERING_PATH_SUMMARY','ZERO_NUMERICS_SUMMARY','DECISION_EQUIVALENCE_SUMMARY','DIAGNOSTIC_AUDIT']:dump(O/(n+'.json'),rd(H/(n+'.json')))
 for n in ['REPORT.md','LITERATURE.md']:(O/n).write_text((H/n).read_text())
 for n in ['DESIGN_LOCK','POSTPROCESS_CODE_LOCK','ZERO_NUMERICS_DIAGNOSTIC_LOCK']:dump(O/(n+'.json'),rd(H/'experiment'/(n+'.json')))
 files={R/p for p in rd(H/'experiment/DESIGN_LOCK.json')['files']};files.update(H.glob('*.py'));files.update({R/'research/round116/audit116.py',R/'research/round118/core118.py',R/'tools/obb_iou.py',R/'research/round6/source_validation_scoring/batch_yaw_iou.py',R/'tools/exploration_guard_round120.py'});sources=[]
 for f in sorted(files):
  assert '/upstream/' not in str(f);rel=f.relative_to(R);dst=O/'own_code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(rel),'public':str(dst.relative_to(O))})
 for f in sorted((H/'attempts').glob('*.py')):
  dst=O/'failed_attempts'/f.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));sources.append({'source':str(f.relative_to(R)),'public':str(dst.relative_to(O))})
 for f in H.glob('*AMENDMENT*.json'):dump(O/f.name,rd(f))
 ff=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':'studies/3eed_live_semantic_RGB_GN_physical_TTA_20261117','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in ff]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'sources':sources});print('EXPORT120',len(ff)+1,flush=True)
if __name__=='__main__':main()
