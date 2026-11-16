"""Anonymous complete results and own code only; no sensor identity or arrays."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round105';B=R/'crossplatform/round105';P=H/'publication';O=P/'public_export_attempt001';S=H/'scores_attempt001';PREFIX='studies/3eed_actual_multiview_bidir_native_matching_TTA_20261116';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;O.mkdir(exist_ok=False,parents=True);scmaps={};fitcount=0
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scmaps[name]=sc;tasks=[];fits=[]
  for i,r in enumerate(rr):
   alias='ROW'+str(i+1).zfill(6);s=r['status'];a={'row':alias,'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3),'fallback':s['fallback'],'fit_attempted':s.get('adaptation_attempted',False),'TTA_enabled':s.get('TTA_output_enabled',False),'B_visible_queries':s.get('B_visible_queries',0),'C_visible_queries':s.get('C_visible_queries',0),'seconds':s['seconds']}
   for arm,x in r['arms'].items():
    for k,v in x.items():
     if k!='query_PRIVATE':a[arm+'__'+k]=v
   tasks.append(a)
   for label,x in r['fits'].items():fits.append({'row':alias,'source':r['source'],'target':r['target'],'fit':label,**x})
  table(O/'tasks'/(name+'.csv'),tasks);table(O/'fit_endpoints'/(name+'.csv'),fits);fitcount+=len(fits)
 for key in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(key+'.csv'),rd(S/(key+'.json')))
 table(O/'results/ALL_SCENE_ROWS.csv',[{k:v for k,v in r.items() if k!='scene_PRIVATE'}|{'scene_cluster':'CLUSTER'+str(scmaps[r['source']+'_to_'+r['target']].index(r['scene_PRIVATE'])+1).zfill(3)} for r in rd(S/'ALL_SCENE_ROWS_PRIVATE.json')])
 for key in ['REPORT.md','DECISION.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json','ENGINEERING.json','COVERAGE_SUMMARY.json','FIT_SUMMARY.json','CONCENTRATION_SUMMARY.json','SUBSET_RESULTS.json','REGISTRATION_SUMMARY.json','FULL_REGISTRATION_SUMMARY.json','OBSERVATION_DIAGNOSTICS.json','ATLAS_REVIEW_SUMMARY.json','LITERATURE.md']:(O/key).write_text((H/key).read_text().replace(str(R),'WORKSPACE_ROOT'))
 dump(O/'FIXED_CONFIGURATION.json',rd(H/'experiment/DESIGN_LOCK.json'));dump(O/'COUNTS.json',{'directed_inputs':768,'unique_target_expressions':384,'arms':len(rd(B/'MATRIX_SEALED.json')['arms']),'fit_endpoint_rows':fitcount,'all_negatives_unknowns_and_fallbacks_retained':True,'private_media_descriptions_sensor_ids_arrays_weights_not_published':True,'FINAL_access':False})
 files={R/f for f in rd(H/'experiment/CODE_LOCK.json')['files']};files.update(H.glob('*.py'));files.update((H/'attempts').glob('*.py'));files.update([R/'crossplatform/round12/sensor_pnp.py',R/'crossplatform/round12/sensor_icp.py',R/'crossplatform/round12/temporal_association.py'])
 for f in sorted(files):
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 files2=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files2]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(files)]});print('EXPORT105',len(files2)+1)
if __name__=='__main__':main()
