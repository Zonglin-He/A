"""Complete anonymized task/fit results and own implementation; no sensor arrays."""
from pathlib import Path
import json,csv,io,hashlib
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);arg=ap.parse_args();tag=str(arg.attempt).zfill(3)
R=Path(__file__).resolve().parents[2];H=R/'research/round106';B=R/'crossplatform/round106';P=H/'publication';O=P/('public_export_attempt'+tag);S=H/'scores_attempt001';PREFIX='studies/3eed_canonical_point_multiview_LiDAR_feature_TTA_20261116';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,z):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(exist_ok=True,parents=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;O.mkdir(exist_ok=False,parents=True);scmap={};fitn=0
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scmap[name]=sc;tasks=[];fits=[]
  for i,r in enumerate(rr):
   alias='ROW'+str(i+1).zfill(6);st=r['status'];a={'row':alias,'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3),'protected_alias_fallback':st['protected_alias_fallback'],'seconds':st['seconds']}
   for k in ['real','randomA','wrong']:
    for v in ['B_available','C_available','updated_output_enabled']:a[k+'__'+v]=st['families'].get(k,{}).get(v,False)
   for k,x in r['arms'].items():
    for v,y in x.items():
     if v!='query_PRIVATE':a[k+'__'+v]=y
   tasks.append(a)
   for k,x in r['fits'].items():fits.append({'row':alias,'source':r['source'],'target':r['target'],'fit':k,**x})
  table(O/'tasks'/(name+'.csv'),tasks);table(O/'fit_endpoints'/(name+'.csv'),fits);fitn+=len(fits)
  su={(x['source'],x['target'],x['id_PRIVATE']):x['arms'] for x in rd(H/'DIRECT_TEACHER_ROWS_PRIVATE.json')};table(O/'direct_teacher_tasks'/(name+'.csv'),[{'row':'ROW'+str(i+1).zfill(6),'source':r['source'],'target':r['target'],**su[(r['source'],r['target'],r['id_PRIVATE'])]} for i,r in enumerate(rr)])
 for k in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(k+'.csv'),rd(S/(k+'.json')))
 table(O/'results/ALL_SCENE_ROWS.csv',[{k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':'CLUSTER'+str(scmap[x['source']+'_to_'+x['target']].index(x['scene_PRIVATE'])+1).zfill(3)} for x in rd(S/'ALL_SCENE_ROWS_PRIVATE.json')])
 for k in ['REPORT.md','DECISION.json','SCALAR_AUDIT.json','NATIVE_AUDIT.json','ENGINEERING.json','COVERAGE_SUMMARY.json','COVERAGE_MODEL_SUMMARY.json','SUBSET_RESULTS.json','CONCENTRATION_SUMMARY.json','FIT_SUMMARY.json','TEACHER_DIAGNOSTIC_SUMMARY.json','ATLAS_REVIEW_SUMMARY.json','BENEFIT_PROVENANCE.json','DIRECT_TEACHER_SUMMARY.json','DIRECT_TEACHER_AUDIT.json','LITERATURE.md']:(O/k).write_text((H/k).read_text().replace(str(R),'WORKSPACE_ROOT'))
 dump(O/'FIXED_CONFIGURATION.json',rd(H/'experiment/DESIGN_LOCK.json'));dump(O/'ENGINEERING_AMENDMENT.json',rd(H/'experiment/ENGINEERING_AMENDMENT.json'));dump(O/'COUNTS.json',{'directed_inputs':768,'unique_target_expressions':384,'main_arms':34,'main_readouts':26112,'supplementary_direct_teacher_arms':3,'supplementary_readouts':2304,'total_readouts':28416,'fit_endpoint_rows':fitn,'all_negatives_unknowns_and_fallbacks_retained':True,'raw_media_text_ids_coordinates_weights_features_excluded':True,'FINAL_access':False})
 files={R/p for p in rd(H/'experiment/CODE_LOCK.json')['files']};files.update(H.glob('*.py'));files.update((H/'attempts').glob('*.py'))
 for f in sorted(files):
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 fs=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in fs]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(files)]});print('EXPORT106',len(fs)+1)
if __name__=='__main__':main()
