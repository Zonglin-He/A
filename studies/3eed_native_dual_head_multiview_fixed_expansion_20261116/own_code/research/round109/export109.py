"""Full anonymous expansion matrix, scene clusters, own code and negative evidence."""
from pathlib import Path
import json,csv,io,hashlib,argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);arg=ap.parse_args();tag=str(arg.attempt).zfill(3)
R=Path(__file__).resolve().parents[2];H=R/'research/round109';B=R/'crossplatform/round109';P=H/'publication';O=P/('public_export_attempt'+tag);S=H/'scores_attempt001';PREFIX='studies/3eed_native_dual_head_multiview_fixed_expansion_20261116';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(exist_ok=True,parents=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==1536;O.mkdir(exist_ok=False,parents=True);scmap={}
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scmap[name]=sc;tasks=[]
  for i,r in enumerate(rr):
   st=r['status'];a={'row':'ROW'+str(i+1).zfill(6),'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3),'protected_alias_fallback':st['protected_alias_fallback'],'seconds':st['seconds']}
   for k in ['real','randomA','wrong']:
    for v in ['B_available','C_available']:a[k+'_'+v]=st.get('families',{}).get(k,{}).get(v,False)
   for k,x in r['arms'].items():
    for metric,val in x.items():
     if metric!='query_PRIVATE':a[k+'__'+metric]=val
   assert not r['fits'];tasks.append(a)
  table(O/'tasks'/(name+'.csv'),tasks)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(n+'.csv'),rd(S/(n+'.json')))
 scenes=[]
 for x in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];sc=x['scene_PRIVATE'];scenes.append({('scene_cluster' if k=='scene_PRIVATE' else k):('CLUSTER'+str(scmap[name].index(sc)+1).zfill(3) if k=='scene_PRIVATE' else v) for k,v in x.items()})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for f in ['REPORT.md','LITERATURE.md','BRIDGE.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json','INPUT_SUMMARY.json','PAIR_SUMMARY.json','REGISTRATION_SUMMARY.json','COVERAGE_MODEL_SUMMARY.json','RANGE_COMPARISON.json','CE_COMPONENT_DIAGNOSTIC.json','SUBSET_RESULTS.json','CONCENTRATION_SUMMARY.json','DECISION.json']:(O/f).write_bytes((H/f).read_bytes())
 dump(O/'FIXED_CONFIGURATION.json',rd(H/'experiment/INPUT_LOCK.json'));dump(O/'CE_COMPONENT_ENGINEERING_AMENDMENT.json',rd(H/'experiment/CE_COMPONENT_ENGINEERING_AMENDMENT.json'));dump(O/'COUNTS.json',{'directed_inputs':1536,'unique_target_expressions':768,'arms':33,'readouts':50688,'TTA_run':False,'all_negatives_unknowns_fallbacks_retained':True,'raw_media_text_IDs_coordinates_features_weights_thirdparty_snapshots_excluded':True,'FINAL_access':False})
 files={R/p for p in rd(H/'experiment/CODE_LOCK.json')['files']};files.update(H.glob('*.py'));files.update((H/'attempts').glob('*.py'))
 for f in sorted(files):
  assert '/upstream/' not in str(f);p=O/'own_code'/f.relative_to(R);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 fs=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in fs]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(files)]});print('EXPORT109',len(fs)+1)
if __name__=='__main__':main()
