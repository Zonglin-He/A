"""Complete public own implementation, anonymous scalar matrix and negative controls."""
from pathlib import Path
import json,csv,io,hashlib,argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);arg=ap.parse_args();tag=str(arg.attempt).zfill(3)
R=Path(__file__).resolve().parents[2];H=R/'research/round110';B=R/'crossplatform/round110';P=H/'publication';O=P/('public_export_attempt'+tag);S=H/'scores_attempt001';PREFIX='studies/3eed_native_head_paired_statistics_adaptation_20261116';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(exist_ok=True,parents=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;O.mkdir(exist_ok=False,parents=True);scmap={}
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scmap[name]=sc;tasks=[];stats=[]
  for i,r in enumerate(rr):
   st=r['status'];a={'row':'ROW'+str(i+1).zfill(6),'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3),'protected_alias_fallback':st['protected_alias_fallback'],'actual_C_available':st['actual_C_available'],'seconds':st['seconds']}
   for k,x in r['arms'].items():
    for metric,val in x.items():
     if metric!='query_PRIVATE':a[k+'__'+metric]=val
   assert not r['fits'];tasks.append(a)
   for label,x in st['calibrations'].items():stats.append({'row':a['row'],'source':r['source'],'target':r['target'],'calibration':label,**{k:v for k,v in x.items() if k!='eligible_channels_per_layer'},'eligible_BN1':x['eligible_channels_per_layer'][0],'eligible_BN2':x['eligible_channels_per_layer'][1]})
  table(O/'tasks'/(name+'.csv'),tasks);table(O/'state_change'/(name+'.csv'),stats)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(n+'.csv'),rd(S/(n+'.json')))
 scenes=[]
 for x in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];sc=x['scene_PRIVATE'];scenes.append({('scene_cluster' if k=='scene_PRIVATE' else k):('CLUSTER'+str(scmap[name].index(sc)+1).zfill(3) if k=='scene_PRIVATE' else v) for k,v in x.items()})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for f in ['REPORT.md','LITERATURE.md','NATIVE_AUDIT.json','SCALAR_AUDIT.json','ENGINEERING.json','ATLAS_REVIEW_SUMMARY.json','SUBSET_RESULTS.json','CONCENTRATION_SUMMARY.json','CALIBRATION_SUMMARY.json','HEAD_COMPATIBILITY_DIAGNOSTIC.json','DECISION.json']:(O/f).write_bytes((H/f).read_bytes())
 dump(O/'FIXED_CONFIGURATION.json',rd(H/'experiment/DESIGN_LOCK.json'));dump(O/'COUNTS.json',{'directed_inputs':768,'unique_target_expressions':384,'arms':58,'readouts':44544,'calibration_states':1448,'running_buffer_values_per_state':1152,'learned_parameters_changed':0,'all_negatives_unknowns_fallbacks_retained':True,'excluded_raw_media_text_IDs_coordinates_features_weights_thirdparty_snapshots':True,'FINAL_access':False,'failed_attempts_retained':['independentCPUaudit001 expected flat seal but original extraction uses nested directional seals; corrected audit only before GT; formal prediction unchanged']})
 files={R/p for p in rd(H/'experiment/CODE_LOCK.json')['files']};files.update(H.glob('*.py'))
 for f in sorted(files):
  assert '/upstream/' not in str(f);p=O/'own_code'/f.relative_to(R);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 fs=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in fs]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(files)]});print('EXPORT110',len(fs)+1)
if __name__=='__main__':main()
