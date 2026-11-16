"""Complete anonymous 75-arm matrix, component tests and retained engineering failures."""
from pathlib import Path
import json,csv,io,hashlib,argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args();tag=f'{a.attempt:03d}';R=Path(__file__).resolve().parents[2];H=R/'research/round111';B=R/'crossplatform/round111';P=H/'publication';O=P/('public_export_attempt'+tag);S=H/'scores_attempt001';PREFIX='studies/3eed_same_evidence_native_path_component_ablation_20261116';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(exist_ok=True,parents=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;O.mkdir(exist_ok=False,parents=True);scmap={}
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  rr=rd(S/(name+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scmap[name]=sc;tasks=[]
  for i,r in enumerate(rr):
   st=r['status'];row={'row':f'ROW{i+1:06d}','source':r['source'],'target':r['target'],'scene_cluster':f"CLUSTER{sc.index(r['scene_PRIVATE'])+1:03d}",'protected_alias_fallback':st['protected_alias_fallback'],'seconds':st['seconds'],'readout_amended_cached_state_reuse':st.get('readout_amended_cached_state_reuse',False)}
   for k in ['real','randomA','wrong']:
    for f in ['B_available','C_available']:row[k+'_'+f]=st['families'].get(k,{}).get(f,False)
   for k,x in r['arms'].items():
    for metric,val in x.items():
     if metric!='query_PRIVATE':row[k+'__'+metric]=val
   assert not r['fits'];tasks.append(row)
  table(O/'tasks'/(name+'.csv'),tasks)
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(name+'.csv'),rd(S/(name+'.json')))
 scenes=[]
 for x in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];scenes.append({('scene_cluster' if k=='scene_PRIVATE' else k):(f"CLUSTER{scmap[name].index(v)+1:03d}" if k=='scene_PRIVATE' else v) for k,v in x.items()})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for f in ['REPORT.md','LITERATURE.md','ENGINEERING.json','READOUT_ENGINEERING_AMENDMENT.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json','HEAD_STATE_IDENTITY.json','SOURCE_MODULE_ACTIVITY.json','ATLAS_REVIEW_SUMMARY.json','HEAD_FACTORIAL_DIAGNOSTIC.json','DECISION.json','CONCENTRATION_SUMMARY.json','SUBSET_RESULTS.json']:(O/f).write_bytes((H/f).read_bytes())
 dump(O/'FIXED_CONFIGURATION.json',rd(H/'experiment/DESIGN_LOCK.json'));dump(O/'COUNTS.json',{'directed_inputs':768,'unique_target_expressions':384,'arms':75,'readouts':57600,'new_readouts':13056,'reused_readouts':44544,'TTA_run':False,'all_negatives_unknowns_fallbacks_retained':True,'protected_alias_fallback':44,'actual_nonalias_hidden_states_reused_after_readout_amendment':62,'raw_media_descriptions_IDs_coordinates_features_weights_thirdparty_snapshots_excluded':True,'FINAL_access':False})
 files={R/p for p in rd(H/'experiment/CODE_LOCK.json')['files']};files.update(H.glob('*.py'));files.update((H/'attempts').glob('*.py'))
 for f in sorted(files):
  assert '/upstream/' not in str(f);p=O/'own_code'/f.relative_to(R);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 fs=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in fs]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(files)]});print('EXPORT111',len(fs)+1)
if __name__=='__main__':main()
