"""Complete anonymous outcomes and fit endpoints, never media/weights/captions."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round98';B=R/'crossplatform/round98';P=H/'publication';O=P/'public_export_attempt001';S=H/'scores_attempt001';PREFIX='studies/3eed_complete_R95_three_source_six_direction_TTA_20261114';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rows):
 assert rows;p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 for f,s in rd(H/'experiment/PUBLIC_EXPORT_LOCK.json')['inputs_sha256'].items():assert sha(R/f)==s,f
 assert rd(H/'SCALAR_AUDIT.json')['state'].startswith('PASS_');assert not O.exists();O.mkdir(parents=True);matrix=rd(B/'MATRIX_SEALED.json');scene_maps={}
 for direction in matrix['directions']:
  rr=rd(S/(direction+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scene_maps[direction]=sc;rows=[];fits=[]
  for r in rr:
   a={'row':r['row'],'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3),'unknown_roles':r['status']['unknown_roles'],'visible_seeds':r['status']['visible_seeds']}
   for arm,v in r['arms'].items():
    for k,x in v.items():
     if k!='query_PRIVATE':a[arm+'__'+k]=x
   rows.append(a)
   for fit,x in r['fits'].items():fits.append({'row':r['row'],'source':r['source'],'target':r['target'],'fit':fit,**x})
  for label,z in [('tasks',rows),('fit_endpoints',fits)]:
   for i in range(0,len(z),1000):table(O/label/(direction+'_part'+str(i//1000+1).zfill(3)+'.csv'),z[i:i+1000])
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(name+'.csv'),rd(S/(name+'.json')))
 scenes=[]
 for x in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):
  sc=scene_maps[x['source']+'_to_'+x['target']];scenes.append({k:v for k,v in x.items() if k!='scene_PRIVATE'}|{'scene_cluster':'CLUSTER'+str(sc.index(x['scene_PRIVATE'])+1).zfill(3)})
 table(O/'results/ALL_SCENE_ROWS.csv',scenes);table(O/'results/FIT_SUMMARY_ROWS.csv',rd(H/'FIT_SUMMARY_ROWS.json'))
 for name in ['REPORT.md','PROTOCOL.md','ITERATION_RECORD.md','DECISION.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json','LEGACY_R95_PARITY_AUDIT.json']:(O/name).write_text((H/name).read_text().replace(str(R),'WORKSPACE_ROOT'))
 dump(O/'SCORING_RECEIPT.json',rd(S/'SCORING_RECEIPT.json'));config=rd(H/'experiment/DESIGN_LOCK.json');dump(O/'FIXED_CONFIGURATION.json',{k:v for k,v in config.items() if k not in ['targets','original_models','source_modules']})
 dump(O/'FAILURES_AND_REPAIRS.json',{'reference_only_stale_gradient':True,'detected_before_any_target_scoring':True,'primary_LM_unchanged':True,'formal_predictions_restarted':True,'old_attempt_retained':True,'legacy_GN_reference_exact_after_fix':True,'overlap32_LM_all12_endpoints_bit_exact':True})
 files=list(H.glob('*.py'))+list((H/'attempts').glob('*.py'))+[R/'tools/manual_r98_guard.py'];files += [R/f for f in rd(H/'experiment/CODE_LOCK.json')['code_sha256']];files=sorted(set(files))
 # Only own research implementation; no official downloaded model snapshots.
 for f in files:
  assert f.suffix=='.py' and f.is_file();p=O/'own_code'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 sources=[];from97=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json');original=rd(R/'research/round96/SOURCE_MODEL_COLLECTION_PRIVATE.json')
 for src in ['waymo','drone','quad']:
  sources.append({'source':src,'original_and_module_provenance':'unchanged_R96_R97','new_training_in_R98':False,'source100_sha256':next(x['checkpoint_sha256'] for x in original['models'] if x['source']==src),'prepared_module_sha256':from97['modules'][src]['state_sha256']})
 dump(O/'source/REUSE_METADATA.json',{'sources':sources,'original_collection_sha256':sha(R/'research/round96/SOURCE_MODEL_COLLECTION_PRIVATE.json'),'module_collection_sha256':sha(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json'),'new_training':False})
 dump(O/'COUNTS.json',{'sources':3,'directions':6,'unique_target_expressions':9310,'direction_inputs':18620,'task_readouts':539980,'fit_endpoint_rows':74480,'FINAL_access':False,'all_task_negatives_retained':True,'accepted_and_rejected_trial_scalars':'retained_locally_in_full','public_trial_content':'all_four_fit_endpoints_per_input','raw_data_weights_or_features_published':False})
 filesout=sorted(p for p in O.rglob('*') if p.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(O).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in filesout]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in files]});print({'files':len(filesout)+1,'bytes':sum(p.stat().st_size for p in O.rglob('*') if p.is_file())})
if __name__=='__main__':main()
