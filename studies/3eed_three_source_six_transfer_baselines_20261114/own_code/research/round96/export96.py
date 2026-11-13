"""Explicit own-code and anonymous saved-scalar export; no models, scoring or network."""
from pathlib import Path
import json,hashlib,csv,io,math
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';P=H/'publication';PREFIX='studies/3eed_three_source_six_transfer_baselines_20261114';rd=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=P/'public_export_attempt001'
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr):
 p.parent.mkdir(parents=True,exist_ok=True);cols=list(rr[0]);f=io.StringIO();w=csv.DictWriter(f,fieldnames=cols,lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(f.getvalue())
def main():
 for f,v in rd(H/'experiment/PUBLIC_EXPORT_LOCK.json')['inputs_sha256'].items():assert sha(R/f)==v,f
 assert rd(H/'MATRIX_SUMMARY_AUDIT.json')['state'].startswith('PASS_') and rd(H/'independent_native_matrix_audit/AUDIT_RESULT.json')['state'].startswith('PASS_') and rd(H/'TRAINING_AUDIT.json')['state'].startswith('PASS_');assert not O.exists();O.mkdir();S=H/'frozen_scores_attempt003';summ=rd(S/'ALL_METHOD_ROWS.json');matrix=rd(B/'frozen_matrix_attempt003/MATRIX_SEALED.json');pubrows=[];allscene=[]
 for d in matrix['directions']:
  src,tgt=d['source'],d['target'];rows=rd(S/(src+'_to_'+tgt+'_ALL_ROWS_PRIVATE.json'));scenes=sorted({r['scene_PRIVATE'] for r in rows});wide=[]
  for i,r in enumerate(rows):
   z={'row':'ROW'+str(i+1).zfill(6),'source':src,'target':tgt,'scene_cluster':'CLUSTER'+str(scenes.index(r['scene_PRIVATE'])+1).zfill(3)}
   for arm,v in r['arms'].items():
    for name,val in v.items():z[arm+'__'+name]=val
   wide.append(z)
  for i in range(0,len(wide),1000):table(O/'tasks'/(src+'_to_'+tgt+'_part'+str(i//1000+1).zfill(3)+'.csv'),wide[i:i+1000])
  for i,sc in enumerate(scenes):
   rr=[r for r in rows if r['scene_PRIVATE']==sc]
   for arm in rr[0]['arms']:
    v=[r['arms'][arm]['IoU'] for r in rr];allscene.append({'source':src,'target':tgt,'scene_cluster':'CLUSTER'+str(i+1).zfill(3),'arm':arm,'n':len(v),'mIoU_percent':sum(v)/len(v)*100,'Acc25_percent':sum(x>.25 for x in v)/len(v)*100,'Acc50_percent':sum(x>.5 for x in v)/len(v)*100})
 table(O/'results/ALL_METHOD_ROWS.csv',summ);table(O/'results/ALL_SCENE_ROWS.csv',allscene)
 curves=[];sources=[];reg=rd(H/'experiment/ACTIVE_TRAINING_REGISTRATION.json')
 for pl in ['drone','quad']:
  d=B/(pl+'_source100');c=rd(d/'TRAINING_COMPLETE.json');plan=rd(Path(reg[pl]['train']['path']));steps={}
  for p in sorted((d/'attempts').glob('*/steps.jsonl')):
   for line in p.read_text().splitlines():
    x=json.loads(line);steps[x['epoch'],x['batch']]=x
  for ep in range(1,101):
   rr=[v for (e,b),v in steps.items() if e==ep];curves.append({'source':pl,'epoch':ep,'optimizer_updates':len(rr),'mean_loss':sum(x['loss'] for x in rr)/len(rr),'min_loss':min(x['loss'] for x in rr),'max_loss':max(x['loss'] for x in rr),'mean_update_seconds':sum(x['forward_backward_update_seconds'] for x in rr)/len(rr),'peak_GPU_bytes':max(x['peak_allocated_bytes'] for x in rr)})
  sources.append({'source':pl,'epoch':100,'training_expressions':c['source_train_contract']['n'],'updates':c['completed_optimizer_steps'],'checkpoint_sha256':c['primary_checkpoint_sha256'],'full_resume_sha256':c['final_resume_checkpoint_sha256'],'training_and_source_validation_seconds':c['elapsed_attempt_seconds'],'initial_model_sha256':plan['expected_model_initial_sha256'],'initial_generic_text_sha256':plan['expected_text_encoder_initial_sha256'],'no_joint_or_other_source_task_initialization':True,'source_train_order_sha256':c['source_train_contract']['order_sha256'],'source_val_n':c['source_val_contract']['n'],'batch_size':24,'precision':'FP32','source_val_frequency':plan['source_validation_every'],'milestones_optimizer_steps':plan['native_effective_scheduler_milestones_steps'],'primary_selection':'fixed epoch100, no target selection'})
 old=rd(R/'crossplatform/round6/waymo_source100/TRAINING_COMPLETE.json');sources.insert(0,{'source':'waymo','epoch':100,'training_expressions':old['source_train_contract']['n'],'updates':old['completed_optimizer_steps'],'checkpoint_sha256':old['primary_checkpoint_sha256'],'training_and_source_validation_seconds':old['elapsed_attempt_seconds'],'reused_existing_audited_run':True,'primary_selection':'fixed epoch100, no target selection'})
 table(O/'training/ALL_EPOCH_ROWS.csv',curves);dump(O/'training/SOURCE100_METADATA.json',sources)
 for src,dest in [(H/'REPORT.md','REPORT.md'),(H/'PROTOCOL.md','protocol/PROTOCOL.md')]:p=O/dest;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(src.read_text().replace(str(R),'WORKSPACE_ROOT'))
 for src,dest in [(H/'TRAINING_AUDIT.json','audits/TRAINING_AUDIT.json'),(H/'MATRIX_SUMMARY_AUDIT.json','audits/MATRIX_SUMMARY_AUDIT.json'),(H/'independent_native_matrix_audit/AUDIT_RESULT.json','audits/NATIVE_MATRIX_AUDIT.json'),(S/'SCORING_RECEIPT.json','audits/SCORING_RECEIPT.json'),(H/'experiment/CORE_COMPATIBILITY_AUDIT.json','audits/CORE_COMPATIBILITY.json'),(H/'experiment/RESUME_AUDIT.json','audits/RESUME_AUDIT.json'),(H/'experiment/ENGINEERING_REVIEW.json','audits/ENGINEERING_REVIEW.json')]:dump(O/dest,rd(src))
 fixed=rd(H/'experiment/MATRIX_REPAIR_LOCK_attempt003.json');dump(O/'protocol/FIXED_CONFIGURATION.json',{k:v for k,v in fixed.items() if k not in ['code_sha256','target_contracts']});dump(O/'history/FAILURES_AND_REPAIRS.json',{'events':[{'stage':'drone_engineering_attempt001','failure':'300 second wrapper timeout after DDP ready at291.33s; zero optimizer steps','repair':'900 second startup and2GiB CUDA compile cache; no training math change'},{'stage':'matrix_original_entry','failure':'wildcard factory import shadowed the repository root; missing Waymo checkpoint before any target predictions','repair':'explicit imports and preflight absolute checkpoint paths; no source retraining or metric change'},{'stage':'slotwise_and_geometry_matched_native_audits','failure':'candidate permutations and numerical replay bounds violated; zero new target IoU scores read','repair':'fixed FP32 inference backend after no-GT repeated-forward preflight; preserve all old predictions and regenerate six directions'},{'stage':'diagnostic_multiprocessing','failure':'missing main guard in diagnostic caused worker startup error; retry blocked by GPU lock until failed diagnostic was terminated','repair':'add main guard; preserve failed source and logs; no training or target scoring involved'},{'stage':'repair_registration_helper','failure':'syntax error followed by missing uncreated entry; no target predictions','repair':'correct registration helper, preserve both failed logs, compile before execution'}],'all_failed_artifacts_retained_locally':True})
 allow=sorted(H.glob('*.py'))+sorted((H/'attempts').glob('*.py'))+sorted(B.glob('*.py'))+[R/'tools/manual_r96_guard.py']
 for p in allow:
  dst=O/'own_code'/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(p.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 dump(O/'COUNTS.json',{'sources_trained':2,'reused_sources':1,'new_training_updates':37600,'cross_domain_directions':6,'direction_task_inputs':18620,'readout_task_values':186200,'all_rows_retained':True,'TTA':False,'protected_frames_excluded':384,'raw_data_and_weights_published':False,'source_model_code_files':len(allow)})
 (O/'README.md').write_text('# R96 three-source frozen baselines\n\nOfficial 3EED source100 training for Drone/Quad, existing Waymo100 reused. All six directed frozen transfers; no TTA claim. Every anonymous task/readout scalar is retained in wide CSVs, so all negative readout comparisons can be reconstructed. No original captions, IDs, scene names, media, coordinates, candidate indices, weights, feature caches or secrets. Source-trained RGB–LiDAR R91 modules are not included. Training and entry failures are disclosed. See REPORT.md and protocol/PROTOCOL.md.\n')
 files=sorted(x for x in O.rglob('*') if x.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'manifest_excludes_itself':True,'files':[{'path':p.relative_to(O).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'own_code_files':[str(x.relative_to(R)) for x in allow],'scalar_inputs':[str(p.relative_to(R)) for p in S.glob('*.json')],'prefix':PREFIX});print({'files':len(files)+1,'bytes':sum(p.stat().st_size for p in O.rglob('*') if p.is_file())})
if __name__=='__main__':main()
