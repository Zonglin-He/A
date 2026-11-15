"""Only own code and anonymous complete scalar evidence; no models or private media."""
from pathlib import Path
import json,hashlib,csv,io
R=Path(__file__).resolve().parents[2];H=R/'research/round97';B=R/'crossplatform/round97';P=H/'publication';O=P/'public_export_attempt001';PREFIX='studies/3eed_fixed_R91_three_source_six_direction_TTA_20261114';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 for f,s in rd(H/'experiment/PUBLIC_EXPORT_LOCK.json')['inputs_sha256'].items():assert sha(R/f)==s,f
 assert rd(H/'SCALAR_AUDIT.json')['state'].startswith('PASS_');assert not O.exists();O.mkdir(parents=True);matrix=rd(B/'MATRIX_SEALED.json');S=H/'scores_attempt001';steps=[];hist=[];sourceinfo=[];scene_maps={}
 for direction in matrix['seals_sha256']:
  rr=rd(S/(direction+'_ALL_ROWS_PRIVATE.json'));sc=sorted({r['scene_PRIVATE'] for r in rr});scene_maps[direction]=sc;rows=[];trials=[]
  for r in rr:
   a={'row':r['row'],'source':r['source'],'target':r['target'],'scene_cluster':'CLUSTER'+str(sc.index(r['scene_PRIVATE'])+1).zfill(3),'unknown_roles':r['status']['unknown_roles'],'visible_seeds':r['status']['visible_seeds']}
   for arm,v in r['arms'].items():
    for key,x in v.items():
     if key!='query_PRIVATE':a[arm+'__'+key]=x
   for arm,u in r['updates'].items():
    for key in ['alpha','actual_parameter_L2','gradient_norm','objective_before','update_applied']:a[arm+'__'+key]=u[key]
    for k,t in enumerate(u['trials']):trials.append({'row':r['row'],'source':r['source'],'target':r['target'],'arm':arm,'trial':k,**t})
   rows.append(a)
  for label,z in [('tasks',rows),('trials',trials)]:
   for i in range(0,len(z),1000):table(O/label/(direction+'_part'+str(i//1000+1).zfill(3)+'.csv'),z[i:i+1000])
 for src in ['drone','quad']:
  c=rd(B/src/'SOURCE_COMPLETE.json');audit=next(v for v in rd(H/'SOURCE_PREPARATION_AUDIT.json')['rows'] if v['source']==src);sourceinfo.append({k:v for k,v in c.items() if k not in ['fusion_selected','head_selected','updates']});sourceinfo[-1].update(fusion_epoch=c['fusion_selected']['epoch'],head_epoch=c['head_selected']['epoch'],source_preparation_attempted_updates=c['updates'],actual_optimizer_updates=audit['actual_optimizer_updates'],no_interface_signal_rows=audit['no_interface_signal_rows'])
  for label,name in [('fusion','FUSION_HISTORY.json'),('head','HEAD_HISTORY.json')]:
   for v in rd(B/src/name):hist.append({'source':src,'stage':label,'epoch':v['epoch'],'source_val_mIoU_percent':v['source_val_mIoU_percent'],'state_sha256':v['state_sha256']})
  steps.extend({'source':src,**v} for v in rd(B/src/'SOURCE_STEPS.json'))
 table(O/'source/ALL_SOURCE_STEPS.csv',steps);table(O/'source/ALL_SOURCE_EPOCHS.csv',hist);dump(O/'source/PREPARATION_METADATA.json',sourceinfo)
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(name+'.csv'),rd(S/(name+'.json')))
 scene=[]
 for v in rd(S/'ALL_SCENE_ROWS_PRIVATE.json'):
  sc=scene_maps[v['source']+'_to_'+v['target']];scene.append({k:x for k,x in v.items() if k!='scene_PRIVATE'}|{'scene_cluster':'CLUSTER'+str(sc.index(v['scene_PRIVATE'])+1).zfill(3)})
 table(O/'results/ALL_SCENE_ROWS.csv',scene);table(O/'results/UPDATE_SUMMARY_ROWS.csv',rd(H/'UPDATE_SUMMARY_ROWS.json'))
 for f in ['REPORT.md','ITERATION_RECORD.md','DECISION.json','NATIVE_AUDIT.json','SCALAR_AUDIT.json','SOURCE_PREPARATION_AUDIT.json','ENGINEERING_R91_PARITY_AUDIT.json']:
  p=O/f;p.parent.mkdir(exist_ok=True);p.write_text((H/f).read_text().replace(str(R),'WORKSPACE_ROOT'))
 dump(O/'SCORING_RECEIPT.json',rd(S/'SCORING_RECEIPT.json'));config=rd(H/'experiment/DESIGN_LOCK.json');dump(O/'FIXED_CONFIGURATION.json',{k:v for k,v in config.items() if k not in ['sources','inputs_sha256']})
 failed=[{'kind':'source_engineering_missing_iou_import','retained':True,'model_or_target_metric_effect':False},{'kind':'source_other_box_7D_9D_schema','retained':True,'repair':'official documented yaw-only7D source others adapter; no target geometry change'}];dump(O/'FAILURES_AND_REPAIRS.json',failed)
 files=list(H.glob('*.py'))+list((H/'attempts').glob('*.py'))+[R/'tools/manual_r97_guard.py']
 lock=rd(H/'experiment/TARGET_CODE_LOCK.json')
 files+= [R/f for f in lock['code_sha256'] if f.startswith(('research/round76/','research/round79/','research/round88/','research/round89/','research/round90/','research/round91/','research/round66/','crossplatform/round51/','crossplatform/round7/','crossplatform/round96/','crossplatform/round48/','research/round85/','research/round86/','research/round87/'))]
 for f in sorted(set(files)):
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 (O/'PROTOCOL.md').write_text((H/'PROTOCOL.md').read_text());dump(O/'COUNTS.json',{'sources':3,'directions':6,'direction_inputs':18620,'task_readouts':148960,'source_new_preparations':2,'FINAL_access':False,'all_negatives_unknown_rejected_trials_retained':True,'raw_data_weights_or_features_published':False})
 filesout=sorted(f for f in O.rglob('*') if f.is_file());dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':f.relative_to(O).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)} for f in filesout]});dump(P/'EXPORT_INDEX_PRIVATE.json',{'prefix':PREFIX,'own_code_files':[str(f.relative_to(R)) for f in sorted(set(files))]});print({'files':len(filesout)+1,'bytes':sum(f.stat().st_size for f in O.rglob('*') if f.is_file())})
if __name__=='__main__':main()
