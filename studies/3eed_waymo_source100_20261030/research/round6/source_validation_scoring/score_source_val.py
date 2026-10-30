"""Epoch-specific postseal CPUsourceVAL scorer. Only approvedWaymoVALindex.
No torch/model/GPU imports, optimizer, targetdomains or checkpoint deserialization.
"""
from pathlib import Path
import argparse,json,hashlib,sys,time,collections
import numpy as np
from core import HEADS,MODES,KS,score_arrays,INDEPENDENT_ABS_TOL
R=Path(__file__).resolve().parent;ROOT=R.parents[2];AUDIT=R.parent/'source_data_audit';INDEX=AUDIT/'OFFICIAL_VAL_ANNO_ORDER_PRIVATE.json';WAYMO=ROOT/'crossplatform/dataset_full/3eed_dataset/waymo'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sealed_path(pred):
 p=pred/'sealed.json'
 if not p.is_file():p=pred/'SEALED.json'
 if not p.is_file():raise FileNotFoundError('SourceVALpredictionseal absent; no GTscoring')
 return p

def verify(a):
 program=R/'SCORER_PROGRAM_LOCK.json';lock=read(program);release=read(a.release)
 assert release['CPU_source_val_scoring_authorized'] is True and release['source_only'] is True
 assert release['scorer_program_lock_sha256']==sha(program)
 assert release['primary_epoch']==100,'No sourceVAL-dependentprimaryepoch selection'
 for name,h in lock['code_sha256'].items():assert sha(R/name)==h,name
 assert sha(Path(lock['source_prediction_writer_path']))==lock['source_prediction_writer_sha256']
 assert sha(AUDIT/'SOURCE_DATA_LOCK.json')==lock['source_data_lock_sha256']
 sp=sealed_path(a.predictions);seal=read(sp);assert sha(sp)==release['prediction_seal_sha256'];assert seal['n']==3782 and len(seal['files'])==3782
 assert seal['source_only'] is True and seal['all_inference_no_grad'] is True and seal['parameters_buffers_unchanged'] is True and seal['dynamic_padding_preserved'] is True
 assert seal['head_names']==HEADS and seal['actual_batch_size']==24 and seal['last_batch']==14
 assert seal['epoch']==release['epoch'] and seal['epoch'] in range(10,101,10)
 assert sha(a.checkpoint)==seal['checkpoint_sha256']==release['checkpoint_sha256']
 assert seal['model_state_fingerprint']==release['model_state_fingerprint']
 assert sha(INDEX)==seal['private_val_index_sha256']==release['private_val_index_sha256']==lock['private_val_index_sha256']
 assert seal['val_order_sha256']==release['val_order_sha256']==lock['val_order_sha256']
 assert set(seal['files'])=={f'{i:06d}.npz' for i in range(3782)}
 for name,h in seal['files'].items():assert sha(a.predictions/name)==h,(name,'sealedpredictionhashmismatch')
 # trace is informative, not part of predictor files map; parent may additionally bind it.
 trace=read(a.predictions/'trace.json');assert len(trace)==158 and [x['batch'] for x in trace]==list(range(158))
 assert all(x['B']==24 for x in trace[:-1]) and trace[-1]['B']==14 and trace[-1]['completed']==3782
 assert all(x['completed']==sum(y['B'] for y in trace[:i+1]) and x['T']<=256 for i,x in enumerate(trace))
 return seal,sp,lock,trace

def summary(rows):
 out=[]
 for name in rows[0]['arms']:
  aa=[r['arms'][name] for r in rows];scenes=sorted({r['scene'] for r in rows});out.append({'arm':name,'n':len(aa),'mean_iou':float(np.mean([a['iou'] for a in aa])),'equal_scene_mean_iou':float(np.mean([np.mean([r['arms'][name]['iou'] for r in rows if r['scene']==scene]) for scene in scenes])),'n_scene_clusters':len(scenes),'mean_candidate_oracle_iou_GT_assisted':float(np.mean([a['oracle_iou_GT_assisted'] for a in aa])),'topk':{str(k):{f'acc{t}':sum(a['topk'][str(k)][f'acc{t}'] for a in aa) for t in (25,50)} for k in KS},'rank_recoverable50':sum(a['rank_recoverable50'] for a in aa),'candidate_coverage_failure50':sum(a['candidate_coverage_failure50'] for a in aa),'sourceVAL':True,'label_assisted':name.startswith('official_'),'native_yaw':0})
 return out

def main(a):
 began=time.perf_counter();seal,sp,lock,trace=verify(a);a.out.mkdir(exist_ok=False,parents=True)
 transaction={'state':'verified_before_sourceGTscoring','prediction_seal_sha256':sha(sp),'checkpoint_sha256':sha(a.checkpoint),'epoch':seal['epoch'],'primary_epoch':100,'primary_checkpoint':seal['epoch']==100,'scorer_program_lock_sha256':sha(R/'SCORER_PROGRAM_LOCK.json'),'release_sha256':sha(a.release),'prediction_trace_sha256':sha(a.predictions/'trace.json'),'private_val_index_sha256':sha(INDEX),'all3782prediction_hashes_verified_before_sourceGTload':True,'source_only':True,'no_model_TTA_or_targetdomain_access':True,'geometry':'ActualsourceGTfloat32 promotedfloat64; predictionyaw0/clamp1e-6; batched2Dpolygon×z overlap +independent selectedgeneral3Dconvex'};(a.out/'SCORING_TRANSACTION.json').write_text(json.dumps(transaction,indent=2));index=read(INDEX);assert len(index)==3782;rows=[];invalid=collections.Counter();maxdiff=0.;ncheck=0;nbools=0;metahashes={};Trows=[]
 for i,s in enumerate(index):
  assert s['anno_index']==i and s['platform']=='waymo' and s['split']=='val' and s['official_eligible'] and s['positive_map_nonzero'];mp=Path(s['meta_path']);assert mp.is_relative_to(WAYMO)
  if str(mp) not in metahashes:metahashes[str(mp)]=sha(mp)
  assert metahashes[str(mp)]==s['meta_sha256'],'SourceVALcontainer changed since auditedjoin'
  with np.load(a.predictions/f'{i:06d}.npz',allow_pickle=False) as z:
   required={'head_names','boxes_raw','boxes','anno_index','point_sample_sha256','positive_map_sha256','T_batch','actual_text_token_length'}|set(MODES)|{m+'_top10' for m in MODES};assert required.issubset(z.files)
   assert int(z['anno_index'])==i and z['anno_index'].shape==();assert z['positive_map_sha256'].shape==(32,) and z['positive_map_sha256'].dtype==np.uint8;assert z['positive_map_sha256'].tobytes().hex()==s['positive_map_sha256'];assert z['point_sample_sha256'].shape==(32,) and z['point_sample_sha256'].dtype==np.uint8
   bidx=i//24;assert int(z['T_batch'])==trace[bidx]['T'];assert int(z['actual_text_token_length'])==s['actual_model_token_length'];assert 1<int(z['actual_text_token_length'])<=int(z['T_batch'])<=256;assert z['T_batch'].shape==() and z['actual_text_token_length'].shape==()
   for mode in MODES:assert z[mode].shape==(2,256) and z[mode].dtype==np.float32 and z[mode+'_top10'].shape==(2,10)
   arms,ious,ind,bad=score_arrays(z,s['source_GT7_float32']);maxdiff=max(maxdiff,ind['max_abs_selected2D_vs3D_difference']);ncheck+=ind['unique_selected_boxes_checked'];nbools+=ind['strict_threshold_booleans_checked'];Trows.append({'anno_index':i,'batch_token_length':int(z['T_batch']),'actual_text_token_length':int(z['actual_text_token_length'])})
   for key,values in bad.items():
    for h,v in zip(HEADS,values):invalid[key+':'+h]+=v
   rows.append({'anno_index':i,'platform':'waymo','scene':s['scene'],'frame':s['frame'],'object_index':s['object_index'],'class_source_GT':s['true_source_class'],'arms':arms,'point_sample_sha256':z['point_sample_sha256'].tobytes().hex(),'actual_text_token_length':int(z['actual_text_token_length']),'T_batch':int(z['T_batch']),'independent_selected_audit':ind})
   np.savez_compressed(a.out/f'{i:06d}_IOU_PRIVATE.npz',**ious)
  if (i+1)%500==0:print(json.dumps({'sourceVAL_scored':i+1,'n':3782,'epoch':seal['epoch'],'seconds':time.perf_counter()-began}),flush=True)
 result={'scope':'sourceWaymoVALmonitoring_only' if seal['epoch']!=100 else 'fixed_epoch100_sourceWaymoVAL_primary','n':3782,'epoch':seal['epoch'],'primary_epoch':100,'checkpoint_sha256':seal['checkpoint_sha256'],'prediction_seal_sha256':sha(sp),'source_val_order_sha256':seal['val_order_sha256'],'private_val_index_sha256':sha(INDEX),'rows':rows,'summaries':summary(rows),'invalid_counts':dict(invalid),'independent_selected_audit':{'all_rows_all8readouts_selectedTop1_verified':True,'unique_selected_boxes_checked':ncheck,'strict_threshold_booleans_checked':nbools,'max_abs_2D_vs_general3D_difference':maxdiff,'predeclared_abs_tolerance':INDEPENDENT_ABS_TOL,'full256pool_independently_rescored':False},'actual_batch_size':24,'last_batch':14,'padding_length_min_max':[min(x['batch_token_length'] for x in Trows),max(x['batch_token_length'] for x in Trows)],'no_optimizer_or_model_or_new_targetdomain_calls':True,'checkpoint_selection_does_not_change_from_fixed100':True,'seconds':time.perf_counter()-began};(a.out/'RESULTS_PRIVATE.json').write_text(json.dumps(result,indent=2,allow_nan=False));(a.out/'SEALED.json').write_text(json.dumps({'state':'sourceVAL_CPU_scoring_complete','n':3782,'epoch':seal['epoch'],'checkpoint_sha256':seal['checkpoint_sha256'],'prediction_seal_sha256':sha(sp),'scorer_program_lock_sha256':sha(R/'SCORER_PROGRAM_LOCK.json'),'files':{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()}},indent=2));print('SOURCE_VAL_SCORE_SEALED',seal['epoch'],sha(a.out/'SEALED.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['predictions','checkpoint','release','out']:p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
