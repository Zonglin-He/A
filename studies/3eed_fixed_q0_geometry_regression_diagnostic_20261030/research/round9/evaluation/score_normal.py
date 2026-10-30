"""R9 gated CPU scorer for fixed-q0 normal11 arms. No model/GPU.
No annotation read until full prediction+program+manifest+checkpoint gate passes.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import collections
from geometry_metrics import ARMS,MODES,eval_box,coordinate_errors,frame_inventory_from_metadata,spatial_confirmation,aggregate,candidate_ious
from independent_core_v2 import validate_native_top10
ROOT=Path(__file__).resolve().parents[3]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def bounded(root,name):
 p=(Path(root)/name).resolve()
 if not p.is_relative_to(Path(root).resolve()):raise ValueError('Path escapes root')
 return p

def gate(a):
 assert sha(a.release)==a.release_sha256
 g=read(a.release)
 assert g.get('CPU_SCORE_RELEASE') is True and g.get('complete_normal_predictions_sealed_before_GT') is True and g.get('stage')=='normal'
 for k,p in [('prediction_seal_sha256',a.predictions/'SEALED.json'),('manifest_sha256',a.manifest),('scoring_index_sha256',a.scoring_index),('source_checkpoint_sha256',a.source_checkpoint),('scorer_program_sha256',Path(__file__)),('geometry_metrics_sha256',Path(__file__).with_name('geometry_metrics.py'))]:assert g.get(k)==sha(p),k
 seal=read(a.predictions/'SEALED.json');assert seal['n_arms']==11 and seal['manifest_sha256']==sha(a.manifest) and seal['source_checkpoint_sha256']==sha(a.source_checkpoint)
 for k in ('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark'):assert seal['strict_FP32_flags'][k] is False
 for name,digest in seal['files'].items():assert sha(bounded(a.predictions,name))==digest
 assert seal['program_sha256_base']=='workspace_root'
 for name,digest in seal['program_sha256'].items():assert sha(bounded(a.program_root,name))==digest
 assert g['source_model_source_only_Waymo'] is True
 assert seal['all_11arms_and_sourcecache_sealed_before_GT'] and seal['all_normal_arm_queries_fixed_q0'] and seal['grounder_restored_and_CLIP_frozen']
 assert seal['source_cache_count']==seal['n_samples'] and seal['source_cache_files']=={k:v for k,v in seal['files'].items() if k.startswith('source_cache/')}
 assert set(g['geometry_dependencies_sha256'])=={'research/round5/audit/independent_core_v2.py','tools/obb_iou.py'}
 for rel,digest in g['geometry_dependencies_sha256'].items():assert sha(bounded(ROOT,rel))==digest
 # Gate reads no scoring index or metadata contents.
 return g,seal

def prediction(path):
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 names=d['arm_names'].tolist();assert tuple(names)==ARMS
 pool=d['source_boxes_raw6'];scores=d['source_text_scores256'];q=int(d['q_source']);raw=d['arm_boxes_raw6'];assert pool.shape==(256,6) and raw.shape==(11,6) and scores.shape==(256,) and np.isfinite(raw).all() and np.isfinite(pool).all() and np.isfinite(scores).all()
 validate_native_top10(scores,d['source_native_top10'])
 assert q==int(scores.argmax()) and np.array_equal(d['arm_query_index'],np.full(11,q))
 assert np.array_equal(raw[0],pool[q]) and np.array_equal(d['source_boxes_eval6'],eval_box(pool)) and np.array_equal(d['arm_boxes_eval6'],eval_box(raw))
 w=d['pseudo_reliability6'];p=d['pseudo_target6'];assert w.shape==p.shape==(2,6) and np.isfinite(w).all() and np.isfinite(p).all() and (w>=0).all() and (w<=1).all() and not w[:,3:].any()
 assert np.array_equal(raw[:,3:],np.broadcast_to(pool[q,3:],(11,3)))
 assert np.array_equal(d['forced_q_source'],np.full(2,q)) and tuple(d['modes'].tolist())==MODES
 assert d['raw_consensus_center'].shape==(2,3) and np.isfinite(d['raw_consensus_center']).all()
 for mi,m in enumerate(MODES):assert np.array_equal(raw[names.index(m+'__direct_G')],np.where(w[mi]>0,p[mi],pool[q]))
 return d

def validate_cache(root,seal,sample,pred,sentinel):
 rel='source_cache/'+sample['id']+'.npz';assert rel in seal['source_cache_files']
 with np.load(root/rel,allow_pickle=False) as z:
  assert str(z['sample_id'])==sample['id'] and str(z['source_checkpoint_sha256'])==seal['source_checkpoint_sha256']
  assert int(z['query_index'])==int(pred['q_source']) and np.array_equal(z['boxes_raw6'][0],pred['source_boxes_raw6']) and np.array_equal(z['text_scores'][0],pred['source_text_scores256'])
  for name,shape in [('base_xyz',(1,256,3)),('center_features',(1,288,256)),('size_features',(1,288,256))]:assert z[name].shape==shape and np.isfinite(z[name]).all()
  for name in ('point_sample_sha256','raw_indices_sha256'):assert np.array_equal(z[name],pred[name])
  assert z['actual_model_text_sha256'].tobytes()==hashlib.sha256((sample['caption']+sentinel).encode()).digest()
  assert z['manifest_row_sha256'].tobytes()==hashlib.sha256(json.dumps(sample,sort_keys=True,separators=(',',':')).encode()).digest()

def direction_diagnostic(d,mi,target,trace):
 source=d['arm_boxes_raw6'][0].astype(float);raw=d['raw_consensus_center'][mi].astype(float)-source[:3];final=d['pseudo_target6'][mi,:3].astype(float)-source[:3];truth=target[:3].astype(float)-source[:3];nr=float(np.linalg.norm(raw));nt=float(np.linalg.norm(truth));nf=float(np.linalg.norm(final));accepted=bool(d['pseudo_reliability6'][mi].any());active=d['pseudo_reliability6'][mi,:3]>0
 direct=source[:3]+np.where(active,final,0.)
 return {'accepted_any_axis':accepted,'actual_pseudo_nonzero':bool(np.any(final[active]!=0)),'raw_consensus_shift_L2_m':nr,'actual_pseudo_shift_L2_m':nf,'r_global':float(d['r_global'][mi]),'raw_consensus_dot_GT_correction_m2':float(raw@truth),'raw_consensus_cosine_GT_correction':float(raw@truth/(nr*nt)) if nr>0 and nt>0 else None,'final_pseudo_dot_GT_correction_m2':float(final@truth),'direct_center_error_delta_m':float(np.linalg.norm(direct-target[:3])-nt),'size_weights_all_zero':not bool(d['pseudo_reliability6'][mi,3:].any()),'normal_pseudo_uses_GT':False,'postseal_direction_diagnostic_only_not_extra_TTAarm':True,'reason':trace['G_diagnostics']['reason']}

def cancellation_summaries(rows):
 out=[]
 for platform in ('drone','quad'):
  rr=[r for r in rows if r['platform']==platform]
  for mode in MODES:
   gs=[r['G'][mode] for r in rr];align=[g['raw_consensus_cosine_GT_correction'] for g in gs if g['raw_consensus_cosine_GT_correction'] is not None]
   for kind in ('direct_G','param1','param3','outputspace1','outputspace3'):
    arm=mode+'__'+kind;ds=[r['arms'][arm]['delta_vs_source'] for r in rr]
    pos=[v for v in ds if v>1e-8];neg=[v for v in ds if v< -1e-8]
    out.append({'platform':platform,'mode':mode,'arm':arm,'n':len(rr),'positive_count_atol1e8':len(pos),'negative_count_atol1e8':len(neg),'ties_atol1e8':sum(abs(v)<=1e-8 for v in ds),'sum_positive_IoU_delta':float(sum(pos)),'sum_negative_IoU_delta':float(sum(neg)),'net_all_IoU_delta':float(sum(ds)),'mean_selected_center_shift_m':float(np.mean([r['arms'][arm]['selected_center_shift_L2_m'] for r in rr])) if rr else None,'size_changed_count':sum(r['arms'][arm]['selected_size_changed'] for r in rr),'failed_status_count':sum(r['arms'][arm]['status'].startswith('failed') for r in rr),'status_histogram':dict(collections.Counter(r['arms'][arm]['status'] for r in rr)),'parameter_steps_executed':sum(r['arms'][arm]['parameter_steps_executed'] for r in rr),'nonzero_parameter_update_cases':sum(r['arms'][arm]['nonzero_parameter_update'] for r in rr),'raw_consensus_alignment_defined':len(align),'raw_consensus_alignment_positive':sum(x>0 for x in align),'raw_consensus_alignment_negative':sum(x<0 for x in align),'raw_consensus_alignment_zero':sum(x==0 for x in align),'raw_consensus_alignment_mean_defined':float(np.mean(align)) if align else None,'postseal_alignment_not_extra_TTAarm':True})
 return out

def main(a):
 g,seal=gate(a);samples=read(a.manifest);assert len(samples)==seal['n_samples'] and len({s['id'] for s in samples})==len(samples)
 # Validate every prediction BEFORE opening GT-bearing containers.
 preds={s['id']:prediction(a.predictions/(s['id']+'.npz')) for s in samples}
 plan=read(a.predictions/'registered_plan.json')
 for sample in samples:validate_cache(a.predictions,seal,sample,preds[sample['id']],plan['sentinel'])
 index={s['id']:s for s in read(a.scoring_index)};a.output.mkdir(parents=True,exist_ok=False)
 dump(a.output/'SCORER_LOCK.json',{'state':'all_normal_source_program_hashes_and_prediction_contracts_checked_before_target_GT','release_sha256':sha(a.release),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'source_checkpoint_sha256':sha(a.source_checkpoint),'scorer_sha256':sha(Path(__file__)),'metrics_sha256':sha(Path(__file__).with_name('geometry_metrics.py'))})
 rows=[];GTindex=[];GTsidecar=[];meta_cache={}
 traces={t['id']:t for t in read(a.predictions/'trace.json')};assert set(traces)=={s['id'] for s in samples}
 for s in samples:
  sid=s['id'];d=preds[sid];ix=index[sid];assert ix['platform']==s['platform'];mp=Path(ix['meta_path']);assert sha(mp)==ix['metadata_container_sha256']
  if str(mp) not in meta_cache:meta_cache[str(mp)]=read(mp)
  meta=meta_cache[str(mp)]
  inventory,ti,refs=frame_inventory_from_metadata(meta,ix['object_index'],s['platform'],ix.get('waymo_view_suffix'));target=inventory[ti]
  GTindex.append({'id':sid,'target6_modelcoords':target[:6].tolist()});GTsidecar.append({'id':sid,'platform':s['platform'],'scene':ix['scene'],'GT7D_model':target.tolist()})
  vals,cost=candidate_ious(eval_box(d['arm_boxes_raw6']),target);assert np.isfinite(vals).all()
  identity=spatial_confirmation(d['arm_boxes_raw6'][0],target,inventory,ti)
  row={'id':sid,'platform':s['platform'],'scene':ix['scene'],'q_source':int(d['q_source']),'identity_diagnostic':identity,'arms':{},'G':{}}
  for mi,m in enumerate(MODES):row['G'][m]=direction_diagnostic(d,mi,target,traces[sid]['G_modes'][m])
  for j,name in enumerate(ARMS):
   steps=[]
   if name.endswith(('__param1','__param3')):steps=traces[sid]['G_modes'][name.split('__')[0]]['steps'].get(name[-1],{}).get('trace',[])
   row['arms'][name]={'iou':float(vals[j]),'Acc25':bool(vals[j]>.25),'Acc50':bool(vals[j]>.5),'coordinate_errors':coordinate_errors(d['arm_boxes_raw6'][j],target),'delta_vs_source':float(vals[j]-vals[0]),'selected_center_shift_L2_m':float(np.linalg.norm(d['arm_boxes_raw6'][j,:3].astype(float)-d['arm_boxes_raw6'][0,:3].astype(float))),'selected_size_changed':not np.array_equal(d['arm_boxes_raw6'][j,3:],d['arm_boxes_raw6'][0,3:]),'status':str(d['arm_status'][j]),'parameter_steps_executed':len(steps),'nonzero_gradient_steps':sum(z['parameter_gradient_norm']>0 and z['nonzero_gradient_parameter_tensors']>0 for z in steps),'nonzero_parameter_update':bool(steps and any(v>0 for v in steps[-1]['parameter_delta_max_by_name'].values()))}
  rows.append(row)
 result={'state':'normal11_fixedq0_scored_postseal','n':len(rows),'summary':aggregate(rows),'correction_and_cancellation_diagnostics':cancellation_summaries(rows),'rows_PRIVATE':rows,'scope':'Nativeyaw0, noPCA/noGTcandidate selection; conditional identity diagnostic is annotated-spatial only. Source-onlyWaymo; targetVAL prior-research-exposed. 2Drone/3Quad scenes; descriptive CI only.'}
 dump(a.output/'RESULTS_PRIVATE.json',result);dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in result.items() if k!='rows_PRIVATE'});dump(a.output/'GT_CONTROL_TARGET_INDEX_PRIVATE.json',GTindex);dump(a.output/'PRIVATE_GT7D_SIDECAR.json',{'scope':'labelled_controlled_and_postseal_overlay_only_NOT_normal_G_input','normal_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'manifest_sha256':sha(a.manifest),'score_release_sha256':sha(a.release),'transform_helper_sha256':sha(ROOT/'research/round5/audit/independent_core_v2.py'),'rows':GTsidecar});dump(a.output/'SEALED.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_samples':len(rows),'normal_prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'manifest_sha256':sha(a.manifest),'GT_release_sha256':sha(a.release),'GTindex_purpose':'labelled_controlled_only_not_normal_G','GT7D_purpose':'postseal_identity_overlay_only_not_model_input'})
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','manifest','scoring-index','source-checkpoint','release','program-root','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
