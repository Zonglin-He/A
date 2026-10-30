"""R12 post-all-prediction-seal selectedbox CPU score. No label API before release."""
from pathlib import Path
import argparse,hashlib,json,time,sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import transform_gt
from metrics import evaluate_case,summaries,eval_boxes
ARMS=('source','grabcut_direct','single_instance_direct','multiview_instance_direct','wrongview_instance_direct','single_instance_half','multiview_instance_half','multiview_maskmatch_only_direct')
DEP={'research/round5/audit/independent_core_v2.py':'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a','tools/obb_iou.py':'45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def member(base,name):
 p=(base/name).resolve();assert p.is_relative_to(base.resolve());return p

def main(a):
 start=time.perf_counter();assert sha(a.release)==a.release_sha256;g=read(a.release)
 assert g['CPU_SCORE_RELEASE'] and g['stage']=='R12_independent_instance_multiview_direct' and g['all_source_masks_geometry_arms_sealed_before_GT']
 for k,p in [('prediction_seal_sha256',a.predictions/'SEALED.json'),('source_seal_sha256',a.source/'SEALED.json'),('mask_seal_sha256',a.masks/'SEALED.json'),('manifest_sha256',a.manifest),('scoring_index_sha256',a.scoring_index),('scorer_sha256',Path(__file__)),('metrics_sha256',Path(__file__).with_name('metrics.py'))]:assert g[k]==sha(p),k
 for rel,h in DEP.items():assert sha(ROOT/rel)==h
 ps=read(a.predictions/'SEALED.json');ss=read(a.source/'SEALED.json');ms=read(a.masks/'SEALED.json')
 assert ss['state']=='sealed_R12_source_forward_and_input_observations_no_G_no_GT' and ss['source_unchanged'] and ss['source_repeat_bitexact']
 assert ms['state']=='sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT' and not ms['trained_or_updated'] and ms['mask_returned_original_image_pixels']
 for seal in (ss,ms):
  assert seal['strict_FP32_flags']=={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
 sourceplan=read(a.source/'registered_plan.json');assert sha(sourceplan['source_checkpoint_path'])==g['source_checkpoint_sha256']==ss['source_checkpoint_sha256']
 assert ms['weights_sha256']==g['mask_weights_sha256']=='73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e'
 assert ps['state']=='sealed_R12_8arm_independent_instance_multiview_direct_before_GT' and tuple(ps['arm_names'])==ARMS and ps['n_arms']==8
 assert ps['source_seal_sha256']==sha(a.source/'SEALED.json') and ps['mask_seal_sha256']==sha(a.masks/'SEALED.json') and ps['manifest_sha256']==ss['manifest_sha256']==sha(a.manifest)
 assert ps['all_queries_fixed_original_q0'] and not ps['GT_model_GPU_or_neural_update_executed'] and not ps['provided_pose_used_for_registration_or_target_fusion']
 assert ps['source_checkpoint_sha256']==ss['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 for base,seal in [(a.predictions,ps),(a.source,ss),(a.masks,ms)]:
  for rel,h in seal['files'].items():assert sha(member(base,rel))==h
 for rel,h in ps['program_sha256'].items():assert sha(member(ROOT,rel))==h
 manifest=read(a.manifest);assert len(manifest)==ps['n_samples']==ss['n_samples']==146;assert len({r['id'] for r in manifest})==146
 assert {r['id']+'.npz' for r in manifest}=={p.name for p in a.predictions.glob('*.npz')}
 preds={}
 for r in manifest:
  cp=a.source/'source_cache'/(r['id']+'.npz')
  with np.load(a.predictions/(r['id']+'.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  with np.load(cp,allow_pickle=False) as z:c={k:z[k].copy() for k in ('boxes_raw6','text_scores','query_index','point_sample_sha256','raw_indices_sha256')}
  q=int(c['query_index']);assert q==int(d['q_source'])==int(c['text_scores'][0].argmax()) and np.array_equal(d['arm_query_index'],np.full(8,q))
  raw=d['arm_boxes_raw6'];assert tuple(d['arm_names'].tolist())==ARMS and raw.shape==(8,6) and np.isfinite(raw).all() and d['acceptance_mask6'].shape==(8,6) and d['arm_status'].shape==(8,)
  assert np.array_equal(d['source_boxes_raw6'],c['boxes_raw6'][0]) and np.array_equal(d['source_text_scores256'],c['text_scores'][0]) and np.array_equal(raw[0],c['boxes_raw6'][0,q])
  assert np.array_equal(d['actual_changed_axes'],raw!=raw[0]) and np.array_equal(d['arm_boxes_eval6'],eval_boxes(raw))
  assert np.array_equal(raw[~d['acceptance_mask6']],np.broadcast_to(raw[0],(8,6))[~d['acceptance_mask6']])
  assert np.array_equal(raw[5],raw[0]+np.float32(.5)*(raw[2]-raw[0])) and np.array_equal(raw[6],raw[0]+np.float32(.5)*(raw[3]-raw[0]))
  assert d['source_cache_sha256'].tobytes().hex()==sha(cp)
  for k in ('point_sample_sha256','raw_indices_sha256'):assert np.array_equal(d[k],c[k])
  preds[r['id']]=d
 # All unlabelled checks complete before scoring index or target annotation read.
 a.output.mkdir(parents=True,exist_ok=False);dump(a.output/'SCORER_LOCK.json',{'state':'all146_8arm_contracts_checked_before_targetGT','prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'release_sha256':sha(a.release),'nativeyaw0_noGTselection':True})
 idx={r['id']:r for r in read(a.scoring_index)};cache={};rows=[];cost=collections.Counter()
 for s in manifest:
  ix=idx[s['id']];assert ix['platform']==s['platform'] and ix['scene']==s['scene'];mp=Path(ix['meta_path']);assert sha(mp)==ix['metadata_container_sha256']
  if str(mp) not in cache:cache[str(mp)]=read(mp)
  target=transform_gt(cache[str(mp)]['ground_info'][ix['object_index']]['bbox_3d'],s['platform'],ix.get('waymo_view_suffix'));d=preds[s['id']]
  r,c=evaluate_case(sid=s['id'],platform=s['platform'],scene=ix['scene'],boxes_raw6=d['arm_boxes_raw6'],arm_names=ARMS,accepted_axes=d['acceptance_mask6'],target7=target,statuses=d['arm_status']);rows.append(r);cost.update(c)
 summary,contrasts=summaries(rows);out={'state':'R12_146_independent_instance_multiview_scored_after_complete_seal','n':len(rows),'summary':summary,'paired_contrasts':contrasts,'rows_PRIVATE':rows,'geometry_counts':dict(cost),'seconds':time.perf_counter()-start,'scope':'New exact-content distinct anchor queries, D50/Q96 within2/3 historically researched scenes; eight direct/no-parameter arms, nativeyaw0. AlltargetGT only postseal. Micro and equal-scene both reported; fewcluster descriptive intervals.'}
 dump(a.output/'RESULTS_PRIVATE.json',out);dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in out.items() if k!='rows_PRIVATE'});dump(a.output/'SEALED.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_samples':146,'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'release_sha256':sha(a.release)});print(json.dumps({'n':146,'seconds':out['seconds']}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','source','masks','manifest','scoring-index','release','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
