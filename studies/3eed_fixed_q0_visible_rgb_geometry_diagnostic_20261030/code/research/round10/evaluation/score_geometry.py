"""R10 fixedq0 selected geometry scorer. Root post-all-prediction-seal release only.
No model/GPU; reads source observations/private target metadata only after gate.
"""
from pathlib import Path
import argparse,hashlib,json,time,traceback,collections
import numpy as np
from metrics import evaluate_case,summaries,eval_boxes
from independent_core_v2 import transform_gt
ROOT=Path(__file__).resolve().parents[3]
ARMS=('source','full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation')
ROLES={'source':'source','full':'full_rgb','no_rgb_coordinate':'no_rgb_coordinate','pointmask_ablated':'rgb_gated_pointmask_ablation'}
DEPS={'research/round5/audit/independent_core_v2.py':'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a','tools/obb_iou.py':'45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def bounded(root,name):
 p=(Path(root)/name).resolve()
 if not p.is_relative_to(Path(root).resolve()):raise ValueError('Path escapes locked root')
 return p
def gate(a):
 assert sha(a.release)==a.release_sha256;g=read(a.release)
 assert g.get('CPU_SCORE_RELEASE') is True and g.get('stage')=='R10_direct_geometry' and g.get('all_source_and_geometry_arms_sealed_before_GT') is True
 for k,p in [('prediction_seal_sha256',a.predictions/'SEALED.json'),('source_capture_seal_sha256',a.source/'SEALED.json'),('manifest_sha256',a.manifest),('scoring_index_sha256',a.scoring_index),('source_checkpoint_sha256',a.source_checkpoint),('scorer_program_sha256',Path(__file__)),('metrics_program_sha256',Path(__file__).with_name('metrics.py'))]:assert g[k]==sha(p),k
 for rel,h in DEPS.items():assert sha(ROOT/rel)==h
 ps=read(a.predictions/'SEALED.json');ss=read(a.source/'SEALED.json')
 assert ps['state']=='sealed_R10_fixedq0_direct_geometry_before_GT' and ps['n_arms']==4 and ps['n_samples']==ss['n_samples']
 assert ps['source_seal_sha256']==sha(a.source/'SEALED.json') and ps['manifest_sha256']==ss['manifest_sha256']==sha(a.manifest)
 assert ps['source_checkpoint_sha256']==ss['source_checkpoint_sha256']==g['source_checkpoint_sha256']
 assert ps['same_support_and_acceptance_full_noRGB'] and ps['all_queries_fixed_original_q0'] and ps['no_GT_or_parameter_update']
 for root,s in [(a.predictions,ps),(a.source,ss)]:
  for rel,h in s['files'].items():assert sha(bounded(root,rel))==h
 assert ps['program_sha256_base']=='workspace_root'
 for rel,h in ps['program_sha256'].items():assert sha(bounded(a.program_root,rel))==h
 sourceplan=read(a.source/'registered_plan.json')
 for rel,h in sourceplan['capture_program_sha256'].items():assert sha(bounded(a.program_root,rel))==h
 for k in ('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark'):assert ss['strict_FP32_flags'][k] is False
 return g,ps,ss

def prediction(path,sourcecache):
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 with np.load(sourcecache,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
 assert tuple(d['arm_names'].tolist())==ARMS and d['arm_boxes_raw6'].shape==(4,6) and d['acceptance_mask6'].shape==(4,6) and d['actual_changed_axes'].shape==(4,6)
 q=int(d['q_source']);assert q==int(c['query_index'])==int(c['text_scores'][0].argmax()) and np.array_equal(d['arm_query_index'],np.full(4,q))
 assert np.array_equal(d['source_boxes_raw6'],c['boxes_raw6'][0]) and np.array_equal(d['source_text_scores256'],c['text_scores'][0]) and np.array_equal(d['arm_boxes_raw6'][0],c['boxes_raw6'][0,q])
 for k in ('source_boxes_raw6','arm_boxes_raw6','pre_constraint_geometry6'):assert np.isfinite(d[k]).all()
 assert d['pre_constraint_geometry6'].shape==(4,6) and np.array_equal(d['pre_constraint_geometry6'][1],d['pre_constraint_geometry6'][2]) and d['provenance'].shape==(4,6) and d['point_support_sha256'].shape==(4,) and d['arm_status'].shape==(4,)
 assert np.array_equal(d['arm_boxes_eval6'],eval_boxes(d['arm_boxes_raw6'])) and np.array_equal(d['source_boxes_eval6'],eval_boxes(d['source_boxes_raw6']))
 axes=d['acceptance_mask6'].astype(bool);source=d['arm_boxes_raw6'][0];assert not axes[0].any() and np.array_equal(axes[1],axes[2]) and np.array_equal(axes[1],axes[3])
 assert d['point_support_sha256'][1]==d['point_support_sha256'][2]
 assert np.array_equal(d['actual_changed_axes'],d['arm_boxes_raw6']!=source) and np.array_equal(d['arm_boxes_raw6'][~axes],np.broadcast_to(source,(4,6))[~axes])
 for k in ('point_sample_sha256','raw_indices_sha256'):assert np.array_equal(d[k],c[k])
 assert d['source_cache_sha256'].tobytes().hex()==sha(sourcecache)
 assert d['RGB_mask2d_private'].dtype==bool and d['RGB_mask2d_private'].ndim==2 and d['RGB_mask_sha256'].tobytes()==hashlib.sha256(np.ascontiguousarray(d['RGB_mask2d_private']).tobytes()).digest()
 return d

def main(a):
 started=time.perf_counter();g,ps,ss=gate(a);manifest=read(a.manifest);assert len(manifest)==ps['n_samples'] and len({r['id'] for r in manifest})==len(manifest)
 assert {r['id']+'.npz' for r in manifest}=={p.name for p in a.predictions.glob('*.npz')}
 # Contract validation for ALL samples before opening any target metadata.
 preds={r['id']:prediction(a.predictions/(r['id']+'.npz'),a.source/'source_cache'/(r['id']+'.npz')) for r in manifest}
 a.output.mkdir(parents=True,exist_ok=False);dump(a.output/'SCORER_LOCK.json',{'state':'source_geometry_codes_and_all_prediction_contracts_checked_before_GT','release_sha256':sha(a.release),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'source_capture_seal_sha256':sha(a.source/'SEALED.json'),'scorer_sha256':sha(Path(__file__)),'metric_sha256':sha(Path(__file__).with_name('metrics.py')),'native_yaw0_noPCA_noGTselection':True})
 index={r['id']:r for r in read(a.scoring_index)};rows=[];cost=collections.Counter();meta_cache={};provenance=collections.defaultdict(collections.Counter)
 for sample in manifest:
  sid=sample['id'];ix=index[sid];assert ix['platform']==sample['platform'];mp=Path(ix['meta_path']);assert sha(mp)==ix['metadata_container_sha256']
  if str(mp) not in meta_cache:meta_cache[str(mp)]=read(mp)
  raw=meta_cache[str(mp)]['ground_info'][ix['object_index']]['bbox_3d'];target=transform_gt(raw,sample['platform'],ix.get('waymo_view_suffix'));d=preds[sid]
  row,c=evaluate_case(sid=sid,platform=sample['platform'],scene=ix['scene'],boxes_raw6=d['arm_boxes_raw6'],arm_names=ARMS,accepted_axes=d['acceptance_mask6'],target7=target,statuses=d['arm_status']);cost.update(c)
  for j,arm in enumerate(ARMS):
   for k,v in enumerate(d['provenance'][j].tolist()):provenance[sample['platform']+'|'+arm+'|axis'+str(k)][str(v)]+=1
  rows.append(row)
 summary,contrasts=summaries(rows,ROLES);out={'state':'R10_direct_geometry_scored_after_complete_seal','n':len(rows),'summary':summary,'paired_contrasts':contrasts,'coordinate_provenance_counts':{k:dict(v) for k,v in provenance.items()},'rows_PRIVATE':rows,'geometry_counts':dict(cost),'scope':'Originalq0 fixed, directgeometry only, nativeyaw0/noPCA. NoTTA/GTselection/oldidentitylabels. NewR10IDs excludeR9 but targetVAL prior-research-exposed. Full/noRGBcoordinate sameRGBpointmask+gate; maskablation inheritsRGBgate, notpurePC deployment.','scene_scope':'Only2Drone/3Quad effectiveclusters; bootstrapdescriptive, not160independentdomains.','seconds':time.perf_counter()-started};dump(a.output/'RESULTS_PRIVATE.json',out);dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in out.items() if k!='rows_PRIVATE'});dump(a.output/'SEALED.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_samples':len(rows),'prediction_seal_sha256':sha(a.predictions/'SEALED.json')});print(json.dumps({'n':len(rows),'seconds':out['seconds']}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','source','manifest','scoring-index','source-checkpoint','release','program-root','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);a=p.parse_args()
 try:main(a)
 except BaseException as e:
  a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output.parent/('SCORER_FAILURE_'+str(time.time_ns())+'.json'),{'state':'failed_no_silent_sample_omission','exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
