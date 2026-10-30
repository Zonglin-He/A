"""R11 post-all-seal scorer. Target metadata cannot open before release+contracts.
No model/GPU. Fixedq0, nativeyaw0, independent float64 general3D selected IoU.
"""
from pathlib import Path
import argparse,hashlib,json,collections,math,time,traceback
import numpy as np
from metrics import ARMS,GATES,CONTROL_TO_ARM,TIE_SALT,eval_boxes,evaluate_case,summarize
from discrimination import deterministic_select
from independent_core_v2 import transform_gt
ROOT=Path(__file__).resolve().parents[3]
CONTROLS=tuple(CONTROL_TO_ARM)
DEPS={'research/round5/audit/independent_core_v2.py':'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a','tools/obb_iou.py':'45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def bounded(root,name):
 p=(Path(root)/name).resolve()
 if not p.is_relative_to(Path(root).resolve()):raise ValueError('Path escapes locked root')
 return p

def verify_seal(root,seal):
 for name,digest in seal['files'].items():assert sha(bounded(root,name))==digest,name

def gate(a):
 assert sha(a.release)==a.release_sha256;g=read(a.release)
 assert g.get('CPU_SCORE_RELEASE') is True and g.get('stage')=='R11_reliability_assay'
 assert g.get('all_source_reliability_atomic_and_policy_arms_sealed_before_GT') is True
 required={'prediction_seal_sha256':a.predictions/'SEALED.json','source_capture_seal_sha256':a.source/'SEALED.json','manifest_sha256':a.manifest,'scoring_index_sha256':a.scoring_index,'source_checkpoint_sha256':a.source_checkpoint,'scorer_program_sha256':Path(__file__),'metrics_program_sha256':Path(__file__).with_name('metrics.py'),'discrimination_program_sha256':Path(__file__).with_name('discrimination.py')}
 for key,p in required.items():assert g[key]==sha(p),key
 for rel,h in DEPS.items():assert sha(ROOT/rel)==h,rel
 ps=read(a.predictions/'SEALED.json');ss=read(a.source/'SEALED.json')
 assert ps['state']=='sealed_R11_reliability11_and_atomic6_before_GT' and ss['state']=='sealed_R11_source_forward_and_input_observations_no_G_no_GT'
 assert ps['n_samples']==ss['n_samples']==320 and ps['n_arms']==11 and ps['atomic_arms_per_sample']==6
 assert ps['source_seal_sha256']==sha(a.source/'SEALED.json')
 assert ps['manifest_sha256']==ss['manifest_sha256']==g['manifest_sha256']
 assert ps['source_checkpoint_sha256']==ss['source_checkpoint_sha256']==g['source_checkpoint_sha256']
 assert ps['source_model_state_fingerprint']==ss['source_model_state_fingerprint']
 assert ps['all_queries_fixed_original_q0'] and ps['all_reliability_and_rank_weights_sealed_before_GT'] and not ps['neural_parameter_updates_executed']
 for field in ('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark'):assert ps['strict_FP32_flags'][field] is False and ss['strict_FP32_flags'][field] is False
 verify_seal(a.predictions,ps);verify_seal(a.source,ss)
 assert ps['program_sha256_base']=='workspace_root'
 for rel,h in ps['program_sha256'].items():assert sha(bounded(a.program_root,rel))==h,rel
 sp=read(a.source/'registered_plan.json')
 for rel,h in sp['capture_program_sha256'].items():assert sha(bounded(a.program_root,rel))==h,rel
 return g,ps,ss

def prediction(path,sourcecache):
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 with np.load(sourcecache,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
 assert tuple(d['arm_names'].tolist())==ARMS and tuple(d['control_score_names'].tolist())==CONTROLS
 assert d['source_boxes_raw6'].shape==(256,6) and d['source_text_scores256'].shape==(256,)
 q=int(d['q_source']);source=d['source_boxes_raw6'][q];full=d['arm_boxes_raw6'][1]
 assert q==int(c['query_index'])==int(c['text_scores'][0].argmax()) and np.array_equal(d['arm_query_index'],np.full(11,q))
 assert np.array_equal(d['source_boxes_raw6'],c['boxes_raw6'][0]) and np.array_equal(d['source_text_scores256'],c['text_scores'][0])
 assert d['arm_boxes_raw6'].shape==(11,6) and d['atomic_boxes_raw6'].shape==(6,6) and np.array_equal(d['arm_boxes_raw6'][0],source)
 assert all(np.isfinite(d[k]).all() for k in ('source_boxes_raw6','source_text_scores256','arm_boxes_raw6','atomic_boxes_raw6','coordinate_reliability6','RGB_reliability','case_reliability','control_scores','face_weights'))
 assert np.array_equal(d['arm_boxes_eval6'],eval_boxes(d['arm_boxes_raw6'])) and np.array_equal(d['atomic_boxes_eval6'],eval_boxes(d['atomic_boxes_raw6']))
 expected=np.tile(source,(6,1));expected[np.arange(6),np.arange(6)]=full
 assert np.array_equal(d['atomic_boxes_raw6'],expected) and np.array_equal(d['atomic_active6'],full!=source)
 active=d['atomic_active6'].astype(bool);accepted=d['base_acceptance_mask6'].astype(bool)
 assert active.shape==accepted.shape==(6,) and not (active&~accepted).any()
 assert d['coordinate_reliability6'].shape==d['coordinate_defined6'].shape==(6,) and d['face_weights'].shape==(6,)
 assert d['control_scores'].shape==d['control_defined'].shape==d['gate_selection'].shape==(6,)
 for k in ('coordinate_reliability6','RGB_reliability','case_reliability','face_weights'):assert ((d[k]>=0)&(d[k]<=1)).all()
 for j,keep in enumerate(d['gate_selection']):assert np.array_equal(d['arm_boxes_raw6'][j+5],full if keep else source)
 if not active.any():assert np.array_equal(d['arm_boxes_raw6'][3:5],np.tile(source,(2,1)))
 status=str(d['association_status']);assert status in ('association_supported','association_conflicted','association_undefined')
 if status!='association_supported':assert not d['face_weights'].any() and float(d['RGB_reliability'])==0
 if status=='association_undefined':assert not d['coordinate_defined6'].any() and not bool(d['RGB_defined']) and not bool(d['case_defined'])
 if status=='association_conflicted' and active.any():assert bool(d['case_defined']) and float(d['case_reliability'])==0
 defined=bool(active.any() and d['coordinate_defined6'][active].all())
 assert bool(d['case_defined'])==defined
 assert float(d['case_reliability'])==(float(d['coordinate_reliability6'][active].min()) if defined else 0.)
 assert bool(d['control_defined'][0])==bool(d['case_defined']) and float(d['control_scores'][0])==float(d['case_reliability'])
 for key in ('point_sample_sha256','raw_indices_sha256'):assert np.array_equal(d[key],c[key])
 assert d['source_cache_sha256'].tobytes().hex()==sha(sourcecache)
 assert d['RGB_mask2d_private'].dtype==bool and d['RGB_mask2d_private'].ndim==2
 return d

def validate_cohort(preds,manifest,control,trace):
 assert set(preds)=={r['id'] for r in manifest}=={r['id'] for r in trace}
 lookup={r['id']:r for r in trace}
 means={}
 for platform in ('drone','quad'):
  pools={'face':[],'RGB':[]};members=[r for r in manifest if r['platform']==platform]
  for sample in members:
   sid=sample['id'];d=preds[sid];r=lookup[sid]['reliability'];eligible=bool(d['atomic_active6'].any())
   assert lookup[sid]['base_actual_change']==eligible
   assert np.array_equal(np.asarray(r['axis_reliability6']),d['coordinate_reliability6']) and np.array_equal(np.asarray(r['axis_reliability_defined6']),d['coordinate_defined6'])
   for key in r['active_face_keys']:
    defined=r['association_status']=='association_conflicted' or (r['association_status']=='association_supported' and r['features']['faces'][key]['defined'])
    if defined:pools['face'].append(float(r['face_weights'][key]))
   if eligible and r['RGB_weight_defined']:pools['RGB'].append(float(r['RGB_weight']))
  means[platform]={key:{'n_defined_active':len(v),'mean':math.fsum(v)/len(v) if v else None,'defined_zero_count':sum(x==0 for x in v)} for key,v in pools.items()}
  ids=[r['id'] for r in members if preds[r['id']]['atomic_active6'].any()]
  for j,name in enumerate(CONTROLS):
   score={sid:0. if name=='hash_random' else float(preds[sid]['control_scores'][j]) for sid in ids};defined={sid:True if name=='hash_random' else bool(preds[sid]['control_defined'][j]) for sid in ids}
   selected=deterministic_select(ids,.5,score,defined,salt=TIE_SALT,platform=platform)
   actual={r['id'] for r in members if preds[r['id']]['gate_selection'][j]}
   assert set(selected['selected'])==actual and len(actual)==(len(ids)+1)//2
   assert set(control['selection'][name]['selected_ids'])=={sid for sid,p in preds.items() if p['gate_selection'][j]}
 assert means==control['uniform_means_perplatform']
 return means

def main(a):
 started=time.perf_counter();release,ps,ss=gate(a);manifest=read(a.manifest)
 assert len(manifest)==320 and len({r['id'] for r in manifest})==320
 assert {r['id']+'.npz' for r in manifest}=={p.name for p in a.predictions.glob('*.npz')}
 preds={r['id']:prediction(a.predictions/(r['id']+'.npz'),a.source/'source_cache'/(r['id']+'.npz')) for r in manifest}
 controls=read(a.predictions/'COHORT_CONTROL_LOCK.json');trace=read(a.predictions/'trace.json');means=validate_cohort(preds,manifest,controls,trace)
 bridges=read(a.predictions/'ACTUAL_WEIGHT_BRIDGES.json');assert {r['id'] for r in bridges}=={sid for sid,p in preds.items() if p['atomic_active6'].any()}
 assert all(r['allone_base_bitexact'] and r['allzero_source_bitexact'] for r in bridges)
 assert controls['hash_random_reference']['salt']=='R11_random_reference_v1' and controls['hash_random_reference']['seed']==0 and controls['hash_random_reference']['draws']==1000
 a.output.mkdir(parents=True,exist_ok=False)
 dump(a.output/'SCORER_LOCK.json',{'state':'all_input_hashes_programs_and320_prediction_contracts_checked_before_GT','release_sha256':sha(a.release),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'source_capture_seal_sha256':sha(a.source/'SEALED.json'),'scorer_sha256':sha(Path(__file__)),'metrics_sha256':sha(Path(__file__).with_name('metrics.py')),'discrimination_sha256':sha(Path(__file__).with_name('discrimination.py')),'native_yaw0_noPCA_no_GTselection':True})
 # First access to annotation index and ground_info is after all contracts above.
 index={r['id']:r for r in read(a.scoring_index)};meta={};rows=[];cost=collections.Counter()
 for sample in manifest:
  sid=sample['id'];ix=index[sid];assert ix['platform']==sample['platform'];mp=Path(ix['meta_path']);assert sha(mp)==ix['metadata_container_sha256']
  if str(mp) not in meta:meta[str(mp)]=read(mp)
  raw=meta[str(mp)]['ground_info'][ix['object_index']]['bbox_3d'];target=transform_gt(raw,sample['platform'],ix.get('waymo_view_suffix'));d=preds[sid]
  row,c=evaluate_case(sid=sid,platform=sample['platform'],scene=ix['scene'],boxes_raw6=d['arm_boxes_raw6'],atomic_boxes_raw6=d['atomic_boxes_raw6'],target7=target,statuses=d['arm_status'],atomic_active6=d['atomic_active6'],coordinate_scores6=d['coordinate_reliability6'],coordinate_defined6=d['coordinate_defined6'],RGB_score=d['RGB_reliability'],RGB_defined=d['RGB_defined'],case_score=d['case_reliability'],case_defined=d['case_defined'],gate_selected6=d['gate_selection'],control_names=d['control_score_names'],control_scores=d['control_scores'],control_defined=d['control_defined']);cost.update(c);rows.append(row)
 result={'state':'R11_reliability_scored_after_all_arm_seal','n':len(rows),**summarize(rows),'rows_PRIVATE':rows,'uniform_means_input_only':means,'geometry_counts':dict(cost),'seconds':time.perf_counter()-started}
 dump(a.output/'RESULTS_PRIVATE.json',result);dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in result.items() if k!='rows_PRIVATE'})
 dump(a.output/'SEALED.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_samples':len(rows),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'source_capture_seal_sha256':sha(a.source/'SEALED.json'),'release_sha256':sha(a.release)})
 print(json.dumps({'n':len(rows),'seconds':result['seconds']}))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('predictions','source','manifest','scoring-index','source-checkpoint','release','program-root','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);args=p.parse_args()
 try:main(args)
 except BaseException as error:
  args.output.parent.mkdir(parents=True,exist_ok=True);dump(args.output.parent/('SCORER_FAILURE_'+str(time.time_ns())+'.json'),{'state':'failed_no_silent_sample_omission','exception':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
