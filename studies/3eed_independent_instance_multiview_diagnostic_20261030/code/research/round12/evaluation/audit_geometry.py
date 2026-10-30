"""Independent R12 actual-geometry audit. No labels/index/model/GPU API.
Run only after root releases actual sealed G output; never poll/inprogress read.
"""
from pathlib import Path
import argparse,json,hashlib,collections,math,re,time
import numpy as np
ARMS=('source','grabcut_direct','single_instance_direct','multiview_instance_direct','wrongview_instance_direct','single_instance_half','multiview_instance_half','multiview_maskmatch_only_direct')
PRED_KEYS={'sample_id','source_boxes_raw6','source_text_scores256','q_source','arm_names','arm_query_index','arm_status','arm_boxes_raw6','arm_boxes_eval6','acceptance_mask6','actual_changed_axes','point_support_sha256','provenance','point_sample_sha256','raw_indices_sha256','source_cache_sha256','anchor_image_file_sha256','anchor_raw_point_file_sha256'}
ROOT=Path(__file__).resolve().parents[3]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bounded(base,name):
 p=(base/name).resolve();assert p.is_relative_to(base.resolve());return p
def verify(base,seal):
 for n,h in seal['files'].items():assert sha(bounded(base,n))==h,n

def check_prediction(d,c,sid):
 q=int(c['query_index']);assert q==int(d['q_source'])==int(c['text_scores'][0].argmax())
 assert str(d['sample_id'])==sid and str(c['sample_id'])==sid
 assert tuple(d['arm_names'].tolist())==ARMS and np.array_equal(d['arm_query_index'],np.full(8,q))
 b=d['arm_boxes_raw6'];a=d['acceptance_mask6'];assert b.shape==a.shape==(8,6) and a.dtype==bool and np.isfinite(b).all() and d['arm_status'].shape==(8,)
 assert np.array_equal(d['source_boxes_raw6'],c['boxes_raw6'][0]) and np.array_equal(d['source_text_scores256'],c['text_scores'][0]) and np.array_equal(b[0],c['boxes_raw6'][0,q])
 assert np.isfinite(d['source_boxes_raw6']).all() and np.isfinite(d['source_text_scores256']).all()
 e=b.copy();e[:,3:]=np.maximum(e[:,3:],np.asarray(1e-6,e.dtype));assert np.array_equal(d['arm_boxes_eval6'],e.astype(float))
 assert d['actual_changed_axes'].dtype==bool and np.array_equal(d['actual_changed_axes'],b!=b[0]) and not a[0].any()
 assert np.array_equal(b[~a],np.broadcast_to(b[0],b.shape)[~a])
 assert np.array_equal(b[5],b[0]+np.float32(.5)*(b[2]-b[0])) and np.array_equal(b[6],b[0]+np.float32(.5)*(b[3]-b[0]))
 assert np.array_equal(a[5],a[2]) and np.array_equal(a[6],a[3])
 assert d['point_support_sha256'].shape==(8,) and d['provenance'].shape==(8,6)
 assert all(re.fullmatch('[0-9a-f]{64}',str(x)) for x in d['point_support_sha256'])
 for k in ('point_sample_sha256','raw_indices_sha256'):assert np.array_equal(d[k],c[k])
 return b,a

def check_multiview(diag,b,a):
 counts=collections.Counter();mv=diag.get('multiview')
 if mv is None:return counts
 attempts=diag.get('neighbor_attempts',[]);assert mv['neighbor_n']==len(attempts)
 verified=sum(v['verified'] for v in attempts);veto=any(v['contradiction'] for v in attempts)
 assert mv['verified_neighbor_n']==verified and mv['contradiction_veto']==veto
 assert mv['point_corroboration_thresholds']=={'nearest_anchor_pixel_px':3.,'camera_depth_difference_m':.25,'raw_3D_distance_m':.25}
 counts['attempted_neighbor_views']=len(attempts);counts['verified_neighbor_views']=verified;counts['static_contradiction_cases']=int(veto)
 for v in attempts:
  counts['neighbor_status|'+v['status']]+=1
  assert v.get('clock_unverified_no_velocity',False)
  if 'registration' in v:
   r=v['registration'];assert not r['provided_pose_used_for_fit_or_selection'];counts['registration_status|'+r['status']]+=1
   if r['accepted']:assert r['observation_diagnostics']['background_only_qualification']
  if v['verified']:
   assert v['registration']['accepted'] and v['target_depthbound_unique_pairs']>=6 and v['static_support']['accepted'] and not v['contradiction']
 if veto or not verified:
  assert np.array_equal(b[[3,6,7]],np.tile(b[0],(3,1))) and not a[[3,6,7]].any();return counts
 assert 'multiview_geometry' in diag and 'unfiltered_multiview_geometry' in diag
 main=diag['multiview_geometry'];unfiltered=diag['unfiltered_multiview_geometry']
 assert main['diagnostics']['canonical_allowed_model_mapping']['applied'] is True and unfiltered['diagnostics']['canonical_allowed_model_mapping']['applied'] is False
 assert np.array_equal(b[3],np.asarray(main['arm_boxes_raw6'],np.float32)[1]) and np.array_equal(b[7],np.asarray(unfiltered['arm_boxes_raw6'],np.float32)[1])
 assert np.array_equal(a[3],np.asarray(main['acceptance_mask6'],bool)[1]) and np.array_equal(a[7],np.asarray(unfiltered['acceptance_mask6'],bool)[1])
 assert mv['unfiltered_ablation_has_same_instance_static_gate_and_views'] and mv['anchor_mask_unchanged']
 qualified=0
 for v in mv['neighbors']:
  chain=[v['visible_instance_points_before_transform'],v['anchor_FOV'],v['anchor_mask'],v['anchor_mask_and_source_ROI'],v['pixel_within3px'],v['pixel_and_depth_within_threshold'],v['fully_corroborated_neighbor_points']]
  assert all(isinstance(x,int) and x>=0 for x in chain) and all(x>=y for x,y in zip(chain,chain[1:]))
  ids=np.asarray(v['matched_anchor_raw_ids_unique'],np.int64);assert np.array_equal(ids,np.unique(ids)) and len(ids)<=chain[-1]
  assert hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()==v['matched_anchor_raw_ids_sha256'];qualified+=chain[-1]
  for name in ('unfiltered_added_raw_support_sha256','corroborated_added_raw_support_sha256','corroborating_anchor_raw_XYZ_sha256'):assert re.fullmatch('[0-9a-f]{64}',v[name])
 counts['qualified_neighbor_point_measurements']=qualified;counts['new_unique_neighbor_support']=mv['new_raw_unique_count'];counts['unfiltered_new_unique_neighbor_support']=mv['unfiltered_new_raw_unique_count']
 assert len(mv['neighbors'])==verified and mv['new_raw_unique_count']>=0 and mv['unfiltered_new_raw_unique_count']>=mv['new_raw_unique_count']
 assert 0<=mv['allowed_corroborated_raw_XYZ_count']<=2*qualified
 if not qualified:
  assert np.array_equal(b[3],b[0]) and not a[3].any();counts['verified_track_but_zero_corroborated_points_cases']+=1
 return counts

def main(args):
 start=time.perf_counter();p=args.predictions;s=args.source;m=args.masks;ps=read(p/'SEALED.json');ss=read(s/'SEALED.json');ms=read(m/'SEALED.json');plan=read(p/'registered_plan.json')
 assert ps['state']=='sealed_R12_8arm_independent_instance_multiview_direct_before_GT' and tuple(ps['arm_names'])==ARMS and ps['n_samples']==146 and ps['n_arms']==8
 assert ss['state']=='sealed_R12_source_forward_and_input_observations_no_G_no_GT' and ss['n_samples']==146 and ss['source_unchanged'] and ss['source_repeat_bitexact']
 assert ms['state']=='sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT' and ms['frame_n']==501 and not ms['trained_or_updated'] and ms['mask_returned_original_image_pixels']
 for seal in (ss,ms):assert seal['strict_FP32_flags']=={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
 assert ps['source_seal_sha256']==sha(s/'SEALED.json') and ps['mask_seal_sha256']==sha(m/'SEALED.json') and ps['plan_sha256']==sha(p/'registered_plan.json')
 assert ps['source_checkpoint_sha256']==ss['source_checkpoint_sha256']==sha(args.source_checkpoint)=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 assert ps['mask_weights_sha256']==ms['weights_sha256']=='73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e'
 assert ps['all_queries_fixed_original_q0'] and not ps['GT_model_GPU_or_neural_update_executed'] and not ps['provided_pose_used_for_registration_or_target_fusion']
 for base,seal in ((p,ps),(s,ss),(m,ms)):verify(base,seal)
 assert ps['program_sha256_base']=='workspace_root' and len(ps['program_sha256'])==9 and ps['program_sha256']==plan['program_sha256']
 for rel,h in ps['program_sha256'].items():assert sha(bounded(ROOT,rel))==h
 inputs={}
 for key in ('manifest','pairmap','frame_index'):
  path=Path(plan[key]);assert sha(path)==plan[key+'_sha256'];inputs[key]=read(path)
 assert plan['stage']=='multiview_direct_geometry' and not plan['GT_read_permitted'] and not plan['model_or_GPU_execution_permitted'] and plan['expected_samples']==146
 assert ps['manifest_sha256']==ss['manifest_sha256']==plan['manifest_sha256'] and ps['pairmap_sha256']==plan['pairmap_sha256'] and ps['frame_index_sha256']==plan['frame_index_sha256']
 assert plan['feedback_config']['registration_backend']=='one_depth_sensor_PnP_v1'
 rows=inputs['manifest'];index=inputs['frame_index'];pair={x['anchor_id_PRIVATE']:x for x in inputs['pairmap']['rows_PRIVATE']};assert len(rows)==len(pair)==146 and collections.Counter(x['platform'] for x in rows)=={'drone':50,'quad':96}
 assert all(re.fullmatch('input_frame_[0-9]{5}',x) for x in index) and len(index)==501
 trace=read(p/'trace.json');lines=[json.loads(x) for x in (p/'trace.jsonl').read_text().splitlines()];assert trace==lines and len(trace)==146;tr={x['id']:x for x in trace};assert len(tr)==146
 assert {x['id']+'.npz' for x in rows}=={x.name for x in p.glob('*.npz')}
 counts={platform:collections.Counter() for platform in ('drone','quad')};solver=collections.Counter()
 for row in rows:
  sid=row['id'];cp=s/'source_cache'/(sid+'.npz')
  with np.load(cp,allow_pickle=False) as z:c={k:z[k].copy() for k in ('boxes_raw6','text_scores','query_index','sample_id','point_sample_sha256','raw_indices_sha256')}
  with np.load(p/(sid+'.npz'),allow_pickle=False) as z:
   assert set(z.files)==PRED_KEYS
   d={k:z[k].copy() for k in z.files}
  b,a=check_prediction(d,c,sid);assert d['source_cache_sha256'].tobytes().hex()==sha(cp)
  x=tr[sid];pi=pair[sid];alias=pi['anchor_frame_alias'];fr=index[alias];C=counts[row['platform']];C['n']+=1
  assert x['platform']==row['platform']==fr['platform'] and x['scene_PRIVATE']==row['scene']==fr['scene_PRIVATE'] and x['anchor_frame_alias']==alias and x['q_source']==int(d['q_source'])
  assert all(fr[k]==row[k] for k in ('pointpath','imagepath','metapath')) and d['anchor_image_file_sha256'].tobytes().hex()==fr['image_sha256'] and d['anchor_raw_point_file_sha256'].tobytes().hex()==fr['point_sha256']
  assert x['source_cache_sha256']==sha(cp) and x['provided_pose_not_primary_input'] and x['direct_no_parameter_update'] and x['geometry_uses_raw_full_points_not_source_resampling']
  for arm,status,axis in zip(ARMS,d['arm_status'].tolist(),d['actual_changed_axes']):C['status|'+arm+'|'+status]+=1;C['changed|'+arm]+=int(axis.any());C['accepted|'+arm]+=int(a[ARMS.index(arm)].any())
  diag=x['diagnostics'];assert diag['q0_fixed'] and not diag['neural_parameter_updates'] and diag['independent_frontend_no_q0_seed'];C['target_noun_scope|'+diag.get('target_noun',{}).get('scope','source_uncomputed')]+=1;C['anchor_association|'+diag.get('anchor_association',{}).get('status','not_attempted')]+=1
  C.update(check_multiview(diag,b,a))
  # Role labels can overwrite shallow solver statuses. Preserve nested numerical states.
  for key in ('grabcut','single_geometry','multiview_geometry','unfiltered_multiview_geometry','wrongview_geometry'):
   g=diag.get(key);g=g.get('geometry') if key=='grabcut' and isinstance(g,dict) else g
   if g:
    st=g['diagnostics'].get('solver',{}).get('full_rgb',{}).get('status',g['arm_status'][1]);solver[row['platform']+'|'+key+'|'+st]+=1
 out={'state':'PASS_R12_ACTUAL146_8ARM_UNLABELLED_AUDIT','n':146,'prediction_seal_sha256':sha(p/'SEALED.json'),'source_seal_sha256':sha(s/'SEALED.json'),'mask_seal_sha256':sha(m/'SEALED.json'),'manifest_sha256':ps['manifest_sha256'],'pairmap_sha256':ps['pairmap_sha256'],'frame_index_sha256':ps['frame_index_sha256'],'plan_sha256':ps['plan_sha256'],'auditor_sha256':sha(Path(__file__)),'all_source_mask_prediction_files_and9programdeps_rehashed':True,'fixedq0_half_acceptance_sourcealiases_trace_provenance_counts_checked':True,'platforms':{k:dict(v) for k,v in counts.items()},'nested_numeric_solver_states':dict(solver),'scope':'Counts/provenance only, not semantic identity/geometry correctness/task benefit. Source/model-helper stages ran earlier; no model executes in this audit.','no_GT_index_annotations_pixels_pointarrays_model_GPU_or_newIoU_reads':True,'seconds':time.perf_counter()-start}
 args.output.parent.mkdir(parents=True,exist_ok=True);assert not args.output.exists();args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':out['state'],'sha256':sha(args.output)}))
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for key in ('predictions','source','masks','source-checkpoint','output'):parser.add_argument('--'+key,type=Path,required=True)
 main(parser.parse_args())
