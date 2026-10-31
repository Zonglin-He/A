"""Actual R13 B no-label contract and support replay; never optimizes G."""
from pathlib import Path
import collections,hashlib,json,sys
import numpy as np
B=Path(__file__).resolve().parents[3];V=B/'research/round13/evaluation';G=B/'crossplatform/round13/geometry116_attempt001';I=B/'research/round13/input'
sys.path.insert(0,str(B/'crossplatform/round7'))
from camera_projection import CameraSpec,project_model_points,raw_to_model_points

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
checks=0;boundfiles=0

def ck(x):
 global checks
 assert x;checks+=1

def verify(folder):
 global boundfiles
 s=read(folder/'SEALED.json')
 for rel,h in s['files'].items():
  p=(folder/rel).resolve();ck(p.is_relative_to(folder.resolve()) and sha(p)==h);boundfiles+=1
 return s
seal=verify(G);ck(sha(G/'SEALED.json')=='963e26389844d2aa90548be83dd6f11132a3b2c12595c3544451d0f238fb96ca')
p=read(G/'registered_plan.json');ck(sha(G/'registered_plan.json')==seal['plan_sha256']);ck(p['stage']=='R13_fixed_q0_partial_positive_direct' and p['GT_permitted'] is False)
for rel,h in p['program_sha256'].items():ck(sha(B/rel)==h)
ck(p['program_sha256']==seal['program_sha256'])
for key in ('manifest','cases','frame_index'):ck(sha(p[key])==p[key+'_sha256'])
for folder,h in p['capture_seals'].items():ck(sha(B/folder/'SEALED.json')==h);verify(B/folder)
rows=read(p['manifest']);cases=read(p['cases']);frames=read(p['frame_index']);tr=read(G/'TRACE_PRIVATE.json');ck(len(rows)==len(cases)==len(tr)==116)
ck(seal['n']==116 and seal['new_n']==108 and seal['regression_n']==8 and seal['all_q_fixed'] and seal['no_neural_update_or_GPU_or_GT'])
ARMS=('source','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality');ck(tuple(seal['arm_names'])==ARMS)
ck({z.name for z in G.glob('*.npz')}=={r['id']+'.npz' for r in rows})
ck(set(p['capture_seals'])=={c['source_dir'] for c in cases}|{c['mask_dir'] for c in cases})
keys={'sample_id','case_alias','cohort','source_boxes_raw6','source_text_scores256','q_source','arm_names','arm_query_index','arm_boxes_raw6','arm_boxes_eval6','arm_status','active_coordinate_mask','actual_changed_axes','B_support_raw_ids','B_support_model_points','selected_mask_index','source_cache_sha256','mask_cache_sha256'}
counts=collections.defaultdict(collections.Counter);hist=collections.defaultdict(collections.Counter);supporthist=collections.defaultdict(collections.Counter);rawfiles=set();maxboundcast=0.;max_point_term_error=0.;oldsame=0
for r,c,t in zip(rows,cases,tr):
 ck(r['id']==c['id']==t['id'] and r['platform']==c['platform']==t['platform'] and r['scene']==c['scene_PRIVATE']==t['scene_PRIVATE']);grp=c['role']+'|'+r['platform'];counts[grp]['n']+=1
 frame=frames[c['frame_alias']];cp=B/c['source_dir']/'source_cache'/(r['id']+'.npz');mp=B/c['mask_dir']/(c['frame_alias']+'.npz');ck(sha(cp)==t['bindings']['source_cache_sha256'] and sha(mp)==t['bindings']['mask_cache_sha256'])
 with np.load(cp,allow_pickle=False) as f:cache={k:f[k].copy() for k in ['boxes_raw6','text_scores','query_index']}
 with np.load(G/(r['id']+'.npz'),allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
 ck(set(z)==keys and str(z['sample_id'])==r['id'] and str(z['case_alias'])==c['case_alias'] and str(z['cohort'])==c['role']);q=int(z['q_source']);ck(q==int(cache['query_index'])==int(cache['text_scores'][0].argmax())==t['q_source'])
 src=cache['boxes_raw6'][0,q];box=z['arm_boxes_raw6'];active=z['active_coordinate_mask'];ck(np.array_equal(z['source_boxes_raw6'],cache['boxes_raw6'][0]) and np.array_equal(z['source_text_scores256'],cache['text_scores'][0]));ck(tuple(z['arm_names'])==ARMS and np.array_equal(z['arm_query_index'],np.full(5,q)))
 ck(box.shape==(5,6) and box.dtype==np.float32 and np.isfinite(box).all() and np.array_equal(box[0],src));ev=box.copy();ev[:,3:]=np.maximum(ev[:,3:],np.float32(1e-6));ck(z['arm_boxes_eval6'].dtype==np.float32 and np.array_equal(ev,z['arm_boxes_eval6']))
 ck(active.shape==(5,6) and active.dtype==bool and not active[0].any() and np.array_equal(active[1],active[2]) and np.array_equal(active[1],active[3]));ck(np.array_equal(z['actual_changed_axes'],box!=src));ck(np.array_equal(box[~active],np.broadcast_to(src,(5,6))[~active]));ck(np.array_equal(box[3],src+np.float32(.5)*(box[1]-src)))
 ck(str(z['source_cache_sha256'])==sha(cp) and str(z['mask_cache_sha256'])==sha(mp));di=t['inequality'];sp=t['support'];ck(di['fixed_q0']==q and di['same_support_gate_active_bounds_full_noRGB'] and sp['q0']==q)
 ck(di['config']==p['observation_config']);ck(sp['diagnostics']['config']==p['support_config']);supporthist[grp][sp['status']]+=1
 for j,a in enumerate(ARMS):
  counts[grp][a+'|changed']+=bool(z['actual_changed_axes'][j].any());counts[grp][a+'|center_changed']+=bool(z['actual_changed_axes'][j,:3].any());counts[grp][a+'|size_changed']+=bool(z['actual_changed_axes'][j,3:].any());counts[grp][a+'|active']+=bool(active[j].any());hist[grp+'|'+a][str(z['arm_status'][j])]+=1
 # Only rawpoint XYZ and sealed inputcalibration are read. Metadata labels and RGB pixels never opened.
 pp=Path(r['pointpath']);ck(sha(pp)==frame['point_sha256']==t['bindings']['point_sha256']);rawfiles.add(str(pp));raw=np.load(pp,allow_pickle=False) if pp.suffix=='.npy' else np.fromfile(pp,np.float32).reshape(-1,4);raw=np.ascontiguousarray(raw[:,:3],np.float32);ck(raw.ndim==2 and raw.shape[1]==3 and np.isfinite(raw).all())
 ids=z['B_support_raw_ids'];model=z['B_support_model_points'];ck(ids.ndim==1 and ids.dtype==np.int64 and model.shape==(len(ids),3) and model.dtype==np.float32 and np.isfinite(model).all());ck((ids>=0).all() and (ids<len(raw)).all() and len(np.unique(ids))==len(ids));ck(np.array_equal(ids,np.asarray(sp['physical_raw_ids'],np.int64)))
 replay=raw[ids].copy()
 if r['platform']=='drone':replay[:,2]+=np.float32(1.8)
 else:ck(r['platform']=='quad')
 ck(np.array_equal(model,replay));ck(np.array_equal(model,np.unique(model,axis=0)));ck(di['dedup']['input_rows']==di['dedup']['unique_float32_XYZ']==len(model))
 ck(int(z['selected_mask_index'])==(-1 if sp['selected_mask_index'] is None else sp['selected_mask_index']))
 if sp['segmentation_valid']:
  counts[grp]['unique_mask_association']+=1;counts[grp]['support_points']+=len(model)
  ck(ah(model)==sp['diagnostics']['support_model_XYZ_sha256']);ck(ah(ids)==sp['diagnostics']['support_original_raw_ids_sha256']);ck(len(model)==sp['diagnostics']['selected_unique_model_XYZ'])
  with np.load(mp,allow_pickle=False) as f:mask=f['masks_bool'][int(z['selected_mask_index'])].copy();hw=f['original_hw'].copy()
  ck(mask.any() and ah(mask)==sp['diagnostics']['mask_sha256']);fields={k:frame[k] for k in ('image_intrinsic','image_extrinsic')}
  if frame['camera_distortion'] is not None:fields['camera_distortion']=frame['camera_distortion']
  cam=CameraSpec.from_camera_fields(r['platform'],fields,missing_distortion_policy='assume_zero_declared');H,W=mask.shape;ck(np.array_equal(hw,[H,W]))
  # Independent whole-cloud nearest-return zbuffer replay, before instance mask.
  uraw,first=np.unique(raw.astype(float),axis=0,return_index=True);projection=project_model_points(raw_to_model_points(uraw,r['platform']),cam,(W,H),min_depth=.1);ok=np.flatnonzero(projection['in_image']);uv=np.floor(projection['uv'][ok]).astype(np.int64);key=uv[:,1]*W+uv[:,0];order=np.lexsort((uraw[ok,2],uraw[ok,1],uraw[ok,0],projection['depth'][ok],key));ok=ok[order];key=key[order];ok=ok[np.r_[True,key[1:]!=key[:-1]]] if len(ok) else ok;vis=first[ok].astype(np.int64);ck(np.array_equal(vis,np.asarray(sp['visible_raw_ids'],np.int64)))
  prem=np.asarray(sp['support_raw_ids_before_model_XYZ_merge'],np.int64);ck(np.isin(prem,vis).all() and np.isin(ids,prem).all());vmodel=raw[prem].copy()
  if r['platform']=='drone':vmodel[:,2]+=np.float32(1.8)
  ck(np.array_equal(np.unique(vmodel,axis=0),model));pr=project_model_points(raw_to_model_points(raw[prem].astype(float),r['platform']),cam,(W,H),min_depth=.1);pix=np.floor(pr['uv']).astype(int);ck(pr['in_image'].all() and mask[pix[:,1],pix[:,0]].all())
  margin=np.minimum(.25*src[3:]+.1,.75);ck((np.abs(vmodel-src[:3])<=src[3:]/2+margin).all())
 else:ck(len(ids)==0 and not di['gate_passed'])
 if di['gate_passed']:
  counts[grp]['B_gate_passed']+=1;initial=di['initial_terms'];av=initial['point_containment']>0 or initial['RGB_positive_containment']>1e-16;ck(np.array_equal(active[1],np.full(6,av)))
  sol=di['solver'];ck(sol['direct_full']['initial_terms']==sol['direct_no_rgb']['initial_terms']==initial)
  for j,a in [(1,'direct_full'),(2,'direct_no_rgb')]:
   dx=box[j,:3].astype(float)-src[:3].astype(float);lim=np.minimum(.25*src[3:].astype(float),.5);tol=float(2*np.spacing(np.abs(src)).max()+1e-6);maxboundcast=max(maxboundcast,float(np.maximum(np.abs(dx)-lim,0).max()));ck((np.abs(dx)<=lim+tol).all());ck((box[j,3:]>=src[3:]).all() and (box[j,3:].astype(float)<=1.2*src[3:].astype(float)+tol).all())
   sd=sol[a];outside=np.maximum(np.abs(model.astype(float)-box[j,:3])-box[j,3:]/2-.03,0);v=outside/.05;loss=float(np.where(v<=1,.5*v*v,v-.5).mean());err=abs(loss-sd['final_terms']['point_containment']);max_point_term_error=max(max_point_term_error,err);ck(err<1e-9)
   ck(str(z['arm_status'][j])==sd['status']);ck(sd['status'] in ('source_already_satisfies_positive_constraints','converged_bounded_result','finite_nonconverged_bounded_result','numerical_invalid_source_fallback'))
   counts[grp][a+'|finite_nonconverged']+=bool(sd.get('finite_nonconverged_retained',False));counts[grp][a+'|numerical_invalid']+=bool(sd.get('numerical_invalid',False))
  if initial['point_containment']==0:ck(np.array_equal(box[2],src))
 else:ck(np.array_equal(box[:4],np.tile(src,(4,1))) and not active[:4].any())
 old=t['old_surface']
 if old is not None:ck(np.array_equal(box[4],np.asarray(old['arm_boxes_raw6'],np.float32)[1]) and np.array_equal(active[4],np.asarray(old['acceptance_mask6'],bool)[1]));oldsame+=1
result={'state':'PASS_R13_ACTUAL116_FIXEDQ0_GEOMETRY_UNLABELLED_AUDIT','prediction_seal_sha256':sha(G/'SEALED.json'),'plan_sha256':sha(G/'registered_plan.json'),'auditor_sha256':sha(Path(__file__)),'checks':checks,'bound_files_checked_including_captures':boundfiles,'n':116,'new_n':108,'regression_n':8,'raw_pointfiles_XYZ_read_without_GT_metadata_or_RGB_decode':len(rawfiles),'original_rawID_to_float32_model_exact_replay_all':True,'visible_zbuffer_and_mask_ROI_replay_on_all_unique_associations':True,'fixedq0_half_raw_eval_active_changed_source_checks_all':True,'same_full_noRGB_input_support_gate_domain_initialterms_checked':True,'delivered_center_bound_roundoff_max_m':maxboundcast,'delivered_pointloss_max_recompute_error':max_point_term_error,'old_surface_outputs_match_nested_trace_n':oldsame,'oldsurface_not_strict_one_loss_ablation':True,'actual_elapsed_seconds':seal['seconds'],'counts':{k:dict(v) for k,v in counts.items()},'support_status_histograms':{k:dict(v) for k,v in supporthist.items()},'arm_status_histograms':{k:dict(v) for k,v in hist.items()},'active6_is_searchdomain_not_six_identified_parameters':True,'no_GT_annotations_index_or_taskscore_new_IoU_G_reoptimization_model_GPU_or_threshold_change':True,'issues':[]}
out=V/'GEOMETRY116_UNLABELLED_AUDIT.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':checks,'counts':result['counts'],'sha256':sha(out)}))
