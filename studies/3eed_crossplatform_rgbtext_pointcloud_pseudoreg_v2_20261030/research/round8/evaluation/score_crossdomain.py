"""R8 post-seal CPU evaluation; q0 source, q1 association/geometry/update distinct.
No model, GPU or GT-assisted online selection. Requires parent full-stage release.
"""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,math,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import candidate_ious,transform_gt,validate_native_top10
_spec=importlib.util.spec_from_file_location('R8_batch_geometry',ROOT/'research/round6/source_validation_scoring/batch_yaw_iou.py');_batch=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_batch)
MODES=('rgb_fulltext','pc_only','semantic_shuffle')
KINDS=('association_only','direct_G','param1','param3','outputspace1','outputspace3')
ARMS=('source',)+tuple(m+'__'+k for m in MODES for k in KINDS)
THRESHOLDS=(.25,.5);BOOTSTRAP_SEED=801;BOOTSTRAP_DRAWS=2000
GEOMETRY_DEPENDENCIES={'research/round5/audit/independent_core_v2.py': 'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a', 'tools/obb_iou.py': '45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083', 'research/round6/source_validation_scoring/batch_yaw_iou.py': '6d1a73e3ada18a41ce04a374c60e76f254aab605f92c9037f740fd48fe4217c9'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def bounded(root,name):
 path=(Path(root)/name).resolve()
 if not path.is_relative_to(Path(root).resolve()):raise ValueError('Path escapes registered artifact root')
 return path

def eval_box(raw):
 raw=np.asarray(raw);out=raw.copy();out[...,3:6]=np.maximum(out[...,3:6],np.asarray(1e-6,dtype=out.dtype));return out.astype(np.float64)

def selected_geometry(boxes,target):
 values=np.full(len(boxes),np.nan);memo={};calls=0
 for i,raw in enumerate(boxes):
  if not np.isfinite(raw).all():continue
  b=eval_box(raw);key=b.tobytes()
  if key not in memo:
   v,d=candidate_ious(b[None],target);memo[key]=float(v[0]);calls+=d['convex_intersection_calls']
  values[i]=memo[key]
 return values,{'unique_finite_selected_boxes':len(memo),'convex_intersection_calls':calls}

def validate_strict_flags(flags):
 expected={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False}
 if any(flags.get(k) is not v for k,v in expected.items()):raise ValueError('New matched strictFP32 source required; R7 frozen cache is not this baseline')

def release_gate(a):
 if sha(a.release)!=a.release_sha256:raise PermissionError('Parent release SHA mismatch')
 grant=read(a.release)
 for name,digest in GEOMETRY_DEPENDENCIES.items():
  if sha(ROOT/name)!=digest:raise PermissionError('Geometry changed before GT')
 if grant.get('CPU_SCORE_RELEASE') is not True or grant.get('both_smoke_and_full_sealed_before_any_stage_GT_score') is not True:raise PermissionError('No post-full-seal GT release')
 sealpath=a.predictions/'SEALED.json'
 for key,p in [('prediction_seal_sha256',sealpath),('manifest_sha256',a.manifest),('scoring_index_sha256',a.scoring_index),('source_checkpoint_sha256',a.source_checkpoint),('scorer_program_sha256',Path(__file__))]:
  if grant.get(key)!=sha(p):raise PermissionError('Unbound '+key)
 seal=read(sealpath);assert seal['manifest_sha256']==grant['manifest_sha256'] and seal['source_checkpoint_sha256']==grant['source_checkpoint_sha256'];validate_strict_flags(seal['strict_FP32_flags'])
 for name,digest in seal['files'].items():assert sha(bounded(a.predictions,name))==digest
 for name,digest in seal['program_sha256'].items():assert sha(bounded(a.program_root,name))==digest
 assert seal['program_sha256_base']=='workspace_root' and seal['n_arms']==19 and seal['grounder_restored_and_CLIP_frozen'] and seal['sourceQ0_not_retagged_as_associationQ1']
 plan=read(a.predictions/'registered_plan.json');assert sha(a.predictions/'registered_plan.json')==seal['plan_sha256'];assets=Path(plan['CLIP_asset_dir']);conversion=assets/'SAFE_CONVERSION_RECEIPT.json';assert sha(conversion)==plan['CLIP_conversion_receipt_sha256'];resource=read(conversion);assert resource['derived_safetensors_sha256']==seal['CLIP_resource']['derived_safetensors_sha256'];assert all(sha(assets/name)==digest for name,digest in resource['runtime_files_sha256'].items())
 return grant,seal,sealpath

def load_prediction(path):
 with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 required={'source_boxes_raw6','source_boxes_eval6','source_text_scores256','source_native_top10','q_source','q_association','arm_names','arm_modes','arm_query_index','arm_status','arm_boxes_raw6','arm_boxes_eval6','modes','raw_consensus_center','r_global','pseudo_target6_float64','pseudo_target6','pseudo_reliability6','dedup_membership','roi_group_ids'}
 if not required<=a.keys():raise ValueError('Missing R8 fields '+str(sorted(required-a.keys())))
 pool=a['source_boxes_raw6'];scores=a['source_text_scores256'];q0=int(a['q_source']);q1=np.asarray(a['q_association']);names=a['arm_names'].tolist();modes=a['arm_modes'].tolist();aq=np.asarray(a['arm_query_index']);status=a['arm_status'].tolist()
 if pool.shape!=(256,6) or scores.shape!=(256,) or not np.isfinite(pool).all() or not np.isfinite(scores).all():raise ValueError('Finite source pool required')
 if q0!=int(np.argmax(scores)) or q1.shape!=(3,) or not np.issubdtype(q1.dtype,np.integer) or ((q1<0)|(q1>=256)).any():raise ValueError('q0/q1 contract violated; only q0 must be argmax')
 validate_native_top10(scores,a['source_native_top10'])
 if tuple(names)!=ARMS or tuple(a['modes'].tolist())!=MODES or len(modes)!=19 or len(status)!=19 or aq.shape!=(19,) or a['arm_boxes_raw6'].shape!=(19,6):raise ValueError('Expected fixed19 arms/schema')
 if not np.array_equal(a['source_boxes_eval6'],eval_box(pool)) or not np.array_equal(a['arm_boxes_eval6'],eval_box(a['arm_boxes_raw6'])):raise ValueError('Official evalclamp mismatch')
 if aq[0]!=q0 or not np.array_equal(a['arm_boxes_raw6'][0],pool[q0]):raise ValueError('Originalq0 source changed')
 w=a['pseudo_reliability6'];p64=a['pseudo_target6_float64'];p32=a['pseudo_target6'];r=a['r_global'];raw=a['raw_consensus_center'];membership=a['dedup_membership'];roi_ids=np.asarray(a['roi_group_ids']);gid=np.array([int(membership[q]) if int(membership[q]) in roi_ids else -1 for q in q1],np.int64);a['selected_group_id']=gid
 if w.shape!=(3,6) or p64.shape!=(3,6) or p32.shape!=(3,6) or r.shape!=(3,) or raw.shape!=(3,3) or membership.shape!=(256,) or gid.shape!=(3,):raise ValueError('InvalidG evidence shape')
 if not all(np.isfinite(v).all() for v in (w,p64,p32,r,raw)) or (w<0).any() or (w>1).any() or w[:,3:].any() or (r<0).any() or (r>1).any():raise ValueError('Invalid trust/axisweights')
 if not np.array_equal(p32,p64.astype(np.float32)):raise ValueError('Saved target must match actual float32 target conversion')
 for mi,mode in enumerate(MODES):
  anchor=pool[q1[mi]];ids=[i for i,m in enumerate(modes) if m==mode]
  if len(ids)!=6 or any(aq[i]!=q1[mi] for i in ids):raise ValueError('All associated arms must use actualq1, never staleq0 cache')
  assoc=names.index(mode+'__association_only');direct=names.index(mode+'__direct_G')
  if not np.array_equal(a['arm_boxes_raw6'][assoc],anchor) or not np.array_equal(p64[mi,3:],anchor[3:]) or not np.array_equal(p32[mi,3:],anchor[3:]):raise ValueError('q1 original anchor/shape prior changed')
  expected_center=np.where(w[mi,:3]>0,anchor[:3].astype(float)+r[mi]*(raw[mi]-anchor[:3].astype(float)),anchor[:3].astype(float))
  if not np.allclose(p64[mi,:3],expected_center,rtol=0,atol=1e-12):raise ValueError('Globaltrust must shrink shared pseudotarget, not cancel in normalizedw')
  if not np.array_equal(a['arm_boxes_raw6'][direct],np.where(w[mi]>0,p32[mi],anchor)):raise ValueError('DirectG not same final pseudotarget used for adaptation')
  if gid[mi]>=0:
   members=np.flatnonzero(membership==gid[mi]);assert len(members)>0 and q1[mi]==members[np.argmax(scores[members])],'q1 must be canonical actual sourceproposal'
  else:assert q1[mi]==q0 and not w[mi].any(),'No selectedgroup must failclosed toq0'
  for i in ids:assert np.array_equal(a['arm_boxes_raw6'][i,3:],anchor[3:]),'Sizeprior must be q1 notq0'
 a.update(q0=q0,q1=q1,names=names,modes=modes,status=status)
 return a

def paired_scene(rows,key):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['scene']].append(r[key])
 means=np.array([np.mean(groups[s]) for s in sorted(groups)]);rng=np.random.default_rng(BOOTSTRAP_SEED);ci=np.quantile(rng.choice(means,size=(BOOTSTRAP_DRAWS,len(means)),replace=True).mean(1),[.025,.975]).tolist() if len(means)>1 else None
 return {'n':len(rows),'scene_clusters':len(groups),'micro_delta':float(np.mean([r[key] for r in rows])) if rows else None,'equal_scene_delta':float(means.mean()) if len(means) else None,'descriptive_scene_CI95':ci,'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_draws':BOOTSTRAP_DRAWS,'small_2_Drone_3_Quad_scene_limit':True}

def source_coverage(pool,scores,native,q0,gt):
 v=_batch.many_iou(np.c_[eval_box(pool),np.zeros(256)],np.asarray(gt,float));order=validate_native_top10(scores,native);metrics={}
 for k in (1,5,10,256):
  best=float(v[q0]) if k==1 else float(v[order[:k]].max()) if k<=10 else float(v.max());metrics[str(k)]={'best_iou':best,'acc25':bool(best>.25),'acc50':bool(best>.5),'GTassisted_if_k_gt1':k>1}
 return v,metrics

def score_sample(s,ix,a,t):
 # Called only after the complete parent gate and source/prediction hash checks.
 meta=Path(ix['meta_path']);assert sha(meta)==ix['metadata_container_sha256'];box=read(meta)['ground_info'][ix['object_index']]['bbox_3d'];gt=transform_gt(box,s['platform'],ix.get('waymo_view_suffix'))
 vals,cost=selected_geometry(a['arm_boxes_raw6'],gt);assert np.isfinite(vals).all(),'Nonfinite arm is an explicit failed stage, never silently excluded';batch=_batch.many_iou(np.c_[eval_box(a['arm_boxes_raw6']),np.zeros(19)],gt.astype(float));err=float(np.max(np.abs(vals-batch)));assert err<1e-8 and all(np.array_equal(vals>threshold,batch>threshold) for threshold in THRESHOLDS)
 pv,cover=source_coverage(a['source_boxes_raw6'],a['source_text_scores256'],a['source_native_top10'],a['q0'],gt);assert abs(pv[a['q0']]-vals[0])<1e-8;row={'id':s['id'],'platform':s['platform'],'scene':ix['scene'],'q_source':a['q0'],'q_association':a['q1'].tolist(),'source_q0_iou':float(vals[0]),'source_coverage':cover,'arms':{},'G':{},'cluster_oracles_PRIVATE':{}}
 for mi,mode in enumerate(MODES):
  anchor=a['source_boxes_raw6'][a['q1'][mi]];w=a['pseudo_reliability6'][mi];target=a['pseudo_target6'][mi];gid=int(a['selected_group_id'][mi]);member=np.flatnonzero(a['dedup_membership']==gid) if gid>=0 else np.array([],int);roi=np.isin(a['dedup_membership'],a['roi_group_ids'])&(a['dedup_membership']>=0)
  row['G'][mode]={'q_changed':int(a['q1'][mi])!=a['q0'],'accepted_axis':bool(w.any()),'raw_consensus_delta_nonzero':bool(np.any(a['raw_consensus_center'][mi]!=anchor[:3])),'final_float64_pseudo_delta_nonzero':bool(np.any(a['pseudo_target6_float64'][mi]!=anchor)),'actual_float32_pseudo_delta_nonzero':bool(np.any((target-anchor)[w>0]!=0)),'r_global':float(a['r_global'][mi]),'selected_group':gid}
  row['cluster_oracles_PRIVATE'][mode]={'selected_group_member_count':len(member),'selected_group_best_iou_GTassisted':float(pv[member].max()) if len(member) else None,'selected_group_best_query_GTassisted':int(member[np.argmax(pv[member])]) if len(member) else None,'all_prepared_group_pool_best_iou_GTassisted':float(pv[a['dedup_membership']>=0].max()) if np.any(a['dedup_membership']>=0) else None,'semantic_ROI_pool_best_iou_GTassisted':float(pv[roi].max()) if roi.any() else None,'not_groundtruth_instance_purity':True}
 for j,name in enumerate(a['names']):
  mode=a['modes'][j];mi=MODES.index(mode) if mode in MODES else None;assoc=vals[a['names'].index(mode+'__association_only')] if mi is not None else vals[0];direct=vals[a['names'].index(mode+'__direct_G')] if mi is not None else vals[0];anchor=a['source_boxes_raw6'][a['q1'][mi]] if mi is not None else a['source_boxes_raw6'][a['q0']];is_param=name.endswith('__param1') or name.endswith('__param3');steps=[]
  if is_param:steps=t['G_modes'][mode]['steps'].get('1' if name.endswith('1') else '3',{}).get('trace',[])
  kernel=vals[a['names'].index(mode+'__outputspace'+name[-1])] if mi is not None and (name.endswith('__param1') or name.endswith('__param3') or name.endswith('__outputspace1') or name.endswith('__outputspace3')) else None
  row['arms'][name]={'mode':mode,'delta_vs_outputspace':float(vals[j]-kernel) if kernel is not None else None,'raw_nonpositive_dimensions':int((a['arm_boxes_raw6'][j,3:]<=0).sum()),'query_index':int(a['arm_query_index'][j]),'status':a['status'][j],'iou':float(vals[j]),'delta_vs_q0':float(vals[j]-vals[0]),'delta_vs_association':float(vals[j]-assoc),'delta_vs_directG':float(vals[j]-direct),'association_iou':float(assoc),'directG_iou':float(direct),'selected_changed_vs_q1_anchor':not np.array_equal(a['arm_boxes_raw6'][j],anchor),'parameter_steps_executed':len(steps),'nonzero_gradient_steps':sum(z['parameter_gradient_norm']>0 and z['nonzero_gradient_parameter_tensors']>0 for z in steps),'nonzero_parameter_update':bool(steps and any(v>0 for v in steps[-1]['parameter_delta_max_by_name'].values()))}
 return row,cost,err

def summaries(rows):
 out=[]
 for platform in ('drone','quad'):
  pp=[r for r in rows if r['platform']==platform]
  for mode in MODES:
   names=('source',)+tuple(mode+'__'+k for k in KINDS)
   for name in names:
    for pop in ('all','q_association_changed','accepted_any_axis','actual_pseudo_nonzero','selected_changed_vs_q1','common_eligible_all_modes'):
     rr=[r for r in pp if pop=='all' or pop=='q_association_changed' and r['G'][mode]['q_changed'] or pop=='accepted_any_axis' and r['G'][mode]['accepted_axis'] or pop=='actual_pseudo_nonzero' and r['G'][mode]['actual_float32_pseudo_delta_nonzero'] or pop=='selected_changed_vs_q1' and r['arms'][name]['selected_changed_vs_q1_anchor'] or pop=='common_eligible_all_modes' and all(v['accepted_axis'] for v in r['G'].values())];aa=[r['arms'][name] for r in rr];n=len(rr);scene=sorted({r['scene'] for r in rr});d=[]
     for r in rr:
      ar=r['arms'][name];d.append({'scene':r['scene'],'delta_vs_q0':ar['iou']-r['source_q0_iou'],'delta_vs_association':ar['iou']-r['arms'][mode+'__association_only']['iou'],'delta_vs_directG':ar['iou']-r['arms'][mode+'__direct_G']['iou']})
     out.append({'platform':platform,'mode':mode,'arm':name,'population':pop,'n':n,'scene_clusters':len(scene),'mean_iou':float(np.mean([a['iou'] for a in aa])) if n else None,'Acc25':sum(a['iou']>.25 for a in aa)/n if n else None,'Acc50':sum(a['iou']>.5 for a in aa)/n if n else None,'equal_scene_mean_iou':float(np.mean([np.mean([r['arms'][name]['iou'] for r in rr if r['scene']==s]) for s in scene])) if scene else None,'mode_association_q_changed':sum(r['G'][mode]['q_changed'] for r in rr),'arm_query_changed_vs_q0':sum(r['arms'][name]['query_index']!=r['q_source'] for r in rr),'accepted':sum(r['G'][mode]['accepted_axis'] for r in rr),'raw_consensus_nonzero':sum(r['G'][mode]['raw_consensus_delta_nonzero'] for r in rr),'actual_pseudo_nonzero':sum(r['G'][mode]['actual_float32_pseudo_delta_nonzero'] for r in rr),'executed_parameter_steps':sum(a['parameter_steps_executed'] for a in aa),'nonzero_parameter_update_runs':sum(a['nonzero_parameter_update'] for a in aa),'status_histogram':dict(collections.Counter(a['status'] for a in aa)),'n_flagged_failed':sum(a['status'].startswith('failed') for a in aa),'raw_nonpositive_dimensions':sum(a['raw_nonpositive_dimensions'] for a in aa),'parameter_vs_outputspace':paired_scene([{'scene':r['scene'],'kernel_delta':r['arms'][name]['delta_vs_outputspace']} for r in rr if r['arms'][name]['delta_vs_outputspace'] is not None],'kernel_delta'),'paired':{key:paired_scene(d,key) for key in ('delta_vs_q0','delta_vs_association','delta_vs_directG')}})
 return out

def coverage_summaries(rows):
 out=[]
 for platform in ('drone','quad'):
  rr=[r for r in rows if r['platform']==platform];n=len(rr);item={'platform':platform,'n':n,'source_pool_topk':{},'selected_group_oracles':{}}
  for k in (1,5,10,256):item['source_pool_topk'][str(k)]={'n':n,'mean_best_iou_GTassisted_if_k_gt1':float(np.mean([r['source_coverage'][str(k)]['best_iou'] for r in rr])) if n else None,'Acc25_successes':sum(r['source_coverage'][str(k)]['acc25'] for r in rr),'Acc50_successes':sum(r['source_coverage'][str(k)]['acc50'] for r in rr)}
  for mode in MODES:
   oracles=[r['cluster_oracles_PRIVATE'][mode] for r in rr];valid=[o for o in oracles if o['selected_group_best_iou_GTassisted'] is not None];item['selected_group_oracles'][mode]={'n_requested':n,'n_with_selected_group':len(valid),'mean_selected_group_best_iou_GTassisted':float(np.mean([o['selected_group_best_iou_GTassisted'] for o in valid])) if valid else None,'not_online_selection_or_true_instance_purity':True}
  out.append(item)
 return out

def main(a):
 started=time.perf_counter();grant,seal,sp=release_gate(a);samples=read(a.manifest);assert len(samples)==seal['n_samples'] and len({s['id'] for s in samples})==len(samples);assert {s['id']+'.npz' for s in samples}=={p.name for p in a.predictions.glob('*.npz')};assert all(s['id']+'.npz' in seal['files'] for s in samples);assert set(s['platform'] for s in samples)<=set(('drone','quad'));traces={r['id']:r for r in read(a.predictions/'trace.json')};assert 'trace.json' in seal['files'] and set(traces)=={s['id'] for s in samples};index={r['id']:r for r in read(a.scoring_index)};a.output.mkdir(parents=True,exist_ok=False)
 lock={'state':'all_hashes_checked_before_target_GT','release_sha256':sha(a.release),'prediction_seal_sha256':sha(sp),'manifest_sha256':sha(a.manifest),'scoring_index_sha256':sha(a.scoring_index),'source_checkpoint_sha256':sha(a.source_checkpoint),'scorer_sha256':sha(Path(__file__)),'geometry':{'independent_selected3D':sha(ROOT/'research/round5/audit/independent_core_v2.py'),'obb3D':sha(ROOT/'tools/obb_iou.py'),'pool_batch':sha(ROOT/'research/round6/source_validation_scoring/batch_yaw_iou.py')},'stage':grant['stage'],'no_GPU_model':True};dump(a.output/'SCORER_LOCK.json',lock);rows=[];maxerr=0.;cost=collections.Counter()
 for s in samples:
  z=load_prediction(a.predictions/(s['id']+'.npz'));ix=index[s.get('scoring_id',s['id'])];assert ix['platform']==s['platform'];row,c,e=score_sample(s,ix,z,traces[s['id']]);rows.append(row);cost.update(c);maxerr=max(maxerr,e)
 result={'state':'R8_scored_after_fullstage_prediction_seal','n':len(rows),'stage':grant['stage'],'summary':summaries(rows),'candidate_and_group_coverage':coverage_summaries(rows),'rows_PRIVATE':rows,'max_selected_3D_vs_batch_abs_error':maxerr,'strict_threshold_disagreements':0,'geometry_counts':dict(cost),'scope':'q0 matched strictFP32 source; q1 association only; nativeyaw0/noPCA/noGTclassspan. CLIP frozen semantic evidence, not3Dteacher. TargetVAL R5/R7research-exposed, sourceWaymo-only.','geometry_scope':'All256 sourcepool batch geometry; every unique selected arm independent3D, notfullpool independently3D.','effective_scene_limit':'Drone2/Quad3 clusters; descriptive CI, no frame/query pseudo-replication.','seconds':time.perf_counter()-started};dump(a.output/'RESULTS_PRIVATE.json',result);dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in result.items() if k!='rows_PRIVATE'});dump(a.output/'SEALED.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n':len(rows)});print(json.dumps({'n':len(rows),'max_geometry_error':maxerr,'seconds':result['seconds']}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for name in ('predictions','manifest','scoring-index','source-checkpoint','release','program-root','output'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--release-sha256',required=True);args=ap.parse_args()
 try:main(args)
 except BaseException as e:
  args.output.parent.mkdir(parents=True,exist_ok=True);dump(args.output.parent/('SCORER_FAILURE_'+str(time.time_ns())+'.json'),{'state':'failed_explicitly_no_silent_skip','exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'scorer_sha256':sha(Path(__file__))});raise
