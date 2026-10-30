"""Post-seal R7 CPU cross-domain scorer. No inference, optimizer or GT selection.
Prediction schema is fixed separately from scoring-index annotation references.
"""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,math,sys,time
import numpy as np
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'research/round5/audit'))
from independent_core_v2 import candidate_ious,transform_gt
_spec=importlib.util.spec_from_file_location('r7_source_coverage_batch',R/'research/round6/source_validation_scoring/batch_yaw_iou.py');_batch=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_batch)
MODES=('rgb_text','pc_only','shuffled_color');BOOTSTRAP_SEED=701;BOOTSTRAP_DRAWS=2000
GEOMETRY_DEPENDENCY_SHA256={'research/round5/audit/independent_core_v2.py': 'b4937fdd89f719fe1d9560f770c01597ff6807c74a75013720b470c43393379a', 'tools/obb_iou.py': '45896d29d1f95167b95a27b044fcc06b4e6fe5f9cb98749096c2f707292d4083', 'research/round6/source_validation_scoring/batch_yaw_iou.py': '6d1a73e3ada18a41ce04a374c60e76f254aab605f92c9037f740fd48fe4217c9'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def bounded_path(root,name):
 path=(Path(root)/name).resolve()
 if not path.is_relative_to(Path(root).resolve()):raise ValueError('Seal path escapes prediction/program root')
 return path

def gate_before_gt(*,predictions,seal_path,manifest,scoring_index,checkpoint,release,release_sha,program_root):
 # This function deliberately never opens annotation paths from scoring_index.
 if sha(release)!=release_sha:raise PermissionError('Release SHA mismatch')
 grant=read(release)
 for name,digest in GEOMETRY_DEPENDENCY_SHA256.items():
  if sha(R/name)!=digest:raise PermissionError('Geometry dependency changed before GT: '+name)
 if grant.get('CPU_SCORE_RELEASE') is not True:raise PermissionError('No stage-specific CPU score release')
 if grant.get('all_stage_predictions_sealed_before_stage_GT_score') is not True:raise PermissionError('Stage seal-before-score transaction not released')
 for key,path in [('prediction_seal_sha256',seal_path),('manifest_sha256',manifest),('scoring_index_sha256',scoring_index),('source_checkpoint_sha256',checkpoint),('scorer_program_sha256',Path(__file__))]:
  if grant.get(key)!=sha(path):raise PermissionError('Unbound '+key)
 seal=read(seal_path)
 if seal['manifest_sha256']!=grant['manifest_sha256'] or seal['source_checkpoint_sha256']!=grant['source_checkpoint_sha256']:raise PermissionError('Seal provenance mismatch')
 for name,digest in seal['files'].items():
  if sha(bounded_path(predictions,name))!=digest:raise PermissionError('Unsealed prediction '+name)
 for name,digest in seal['program_sha256'].items():
  if sha(bounded_path(program_root,name))!=digest:raise PermissionError('Changed producer '+name)
 return grant,seal

def evaluation_box(raw):
 b=np.asarray(raw).copy()
 if b.shape[-1]!=6:raise ValueError('Native yaw0 six dimensional box required')
 b[...,3:6]=np.maximum(b[...,3:6],np.asarray(1e-6,dtype=b.dtype))
 return b.astype(np.float64)

def selected_ious_independent(raw_boxes,gt):
 """General3D convex on distinct selected boxes only; repeat identities reused."""
 out=np.full(len(raw_boxes),np.nan);unique={};diagnostic={'convex_intersection_calls':0,'broadphase_proven_zero_count':0,'unique_finite_boxes':0}
 for i,b in enumerate(raw_boxes):
  if not np.isfinite(b).all():continue
  e=evaluation_box(b);key=e.tobytes()
  if key not in unique:
   val,d=candidate_ious(e[None],gt);unique[key]=float(val[0]);diagnostic['convex_intersection_calls']+=d['convex_intersection_calls'];diagnostic['broadphase_proven_zero_count']+=d['broadphase_proven_zero_count'];diagnostic['unique_finite_boxes']+=1
  out[i]=unique[key]
 return out,diagnostic

def load_prediction(path):
 with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 # Runner schema uses selected raw boxes for every arm, never GT-reselected pools.
 required={'source_boxes_raw6','source_text_scores','fixed_query_index','arm_names','arm_boxes_raw6','arm_modes','arm_status','mode_names','pseudo_target6','pseudo_reliability6'}
 if not required<=a.keys():raise ValueError('Missing prediction fields: '+str(sorted(required-a.keys())))
 boxes=a['source_boxes_raw6'];scores=a['source_text_scores'];q=int(a['fixed_query_index']);names=a['arm_names'].tolist();modes=a['arm_modes'].tolist();status=a['arm_status'].tolist();mnames=a['mode_names'].tolist()
 if boxes.shape!=(256,6) or scores.shape!=(256,) or not np.isfinite(scores).all() or q!=int(np.argmax(scores)):raise ValueError('Invalid fixed pure-text source pool/query')
 if len(set(names))!=len(names) or a['arm_boxes_raw6'].shape!=(len(names),6) or len(modes)!=len(names) or len(status)!=len(names):raise ValueError('Invalid arm schema')
 if 'source' not in names or not np.array_equal(a['arm_boxes_raw6'][names.index('source')],boxes[q],equal_nan=True):raise ValueError('Source arm must be fixed source query')
 if set(mnames)!=set(MODES) or a['pseudo_target6'].shape!=(3,6) or a['pseudo_reliability6'].shape!=(3,6):raise ValueError('Expected three G modes and target6/w6')
 if not np.isfinite(a['pseudo_reliability6']).all() or np.any((a['pseudo_reliability6']<0)|(a['pseudo_reliability6']>1)) or np.any(a['pseudo_reliability6'][:,3:]!=0):raise ValueError('First edition must retain source shape prior with size weights0')
 if not all(m=='source' or m in MODES for m in modes):raise ValueError('Unknown arm mode')
 a.update(q=q,names=names,modes=modes,status=status,mode_names_list=mnames)
 return a

def source_coverage(pool,scores,gt):
 if not np.isfinite(pool).all():return None
 values=_batch.many_iou(np.c_[evaluation_box(pool),np.zeros(256)],np.asarray(gt,np.float64));order=np.argsort(-scores,kind='stable');result={}
 for k in (1,5,10,256):
  v=float(values[order[:k]].max());result[str(k)]={'best_iou':v,'acc25':bool(v>.25),'acc50':bool(v>.5),'GTassisted_if_k_gt_1':k>1}
 return result,values

def paired_scene_summary(records):
 if not records:return {'n':0,'scenes':0,'mean_delta_iou':None,'equal_scene_delta_iou':None,'scene_bootstrap_CI95':None}
 groups=collections.defaultdict(list)
 for r in records:groups[r['scene']].append(r['delta_iou'])
 means=np.array([np.mean(groups[s]) for s in sorted(groups)]);rng=np.random.default_rng(BOOTSTRAP_SEED);draw=rng.choice(means,size=(BOOTSTRAP_DRAWS,len(means)),replace=True).mean(1);ci=np.quantile(draw,[.025,.975]).tolist() if len(means)>1 else None
 return {'n':len(records),'scenes':len(groups),'mean_delta_iou':float(np.mean([r['delta_iou'] for r in records])),'equal_scene_delta_iou':float(means.mean()),'scene_bootstrap_CI95':ci,'bootstrap_draws':BOOTSTRAP_DRAWS,'bootstrap_seed':BOOTSTRAP_SEED,'small_scene_count_descriptive_only':True,'improved':sum(r['delta_iou']>0 for r in records),'worsened':sum(r['delta_iou']<0 for r in records),'unchanged':sum(r['delta_iou']==0 for r in records)}

def summarize(rows,arm_names):
 result=[]
 for platform in ('drone','quad'):
  rr=[r for r in rows if r['platform']==platform]
  for mode in MODES:
   relevant=[n for n in arm_names if n=='source' or any(n in r['arms'] and r['arms'][n]['mode']==mode for r in rr)]
   for name in relevant:
    for population in ('all_requested','accepted_any_axis','changed_nonzero','common_eligible_all_modes'):
     selected=[]
     for r in rr:
      if name not in r['arms']:continue
      ar=r['arms'][name]
      yes=population=='all_requested' or (population=='accepted_any_axis' and r['mode_accepted'][mode]) or (population=='changed_nonzero' and ar['changed_nonzero']) or (population=='common_eligible_all_modes' and r['common_eligible_all_modes'])
      if yes:selected.append((r,ar))
     valid=[(r,ar) for r,ar in selected if ar['iou'] is not None];n=len(selected);nv=len(valid);vv=[ar['iou'] for r,ar in valid];source=[r['source_iou'] for r,ar in valid];scenes=sorted({r['scene'] for r,ar in valid});sm=[np.mean([ar['iou'] for r,ar in valid if r['scene']==s]) for s in scenes]
     pair=[{'scene':r['scene'],'delta_iou':ar['delta_iou']} for r,ar in valid]
     result.append({'platform':platform,'mode':mode,'arm':name,'population':population,'n_requested_in_population':n,'n_scored':nv,'n_failed_or_nonfinite':n-nv,'n_flagged_failed':sum(ar['status']=='failed' for r,ar in selected),'n_skip_identity':sum(ar['status'].startswith('skip') for r,ar in selected),'scene_clusters':len(scenes),'mean_iou':float(np.mean(vv)) if nv else None,'Acc25':sum(v>.25 for v in vv)/nv if nv else None,'Acc50':sum(v>.5 for v in vv)/nv if nv else None,'Acc25_success_count':sum(v>.25 for v in vv),'Acc50_success_count':sum(v>.5 for v in vv),'source_mean_iou_on_same_scored_rows':float(np.mean(source)) if nv else None,'equal_scene_mean_iou':float(np.mean(sm)) if sm else None,'all_requested_pessimistic_mean_iou_if_missing_as_zero':sum(vv)/n if n else None,'paired_source_comparison':paired_scene_summary(pair)})
 return result

def score_rows(samples,index,predictions):
 rows=[];names_all=[];counts=collections.Counter();max_geometry_difference=0.;max_threshold_disagreement=0
 for s in samples:
  sid=s['id'];idx=index[s.get('scoring_id',sid)];platform=s['platform'];assert platform in ('drone','quad') and idx['platform']==platform
  mp=Path(idx['meta_path']);assert sha(mp)==idx['metadata_container_sha256'];obj=read(mp)['ground_info'][idx['object_index']];gt=transform_gt(obj['bbox_3d'],platform,idx.get('waymo_view_suffix'));del obj
  a=load_prediction(predictions/(sid+'.npz'));source=a['source_boxes_raw6'][a['q']];selected=a['arm_boxes_raw6'];ious,geo=selected_ious_independent(selected,gt);root_selected=_batch.many_iou(np.c_[evaluation_box(selected[np.isfinite(selected).all(1)]),np.zeros(int(np.isfinite(selected).all(1).sum()))],gt.astype(float)) if np.isfinite(selected).all(1).any() else np.array([]);finite=np.isfinite(ious)
  if finite.any():
   diff=float(np.max(np.abs(root_selected-ious[finite])));max_geometry_difference=max(max_geometry_difference,diff);disagree=sum(int(np.any((root_selected>t)!=(ious[finite]>t))) for t in (.25,.5));max_threshold_disagreement+=disagree;assert diff<1e-8 and disagree==0,'Independent selected geometry differs from batch primary'
  counts.update(geo);source_iou=float(ious[a['names'].index('source')]);assert math.isfinite(source_iou),'Source failure must be handled explicitly, never silently dropped'
  mode_accepted={m:bool(a['pseudo_reliability6'][a['mode_names_list'].index(m)].any()) for m in MODES};r={'id':sid,'platform':platform,'scene':idx['scene'],'source_query':a['q'],'source_iou':source_iou,'mode_accepted':mode_accepted,'common_eligible_all_modes':all(mode_accepted.values()),'arms':{},'pseudo_diagnostic':{}}
  coverage=source_coverage(a['source_boxes_raw6'],a['source_text_scores'],gt);r['source_coverage']=coverage[0] if coverage is not None else None
  if coverage is not None:assert abs(coverage[1][a['q']]-source_iou)<1e-8
  for j,name in enumerate(a['names']):
   if name not in names_all:names_all.append(name)
   value=float(ious[j]) if np.isfinite(ious[j]) else None;delta=value-source_iou if value is not None else None;changed=not np.array_equal(selected[j],source,equal_nan=True)
   r['arms'][name]={'mode':a['modes'][j],'status':a['status'][j],'iou':value,'delta_iou':delta,'negative_case':bool(delta<0) if delta is not None else None,'changed_nonzero':changed,'raw_nonpositive_dimensions':int((selected[j,3:]<=0).sum()),'clamped_dimensions':int((selected[j,3:]<1e-6).sum())}
  for m in MODES:
   mi=a['mode_names_list'].index(m);w=a['pseudo_reliability6'][mi];pseudo=a['pseudo_target6'][mi];v=pseudo[:3]-source[:3];toward=gt[:3].astype(float)-source[:3];observed=w[:3]>0;den=np.linalg.norm(v)*np.linalg.norm(toward);pseudo_iou=selected_ious_independent(np.asarray([pseudo]),gt)[0][0] if np.isfinite(pseudo).all() else None
   r['pseudo_diagnostic'][m]={'accepted_axes':observed.tolist(),'accepted_axis_count':int(observed.sum()),'nonzero_displacement_axes':int(np.count_nonzero(v)),'source_center_error_m':float(np.linalg.norm(toward)),'pseudo_center_error_m':float(np.linalg.norm(gt[:3]-pseudo[:3])),'direction_cosine_to_GT_center':float(np.dot(v,toward)/den) if den>0 else None,'direction_cosine_undefined_reason':'zero_displacement_or_zero_source_error' if den==0 else None,'accepted_axes_move_toward_GT':int(np.sum(observed&(v*toward>0))),'accepted_axes_move_away_from_GT':int(np.sum(observed&(v*toward<0))),'pseudo_iou':float(pseudo_iou) if pseudo_iou is not None else None,'postseal_descriptive_only_never_gate_or_select_alpha':True}
  rows.append(r)
 return rows,names_all,dict(counts),max_geometry_difference,max_threshold_disagreement

def main(a):
 start=time.perf_counter();sp=a.predictions/'SEALED.json'
 if not sp.exists():sp=a.predictions/'sealed.json'
 grant,se=gate_before_gt(predictions=a.predictions,seal_path=sp,manifest=a.manifest,scoring_index=a.scoring_index,checkpoint=a.source_checkpoint,release=a.release,release_sha=a.release_sha256,program_root=a.program_root)
 samples_doc=read(a.manifest);samples=samples_doc['samples'] if isinstance(samples_doc,dict) else samples_doc;assert len({s['id'] for s in samples})==len(samples) and len(samples)==se['n_samples'];assert set(s['platform'] for s in samples)<=set(('drone','quad'));idx={r['id']:r for r in read(a.scoring_index)}
 expected_npz={s['id']+'.npz' for s in samples};assert expected_npz<=set(se['files']) and expected_npz=={p.name for p in a.predictions.glob('*.npz')},'Every consumed NPZ must be sealed; no missing/extra sample silently ignored'
 a.output.mkdir(parents=True,exist_ok=False);lock={'state':'all_prediction_code_manifest_checkpoint_index_hashes_verified_before_GT','release_sha256':sha(a.release),'prediction_seal_sha256':sha(sp),'scorer_sha256':sha(Path(__file__)),'core3D_sha256':sha(R/'research/round5/audit/independent_core_v2.py'),'obb3D_sha256':sha(R/'tools/obb_iou.py'),'pool_batch_geometry_sha256':sha(R/'research/round6/source_validation_scoring/batch_yaw_iou.py'),'stage':grant['stage'],'no_GPU_model_call':True};dump(a.output/'SCORER_LOCK.json',lock)
 rows,names,counts,diff,disagree=score_rows(samples,idx,a.predictions);summaries=summarize(rows,names);coverage=[]
 for p in ('drone','quad'):
  rr=[r for r in rows if r['platform']==p];cv=[r['source_coverage'] for r in rr if r['source_coverage'] is not None];coverage.append({'platform':p,'n_requested':len(rr),'n_coverage_scored':len(cv),'scene_clusters':len({r['scene'] for r in rr}),'topk':{str(k):{'mean_best_iou_GTassisted_if_k_gt1':float(np.mean([c[str(k)]['best_iou'] for c in cv])) if cv else None,'Acc25_success_count':sum(c[str(k)]['acc25'] for c in cv),'Acc50_success_count':sum(c[str(k)]['acc50'] for c in cv),'denominator':len(cv)} for k in (1,5,10,256)}})
 result={'state':'scored_post_stage_seal','n_samples':len(rows),'stage':grant['stage'],'summary':summaries,'source_candidate_coverage':coverage,'geometry_audit_counts':counts,'max_selected_3D_vs_batch_abs_error':diff,'strict_threshold_disagreements':disagree,'negative_case_count_all_arms':sum(ar['negative_case'] is True for r in rows for ar in r['arms'].values()),'rows_PRIVATE':rows,'scope':'Waymo-only source model on targetDrone/Quad VAL; nativeyaw0, noPCA/noGTyaw/noGTqueryselection; puretextfixedq. TargetVAL already exposed inR5, not freshresearchholdout.','effective_scene_warning':'At most2Drone+3Quad targetVALscene clusters; frames/expressions/arms are not independent scenes.','coverage_geometry_boundary':'All256 source coverage uses audited float64 batched2Dpolygon×height; independent general3D convex covers every unique selected arm, not every pool candidate.','failure_scope':'Finite scored and requested counts explicit; missing outputs never silently drop. Source inference failures currently fail loudly for caller resolution, not converted to zero.','seconds':time.perf_counter()-start}
 dump(a.output/'RESULTS_PRIVATE.json',result);dump(a.output/'SEALED.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_samples':len(rows)});print(json.dumps({'state':result['state'],'n':len(rows),'max_error':diff,'seconds':result['seconds']}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for name in ('predictions','manifest','scoring-index','source-checkpoint','release','program-root','output'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--release-sha256',required=True);main(ap.parse_args())
