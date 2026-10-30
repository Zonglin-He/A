"""Independent CPU audit after coordinator completion/seals. Never runs a model.
Uses explicit math coordinate transforms and 3D convex intersection, distinct from
parent's SciPy-transform + XY polygon clipping. Writes only this audit directory.
"""
import pathlib,json,hashlib,math,sys,traceback,warnings
from collections import Counter,defaultdict
import numpy as np
R=pathlib.Path(__file__).resolve().parent;B=R.parents[2];P=B/'crossplatform/round3';OLD=P.parent/'round2'
sys.path.insert(0,str(B/'tools'));from obb_iou import oriented_iou
import orientation_contract as contract
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def rotation(theta):
 c,s=math.cos(float(theta)),math.sin(float(theta));m=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=float)
 assert np.max(abs(m.T@m-np.eye(3)))<1e-12;return m
def convex_iou(box,gt):
 box=np.asarray(box,dtype=float);gt=np.asarray(gt,dtype=float)
 assert box.shape==gt.shape==(7,) and np.isfinite(box).all() and np.isfinite(gt).all()
 if (box[3:6]<=0).any():return 0.
 assert (gt[3:6]>0).all()
 return float(oriented_iou(box[:3],box[3:6],rotation(box[6]),gt[:3],gt[3:6],rotation(gt[6])))
def manual_target(sample):
 meta=read(sample['meta_path']);gt=np.array(meta['ground_info'][sample['object_index']]['bbox_3d'][:7],dtype=float)
 if sample['platform']=='drone':gt[2]=gt[2]+1.8
 if sample['platform']=='waymo':
  angle=math.radians([0,-45,45,-90,90][sample['Waymo_view_suffix']]);x,y=float(gt[0]),float(gt[1]);c,s=math.cos(angle),math.sin(angle);gt[0]=c*x-s*y;gt[1]=s*x+c*y;gt[6]+=angle
 return gt

def average_ranks(x):
 x=np.asarray(x);order=np.argsort(x,kind='stable');rank=np.empty(len(x),float);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  rank[order[i:j]]=(i+1+j)/2;i=j
 return rank

def rho(x,y):
 if len(x)<3 or len(set(x))<2 or len(set(y))<2:return None
 a=average_ranks(x);b=average_ranks(y);a-=a.mean();b-=b.mean();return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
def auc_pairwise(x,positive):
 x=np.asarray(x);positive=np.asarray(positive,dtype=bool);a=x[positive];b=x[~positive]
 if not len(a) or not len(b):return None
 return float(np.mean((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])))
def compare_optional(a,b,tolerance=1e-10):
 if a is None or b is None:return a is None and b is None
 return abs(a-b)<=tolerance

def main(report):
 diag=P/'diagnostics';assert (diag/'SEALED.json').is_file(),'Wait for completed parent diagnostic seal before audit'
 ds=read(diag/'SEALED.json')
 for name,h in ds['files'].items():assert sha(diag/name)==h
 parent=read(diag/'RESULTS_PRIVATE.json');sc=read(diag/'SCORER_LOCK.json');report['parent_result_sha256']=sha(diag/'RESULTS_PRIVATE.json')
 assert sha(P/'score_diagnostics.py')==sc['scorer_sha256'];assert sha(B/'tools/audit_round2_3eed.py')==sc['polygon_iou_sha256'];assert sha(P/'PLAN_DIAGNOSTIC_LOCK.json')==sc['plan_sha256']
 pp=P/'p3/predictions';ps=read(pp/'SEALED.json');assert sha(pp/'SEALED.json')=='9813c7b75b1b7058db3dffc42e499b411af9a6acd268ef60a40c1f77d3c8250e'==sc['p3_seal_sha256']
 assert ps['n']==54 and len(ps['files'])==54
 for name,h in ps['files'].items():assert sha(pp/name)==h
 assert sha(pp/'trace.json')==ps['trace_sha256'];assert sha(pp/'input_index.json')==ps['input_index_sha256'];assert sha(P/'p3/RUN_LOCK.json')==ps['run_lock_sha256']
 for name,h in ps['source_hashes'].items():assert sha(name)==h
 fs=read(P/'p2/SEALED.json');assert fs['n']==54 and sha(P/'p2/SEALED.json')==sc['p2_seal_sha256']
 for name,h in fs['files'].items():assert sha(P/'p2'/name)==h
 assert sha(P/'p2/FEATURES_PRIVATE.json')=='b3c13a44a249f068eeb69d10594b35761b86a787b4c5d02281093b70a37bb87b'
 fl=read(P/'p2/CODE_INPUT_LOCK.json');assert sha(P/'p2_features.py')==fl['source_sha256'];assert sha(P/'p2/CODE_INPUT_LOCK.json')==fs['code_lock_sha256']
 report['all_P2_P3_parent_completion_source_code_and_prediction_seals_verified_before_target_access']=True
 receipt=read(OLD/'INPUT_PREPARATION_RECEIPT.json');features={x['id']:x for x in read(P/'p2/FEATURES_PRIVATE.json')['rows']};traces={x['id']:x for x in read(pp/'trace.json')}
 parentrows={(x['id'],x['arm']):x for x in parent['rows']};assert len(parentrows)==594 and len(parent['rows'])==594
 arms=parent['arms'];assert len(arms)==11 and len(ps['arms'])==9
 allrows=[];errors=[];bounds=[];fixed=[];perturb_errors=[];field_values={};scene_ids=defaultdict(set);gt_transform_sources=Counter();parent_pert={(x['id'],x['perturbation'],x['variant_index']):x for x in parent.get('P2_controlled_perturbation_rows',[])}
 for cohort in ['dev','confirm']:
  index=OLD/(cohort+'_scoring_index_PRIVATE.json');assert sha(index)==receipt['cohorts'][cohort]['scoring_index_sha256'];samples=read(index)
  ss=read(OLD/(cohort+'_source/sealed.json'));assert sha(OLD/'inputs'/(cohort+'_manifest.json'))==ss['input_manifest_sha256']==fl['input_manifests'][cohort]
  for sample in samples:
   sid=sample['id'];assert sid in features and sid in traces;scene_ids[sample['platform']].add(sample['scene']);gt_transform_sources[sample['platform']]+=1
   sourcefile=OLD/(cohort+'_source')/(sid+'.npz');assert sha(sourcefile)==ss['files'][sid+'.npz']
   with np.load(sourcefile,allow_pickle=False) as z:query=int(np.argmax(z['semantic_scores']));original=np.r_[z['boxes'][query].astype(float),0.]
   assert query==traces[sid]['source_query']==features[sid]['selected_query']
   with np.load(pp/(sid+'.npz'),allow_pickle=False) as z:boxes={('source' if a=='source_yaw0' else a):z[a].astype(float) for a in ps['arms']}
   source=boxes['source'];assert np.array_equal(original,source)
   with np.load(diag/(sid+'_ORACLE_PRIVATE.npz'),allow_pickle=False) as z:boxes.update({a:z[a].astype(float) for a in ['GT_yaw_injection','finite_yaw_search_best']})
   for a,box in boxes.items():
    if a!='legacy_joint_adapter3_reference':assert np.array_equal(box[:6],source[:6]),(sid,a,'Changed orientation-only center/size')
   fixed.append({'id':sid,'source_query':query,'nonhistorical_center_extents_unchanged':True,'legacy_joint_separate_capacity':True})
   # First numeric target read occurs here, after parent/P2/P3 seals and all source/index checks above.
   gt=manual_target(sample);values={a:convex_iou(b,gt) for a,b in boxes.items()};base=values['source']
   for a,value in values.items():
    given=parentrows[sid,a];err=abs(value-given['iou']);errors.append(err)
    allrows.append({'id':sid,'cohort':cohort,'platform':sample['platform'],'scene':sample['scene'],'arm':a,'iou':value,'delta':value-base})
   best=values['finite_yaw_search_best'];feasible=[v for a,v in values.items() if a!='legacy_joint_adapter3_reference']
   upper=contract.volume_height_upper_bound(source,gt);bound_violation=max(max(feasible)-best, best-upper,0.);bounds.append(bound_violation)
   assert boxes['GT_yaw_injection'][6]==gt[6]
   probe=features[sid]['trace'];baseyaw=probe.get('PCA_yaw');probe_box=source.copy();probe_box[6]=baseyaw if baseyaw is not None else source[6];probe_iou=convex_iou(probe_box,gt)
   for field in ['bootstrap_yaws','partial_visibility_yaws','expanded_yaws','background_yaws','shifted_yaws']:
    for j,theta in enumerate(probe.get(field,[])):
     box=source.copy();box[6]=theta if theta is not None else source[6];value=convex_iou(box,gt);given=parent_pert[sid,field,j]
     perturb_errors += [abs(value-given['perturbed_iou']),abs(probe_iou-given['unperturbed_probe_iou']),abs((value-probe_iou)-given['delta_unperturbed_probe'])]
     assert given['valid_PCA_yaw']==(theta is not None) and given['probe_min_unique_points']==3
 report.update(n_parents=54,n_arms=11,n_primary_IoUs=len(errors),max_abs_IoU_error=max(errors),P1_max_feasible_bound_violation=max(bounds),scene_clusters={p:len(v) for p,v in scene_ids.items()},platform_frames=dict(gt_transform_sources),orientation_contract_checks=fixed,P2_perturbation_rows_checked=len(parent_pert),P2_max_abs_perturbation_IoU_or_delta_error=max(perturb_errors,default=0.))
 assert report['scene_clusters']=={'waymo':18,'drone':2,'quad':3}
 # Independent primary scene estimands and stratified bootstrap using integer resampling.
 comparisons={x['arm']:x for x in parent['comparisons']};stats_checks=[];ci_errors=[];summary_checks=[]
 for arm in arms:
  sub=[x for x in allrows if x['arm']==arm];given=comparisons[arm];groups=defaultdict(list)
  for r in sub:groups[r['platform'],r['scene']].append(r['delta'])
  byplatform={p:np.array([np.mean(v) for (q,s),v in sorted(groups.items()) if q==p]) for p in ['waymo','drone','quad']};estimate=sum(float(v.mean())/3 for v in byplatform.values());rng=np.random.default_rng(20261030);replicates=np.zeros(10000)
  for p,v in byplatform.items():indices=rng.integers(0,len(v),size=(10000,len(v)));replicates+=v[indices].mean(1)/3
  ci=np.quantile(replicates,[.025,.975]);ci_errors.extend(abs(ci-np.asarray(given['platform_stratified_scene_bootstrap_95ci_descriptive'])))
  stats_checks.append({'arm':arm,'primary_scene_macro_delta':estimate,'parent_primary_delta':given['equalplatform_scene_macro_delta'],'primary_absolute_error':abs(estimate-given['equalplatform_scene_macro_delta']),'CI_abs_error_max':float(np.max(abs(ci-np.asarray(given['platform_stratified_scene_bootstrap_95ci_descriptive']))))})
  for p in ['all','waymo','drone','quad']:
   ar=[x for x in sub if p=='all' or x['platform']==p];g=next(x for x in parent['summaries'] if x['arm']==arm and x['platform']==p);assert len(ar)==g['n']
   assert sum(x['iou']>.25 for x in ar)==g['acc25_count'] and sum(x['iou']>.5 for x in ar)==g['acc50_count']
   summary_checks.append(abs(np.mean([x['iou'] for x in ar])-g['mean_iou']))
 report.update(scene_estimand_checks=stats_checks,max_abs_scene_macro_delta_error=max(x['primary_absolute_error'] for x in stats_checks),max_abs_scene_bootstrap_CI_error=max(ci_errors),max_abs_frame_summary_error=max(summary_checks))
 # Pairwise AUROC and hand-ranked Spearman; no sklearn/scipy rank/AUC implementations.
 associations=parent['P2_all_predeclared_associations'];assoc_checks=[];undefined_counts=Counter();classification_mismatches=[];max_rho_err=0.;max_auc_err=0.
 lookup={(r['id'],r['arm']):r for r in allrows}
 for g in associations:
  arm,name=g['arm'],g['feature'];eligible=[]
  for sid,f in features.items():
   x=f['features'].get(name)
   if x is not None:eligible.append((float(x),lookup[sid,arm]))
  xs=np.array([x for x,_ in eligible]);ys=np.array([r['delta'] for _,r in eligible]);changed=abs(ys)>1e-10;pos=ys[changed]>0;auc=auc_pairwise(xs[changed],pos);rr=rho(xs,ys)
  if int(pos.sum())!=g['positive_n'] or int((~pos).sum())!=g['negative_n'] or int((~changed).sum())!=g['unchanged_n']:classification_mismatches.append({'arm':arm,'feature':name,'independent_posneg_unchanged':[int(pos.sum()),int((~pos).sum()),int((~changed).sum())],'parent':[g['positive_n'],g['negative_n'],g['unchanged_n']]})
  assert len(eligible)==g['available_n'] and 54-len(eligible)==g['missing_n']
  if auc is None:undefined_counts['global_AUC']+=1;assert g['higher_feature_AUROC_no_direction_flip'] is None
  else:max_auc_err=max(max_auc_err,abs(auc-g['higher_feature_AUROC_no_direction_flip']))
  if rr is None:undefined_counts['global_Spearman']+=1;assert g['spearman_vs_delta'] is None
  else:max_rho_err=max(max_rho_err,abs(rr-g['spearman_vs_delta']))
  actual_scenes=[]
  for p,scene in sorted({(r['platform'],r['scene']) for _,r in eligible}):
   ee=[(x,r['delta']) for x,r in eligible if (r['platform'],r['scene'])==(p,scene) and abs(r['delta'])>1e-10];labels=np.array([d>0 for _,d in ee],dtype=bool);value=auc_pairwise([x for x,_ in ee],labels);oldscene=next(v for v in g['within_scene_AUROC'] if (v['platform'],v['scene'])==(p,scene));assert len(ee)==oldscene['n_changed']
   if value is None:undefined_counts['scene_AUC']+=1;assert oldscene['AUROC'] is None
   else:max_auc_err=max(max_auc_err,abs(value-oldscene['AUROC']))
   actual_scenes.append({'platform':p,'scene':scene,'positive':int(labels.sum()),'negative':int((~labels).sum()),'n_changed':len(ee),'AUROC':value})
  assoc_checks.append({'arm':arm,'feature':name,'available':len(eligible),'missing':54-len(eligible),'positive':int(pos.sum()),'negative':int((~pos).sum()),'unchanged':int((~changed).sum()),'AUROC_raw_higher_feature':auc,'Spearman':rr,'scene_AUC_denominators':actual_scenes})
 report.update(P2_association_rows_checked=len(associations),P2_max_AUROC_error=max_auc_err,P2_max_Spearman_error=max_rho_err,P2_numeric_classification_mismatches=classification_mismatches,P2_undefined_counts=dict(undefined_counts),P2_association_checks=assoc_checks)
 report['failures']=[]
 if max(errors)>1e-6:report['failures'].append('Independent3Dconvex primary IoU error exceeds1e-6')
 if max(bounds)>1e-6:report['failures'].append('Finite yaw lowerbound failed source/GTyaw/orientation-only feasible value or loose upper check')
 if max(perturb_errors,default=0)>1e-6:report['failures'].append('Perturbation IoU/delta error exceeds1e-6')
 if max(ci_errors)>1e-6 or max(summary_checks)>1e-6:report['failures'].append('Scene estimand/bootstrap/frame summary differs above1e-6')
 if classification_mismatches or max_auc_err>1e-10 or max_rho_err>1e-10:report['failures'].append('Association classification or rank statistic differs; do not silently relabel numerical ties')
 report['status']='passed' if not report['failures'] else 'measurement_difference_requires_parent_review'
 report['claims']={'GTyaw_is_not_maximum_or_guaranteed_improvement':True,'finite_best_feasible_lower_bound_not_certified_maximum':True,'volume_height_upper_separately_named_and_usually_loose':True,'all_old54_outcomes_exposed_posthoc_only':True,'perturbation_variants_not_new_independent_scenes':True,'no_descriptor_sign_flip_selector_fit_or_parameter_search_by_auditor':True,'source_query_identity_not_proven_correct_by_fixed_slot':True,'legacy_joint_capacity_group_kept_separate':True,'P0_long_training_cancelled_no_restart':True}
 report['new_model_calls']=0;report['source_or_prediction_or_parent_score_files_modified']=False
 report['reviewed_parent_SCORER_LOCK_sha256']=sha(diag/'SCORER_LOCK.json');report['auditor_source_sha256']=sha(pathlib.Path(__file__));report['independent_obb_source_sha256']=sha(B/'tools/obb_iou.py')
if __name__=='__main__':
 report={'kind':'independent_round3_CPU_3Dconvex_and_scene_rank_audit','status':'started'}
 try:main(report)
 except Exception as exc:report.update(status='failed_no_silent_skip',exception=repr(exc),traceback=traceback.format_exc());raise
 finally:
  (R/'INDEPENDENT_RESULTS_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:report.get(k) for k in ['status','n_parents','n_arms','max_abs_IoU_error','P1_max_feasible_bound_violation','scene_clusters','max_abs_scene_macro_delta_error','max_abs_scene_bootstrap_CI_error','P2_max_AUROC_error','P2_max_Spearman_error','P2_numeric_classification_mismatches','failures','exception']},indent=2))
