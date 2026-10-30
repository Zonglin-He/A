"""Independent3Dconvex verification of five separately sealed posthoc full-box arms.
Does not alter primary594 audit, predictions, scorer, parameters or any old files.
"""
import pathlib,json,hashlib,traceback
from collections import defaultdict
import numpy as np
from independent_results_audit import manual_target,convex_iou
R=pathlib.Path(__file__).resolve().parent;B=R.parents[2];Q=B/'crossplatform/round3';OLD=Q.parent/'round2'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def main(out):
 primary=read(R/'INDEPENDENT_RESULTS_AUDIT.json');assert primary['status']=='passed';out['primary594_audit_sha256_preserved']=sha(R/'INDEPENDENT_RESULTS_AUDIT.json')
 pred=Q/'p3/supplement/predictions';seal=read(pred/'SEALED.json');expected='166117e309e72d54fbf07b550fa865331c3a4705b3d10c3e1ef4f6dbb8da6d9d';assert sha(pred/'SEALED.json')==expected and seal['n']==54 and len(seal['arms'])==5 and not seal['orientation_only']
 for n,h in seal['files'].items():assert sha(pred/n)==h
 assert sha(pred/'trace.json')==seal['trace_sha256'];assert sha(Q/'p3/supplement/PLAN_LOCK.json')==seal['plan_sha256'];assert sha(Q/'p3/RUN_LOCK.json')==seal['original_run_lock_sha256'];assert sha(Q/'p3/predictions/SEALED.json')==seal['original_prediction_seal_sha256']
 for f,h in seal['code_hashes'].items():assert sha(f)==h
 result=Q/'full_refit_diagnostics/RESULTS_PRIVATE.json';raw=result.read_bytes();scores=json.loads(raw);sc=read(Q/'full_refit_diagnostics/SCORER_LOCK.json');assert sha(Q/'score_full_refit.py')==sc['scorer_sha256'] and sha(Q/'score_diagnostics.py')==sc['imported_scorer_sha256'];assert sc['prediction_seal_sha256']==expected;assert sha(Q/'diagnostics/RESULTS_PRIVATE.json')==sc['previous_result_sha256']==primary['parent_result_sha256']
 by={(x['id'],x['arm']):x for x in scores['rows']};assert len(by)==270;trace={x['id']:x for x in read(pred/'trace.json')};original={x['id']:x for x in read(Q/'p3/predictions/trace.json')}
 out['all_new5arm_prediction_plan_and_code_seals_verified_before_targets']=True
 errors=[];rows=[];volume=defaultdict(list);valid_counts=defaultdict(int);receipt=read(OLD/'INPUT_PREPARATION_RECEIPT.json')
 for cohort in ['dev','confirm']:
  index=OLD/(cohort+'_scoring_index_PRIVATE.json');assert sha(index)==receipt['cohorts'][cohort]['scoring_index_sha256']
  for s in read(index):
   sid=s['id'];t=trace[sid];old=original[sid];assert t['source_query']==old['source_query'] and t['crop_sha256_float64_xyz']==old['diagnostics']['crop']['crop_sha256_float64_xyz'] and t['crop_points']==old['diagnostics']['crop']['crop_points']
   gt=manual_target(s)
   with np.load(Q/'p3/predictions'/(sid+'.npz'),allow_pickle=False) as z:source=z['source_yaw0'].astype(float)
   base=convex_iou(source,gt)
   with np.load(pred/(sid+'.npz'),allow_pickle=False) as z:
    for a in seal['arms']:
     box=z[a].astype(float);assert box.shape==(7,) and np.isfinite(box).all();v=convex_iou(box,gt);g=by[sid,a];errors.extend([abs(v-g['iou']),abs(base-g['source_iou']),abs(v-base-g['delta_source'])]);assert g['capability_group']=='classical_full_center_size_yaw_refit';rows.append({'id':sid,'platform':s['platform'],'scene':s['scene'],'arm':a,'iou':v,'delta':v-base});volume[a].append(float(np.prod(box[3:6])/np.prod(source[3:6])));valid_counts[a]+=int((box[3:6]>0).all())
 means=[];ci_errors=[];macro_errors=[];counts=[]
 for a in seal['arms']:
  rr=[x for x in rows if x['arm']==a];grouped=defaultdict(list)
  for x in rr:grouped[x['platform'],x['scene']].append(x['delta'])
  byp={p:np.array([np.mean(v) for (q,s),v in sorted(grouped.items()) if q==p]) for p in ['waymo','drone','quad']};est=sum(v.mean()/3 for v in byp.values());rng=np.random.default_rng(20261030);boot=np.zeros(10000)
  for p,v in byp.items():boot+=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)/3
  g=next(x for x in scores['comparisons'] if x['arm']==a);ci_errors.extend(abs(np.quantile(boot,[.025,.975])-g['platform_stratified_scene_bootstrap_95ci_descriptive']));macro_errors.append(abs(est-g['equalplatform_scene_macro_delta']))
  for p in ['all','waymo','drone','quad']:
   subset=[x for x in rr if p=='all' or x['platform']==p];summary=next(x for x in scores['summaries'] if x['arm']==a and x['platform']==p);assert len(subset)==summary['n'];errors.append(abs(np.mean([x['iou'] for x in subset])-summary['mean_iou']));assert sum(x['iou']>.25 for x in subset)==summary['acc25_count'] and sum(x['iou']>.5 for x in subset)==summary['acc50_count']
  means.append({'arm':a,'frame_mean_iou':float(np.mean([x['iou'] for x in rr])),'equalplatform_scene_delta':float(est),'sceneCI':np.quantile(boot,[.025,.975]).tolist(),'valid_geometry_n':valid_counts[a],'volume_ratio_vs_source_mean':float(np.mean(volume[a])),'volume_ratio_vs_source_median':float(np.median(volume[a])),'volume_smaller_source_n':int(sum(v<1-1e-10 for v in volume[a]))})
 assert result.read_bytes()==raw and sha(R/'INDEPENDENT_RESULTS_AUDIT.json')==out['primary594_audit_sha256_preserved']
 out.update(status='passed',n_parents=54,n_new_fullbox_arms=5,n_independent_IoUs=270,max_abs_IoU_source_delta_or_summary_error=max(errors),max_abs_scene_primary_delta_error=max(macro_errors),max_abs_sceneCI_error=max(ci_errors),arms=means,prediction_seal_sha256=expected,scored_result_sha256=hashlib.sha256(raw).hexdigest(),scope='Explicit posthoc optional full-refit after initial54outcomes exposed; same sourcequery/crop, center/size/yawinclZ maychange; notorientationonly, notTTA or newconfirmation',P0_training_cancelled_no_restart=True,raw_GT_or_coordinate_arrays_written=False,new_model_calls=0,old_or_parent_files_modified=False,auditor_source_sha256=sha(pathlib.Path(__file__)))
 assert max(errors)<1e-6 and max(macro_errors)<1e-6 and max(ci_errors)<1e-6
if __name__=='__main__':
 out={'status':'started','kind':'independent_supplemental_fullbox_3Dconvex_audit'}
 try:main(out)
 except Exception as e:out.update(status='failed_no_silent_skip',exception=repr(e),traceback=traceback.format_exc());raise
 finally:
  (R/'INDEPENDENT_FULL_REFIT_AUDIT.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
