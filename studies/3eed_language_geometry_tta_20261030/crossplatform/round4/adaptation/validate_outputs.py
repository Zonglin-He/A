"""Post-seal label-free verification only; never imports a scorer or targetmetadata."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def validate(cohort):
 p=R/f'{cohort}_predictions';s=json.loads((p/'SEALED.json').read_text());t=json.loads((p/'trace.json').read_text());assert s['n']==len(t)
 assert sha(p/'trace.json')==s['trace_sha256'];assert sha(R/'CODE_LOCK.json')==s['code_lock_sha256']
 lock=json.loads((R/'CODE_LOCK.json').read_text())
 for name,h in lock['source_hashes'].items():assert sha(Path(name))==h,name
 for name,h in s['files'].items():assert sha(p/name)==h
 for name,h in s['diagnostic_files'].items():assert sha(p/'diagnostics'/name)==h
 counts={'region_parsed':0,'spatial_informative':0,'geometry_observed':0,'geometry_violated':0,'query_nonzero_gradient':0,'query_changed_scores':0,'query_changed_selection_vs_aug':0,'geometry_nonzero_gradient':0,'geometry_changed_center':0};axes=0;qmax=0.;gmax=0.;lossdiffq=0.;lossdiffg=0.;geomhist={};platform={}
 for r in t:
  sid=r['id'];counts['region_parsed']+=r['caption_region'] is not None;counts['spatial_informative']+=r['spatial_teacher']['informative'];ng=r['geometry_teacher'].get('observable_axes',0);nv=r['geometry_teacher'].get('violated_observable_axes',0);axes+=ng;counts['geometry_observed']+=ng>0;counts['geometry_violated']+=nv>0;geomhist[str(ng)]=geomhist.get(str(ng),0)+1
  pc=platform.setdefault(r['platform'],{'n':0,'spatial_informative':0,'geometry_observed':0});pc['n']+=1;pc['spatial_informative']+=int(r['spatial_teacher']['informative']);pc['geometry_observed']+=int(ng>0)
  q=r['query'];g=r['geometry'];qc=r['query_no_spatial'];gc=r['geometry_control'];assert q['zero_bridge_to_TTAug_exact'] and q['reset_to_TTAug_exact'] and g['episodic_reset_source_exact'] and g['unobserved_axes_exact_zero'];assert qc['reset_to_TTAug_exact'] and gc['episodic_reset_source_exact']
  counts['query_nonzero_gradient']+=any(x['gradient_norm']>0 for x in q['trace']);counts['geometry_nonzero_gradient']+=any(x['gradient_norm']>0 for x in g['trace'])
  for x in q['trace']:
   assert x['max_embedding_step_L2']<=.05+1e-7 and x['max_embedding_total_L2']<=.1+1e-7;assert x['loss_after']<=x['loss_before']+1e-10;qmax=max(qmax,x['max_embedding_total_L2']);lossdiffq=max(lossdiffq,x['loss_after']-x['loss_before'])
  for x in g['trace']:
   assert x['latent_delta_L2_m']<=.15+1e-12;assert x['total_loss_after']<=x['total_loss_before']+1e-12;gmax=max(gmax,x['latent_delta_L2_m']);lossdiffg=max(lossdiffg,x['total_loss_after']-x['total_loss_before'])
  for x in qc['trace']:assert x['gradient_norm']==0 and x['parameter_delta_l2']==0 and x['max_embedding_total_L2']==0
  for x in gc['trace']:assert x['gradient_norm']==0 and x['latent_delta_L2_m']==0
  with np.load(p/(sid+'.npz'),allow_pickle=False) as z:
   assert set(z.files)==set(s['arms']);base=z['frozen_PCA_primarybaseline'];assert np.array_equal(z['query0_control'],z['TTAug_3fullviews_PCA_no_update']) and np.array_equal(z['query_no_spatial3_control'],z['TTAug_3fullviews_PCA_no_update']) and np.array_equal(z['geometry_no_observation3_control'],base)
   for a in s['arms']:
    assert z[a].shape==(7,) and np.isfinite(z[a]).all()
    if a.startswith('geometry'):assert np.array_equal(z[a][3:],base[3:])
   counts['geometry_changed_center']+=not np.array_equal(z['geometry3_primary'][:3],base[:3])
  with np.load(p/'diagnostics'/(sid+'_scores.npz'),allow_pickle=False) as z:
   counts['query_changed_scores']+=not np.array_equal(z['query3'],z['query0']);counts['query_changed_selection_vs_aug']+=int(z['query3'].argmax())!=int(z['TTAug'].argmax());assert np.array_equal(z['query0'],z['TTAug']);assert np.array_equal(z['query_no_spatial3'],z['TTAug'])
 out={'state':'GTfree_postseal_engineering_pass','cohort':cohort,'n':s['n'],'arms':len(s['arms']),'seal_sha256':sha(p/'SEALED.json'),'code_lock_sha256':sha(R/'CODE_LOCK.json'),'all_bound_code_source_prediction_diagnostic_hashes_verified':True,'counts':counts,'total_observed_axes':axes,'geometry_axis_histogram':geomhist,'platform_counts':platform,'max_embedding_total_L2':qmax,'max_center_local_L2_m':gmax,'max_unlabeled_loss_increase_query':lossdiffq,'max_unlabeled_loss_increase_geometry':lossdiffg,'all_episodic_resets_exact':True,'zero_and_noinfo_controls_bitexact':True,'sizes_yaw_unchanged':True,'seconds_total_cached_CPU_all_13_arms':sum(x['seconds'] for x in t),'seconds_median':float(np.median([x['seconds'] for x in t])),'cost_scope':'cached CPU downstream arms only; excludes frozen neural forwards, preprocessing, calibration, prior engineering; not deployment pipeline latency','numeric_GT_or_target_IoU_scores_read':False,'method_source_or_predictions_modified':False,'support_is_not_task_correctness':True,'validator_sha256':sha(Path(__file__))}
 (R/f'{cohort.upper()}_OUTPUT_VALIDATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':validate(sys.argv[1])
