"""Read-only independent final-code/CONFIRM audit. No model or adaptation; confirmation GT explicitly parent-authorized afterallarmsseal.
Raw/private GT-derived diagnostics stay here and never become adapter inputs.
"""
from pathlib import Path
import json,hashlib,sys,math
from collections import Counter
import numpy as np
R=Path(__file__).resolve().parent;B=R.parents[2];P=B/'crossplatform/round4';S=P/'source';A=P/'adaptation';D=A/'confirm_predictions';C=P/'confirm_readout_controls'
sys.path.insert(0,str(R));from candidate_decomposition import metric,rot,core,aggregate
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,x):
 p=R/name;assert not p.exists(),str(p);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(folder,seal):
 for n,h in seal['files'].items():assert sha(folder/n)==h,n
 if 'trace_sha256' in seal:assert sha(folder/'trace.json')==seal['trace_sha256']
def pair(rows,a,b):
 v=np.array([x['iou'][a]-x['iou'][b] for x in rows]);sc=sorted({x['scene'] for x in rows});sv=np.array([np.mean([x['iou'][a]-x['iou'][b] for x in rows if x['scene']==s]) for s in sc]);rng=np.random.default_rng(20261030)
 ci=np.quantile(rng.choice(sv,(10000,len(sv)),replace=True).mean(1),[.025,.975]).tolist() if len(sc)>1 else None
 return {'a':a,'b':b,'mean_delta':float(v.mean()),'scene_macro_delta':float(sv.mean()),'scene_cluster_descriptive95':ci,'n_scenes':len(sc),'positive':int((v>1e-10).sum()),'negative':int((v< -1e-10).sum()),'same':int((abs(v)<=1e-10).sum()),'all_signed_tails':[{'id':x['id'],'delta':float(x['iou'][a]-x['iou'][b])} for x in sorted(rows,key=lambda x:x['iou'][a]-x['iou'][b])]}
def main():
 master=P/'CONFIRMATION_ALL_ARMS_SEALED.json';assert sha(master)=='080f4a00d1d4a77323a1012e2a8ee0f0cc4d858f67aa8dfcc06ef8dd422c786a';masterdata=read(master);assert masterdata['all_confirm_arms_sealed_before_numeric_GT'] and masterdata['no_confirm_numeric_GT_access_before_this_gate']
 for f,h in masterdata['all_sealed_files_to_verify'].items():assert sha(f)==h,f
 assert sha(A/'CODE_LOCK.json')=='f409fa13165631707703112cb4a44da8adfd80f6bfb1b7a406fdf8f779413fff';lock=read(A/'CODE_LOCK.json')
 for f,h in lock['source_hashes'].items():assert sha(f)==h,f
 for f,h in lock['fixtures'].items():assert sha(A/f)==h,f
 assert sha(D/'SEALED.json')=='31f3ffbe49c9f59c0588d98e4fc1292c52f670bf049cb3a49617ea04ec58cb5f'
 ds=read(D/'SEALED.json');cs=read(C/'sealed.json');ss=read(S/'confirm_source/sealed.json')
 for folder,se in [(D,ds),(C,cs),(S/'confirm_source',ss)]:verify(folder,se)
 for n,h in ds['diagnostic_files'].items():assert sha(D/'diagnostics'/n)==h
 assert sha(C/'CODE_INPUT_LOCK.json')==cs['code_lock_sha256'];cl=read(C/'CODE_INPUT_LOCK.json');assert sha(P/'build_readout_controls.py')==cl['source_sha256']
 bbs=S/'bbs_head_supplement/confirm_replayed_gpu_confirm';bs=read(bbs/'sealed.json');assert sha(bbs/'sealed.json')==cl['BBS_seal_sha256'];verify(bbs,bs)
 mp=S/'inputs/confirm_manifest.json';assert sha(mp)==ss['manifest_sha256'];manifest=read(mp);trace={x['id']:x for x in read(D/'trace.json')};ctrace={x['id']:x for x in read(C/'trace.json')}
 assert set(trace)==set(ctrace)=={x['id'] for x in manifest} and len(manifest)==64
 # Verify complete prediction universe and final source receipts BEFORE opening scoring index or target metadata.
 old=R/'confirm_decomposition_v1';old.mkdir(exist_ok=False)
 idx=S/'confirm_scoring_index_PRIVATE.json';assert sha(idx)==read(S/'INPUT_RECEIPT.json')['cohorts']['confirm']['scoring_index_sha256']
 dump('STAGE2_CONFIRM_SCORER_LOCK.json',{'scorer_sha256':sha(__file__),'metric_source_sha256':sha(R/'candidate_decomposition.py'),'3Dconvex_sha256':sha(B/'tools/obb_iou.py'),'code_lock_sha256':sha(A/'CODE_LOCK.json'),'prediction_seal_sha256':sha(D/'SEALED.json'),'readout_seal_sha256':sha(C/'sealed.json'),'all384_source_832_adaptation_384_readout_predictions_and_diagnostics_verified_before_CONFIRM_GT':True,'master981_all_files_verified_before_target_GT':True,'all_arms_master_sha256':sha(master),'source_BBS_files_verified':len(bs['files']),'CONFIRM_scoring_index_sha256':sha(idx),'confirm_GT_access':True,'authorization':'Parent all-arms master080f4a + explicit64confirmationGT authorization; independentaudit afterpredictionseal'})
 # Numeric engineering from sealed outputs/traces, independent direct formulas.
 eng=[];teacher=[];rows=[];decomprows=[];associations=[];byid={x['id']:x for x in manifest};samples=read(idx);arms=ds['arms']+cs['arms'];assert len(set(arms))==len(arms)
 for s in samples:
  sid=s['id'];tr=trace[sid];pred=arrays(D/(sid+'.npz'));controls=arrays(C/(sid+'.npz'));diag=arrays(D/'diagnostics'/(sid+'_scores.npz'));ref=arrays(S/'confirm_source'/(sid+'_v0_full.npz'));allpred={**pred,**controls};canonical=ref['boxes_pca'];q0=int(ref['semantic_scores'].argmax());base=canonical[q0]
  assert np.array_equal(pred['query0_control'],pred['TTAug_3fullviews_PCA_no_update']) and np.array_equal(pred['query_no_spatial3_control'],pred['query0_control']);assert np.array_equal(pred['geometry_no_observation3_control'],base)
  assert np.array_equal(controls['frozen_PCA'],base) and np.array_equal(pred['frozen_PCA_primarybaseline'],base)
  assert np.array_equal(diag['query0'],diag['TTAug']) and np.array_equal(diag['query_no_spatial3'],diag['TTAug'])
  for name,q in tr['selected_queries'].items():
   if name.startswith('geometry'):assert q==q0 and np.array_equal(pred[name][3:],base[3:])
   elif name!='frozen_raw_diagnosticonly':assert np.array_equal(pred[name],canonical[q])
  for name,q in ctrace[sid]['chosen_queries'].items():
   if name!='frozen_raw':assert np.array_equal(controls[name],canonical[q])
  qt=tr['query'];qc=tr['query_no_spatial'];gt=tr['geometry'];gc=tr['geometry_control'];obs=diag['geometry_observed'];lo=diag['geometry_interval_lower'];hi=diag['geometry_interval_upper'];gdelta=np.asarray(gt['trace'][-1]['center_offset_local_m']);direct=np.clip(np.zeros(3),lo,hi)*obs/1.25;direct*=min(1,.15/max(np.linalg.norm(direct),1e-99))
  assert qt['parameter_count']==512 and qt['zero_bridge_to_TTAug_exact'] and qt['reset_to_TTAug_exact'];assert gt['parameter_count']==3 and gt['episodic_reset_source_exact'];assert np.array_equal(gdelta[~obs],np.zeros((~obs).sum()))
  expected=base.copy();expected[:3]=(base[:3].astype(float)+direct@rot(base[6]).T).astype(base.dtype);assert np.array_equal(expected,pred['geometry_direct_PCA_no_update'])
  maxstep=max(x['max_embedding_step_L2'] for x in qt['trace']);maxtotal=max(x['max_embedding_total_L2'] for x in qt['trace']);assert maxstep<=.0500001 and maxtotal<=.1000001
  assert all(x['loss_after']<=x['loss_before']+1e-10 for x in qt['trace']);assert all(x['total_loss_after']<=x['total_loss_before']+1e-12 for x in gt['trace']);assert all(x['parameter_delta_l2']==0 and x['gradient_norm']==0 for x in qc['trace']);assert all(x['latent_delta_L2_m']==0 and x['gradient_norm']==0 for x in gc['trace'])
  assert np.linalg.norm(gdelta)<=.15000001
  # Output float32 center projection may add source-coordinate rounding; report actual deviation separately.
  observed_output_delta=(pred['geometry3_primary'][:3].astype(float)-base[:3].astype(float))@rot(base[6]);rounderr=float(np.linalg.norm(observed_output_delta-gdelta));assert rounderr<1e-5
  prior=diag['TTAug'].astype(float);prior=np.maximum(prior,1e-12);prior/=prior.sum();pen=diag['spatial_penalty'];expected_teacher=prior*np.exp(-pen);expected_teacher/=expected_teacher.sum();expected_teacher=.9*expected_teacher+.1*prior
  # Original prior normalized in torch float32 then teacher float64; preserve native math tolerance explicitly.
  teachererr=float(np.max(abs(expected_teacher-diag['teacher'])));assert teachererr<2e-8
  informative=bool(tr['spatial_teacher']['informative']);assert informative==qt['informative_teacher']
  if not informative:assert np.array_equal(diag['query3'],diag['TTAug'])
  eng.append({'id':sid,'platform':s['platform'],'informative_spatial':informative,'parsed':tr['caption_region'] is not None,'query_nonzero_gradient_steps':sum(x['gradient_norm']>0 for x in qt['trace']),'query_actual_parameter_update':qt['trace'][-1]['parameter_delta_l2']>0,'query_parameter_delta_l2':qt['trace'][-1]['parameter_delta_l2'],'max_embedding_step_L2':maxstep,'max_embedding_total_L2':maxtotal,'geometry_observed_axes':int(obs.sum()),'geometry_gradient_steps':sum(x['gradient_norm']>0 for x in gt['trace']),'geometry_actual_latent_norm_m':float(np.linalg.norm(gdelta)),'geometry_output_rounding_error_m':rounderr,'teacher_max_reconstruction_abs_error':teachererr,'allzero_reset_bounds_and_direct_formula_assertions':True,'seconds_cpu_cached_adaptation':tr['seconds']})
  meta=read(s['meta_path']);target=np.asarray(meta['ground_info'][s['object_index']]['bbox_3d'][:7],float)
  if s['platform']=='drone':target[2]+=1.8
  if s['platform']=='waymo':angle=math.radians([0,-45,45,-90,90][s['waymo_view_suffix']]);target[:3]=rot(angle)@target[:3];target[6]+=angle
  values={name:metric(b,target) for name,b in allpred.items()};row={'id':sid,'platform':s['platform'],'split':s['split'],'scene':s['scene'],'iou':values,'invalid_counts':{name:int((b[3:6]<=0).any() or not np.isfinite(b).all()) for name,b in allpred.items()}};rows.append(row)
  # Sealed per-candidate IoUs from independently scored reference pool; no candidate changes.
  decomp,inter,err,allious=core(ref['boxes'],canonical,ref['semantic_scores'],target);val=allious['fixedPCA'];np.savez_compressed(old/(sid+'_GT_IOU_PRIVATE.npz'),**allious);decomprows.append({'id':sid,'platform':s['platform'],'split':s['split'],'scene':s['scene'],'geometries':decomp,'interventions':inter,'parameter_error_conditioned_on_selected_query':err,'annotated_object_count':len(meta['ground_info'])});assert abs(values['frozen_PCA']-val[q0])<1e-12
  tq=int(diag['teacher'].argmax());aq=int(diag['TTAug'].argmax());sq=int(diag['source'].argmax());best=int(val.argmax());post=diag['teacher']/diag['teacher'].sum();trueoff=(target[:3]-base[:3])@rot(base[6]);viol=lambda d:np.maximum(np.maximum(lo-d,d-hi),0)*obs
  teacher.append({'id':sid,'platform':s['platform'],'scene':s['scene'],'parsed':tr['caption_region'] is not None,'informative_spatial':informative,'spatial_direct_changed_query_vs_aug':tq!=aq,'spatial_direct_top1_iou':float(val[tq]),'aug_top1_iou':float(val[aq]),'source_top1_iou':float(val[sq]),'teacher_expected_candidate_iou':float(post@val),'prior_expected_candidate_iou':float(prior@val),'teacher_mass_IoU_gt50':float(post[val>.5].sum()),'prior_mass_IoU_gt50':float(prior[val>.5].sum()),'teacher_mass_best256_candidate':float(post[best]),'prior_mass_best256_candidate':float(prior[best]),'geometry_observed_axes':int(obs.sum()),'geometry_interval_contains_GTcenter_axis_count':int(((trueoff>=lo)&(trueoff<=hi)&obs).sum()),'geometry_interval_violation_squared_at_source':float(viol(np.zeros(3))@viol(np.zeros(3))),'geometry_interval_violation_squared_at_GTcenter':float(viol(trueoff)@viol(trueoff)),'geometry_direct_center_distance_to_GT_change_m':float(np.linalg.norm(direct-trueoff)-np.linalg.norm(trueoff)),'geometry3_center_distance_to_GT_change_m':float(np.linalg.norm(gdelta-trueoff)-np.linalg.norm(trueoff)),'interpretation':'postsealCONFIRMdescription; selectedquery fixed forgeometry, GTcenter may be anotherinstance; no tuning/confirmselection'})
  obj=meta['ground_info'][s['object_index']];matches=[]
  for other in obj.get('others',[]):
   ob=np.asarray(other['bbox_3d_other'][:7],float)
   if s['platform']=='drone':ob[2]+=1.8
   if s['platform']=='waymo':ob[:3]=rot(angle)@ob[:3];ob[6]+=angle
   if metric(ob,target)>1-1e-8:continue
   matches.append({'iou_to_q0':metric(base,ob),'same_class':str(other.get('class_other','')).strip().lower()==str(obj.get('class','')).strip().lower()})
  bestother=max(matches,key=lambda x:x['iou_to_q0']) if matches else None;associations.append({'id':sid,'platform':s['platform'],'scene':s['scene'],'target_iou':values['frozen_PCA'],'others_n':len(obj.get('others',[])),'best_nontarget_annotation':bestother,'rankrecoverable50':bool(val[q0]<=.5 and val.max()>.5),'q0_other_gt50_target_le50':bool(bestother and bestother['iou_to_q0']>.5 and val[q0]<=.5),'identity_not_proven':True})
  print('INDEPENDENT_CONFIRM',sid,flush=True)
 summaries=[];pairs=[]
 comparisons=[('query3_primary','frozen_PCA'),('query3_primary','TTAug_3fullviews_PCA_no_update'),('query3_primary','spatial_teacher_direct_PCA_no_update'),('query1','query3_primary'),('geometry3_primary','frozen_PCA'),('geometry3_primary','geometry_direct_PCA_no_update'),('geometry1','geometry3_primary'),('frozen_BBS_PCA','frozen_PCA'),('simple_TTAug_PCA','frozen_PCA')]
 for plat in ['waymo','drone','quad']:
  part=[r for r in rows if r['platform']==plat]
  for arm in arms:
   v=np.array([r['iou'][arm] for r in part]);sc=sorted({r['scene'] for r in part});summaries.append({'platform':plat,'scope':'primaryVAL' if plat=='waymo' else 'TRAIN_unionexposed_auxiliary','arm':arm,'n':len(v),'scene_clusters':len(sc),'mean_iou':float(v.mean()),'scene_macro_iou':float(np.mean([np.mean([r['iou'][arm] for r in part if r['scene']==s]) for s in sc])),'acc25_count':int((v>.25).sum()),'acc50_count':int((v>.5).sum()),'invalid_count':sum(r['invalid_counts'][arm] for r in part)})
  pairs.extend([dict(platform=plat,**pair(part,a,b)) for a,b in comparisons])
 er={'final_code_and_all_input_prediction_hashes_passed':True,'n':64,'rows':eng,'aggregate':{'parsed':sum(x['parsed'] for x in eng),'informative_spatial':sum(x['informative_spatial'] for x in eng),'episodes_query_parameter_updated':sum(x['query_actual_parameter_update'] for x in eng),'episodes_geometry_latent_updated':sum(x['geometry_actual_latent_norm_m']>0 for x in eng),'observed_geometry_axes':sum(x['geometry_observed_axes'] for x in eng),'max_embedding_step_L2':max(x['max_embedding_step_L2'] for x in eng),'max_embedding_total_L2':max(x['max_embedding_total_L2'] for x in eng),'max_geometry_rounding_error_m':max(x['geometry_output_rounding_error_m'] for x in eng)},'evidence_limit':'Read actual sealed outputs/grad/reset traces plus final code assertions; auditor did not rerun adaptation or inspect serialized trainable tensors (none stored). No fullmodel adaptation. Query512 is featureadapter; geometry3 is outputlatent optimization.'}
 dump('CONFIRM_STAGE2_IMPLEMENTATION_REVIEW.json',er);dump('CONFIRM_STAGE2_INDEPENDENT_RESULTS_PRIVATE.json',{'scope':'CONFIRM64 after all-arms master981 seal; independent3Dconvex; Waymo48 primary vsTRAINaux16','lock_sha256':sha(R/'STAGE2_CONFIRM_SCORER_LOCK.json'),'arms':arms,'rows':rows,'summaries':summaries,'paired_comparisons':pairs});dump('CONFIRM_TEACHER_QUALITY_PRIVATE.json',{'scope':'postseal descriptive CONFIRMonly; GTneverfedtoadapters; no parameterselection','rows':teacher,'boundaries':['candidatebest is geometric recall notidentity','globalnormalization can increase unsupported mass','nearface normal support doesnot prove sameinstance/trueocclusion','parser uses finiteobjectwordlist and doesnot rejectallnegations; examples recorded separately']})
 decompresult={'scope':'CONFIRM64 source-candidate independent3D afterallarmsseal; no adaptationinput','summaries':aggregate(decomprows),'rows':decomprows};(old/'RESULTS_PRIVATE.json').write_text(json.dumps(decompresult,indent=2)+'\n');(old/'SEALED.json').write_text(json.dumps({'files':{p.name:sha(p) for p in old.iterdir() if p.is_file()}},indent=2)+'\n');dump('CONFIRM_ANNOTATION_ASSOCIATION_PRIVATE.json',{'scope':'postsealCONFIRMdescriptive; no stableIDs/exhaustiveness; no methodchange','rows':associations})
 print(json.dumps([x for x in summaries if x['platform']=='waymo'],indent=2));print(json.dumps(er['aggregate'],indent=2))
if __name__=='__main__':main()
