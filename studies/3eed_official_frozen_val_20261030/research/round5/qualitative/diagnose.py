"""Post-seal CPU descriptive error decomposition. Never modifies frozen predictions.
Full execution requires parent explicit qualitative release. Smoke only is authorized
by the existing48-expression smoke score release. No media pixels are read here.
"""
from pathlib import Path
import argparse,json,hashlib,sys,time,collections,math
import numpy as np
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parent;B=R.parents[2];E=B/'crossplatform/round5/baseline_engineering'
sys.path.insert(0,str(R.parent/'scoring'))
from batch_yaw_iou import many_iou
PRIMARY='official_full_last_bbf';FULL='text_full_last_bbf';NOUN='text_noun_last_bbf'
ARMS=[PRIMARY,FULL,NOUN];SALT='3eed-round5-qualitative-v1|'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def target(raw,platform,suffix):
 b=np.asarray(raw[:7],np.float64).copy()
 if platform=='drone':b[2]+=1.8
 if platform=='waymo':
  a=np.deg2rad([0,-45,45,-90,90][suffix]);c,s=np.cos(a),np.sin(a);b[:2]=np.array([[c,-s],[s,c]])@b[:2];b[6]+=a
 return b.astype(np.float32).astype(np.float64)
def aabb_pair_iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);inter=np.maximum(0,np.minimum(a[:,:3]+a[:,3:6]/2,b[:,:3]+b[:,3:6]/2)-np.maximum(a[:,:3]-a[:,3:6]/2,b[:,:3]-b[:,3:6]/2)).prod(-1);return inter/(a[:,3:6].prod(-1)+b[:,3:6].prod(-1)-inter)
def correspond(a,b):
 # Identical R4 scale form, no class/target/score used to assign proposal rows.
 scale=np.maximum(np.median(np.abs(a[:,3:6]),axis=0),.5);cost=np.linalg.norm((a[:,None,:3]-b[None,:,:3])/scale,axis=-1)+np.linalg.norm((a[:,None,3:6]-b[None,:,3:6])/scale,axis=-1);rr,cc=linear_sum_assignment(cost);assert np.array_equal(rr,np.arange(len(a)));return cc,cost[rr,cc]
def rank_bounds(scores,q):
 return [int(np.sum(scores>scores[q])+1),int(np.sum(scores>=scores[q]))]
def stratum(iou,oracle,other):
 if iou>.5:return 'success'
 if iou<=.25 and other>.5:return 'annotated_sameclass_other_instance'
 if .25<iou<=.5 and oracle>.5 and other<=.5:return 'localization_boundary'
 if oracle<=.5:return 'candidate_coverage_failure50'
 return 'other_failure'
def verify_inputs(cache,manifest,release,scored,mode):
 rel=read(release);assert rel['all_predictions_sealed_before_numeric_GT'];sp=cache/'sealed.json'
 if not sp.exists():sp=cache/'SEALED.json'
 seal=read(sp);assert sha(sp)==rel['prediction_seal_sha256'];assert sha(manifest)==rel['manifest_sha256']
 if mode=='full':assert rel.get('qualitative_and_GT_analysis_authorized') is True,'Full media/diagnosis needs explicit parent qualitative release'
 else:
  assert len(read(manifest))==48 and sha(manifest)==sha(E/'inputs/smoke48.json');assert sha(release)==sha(R.parent/'SMOKE_SCORE_RELEASE.json')
 ss=read(scored/'SEALED.json');assert sha(scored/'RESULTS_PRIVATE.json')==ss['files']['RESULTS_PRIVATE.json'];result=read(scored/'RESULTS_PRIVATE.json');assert result['prediction_seal_sha256']==sha(sp)
 for name,h in seal['files'].items():assert sha(cache/name)==h,name
 for row in result['rows']:
  ip=scored/(row['id']+'_IOU_PRIVATE.npz');assert sha(ip)==ss['files'][ip.name]
 return seal,result,sp

def main(a):
 began=time.perf_counter();seal,result,sp=verify_inputs(a.cache,a.manifest,a.release,a.scored,a.mode);inputs={s['id']:s for s in read(a.manifest)};assert len(inputs)==result['n'];idx={s['id']:s for s in read(E/'SCORING_INDEX_PRIVATE.json')};exposure={(s['platform'],s['scene'],s['frame']):s['research_exposure'] for s in read(B/'crossplatform/round5/coverage/EXPOSURE_FRAMES_PRIVATE.json')};a.out.mkdir(exist_ok=False,parents=True);rows=[];metadata={}
 for j,row in enumerate(result['rows']):
  sid=row['id'];s=inputs[sid];ix=idx[sid];mp=Path(ix['meta_path']);assert sha(mp)==ix['metadata_container_sha256']
  if str(mp) not in metadata:metadata={str(mp):read(mp)} # bounded1frame cache; never loads any unscored frame
  obj=metadata[str(mp)]['ground_info'][s['object_index']];cls=obj['class'].lower();gt=target(obj['bbox_3d'],s['platform'],s['waymo_view_suffix']);others=obj.get('others') or [];same=[];allother=[]
  for o in others:
   if 'bbox_3d_other' not in o:continue
   ob=target(o['bbox_3d_other'],s['platform'],s['waymo_view_suffix']);allother.append(ob)
   if str(o.get('class_other','')).lower()==cls:same.append(ob)
  with np.load(a.cache/(sid+'_full.npz'),allow_pickle=False) as z:
   h=z['head_names'].tolist().index('last_');fb=z['boxes'][h].astype(float);fs=z['firstnoun_bbf'][h].copy();fulltokens=len(z['input_ids']);cl=z['class_names'].tolist().index(cls);primary_scores=z['bbf_scores_by_class'][h,cl].copy()
  with np.load(a.cache/(sid+'_noun.npz'),allow_pickle=False) as z:
   h=z['head_names'].tolist().index('last_');nb=z['boxes'][h].astype(float);ns=z['firstnoun_bbf'][h].copy();nountokens=len(z['input_ids']);noun_targetclass_span=bool(z['class_synonym_found'][cl])
  with np.load(a.scored/(sid+'_IOU_PRIVATE.npz'),allow_pickle=False) as z:fi=z['full_last'].copy();ni=z['noun_last'].copy()
  assert np.max(np.abs(fi-many_iou(np.c_[fb,np.zeros(256)],gt)))<1e-9
  cc,cost=correspond(fb,nb);inverse=np.argsort(cc);pairiou=aabb_pair_iou(fb,nb[cc]);qf=row['arms'][FULL]['query'];qn=row['arms'][NOUN]['query'];back=int(inverse[qn]);forward=int(cc[qf]);out={'id':sid,'platform':s['platform'],'scene':s['scene'],'frame':s['frame'],'object_index':s['object_index'],'official_eligible':row['official_eligible'],'official_map_nonzero':row['official_map_nonzero'],'prior_exposure':exposure[s['platform'],s['scene'],s['frame']],'sameclass_annotated_other_count':len(same),'allclass_annotated_other_count':len(allother),'noun_fallback_identity':s['noun_fallback_identity'],'noun_input_has_targetclass_dictionary_span_GT_assisted_audit':noun_targetclass_span,'actual_token_counts':{'full':fulltokens,'noun':nountokens},'arms':{}}
  for name in ARMS:
   if name not in row['arms']:continue
   arm=row['arms'][name];boxes=nb if name==NOUN else fb;q=arm['query'];selected=np.r_[boxes[q],0];other_iou=[float(many_iou(selected[None],o)[0]) for o in same];anyother_iou=[float(many_iou(selected[None],o)[0]) for o in allother];mx=max(other_iou,default=0.);diag={'selected_query':q,'target_iou':arm['iou'],'oracle_iou_GT_assisted':arm['oracle_iou'],'max_sameclass_annotated_other_iou':mx,'max_anyclass_annotated_other_iou':max(anyother_iou,default=0.),'overlaps_annotated_sameclass_other50':mx>.5,'confirmed_lowtarget_sameclass_other_error':arm['iou']<=.25 and mx>.5,'success50':arm['iou']>.5,'rank_recoverable50':arm['rank_recoverable50'],'candidate_coverage_failure50':arm['candidate_coverage_failure50'],'selection_primary_stratum':stratum(arm['iou'],arm['oracle_iou'],mx),'best_target_overlap_candidate_score_rank_interval_GT_assisted':rank_bounds(ns if name==NOUN else primary_scores if name==PRIMARY else fs,arm['oracle_query_GT_assisted'])}
   out['arms'][name]=diag
  replay=float(fi[back]);base=float(fi[qf]);noun=float(ni[qn]);out['full_noun_geometry_correspondence']={'algorithm':'Hungarian one-to-one normalizedcenterL2+normalizedsizeL2; scale=max(median(fullsize),.5); noGT/scores in matching','not_true_instance_identity_or_causal_decomposition':True,'mean_matched_center_L2_m':float(np.linalg.norm(fb[:,:3]-nb[cc,:3],axis=-1).mean()),'mean_matched_size_L1_m':float(np.abs(fb[:,3:6]-nb[cc,3:6]).mean()),'mean_matched_AABB_iou':float(pairiou.mean()),'fraction_matched_AABB_iou_above50':float((pairiou>.5).mean()),'same_row_candidate_geometry_bitexact':bool(np.array_equal(fb,nb)),'selected_text_full_query':qf,'selected_text_noun_query':qn,'full_selected_matches_noun_selected':forward==qn,'full_selected_matched_noun_query':forward,'noun_selected_matched_full_query':back,'full_selected_pair_AABB_iou':float(pairiou[qf]),'noun_selected_pair_AABB_iou':float(pairiou[back]),'full_selected_match_rank_in_noun_score_tie_interval':rank_bounds(ns,forward),'noun_selected_match_rank_in_full_score_tie_interval':rank_bounds(fs,back),'full_selected_target_iou':base,'noun_selected_target_iou':noun,'noun_selection_replayed_on_full_geometry_target_iou':replay,'full_to_noun_total_delta':noun-base,'matched_selection_replay_component':replay-base,'matched_geometry_remainder':noun-replay,'exact_algebra_sum_verified':abs((replay-base)+(noun-replay)-(noun-base))<1e-12,'own_pool_oracle_delta':float(ni.max()-fi.max())};out['postseal_geometry_diagnostic']=row['postseal_geometry_diagnostic'];g=row['postseal_geometry_diagnostic'];ref=row['arms'][g['reference_readout']];out['nonexclusive_GT_geometry_oracle_flags']={**{field+'_injection_recovers50':bool(ref['iou']<=.5 and val>.5) for field,val in g['selected_single_GT_field_iou'].items()},'yaw_pool_coverage_recovers50':bool(ref['oracle_iou']<=.5 and g['all256_GTyaw_oracle_iou']>.5),'reference_readout':g['reference_readout'],'not_additive_not_causal':True};rows.append(out)
  if (j+1)%500==0:print(json.dumps({'diagnosed':j+1,'n':result['n'],'seconds':time.perf_counter()-began}),flush=True)
 summary=[]
 for plat in ['waymo','drone','quad']:
  for population in ['official_eligible','all_raw']:
   rr=[x for x in rows if x['platform']==plat and (population=='all_raw' or x['official_eligible'])]
   for arm in ARMS:
    if arm==PRIMARY and population=='all_raw':continue
    xx=[x['arms'][arm] for x in rr if arm in x['arms']]
    if not xx:continue
    summary.append({'platform':plat,'population':population,'arm':arm,'n':len(xx),'exclusive_strata':dict(collections.Counter(x['selection_primary_stratum'] for x in xx)),'nonexclusive_counts':{k:sum(x[k] for x in xx) for k in ['success50','overlaps_annotated_sameclass_other50','confirmed_lowtarget_sameclass_other_error','rank_recoverable50','candidate_coverage_failure50']}})
 paired=[];geometry_summary=[];exposure_summary=[]
 for plat in ['waymo','drone','quad']:
  for population in ['official_eligible','all_raw']:
   rr=[x for x in rows if x['platform']==plat and (population=='all_raw' or x['official_eligible'])]
   if not rr:continue
   gg=[x['full_noun_geometry_correspondence'] for x in rr];delta=np.array([x['full_to_noun_total_delta'] for x in gg]);paired.append({'platform':plat,'population':population,'n':len(rr),'noun_minus_full_text_only_mean_iou':float(delta.mean()),'positive_negative_equal_counts':[int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum())],'mean_matched_selection_replay_component':float(np.mean([x['matched_selection_replay_component'] for x in gg])),'mean_matched_geometry_remainder':float(np.mean([x['matched_geometry_remainder'] for x in gg])),'mean_own_pool_oracle_delta':float(np.mean([x['own_pool_oracle_delta'] for x in gg])),'mean_matched_AABB_iou':float(np.mean([x['mean_matched_AABB_iou'] for x in gg])),'matched_selected_candidate_same_count':sum(x['full_selected_matches_noun_selected'] for x in gg),'noun_fallback_identity_count':sum(x['noun_fallback_identity'] for x in rr),'noun_has_targetclass_dictionary_span_GT_audit_count':sum(x['noun_input_has_targetclass_dictionary_span_GT_assisted_audit'] for x in rr),'not_causal_or_grounding_proof':True})
   for othererror in [False,True]:
    zz=[x for x in rr if x['arms'].get(PRIMARY,x['arms'][FULL])['confirmed_lowtarget_sameclass_other_error']==othererror];geometry_summary.append({'platform':plat,'population':population,'reference_selected_confirmed_sameclass_other_error':othererror,'n':len(zz),'GT_injection_recovery_counts':{k:sum(x['nonexclusive_GT_geometry_oracle_flags'][k] for x in zz) for k in ['center_injection_recovers50','size_injection_recovers50','yaw_injection_recovers50','yaw_pool_coverage_recovers50']},'not_additive_not_causal_not_method':True})
   for exposure in sorted({x['prior_exposure'] for x in rr}):
    zz=[x for x in rr if x['prior_exposure']==exposure];aa=[x['arms'][PRIMARY] for x in zz if PRIMARY in x['arms']];exposure_summary.append({'platform':plat,'population':population,'exposure':exposure,'n_queries':len(zz),'n_unique_frames':len({(x['scene'],x['frame']) for x in zz}),'n_scenes':len({x['scene'] for x in zz}),'primary_eligible_n':len(aa),'primary_mean_iou':float(np.mean([x['target_iou'] for x in aa])) if aa else None,'primary_acc50_count':sum(x['success50'] for x in aa)})
 selection=[]
 for plat in ['waymo','drone','quad']:
  for st in ['success','annotated_sameclass_other_instance','localization_boundary','candidate_coverage_failure50']:
   eligible=[x for x in rows if x['platform']==plat and PRIMARY in x['arms'] and x['arms'][PRIMARY]['selection_primary_stratum']==st];chosen=min(eligible,key=lambda x:hashlib.sha256((SALT+x['id']).encode()).hexdigest()) if eligible else None;selection.append({'platform':plat,'stratum':st,'denominator':len(eligible),'id':chosen['id'] if chosen else None,'choice':'minimum frozen saltedSHA256 among qualifying primaryofficialfull rows; empty not substituted'})
 payload={'scope':a.mode,'primary':PRIMARY,'n':len(rows),'rows':rows,'summaries':summary,'paired_language_summary':paired,'nonexclusive_GT_geometry_summary':geometry_summary,'research_exposure_summary':exposure_summary,'prediction_seal_sha256':sha(sp),'scores_result_sha256':sha(a.scored/'RESULTS_PRIVATE.json'),'numericGT_read_only_after_verified_release_and_predictionseal':True,'no_predictions_updated':True,'label_assisted_official_vs_puretext_not_matched_information':True,'annotations_may_not_exhaust_all_other_instances':True,'GT_field_injections_are_nonadditive_offline_oracles_not_methods':True,'seconds':time.perf_counter()-began};(a.out/'DIAGNOSES_PRIVATE.json').write_text(json.dumps(payload,indent=2,allow_nan=False));(a.out/'FIGURE_SELECTION_PRIVATE.json').write_text(json.dumps({'salt':SALT,'primary_arm':PRIMARY,'cases':selection,'design_sha256':sha(R/'DESIGN_BEFORE_SCORES.md')},indent=2));(a.out/'SEALED.json').write_text(json.dumps({'n':len(rows),'files':{p.name:sha(p) for p in a.out.iterdir() if p.is_file()},'source_code_sha256':sha(Path(__file__)),'release_sha256':sha(a.release),'media_pixels_read':False,'point_clouds_loaded':False},indent=2));print(json.dumps({'n':len(rows),'seconds':time.perf_counter()-began,'case_denominators':[(x['platform'],x['stratum'],x['denominator']) for x in selection],'seal_sha256':sha(a.out/'SEALED.json')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','full'],required=True)
 for name in ['cache','manifest','release','scored','out']:p.add_argument('--'+name,type=Path,required=True)
 main(p.parse_args())
