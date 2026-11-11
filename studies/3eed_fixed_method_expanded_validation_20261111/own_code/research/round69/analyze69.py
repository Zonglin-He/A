"""Fixed-validation conclusions; no method/hyperparameter selection from new labels."""
from common69 import *
H=ROOT/'research/round69';p=read(H/'experiment/DESIGN_LOCK.json');tasks=read(H/'score_all2304_attempt001/ALL_TASK_ROWS_PRIVATE.json');labels={x['id_PRIVATE']:x for x in read(H/'score_all2304_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};hashes=read(ROOT/p['media_hashes']);methods=[];scenes=[];updates=[];restoration=[];repeats=[];frames=[];diagnostics=[]
for co in ['VAL1280','TRAIN1024']:
 tt=[t for t in tasks if t['part']==co];n=len(tt);ref=np.array([t['IoU']['R59'] for t in tt]);source=np.array([t['IoU']['source'] for t in tt]);adapt=np.array([t['IoU']['R66_R59'] for t in tt]);delta=adapt-ref
 for a in ARMS:
  x=np.array([t['IoU'][a] for t in tt]);dd=x-ref;methods.append({'cohort':co,'method':a,'n':n,'Drone_mIoU':float(np.mean([t['IoU'][a] for t in tt if t['platform']=='drone'])*100),'Quad_mIoU':float(np.mean([t['IoU'][a] for t in tt if t['platform']=='quad'])*100),'balanced_mIoU':float(x.mean()*100),'Acc25':float((x>.25).mean()*100),'Acc50':float((x>.5).mean()*100),'gain_vs_R59_pp':float(dd.mean()*100),'improved_vs_R59':int((dd>1e-10).sum()),'worsened_vs_R59':int((dd< -1e-10).sum()),'query_changes_vs_R59':sum(t['queries'][a]!=t['queries']['R59'] for t in tt),'largest_positive_full_denominator_pp':float(max(0.,dd.max())*100/n),'rest_after_largest_full_denominator_pp':float((dd.sum()-max(0.,dd.max()))*100/n)})
  for pl in ['drone','quad']:
   for j,sc in enumerate(sorted({t['scene_PRIVATE'] for t in tt if t['platform']==pl})):
    rr=[t for t in tt if t['platform']==pl and t['scene_PRIVATE']==sc];scenes.append({'cohort':co,'method':a,'platform':pl,'scene_alias':('D' if pl=='drone' else 'Q')+'_S'+str(j+1).zfill(2),'n':len(rr),'mIoU':float(np.mean([t['IoU'][a] for t in rr])*100),'gain_vs_R59_pp':float(np.mean([t['IoU'][a]-t['IoU']['R59'] for t in rr])*100)})
 cat={k:{'n':0,'contribution_pp':0.} for k in ['exact_source_query_restored','source_level_IoU_other_query','beyond_source_IoU','improved_R59_but_below_source']}
 for t in tt:
  d=t['IoU']['R66_R59']-t['IoU']['R59']
  if d<=1e-10:continue
  if t['queries']['R66_R59']==t['queries']['source'] and t['IoU']['source']>t['IoU']['R59']+1e-10:k='exact_source_query_restored'
  elif abs(t['IoU']['R66_R59']-t['IoU']['source'])<=1e-10 and t['IoU']['source']>t['IoU']['R59']+1e-10:k='source_level_IoU_other_query'
  elif t['IoU']['R66_R59']>t['IoU']['source']+1e-10:k='beyond_source_IoU'
  else:k='improved_R59_but_below_source'
  cat[k]['n']+=1;cat[k]['contribution_pp']+=d*100/n
 for k,val in cat.items():restoration.append({'cohort':co,'category':k,**val})
 # Repeated descriptions may share raw media. Equal-frame statistics are
 # diagnostic; the prelocked expression-level metric remains primary.
 byframe={}
 for t in tt:byframe.setdefault((t['platform'],hashes[t['id_PRIVATE']]['point_sha256'],hashes[t['id_PRIVATE']]['image_sha256']),[]).append(t)
 for i,((pl,_,__),rr) in enumerate(sorted(byframe.items())):
  for a in ARMS:frames.append({'cohort':co,'platform':pl,'frame_alias':co+'_'+pl+'_F'+str(i+1).zfill(5),'descriptions':len(rr),'method':a,'mean_frame_IoU':float(np.mean([t['IoU'][a] for t in rr])),'frame_gain_vs_R59_pp':float(np.mean([t['IoU'][a]-t['IoU']['R59'] for t in rr])*100)})
 vals=[next(x['balanced_mIoU'] for x in methods if x['cohort']==co and x['method']=='random_s'+str(k)) for k in range(3)];rep={'cohort':co,'seed0_mIoU':vals[0],'seed1_mIoU':vals[1],'seed2_mIoU':vals[2],'repeat_mean_mIoU':float(np.mean(vals)),'repeat_min_mIoU':min(vals),'repeat_max_mIoU':max(vals),'repeat_std_population':float(np.std(vals)),'mean_gain_vs_R59_pp':float(np.mean(vals)-ref.mean()*100),'all_three_over_R59':all(x>ref.mean()*100+1e-10 for x in vals),'R66_minus_random_policy_mean_pp':float(adapt.mean()*100-np.mean(vals)),'not_ensemble_or_best_seed':True};repeats.append(rep)
 eligible=accepted=0
 for t in tt:
  r=read(ROOT/p['fit_root']/'cases_PRIVATE'/(t['id_PRIVATE']+'.json'));e=load(ROOT/p['evidence_root']/'samples_PRIVATE'/(t['id_PRIVATE']+'.npz'));eligible+=int(e['update_allowed']);accepted+=int(r['accepted']);updates.append({'row':t['row'],'cohort':co,'platform':t['platform'],'eligible':bool(e['update_allowed']),'accepted':r['accepted'],'step_L2':r['step_L2'],'source_IoU':t['IoU']['source'],'R59_IoU':t['IoU']['R59'],'R66_IoU':t['IoU']['R66_R59'],'R66_gain_vs_R59_pp':100*(t['IoU']['R66_R59']-t['IoU']['R59'])})
 diagnostics.append({'cohort':co,'n':n,'unique_media_frames':len(byframe),'update_eligible':eligible,'accepted_nonzero':accepted,'coverage_percent':eligible/n*100,'R66_gain_vs_R59_pp':float(delta.mean()*100),'R66_gain_vs_source_pp':float((adapt-source).mean()*100),'R66_rescue_source_failure25':int(((source<=.25)&(adapt>.25)).sum()),'R66_break_source_success25':int(((source>.25)&(adapt<=.25)).sum()),'R66_C_gain_vs_C_source_pp':float(np.mean([t['IoU']['C_R66']-t['IoU']['C_source'] for t in tt])*100),'R59_frozenABC_gain_pp':float(np.mean([t['IoU']['R59']-t['IoU']['frozen_ABC'] for t in tt])*100),'A256_oracle_good25':sum(max(labels[t['id_PRIVATE']]['candidate_IoUs_PRIVATE'])>.25 for t in tt),'R59_failure_recoverable25':sum(t['IoU']['R59']<=.25 and max(labels[t['id_PRIVATE']]['candidate_IoUs_PRIVATE'])>.25 for t in tt)})
by={co:{x['method']:x['balanced_mIoU'] for x in methods if x['cohort']==co} for co in ['VAL1280','TRAIN1024']};dec={'state':'R69_FIXED_VALIDATION_COMPLETE_NO_RESELECT_OR_RETRAIN','R66_gain_VAL_pp':by['VAL1280']['R66_R59']-by['VAL1280']['R59'],'R66_gain_TRAIN_pp':by['TRAIN1024']['R66_R59']-by['TRAIN1024']['R59'],'R66_exceeds_R59_both_cohorts':all(by[co]['R66_R59']>by[co]['R59']+1e-10 for co in by),'R66_exceeds_random_mean_both_cohorts':all(r['R66_minus_random_policy_mean_pp']>1e-10 for r in repeats),'new_method_or_hyperparameter_selected':False,'C_diagnostic_not_veto':True,'source_training':False,'FINAL_model_scoring':False,'VAL_historical_fullVAL_exposure_disclosed':True,'TRAIN_scene_claim':'new_recording_sequences_and_nominal_place_families_not_certified_global_geographic_holdout'}
for name,value in [('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS',scenes),('ALL_UPDATE_PROCESS_ROWS',updates),('SOURCE_RESTORATION_BREAKDOWN',restoration),('ALL_FRAME_CLUSTER_ROWS',frames),('RANDOM_REPEAT_SUMMARY',repeats),('COHORT_DIAGNOSTICS',diagnostics),('DECISION',dec)]:dump(H/(name+'.json'),value)
print(dec);print(diagnostics);print(repeats)

fb=[]
for co in ['VAL1280','TRAIN1024']:
 for a in ARMS:
  by={pl:[z['mean_frame_IoU'] for z in frames if z['cohort']==co and z['method']==a and z['platform']==pl] for pl in ['drone','quad']};fb.append({'cohort':co,'method':a,'Drone_unique_frames':len(by['drone']),'Quad_unique_frames':len(by['quad']),'equal_frame_then_platform_mIoU':float(np.mean([np.mean(v) for v in by.values()])*100)})
dump(H/'FRAME_BALANCED_SUMMARY.json',fb)
