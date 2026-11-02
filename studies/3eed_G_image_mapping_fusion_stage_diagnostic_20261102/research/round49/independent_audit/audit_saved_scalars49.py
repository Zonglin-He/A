"""Independent reaggregation of sealed saved scalars only; no GT/geometry/model IO."""
from pathlib import Path
import collections,hashlib,itertools,json,math,statistics,time
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
JOIN=ROOT/'crossplatform/round49/blindjoin48_attempt001'
OUT=ROOT/'research/round49/independent_scalar_recheck'
ARMS=[f'legacy_V{v}P{p}R{r}' for v in (0,1) for p in (0,1) for r in (0,1)]+['legacy_rawvv_mapped_cue_only','legacy_postclipV_mapped_cue_only','legacy_fullG_mapped_only','legacy_fullG_V_unclipped','legacy_fullG_no_gray','joint_e_fullG_same_operator','joint_literal_source_plus_rawcue','joint_rawcue_mapped_only']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
ERROR=0.;COUNT=0
def same(a,b,path='root'):
 global ERROR,COUNT
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),(path,'keys');[same(a[k],b[k],path+'.'+k) for k in a]
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),(path,'length');[same(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b))]
 elif isinstance(a,(int,float)) and not isinstance(a,bool):
  assert isinstance(b,(int,float)) and not isinstance(b,bool) and math.isfinite(a) and math.isfinite(b),path
  d=abs(a-b);ERROR=max(ERROR,d);COUNT+=1;assert d<=1e-10,(path,a,b,d)
 else:assert a==b,(path,a,b)
def mean(xs):return statistics.fmean(xs)
def effect(m):
 out={}
 for k,name in enumerate(['V','P','R']):
  others=[j for j in range(3) if j!=k];dd=[]
  for x,y in itertools.product((0,1),repeat=2):
   a=[0,0,0];a[others[0]]=x;a[others[1]]=y;b=a.copy();b[k]=1
   key=lambda v:f'legacy_V{v[0]}P{v[1]}R{v[2]}'
   dd.append(m[key(b)]-m[key(a)])
  out[name+'_average_conditional_task_effect_pp']=mean(dd);out[name+'_four_conditional_effects_pp']=dd
 for key,a,b in [('full_minus_source_pp',7,0),('fullmapped_minus_full_pp',10,7),('unclipV_minus_full_pp',11,7),('no_gray_minus_full_pp',12,7),('joint_same_operator_minus_full_pp',13,7),('joint_literal_minus_full_pp',14,7),('postclip_cue_minus_rawcue_pp',9,8)]:out[key]=m[ARMS[a]]-m[ARMS[b]]
 out['not_additive_identity_error_attribution']=True;return out
def counts(rows,key):return dict(collections.Counter(r[key] for r in rows))
def blind_overall(rr):
 out={'n':len(rr),'phase1_status':counts(rr,'phase1_status'),'legacy_vs_fullcaption':counts(rr,'legacy_visual_vs_fullcaption_identity'),'joint_vs_fullcaption':counts(rr,'joint_visual_vs_fullcaption_identity'),'pool_identity_coverage':counts(rr,'legacy_pool_identity_coverage_status')}
 strata={}
 for name,condition in [('no_pool',lambda r:not r['has_legacy_visual_pool']),('pool_with_compatible',lambda r:r['has_legacy_visual_pool'] and r['partial_compatible_candidate_exists_in_compared_pool']),('pool_without_confirmed_compatible',lambda r:r['has_legacy_visual_pool'] and not r['partial_compatible_candidate_exists_in_compared_pool'])]:
  r=[x for x in rr if condition(x)];strata[name]={'n':len(r),'rawfull_partial_status':counts(r,'legacy_raw_full_vs_actual_partial_subject'),'gray_corrected_partial_status':counts(r,'legacy_gray_corrected_vs_actual_partial_subject'),'rawfull_all_maskmeans_negative':sum(x['legacy_rawfull_all_mask_means_negative'] is True for x in r)}
 out['partial_pool_strata']=strata
 ss=[r for r in rr if r['phase1_status']=='SINGLE_VISIBLE_TARGET'];inv=[r['original256_target_support_inventory'] for r in ss]
 out['single_target']={'n':len(ss),'outside_compared_pool':sum(r['single_target_outside_legacy_pool'] for r in ss),'inpool_legacy_fullcaption_status':counts([r for r in ss if not r['single_target_outside_legacy_pool']],'legacy_visual_vs_fullcaption_identity'),'inpool_joint_fullcaption_status':counts([r for r in ss if not r['single_target_outside_legacy_pool']],'joint_visual_vs_fullcaption_identity'),'original256_support_exists':sum(x['any_original_query_target_supported'] for x in inv),'original256_no_supported_query':sum(not x['any_original_query_target_supported'] for x in inv),'support_exists_but_no_target_W_mass':sum(x['any_original_query_target_supported'] and x['target_supported_queries_with_positive_target_W_mass']==0 for x in inv),'unique_frame_target_points_zero':sum(x['target_fullframe_exclusive_points']==0 for x in inv),'unique_frame_target_points_1to4':sum(1<=x['target_fullframe_exclusive_points']<=4 for x in inv)}
 out['point_support_by_arm']={a:dict(collections.Counter(r['arms'][a]['status'] for r in rr)) for a in ARMS}
 out['cue_maxsets']={}
 for key in ['rawcue','postclipcue']:
  tt=[r['cue_maxset_support_diagnostics'][key] for r in rr];st=[r['cue_maxset_support_diagnostics'][key] for r in ss];nonempty=[x for x in st if not x['no_visual_maxset']]
  out['cue_maxsets'][key]={'n':len(tt),'empty_maxset':sum(x['no_visual_maxset'] for x in tt),'size_distribution':dict(collections.Counter(str(x['maxset_size']) for x in tt)),'single_n':len(st),'single_nonempty_n':len(nonempty),'single_maxset_contains_target_supported':sum(x['contains_target_supported'] is True for x in st),'single_representative_target_supported':sum(x['representative_is_target_supported_member'] is True for x in st),'single_target_supported_present_but_representative_not':sum(x['contains_target_supported'] is True and x['representative_is_target_supported_member'] is False for x in st),'single_mixed_support_maxsets':sum(x['maxset_mixed_support_statuses'] for x in st),'representative_status':dict(collections.Counter(x['representative_support_status'] for x in tt)),'all_maxset_member_status':dict(sum((collections.Counter(x['maxset_support_status_counts']) for x in tt),collections.Counter()))}
 out['rawfull_gray_topsets_different']=sum(not r['raw_full_and_gray_corrected_topsets_equal'] for r in rr)
 out['limits']=['nonexpert48_not_population_or_gold','partial_subject_is_not_fullcaption_identity','point_support_uses_shared_camera_and_masks_not_independent_worldID','representative_failure_not_entire_W_maxset_failure','conditional_counts_not_additive_error_rates']
 return out
def main():
 start=time.perf_counter();assert not OUT.exists();OUT.mkdir()
 seal=read(HERE/'SCORING_NUMERIC_ARTIFACTS_SEALED.json');assert seal['numeric_artifacts_written_before_final_console_print_NameError'] and seal['GT_or_model_rerun_forbidden']
 expected={'ALL336_TASK_ROWS_PRIVATE.json','TASK_SUMMARY.json','ALL_NEGATIVES.json','ALL_SCENE_SUMMARIES_PRIVATE.json','FACTORIAL_AND_INTERFACE_CONTRASTS.json','SCORING_RECEIPT.json'};assert set(seal['files'])==expected
 for f,h in seal['files'].items():assert sha(HERE/f)==h
 tasks=read(HERE/'ALL336_TASK_ROWS_PRIVATE.json');assert len(tasks)==336 and len({r['alias'] for r in tasks})==336
 for r in tasks:
  assert set(r['IoU'])==set(r['queries'])==set(ARMS) and r['platform'] in ('drone','quad') and r['split'] in ('D208','V64','M64')
  assert all(math.isfinite(v) and 0<=v<=1 for v in r['IoU'].values()) and all(type(v) is int and 0<=v<256 for v in r['queries'].values())
 assert collections.Counter(r['split'] for r in tasks)=={'D208':208,'V64':64,'M64':64}
 summary={};scenes=[];effects={};neg=[]
 for part in ['ALL336','D208','V64','M64']:
  rr=tasks if part=='ALL336' else [r for r in tasks if r['split']==part];ss={}
  for pl in ['drone','quad']:
   pr=[r for r in rr if r['platform']==pl];ss[pl]={}
   for a in ARMS:
    vv=[r['IoU'][a] for r in pr];dd=[r['IoU'][a]-r['IoU'][ARMS[0]] for r in pr]
    ss[pl][a]={'n':len(pr),'mIoU_percent':mean(vv)*100,'Acc25_percent':sum(x>.25 for x in vv)/len(pr)*100,'Acc50_percent':sum(x>.5 for x in vv)/len(pr)*100,'gain_vs_source_pp':mean(dd)*100,'improved':sum(x>1e-10 for x in dd),'worsened':sum(x< -1e-10 for x in dd),'query_changes':sum(r['queries'][a]!=r['queries'][ARMS[0]] for r in pr),'scene_clusters':len({r['scene_PRIVATE'] for r in pr})}
   for scene in sorted({r['scene_PRIVATE'] for r in pr}):
    sr=[r for r in pr if r['scene_PRIVATE']==scene];scenes.append({'part':part,'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{a:mean(r['IoU'][a] for r in sr)*100 for a in ARMS}})
  ss['balanced_mIoU_percent']={a:mean(ss[pl][a]['mIoU_percent'] for pl in ('drone','quad')) for a in ARMS};summary[part]=ss;effects[part]={pl:effect(ss['balanced_mIoU_percent'] if pl=='balanced' else {a:ss[pl][a]['mIoU_percent'] for a in ARMS}) for pl in ['drone','quad','balanced']}
 for r in tasks:
  for a,ref in itertools.product(ARMS,repeat=2):
   d=r['IoU'][a]-r['IoU'][ref]
   if a!=ref and d< -1e-10:neg.append({'alias':r['alias'],'split':r['split'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':d*100})
 same(summary,read(HERE/'TASK_SUMMARY.json'));same(scenes,read(HERE/'ALL_SCENE_SUMMARIES_PRIVATE.json'));same(effects,read(HERE/'FACTORIAL_AND_INTERFACE_CONTRASTS.json'));same(neg,read(HERE/'ALL_NEGATIVES.json'));assert len(neg)==4303
 receipt=read(HERE/'SCORING_RECEIPT.json');assert receipt['task_values']==5376 and receipt['n']==336 and receipt['arms']==16 and receipt['negative_ordered_comparisons']==4303
 jseal=read(JOIN/'SEALED.json')
 for f,h in jseal['files'].items():assert sha(JOIN/f)==h
 ba=read(HERE/'BLIND_JOIN_AUDIT_RESULT.json');assert ba['state']=='PASS_R49_INDEPENDENT_POSTSEAL_BLIND_JOIN_ARITHMETIC_AUDIT_NO_GT' and ba['join_seal_sha256']==sha(JOIN/'SEALED.json')
 br=read(JOIN/'ROWS_PRIVATE.json');assert len(br)==48
 overall={'ALL48':blind_overall(br),'by_platform':{pl:blind_overall([r for r in br if r['platform']==pl]) for pl in ('drone','quad')}}
 dump(OUT/'RECOMPUTED_TASK_SUMMARY.json',summary);dump(OUT/'RECOMPUTED_FACTOR_CONTRASTS.json',effects);dump(OUT/'BLIND_STAGE_OVERALL_DIAGNOSTIC.json',overall)
 result={'state':'PASS_R49_INDEPENDENT_SAVED_SCALAR_REAGGREGATION_NO_GT_RERUN','completed_numeric_artifacts_despite_terminal_print_NameError':True,'original_scoring_process_exit1_not_ignored':True,'scientific_numeric_artifacts_complete_and_consistent':True,'n':336,'arms':16,'IoU_values_reaggregated':5376,'ordered_negative_comparisons_reproduced':len(neg),'scene_summary_rows_reproduced':len(scenes),'summary_parts':4,'factor_coordinate_groups':12,'numeric_comparisons':COUNT,'maximum_absolute_scalar_error':ERROR,'blind48_rows_summarized_without_rejudging_labels':48,'GT_geometry_metadata_media_model_FINAL_reads':0,'scoring_numeric_seal_sha256':sha(HERE/'SCORING_NUMERIC_ARTIFACTS_SEALED.json'),'join_seal_sha256':sha(JOIN/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
