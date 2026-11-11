"""Independent saved-only scientific-summary audit; no producers or new GT."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round69'
def rd(p):return json.loads(Path(p).read_text())
def same(a,b):
 if isinstance(b,dict):
  assert set(a)==set(b)
  for k in b:same(a[k],b[k])
 elif isinstance(b,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 elif type(b) in (int,float):assert abs(a-b)<1e-10,(a,b)
 else:assert a==b,(a,b)
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tasks=rd(H/'score_all2304_attempt001/ALL_TASK_ROWS_PRIVATE.json');plan=rd(H/'experiment/DESIGN_LOCK.json');hashes=rd(R/plan['media_hashes']);methods=rd(H/'ALL_METHOD_ROWS.json');scenes=rd(H/'ALL_SCENE_ROWS.json');updates=rd(H/'ALL_UPDATE_PROCESS_ROWS.json');cats=rd(H/'SOURCE_RESTORATION_BREAKDOWN.json');frames=rd(H/'ALL_FRAME_CLUSTER_ROWS.json');repeats=rd(H/'RANDOM_REPEAT_SUMMARY.json');diagnostics=rd(H/'COHORT_DIAGNOSTICS.json');decision=rd(H/'DECISION.json');checked=0
frame_lookup={(x['frame_alias'],x['method']):x for x in frames};update_lookup={x['row']:x for x in updates}
for co in ['VAL1280','TRAIN1024']:
 tt=[t for t in tasks if t['part']==co];n=len(tt);arms=list(tt[0]['IoU']);baseline=np.array([t['IoU']['R59'] for t in tt]);source=np.array([t['IoU']['source'] for t in tt]);adapt=np.array([t['IoU']['R66_R59'] for t in tt])
 for a in arms:
  x=np.array([t['IoU'][a] for t in tt]);d=x-baseline;row=next(z for z in methods if z['cohort']==co and z['method']==a)
  same(row,dict(cohort=co,method=a,n=n,Drone_mIoU=float(np.mean([t['IoU'][a] for t in tt if t['platform']=='drone'])*100),Quad_mIoU=float(np.mean([t['IoU'][a] for t in tt if t['platform']=='quad'])*100),balanced_mIoU=float(np.mean(x)*100),Acc25=float(np.mean(x>.25)*100),Acc50=float(np.mean(x>.5)*100),gain_vs_R59_pp=float(np.mean(d)*100),improved_vs_R59=int(np.sum(d>1e-10)),worsened_vs_R59=int(np.sum(d< -1e-10)),query_changes_vs_R59=sum(t['queries'][a]!=t['queries']['R59'] for t in tt),largest_positive_full_denominator_pp=float(max(0.,max(d))*100/n),rest_after_largest_full_denominator_pp=float((sum(d)-max(0.,max(d)))*100/n)));checked+=1
  for pl in ['drone','quad']:
   for j,sc in enumerate(sorted({t['scene_PRIVATE'] for t in tt if t['platform']==pl})):
    rr=[t for t in tt if t['platform']==pl and t['scene_PRIVATE']==sc];alias=('D' if pl=='drone' else 'Q')+'_S'+str(j+1).zfill(2);row=next(z for z in scenes if z['cohort']==co and z['method']==a and z['platform']==pl and z['scene_alias']==alias);same(row,dict(cohort=co,method=a,platform=pl,scene_alias=alias,n=len(rr),mIoU=float(np.mean([t['IoU'][a] for t in rr])*100),gain_vs_R59_pp=float(np.mean([t['IoU'][a]-t['IoU']['R59'] for t in rr])*100)));checked+=1
 expected={k:[0,0.] for k in ['exact_source_query_restored','source_level_IoU_other_query','beyond_source_IoU','improved_R59_but_below_source']}
 framegroups={};eligible=accepted=0
 for t in tt:
  d=t['IoU']['R66_R59']-t['IoU']['R59'];sc=t['IoU']['source'];ad=t['IoU']['R66_R59']
  if d>1e-10:
   k='exact_source_query_restored' if t['queries']['R66_R59']==t['queries']['source'] and sc>t['IoU']['R59']+1e-10 else 'source_level_IoU_other_query' if abs(ad-sc)<=1e-10 and sc>t['IoU']['R59']+1e-10 else 'beyond_source_IoU' if ad>sc+1e-10 else 'improved_R59_but_below_source';expected[k][0]+=1;expected[k][1]+=d*100/n
  r=rd(R/plan['fit_root']/'cases_PRIVATE'/(t['id_PRIVATE']+'.json'))
  with np.load(R/plan['evidence_root']/'samples_PRIVATE'/(t['id_PRIVATE']+'.npz'),allow_pickle=False) as e:ok=bool(e['update_allowed'])
  eligible+=int(ok);accepted+=int(r['accepted']);u=update_lookup[t['row']];same(u,dict(row=t['row'],cohort=co,platform=t['platform'],eligible=ok,accepted=r['accepted'],step_L2=r['step_L2'],source_IoU=sc,R59_IoU=t['IoU']['R59'],R66_IoU=ad,R66_gain_vs_R59_pp=100*d));checked+=1
  hh=hashes[t['id_PRIVATE']];framegroups.setdefault((t['platform'],hh['point_sha256'],hh['image_sha256']),[]).append(t)
 for k,(nn,v) in expected.items():same(next(z for z in cats if z['cohort']==co and z['category']==k),dict(cohort=co,category=k,n=nn,contribution_pp=v));checked+=1
 for i,((pl,_,__),rr) in enumerate(sorted(framegroups.items())):
  alias=co+'_'+pl+'_F'+str(i+1).zfill(5)
  for a in arms:
   row=frame_lookup[(alias,a)];same(row,dict(cohort=co,platform=pl,frame_alias=alias,descriptions=len(rr),method=a,mean_frame_IoU=float(np.mean([t['IoU'][a] for t in rr])),frame_gain_vs_R59_pp=float(np.mean([t['IoU'][a]-t['IoU']['R59'] for t in rr])*100)));checked+=1
 vals=[np.mean([t['IoU']['random_s'+str(k)] for t in tt])*100 for k in range(3)];rep=next(z for z in repeats if z['cohort']==co)
 same(rep,dict(cohort=co,seed0_mIoU=vals[0],seed1_mIoU=vals[1],seed2_mIoU=vals[2],repeat_mean_mIoU=float(np.mean(vals)),repeat_min_mIoU=min(vals),repeat_max_mIoU=max(vals),repeat_std_population=float(np.std(vals)),mean_gain_vs_R59_pp=float(np.mean(vals)-np.mean(baseline)*100),all_three_over_R59=all(v>np.mean(baseline)*100+1e-10 for v in vals),R66_minus_random_policy_mean_pp=float(np.mean(adapt)*100-np.mean(vals)),not_ensemble_or_best_seed=True));checked+=1
 diag=next(z for z in diagnostics if z['cohort']==co);assert diag['update_eligible']==eligible and diag['accepted_nonzero']==accepted and diag['unique_media_frames']==len(framegroups)
 assert abs(diag['R66_C_gain_vs_C_source_pp']-np.mean([t['IoU']['C_R66']-t['IoU']['C_source'] for t in tt])*100)<1e-10
 assert diag['R66_rescue_source_failure25']==int(np.sum((source<=.25)&(adapt>.25))) and diag['R66_break_source_success25']==int(np.sum((source>.25)&(adapt<=.25)))
for fb in rd(H/'FRAME_BALANCED_SUMMARY.json'):
 vals=[np.mean([z['mean_frame_IoU'] for z in frames if z['cohort']==fb['cohort'] and z['method']==fb['method'] and z['platform']==pl]) for pl in ['drone','quad']];assert abs(fb['equal_frame_then_platform_mIoU']-np.mean(vals)*100)<1e-10;checked+=1
assert decision['new_method_or_hyperparameter_selected'] is False and decision['FINAL_model_scoring'] is False
assert decision['R66_exceeds_R59_both_cohorts']==all(z['gain_vs_R59_pp']>1e-10 for z in methods if z['method']=='R66_R59')
assert decision['R66_exceeds_random_mean_both_cohorts']==all(z['R66_minus_random_policy_mean_pp']>1e-10 for z in repeats)
out=H/'independent_derived_audit_attempt001';assert not out.exists();out.mkdir();result={'state':'PASS_R69_ALL_SAVED_METHOD_SCENE_UPDATE_FRAME_REPEAT_AND_RESTORATION_TABLES','n':len(tasks),'checked_rows':checked,'no_new_GT_or_models_or_selection':True,'parent_scalar_audit_sha256':sh(H/'independent_scalar_audit_attempt001/AUDIT_RESULT.json')};(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
