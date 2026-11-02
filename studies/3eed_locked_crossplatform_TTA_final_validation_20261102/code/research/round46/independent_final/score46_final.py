"""Unexecuted final scoring draft. No data is opened before exact root grants."""
import argparse,json,hashlib,time,datetime,sys,signal
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def hardstop():
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,2,1,30,tzinfo=datetime.timezone.utc),'09:30 Asia/Shanghai hard stop'
def bounded_source_iou(pool,gt,remaining,core):
 def expired(*_):raise TimeoutError('Source-pool diagnostic120second budget')
 assert remaining>0;old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
 try:return core(pool,gt)[0]
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--predictions',required=True);p.add_argument('--authorization',required=True);p.add_argument('--audit-result',required=True);p.add_argument('--code-lock',required=True);p.add_argument('--out',required=True);a=p.parse_args();hardstop()
 # These grants are checked BEFORE opening any final manifest or annotation.
 auth=read(a.authorization);assert auth['final_task_GT_scoring_authorized'] is True;assert auth['independent_code_lock_sha256']==sha(a.code_lock)
 locked=read(a.code_lock)
 for rel,h in {**locked['program_sha256'],**locked['fixed_provenance_sha256']}.items():assert sha(R/rel)==h,rel
 assert auth['plan_sha256']==sha(a.plan) and auth['prediction_seal_sha256']==sha(Path(a.predictions)/'SEALED.json') and auth['independent_audit_sha256']==sha(a.audit_result)
 audit=read(a.audit_result);assert audit['state']=='PASS_R46_INDEPENDENT_FINAL_PREDICTION_CONTRACT' and audit['prediction_seal_sha256']==auth['prediction_seal_sha256'];plan=read(a.plan);assert plan['execution_mode']=='final_once' and auth['manifest_sha256']==plan['manifest_sha256'];assert sha(plan['manifest_path'])==plan['manifest_sha256'];rows=read(plan['manifest_path']);assert len(rows)==plan['expected_n']==384;B=Path(a.predictions);O=Path(a.out);O.mkdir(exist_ok=False,parents=True)
 sys.path.insert(0,str(R/'research/round5/audit'));from independent_core_v2 import candidate_ious,transform_gt,fixtures
 fixtures();start=time.perf_counter();indexpath=Path(auth['scoring_index_path']);assert indexpath.resolve()==(R/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json').resolve() and sha(indexpath)==auth['scoring_index_sha256'];prep=read(R/'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json');assert prep['scoring_index_sha256']==auth['scoring_index_sha256'];index={x['id']:x for x in read(indexpath)};metadata={};tasks=[];diagnostic=[];diag_seconds=0.;diag_complete=True;diag_status='complete';arms=None
 for r in rows:
  hardstop();idx=index[r['id']];mp=Path(idx['meta_path']);assert idx['platform']==r['platform'] and idx['scene']==r['scene'] and mp.resolve()==Path(r['metapath']).resolve() and isinstance(idx['object_index'],int) and not isinstance(idx['object_index'],bool) and idx['object_index']>=0
  if str(mp) not in metadata:assert sha(mp)==idx['metadata_container_sha256'];metadata[str(mp)]=read(mp)
  obj=metadata[str(mp)]['ground_info'][idx['object_index']];gt=transform_gt(obj['bbox_3d'],r['platform'],idx.get('waymo_view_suffix'));m=npz(B/'main/predictions_PRIVATE'/(r['id']+'.npz'));e=npz(B/'external/predictions_PRIVATE'/(r['id']+'.npz'));current=['main::'+x for x in m['arm_names'].tolist()]+['episodic::'+x for x in e['arm_names'].tolist()]
  if arms is None:arms=current;assert len(arms)==24
  assert current==arms;boxes=np.r_[m['boxes'],e['boxes']].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);iou=dict(zip(arms,map(float,candidate_ious(boxes,gt)[0])));assert iou['main::source']==iou['episodic::source'];q=dict(zip(arms,map(int,np.r_[m['queries'],e['queries']])));row={'alias':r.get('alias',r['id']),'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':q,'IoU':iou};tasks.append(row);dump(O/(f'TASK_PROGRESS_PRIVATE.json'),{'completed':len(tasks),'expected':384})
  # Method-independent sealed source-pool diagnostic, never used by a method.
  if diag_complete:
   if diag_seconds>=120:diag_complete=False;diag_status='incomplete_120second_budget_all_partial_rows_retained_no_aggregate'
   else:
    tick=time.perf_counter();z=npz(B/'source/samples_PRIVATE'/(r['id']+'.npz'));pool=z['boxes'].copy();pool[:,3:]=np.maximum(pool[:,3:],1e-6)
    try:v=bounded_source_iou(pool,gt,120-diag_seconds,candidate_ious)
    except TimeoutError:diag_seconds+=time.perf_counter()-tick;diag_complete=False;diag_status='incomplete_120second_budget_all_partial_rows_retained_no_aggregate';continue
    order=np.argsort(-z['scores'],kind='stable');assert int(order[0])==q['main::source'] and abs(float(v[order[0]])-iou['main::source'])<1e-12;top={str(k):{'bestIoU_GTassisted':float(v[order[:k]].max()),'Acc25_GTassisted':bool(v[order[:k]].max()>.25),'Acc50_GTassisted':bool(v[order[:k]].max()>.5)} for k in [1,5,10]};oracle=float(v.max());diagnostic.append({'alias':row['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'TopK':top,'all256_oracle_IoU_GTassisted':oracle,'all256_oracle_Acc25':bool(oracle>.25),'all256_oracle_Acc50':bool(oracle>.5),'source_top_score_tie_count':int((z['scores']==z['scores'][order[0]]).sum()),'topK_are_queries_not_distinct_instances':True,'only_frozen_sourcepool_selection_upperbound_not_MEMO_Reg_newboxes':True});diag_seconds+=time.perf_counter()-tick
    if diag_seconds>120:diag_complete=False;diag_status='incomplete_120second_budget_all_partial_rows_retained_no_aggregate'
 assert len(tasks)==384;summary={};scenes=[];negative=[]
 for pl in ['drone','quad']:
  rr=[x for x in tasks if x['platform']==pl];summary[pl]={}
  for arm in arms:
   v=np.array([x['IoU'][arm] for x in rr]);s=np.array([x['IoU']['main::source'] for x in rr]);summary[pl][arm]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float((v-s).mean()*100),'improved_vs_source':int((v>s+1e-10).sum()),'harmed_vs_source':int((v<s-1e-10).sum()),'query_changes_vs_source':sum(x['queries'][arm]!=x['queries']['main::source'] for x in rr)}
  for scene in sorted({x['scene_PRIVATE'] for x in rr}):
   part=[x for x in rr if x['scene_PRIVATE']==scene];scenes.append({'platform':pl,'scene_PRIVATE':scene,'n':len(part),'mIoU_percent':{arm:float(np.mean([x['IoU'][arm] for x in part])*100) for arm in arms},'descriptive_only_no_crossscene_CI':True})
 summary['equal_platform_mIoU_percent']={arm:.5*(summary['drone'][arm]['mIoU_percent']+summary['quad'][arm]['mIoU_percent']) for arm in arms}
 for r in tasks:
  for arm in arms:
   for ref in arms:
    if arm!=ref and r['IoU'][arm]<r['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':100*(r['IoU'][arm]-r['IoU'][ref])})
 diag_summary={'state':diag_status,'completed_n':len(diagnostic),'expected_n':384,'seconds_sourcepool_geometry_only':diag_seconds,'budget_seconds':120,'all_partial_rows_preserved_if_incomplete':True,'GT_only_postseal_not_in_G_TTA_or_readout':True,'stable_low_index_ties':True,'queries_not_distinct_instances':True,'not_deployable_TopK_or_oracle_results':True,'upperbound_only_for_frozen_sourcepool_selection_not_MEMO_Reg_newboxes':True}
 if diag_complete and len(diagnostic)==384:
  for pl in ['drone','quad']:
   rr=[x for x in diagnostic if x['platform']==pl];diag_summary[pl]={str(k):{'mIoU_percent':float(np.mean([x['TopK'][str(k)]['bestIoU_GTassisted'] for x in rr])*100),'Acc25_percent':float(np.mean([x['TopK'][str(k)]['Acc25_GTassisted'] for x in rr])*100),'Acc50_percent':float(np.mean([x['TopK'][str(k)]['Acc50_GTassisted'] for x in rr])*100)} for k in [1,5,10]};diag_summary[pl]['all256_oracle']={'mIoU_percent':float(np.mean([x['all256_oracle_IoU_GTassisted'] for x in rr])*100),'Acc25_percent':float(np.mean([x['all256_oracle_Acc25'] for x in rr])*100),'Acc50_percent':float(np.mean([x['all256_oracle_Acc50'] for x in rr])*100)}
 pairs=[('main::R30_beta1','main::source'),('main::R30_beta1','main::R30_memory_allA_zeroU'),('main::R30_beta1','main::R30_memory_E_only'),('main::R24_query_mlp_lr1_s16','main::source'),('main::R24_query_mlp_lr1_s16','main::R24_memory_alpha1'),('main::R30_beta1_plus_current_oldG','main::source_plus_current_oldG'),('main::R30_beta1_plus_current_oldG','main::R30_memory_allA_zeroU_plus_current_oldG'),('main::R30_beta1_plus_current_oldG','main::R30_memory_E_only_plus_current_oldG'),('main::R24_query_mlp_lr1_s16_plus_current_oldG','main::source_plus_current_oldG'),('main::R24_query_mlp_lr1_s16_plus_current_oldG','main::R24_memory_alpha1_plus_current_oldG'),('episodic::Tent_clean_post_port','episodic::Tent_BN_statistics_first_pre'),('episodic::SAR_LN_clean_post_port','episodic::SAR_LN_first_pre'),('episodic::MEMO_AdamW1e5_clean_post_port','episodic::MEMO_BN_prior_only'),('episodic::MEMO_AdamW1e5_clean_post_port','episodic::MEMO_augmentation_only'),('episodic::Reg_inspired_teacher_post','episodic::Reg_inspired_direct_pseudo')];paired=[]
 for arm,reference in pairs:
  for pl in ['drone','quad']:
   rr=[r for r in tasks if r['platform']==pl];d=np.array([r['IoU'][arm]-r['IoU'][reference] for r in rr]);paired.append({'platform':pl,'arm':arm,'reference':reference,'n':len(rr),'delta_mIoU_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'harmed':int((d< -1e-10).sum()),'task_tied':int((abs(d)<=1e-10).sum()),'descriptive_paired_no_CI':True})
 receipt={'state':'R46_FINAL_SINGLE_UNIFIED_SCORING_COMPLETE','n':384,'main_arms':12,'external_separate_protocol_controls':12,'task_values':384*24,'all_ordered_negative_comparisons':len(negative),'primary_fixed_before_access':'main::R30_beta1','secondary_fixed_before_access':'main::R24_query_mlp_lr1_s16','source_pool_diagnostic_complete':diag_complete and len(diagnostic)==384,'source_pool_diagnostic_n':len(diagnostic),'prediction_seal_sha256':auth['prediction_seal_sha256'],'plan_sha256':auth['plan_sha256'],'manifest_sha256':auth['manifest_sha256'],'root_postseal_label_auth_sha256':sha(a.authorization),'independent_audit_sha256':sha(a.audit_result),'no_method_selection_or_refitting_after_scores':True,'no_overall_crossscene_CI_claim':True,'seconds':time.perf_counter()-start}
 dump(O/'ALL384_TASK_ROWS_PRIVATE.json',tasks);dump(O/'TASK_SUMMARY.json',summary);dump(O/'SCENE_SUMMARIES_PRIVATE.json',scenes);dump(O/'ALL_ORDERED_NEGATIVE_COMPARISONS.json',negative);dump(O/'FIXED_SAME_INFORMATION_PAIRED_OUTCOMES.json',paired);dump(O/'SOURCE_POOL_ALL_ROWS_PRIVATE.json',diagnostic);dump(O/'SOURCE_POOL_DIAGNOSTIC_SUMMARY.json',diag_summary);dump(O/'SCORING_RECEIPT.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
