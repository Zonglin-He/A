"""One root-authorized old336 five-arm task score after independent audit PASS."""
import sys,time,json,argparse
from pathlib import Path
import numpy as np
from audit48_independent import ROOT,HERE,EXP,CAP,ARMS,read,sha,dump,load,verify,cohort,old_index,verify_independent_code
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import transform_gt,candidate_ious

def main():
 start=time.perf_counter();verify_independent_code();a=read(EXP/'ROOT_SINGLE_TASK_SCORING_AUTHORIZATION.json');assert a['state']=='ROOT_AUTHORIZED_R48_SINGLE_POSTSEAL_TASK_SCORE_OLD336_ONLY';plan=read(EXP/'PLAN_LOCK.json');assert a['plan_sha256']==sha(EXP/'PLAN_LOCK.json');P=ROOT/plan['prediction_output'];assert a['prediction_seal_sha256']==sha(P/'SEALED.json');assert a['independent_audit_sha256']==sha(HERE/'AUDIT_RESULT.json');audit=read(HERE/'AUDIT_RESULT.json');assert audit['state']=='PASS_R48_INDEPENDENT_CPU_ARRAY_SPAN_AUDIT_NO_GTBOX' and audit['prediction_seal_sha256']==a['prediction_seal_sha256'];assert a['scoring_program_sha256']==sha(__file__);verify(P)
 rr,member=cohort();index=old_index(rr);diags={r['id_PRIVATE']:r for r in read(HERE/'ALL336_READOUT_AUDIT_ROWS_PRIVATE.json')};containers={};tasks=[]
 for r in rr:
  i=index[r['id']];path=i['meta_path']
  if path not in containers:assert sha(path)==i['metadata_container_sha256'];containers[path]=read(path)
  raw=containers[path]['ground_info'][i['object_index']];assert ' '.join(raw['caption'].replace(',',' ,').split())==r['caption'];gt=transform_gt(raw['bbox_3d'],r['platform']);p=load(P/'predictions_PRIVATE'/(r['id']+'.npz'));b=p['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);iou=candidate_ious(b,gt)[0];tasks.append({**diags[r['id']],'IoU':dict(zip(ARMS,map(float,iou))),'GT_used_for_taskscore_only_after_seal_and_audit':True})
 assert len(tasks)==336;summary={};scenes=[];neg=[];contrasts={}
 for split in ['ALL336','D208','V64','M64']:
  rows=tasks if split=='ALL336' else [x for x in tasks if x['split']==split];s={}
  for pl in ['drone','quad']:
   part=[x for x in rows if x['platform']==pl];s[pl]={}
   for arm in ARMS:
    v=np.array([x['IoU'][arm] for x in part]);base=np.array([x['IoU'][ARMS[0]] for x in part]);d=v-base;s[pl][arm]={'n':len(part),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_CPU64_current_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(x['queries'][arm]!=x['queries'][ARMS[0]] for x in part),'scene_clusters':len({x['scene_PRIVATE'] for x in part})}
   for scene in sorted({x['scene_PRIVATE'] for x in part}):
    sr=[x for x in part if x['scene_PRIVATE']==scene];scenes.append({'split':split,'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{arm:float(np.mean([x['IoU'][arm] for x in sr])*100) for arm in ARMS}})
  s['balanced_mIoU_percent']={arm:float(np.mean([s[pl][arm]['mIoU_percent'] for pl in ['drone','quad']])) for arm in ARMS};summary[split]=s;contrasts[split]={}
  for pl in ['drone','quad','balanced']:
   m=s['balanced_mIoU_percent'] if pl=='balanced' else {arm:s[pl][arm]['mIoU_percent'] for arm in ARMS};contrasts[split][pl]={'parser_minus_current_CPU64_pp':m[ARMS[2]]-m[ARMS[0]],'official_LITERAL_minus_current_CPU64_pp':m[ARMS[1]]-m[ARMS[0]],'official_REMAP_minus_current_CPU64_pp':m[ARMS[3]]-m[ARMS[0]],'official_REMAP_minus_LITERAL_pp':m[ARMS[3]]-m[ARMS[1]],'current_CPU64_minus_stored_native_pp':m[ARMS[0]]-m[ARMS[4]]}
 for r in tasks:
  for arm in ARMS:
   for ref in ARMS:
    d=r['IoU'][arm]-r['IoU'][ref]
    if arm!=ref and d< -1e-10:neg.append({'alias':r['alias'],'split':r['split'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':float(d*100)})
 strata=[]
 for dim in ['structured','parser_status','parser_fallback','official_no_match','literal_remap_mask_equal']:
  for value in sorted({str(x[dim]) for x in tasks}):
   part=[x for x in tasks if str(x[dim])==value];strata.append({'dimension':dim,'value':value,'n':len(part),'mIoU_percent':{arm:float(np.mean([x['IoU'][arm] for x in part])*100) for arm in ARMS},'diagnostic_descriptive_not_selection':True})
 dump(HERE/'ALL336_TASK_ROWS_PRIVATE.json',tasks);dump(HERE/'TASK_SUMMARY.json',summary);dump(HERE/'FACTOR_CONTRASTS.json',contrasts);dump(HERE/'SCENE_SUMMARIES_PRIVATE.json',scenes);dump(HERE/'ALL_NEGATIVES.json',neg);dump(HERE/'COVERAGE_STRATA.json',strata);result={'state':'PASS_R48_SINGLE_OLD336_TASK_SCORE','n':336,'task_values':1680,'negative_ordered_comparisons':len(neg),'strict_thresholds':[.25,.5],'unchanged_source_6D_yaw0_boxes_vs_transformed_P5_7D_GT':True,'original_metadata_index_hash_and_exact_object_index_join':True,'all_unknowns_retained_in_denominator':True,'official_spans_annotation_informed_not_oracle':True,'no_parser_or_mask_change_after_task_labels':True,'no_FINAL_access':True,'prediction_seal_sha256':a['prediction_seal_sha256'],'independent_audit_sha256':a['independent_audit_sha256'],'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(HERE/'SCORING_RECEIPT.json',result);print(json.dumps({'receipt':result,'contrasts':contrasts},ensure_ascii=False))
if __name__=='__main__':main()
