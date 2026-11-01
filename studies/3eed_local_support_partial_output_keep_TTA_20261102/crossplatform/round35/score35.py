"""Postseal all-A/DEV task summaries and fixed feasibility; no method selection."""
import sys,numpy as np
from common35 import *
CONFIGS=['R30_U_rows','U_rows_matched_denominator','partial_output_keep']
def main():
 out=BASE/'partial_keep_attempt001';assert (out/'SEALED.json').exists(),'Training not sealed';verify(out);audit=read(EXP/'CPU_AUDIT.json');assert audit['state']=='PASSED' and audit['training_seal_sha256']==sha(out/'SEALED.json');plan=read(EXP/'PLAN_LOCK.json');assert plan['configurations']==CONFIGS
 # Label access only after all training heads and A/DEV predictions were sealed.
 sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 from audit35 import group_readout_local
 feedback=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');fit=read(out/'FIT_RECEIPTS.json');records=[];summary={};neg=[];scene=[];pair=[];feas=[]
 for split in ['A208','DEV64']:
  summary[split]={}
  for pl in ['drone','quad']:
   rr=[r for r in rows(split) if r['platform']==pl];pred={name:load_npz(out/'predictions_PRIVATE'/(pl+'_'+name+'_'+split+'.npz')) for name in CONFIGS}
   for name in CONFIGS:assert pred[name]['row_ids_PRIVATE'].tolist()==[r['id'] for r in rr]
   local=[]
   for j,r in enumerate(rr):
    z=feature(r);queries={'source':int(z['scores'].argmax())};queries.update({name:int(pred[name]['queries'][j]) for name in CONFIGS});unique,inv=np.unique(list(queries.values()),return_inverse=True);v=ious(r,z['boxes'][unique])[inv];rec={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':pl,'scene_PRIVATE':r['scene'],'split':split,'queries':queries,'IoU':dict(zip(queries,map(float,v)))};records.append(rec);local.append(rec)
    for name in queries:
     for ref in ['source']+CONFIGS:
      if name!=ref and rec['IoU'][name]<rec['IoU'][ref]-1e-10:neg.append({'alias':r['alias'],'platform':pl,'split':split,'configuration':name,'reference':ref,'delta_pp':100*(rec['IoU'][name]-rec['IoU'][ref])})
   summary[split][pl]={}
   for name in ['source']+CONFIGS:
    v=np.array([r['IoU'][name] for r in local]);d=v-np.array([r['IoU']['source'] for r in local]);summary[split][pl][name]={'n':len(local),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'delta_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][name]!=r['queries']['source'] for r in local),'scene_clusters':len({r['scene_PRIVATE'] for r in local})}
   for sj,s in enumerate(sorted({r['scene_PRIVATE'] for r in local})):
    ss=[r for r in local if r['scene_PRIVATE']==s];scene.append({'split':split,'platform':pl,'scene_alias':pl+'_scene'+str(sj),'n':len(ss),'mIoU_percent':{name:float(np.mean([r['IoU'][name] for r in ss])*100) for name in ['source']+CONFIGS}})
  summary[split]['balanced_mIoU_percent']={n:float(np.mean([summary[split][pl][n]['mIoU_percent'] for pl in ['drone','quad']])) for n in ['source']+CONFIGS}
 for name in CONFIGS:
  fits=[r for r in fit if r['name']==name];directions=sum(r['B_pairs_satisfied'] for r in fits);final=sum(r['B_final_positive'] for r in fits);feas.append({'configuration':name,'B_pairs_satisfied':directions,'B_final_positive':final,'B_pairs_total':sum(f['label']=='B' for f in feedback),'all_pairs_satisfied':sum(r['pairs_satisfied'] for r in fits),'feasible':directions>=5 and final>=4,'criterion':'fixed original >=5/6 Bdirections and >=4/6 Bfinalpositive','DEV_balanced_mIoU_percent':summary['DEV64']['balanced_mIoU_percent'][name],'selected':False})
  for pl in ['drone','quad']:
   A=[r for r in rows() if r['platform']==pl];lookup={r['id']:i for i,r in enumerate(A)};ff=[f for f in feedback if f['platform']==pl];p=load_npz(out/'predictions_PRIVATE'/(pl+'_'+name+'_A208.npz'));rec=next(r for r in fit if r['platform']==pl and r['name']==name)
   for j,f in enumerate(ff):
    i=lookup[f['id_PRIVATE']];r=A[i];q=int(p['queries'][i]);pair.append({'configuration':name,'platform':pl,'alias':r['alias'],'pair_alias':f['pair_alias'],'feedback_label':f['label'],'exact_logodds_gap_GPU':rec['all_original_pair_gaps'][j],'positive_over_negative':bool(rec['all_original_pair_gaps'][j]>0),'final_positive':q in f['positive_group_PRIVATE'],'final_negative':q in f['negative_group_PRIVATE'],'final_unjudged_query':q not in f['positive_group_PRIVATE']+f['negative_group_PRIVATE'],'source_final_positive':int(feature(r)['scores'].argmax()) in f['positive_group_PRIVATE'],'no_GT_used_to_build_or_filter_preference':True})
 comparisons=[]
 for split in ['A208','DEV64']:
  for arm,ref in [('partial_output_keep','R30_U_rows'),('partial_output_keep','U_rows_matched_denominator'),('U_rows_matched_denominator','R30_U_rows')]:
   by={pl:float(np.mean([r['IoU'][arm]-r['IoU'][ref] for r in records if r['split']==split and r['platform']==pl])*100) for pl in ['drone','quad']};comparisons.append({'split':split,'arm':arm,'reference':ref,'platform_delta_pp':by,'balanced_delta_pp':float(np.mean(list(by.values()))),'diagnostic_only_no_new_scene_independence_claim':True})
 dump(EXP/'ADEV_ALL_ROWS_PRIVATE.json',records);dump(EXP/'ADEV_SUMMARY.json',summary);dump(EXP/'ADEV_ALL_NEGATIVES.json',neg);dump(EXP/'ADEV_SCENE_ROWS.json',scene);dump(EXP/'ADEV_PAIRED_COMPARISONS.json',comparisons);dump(EXP/'A_ALL_PAIR_EXECUTION_ROWS.json',pair);dump(EXP/'DEV_FEASIBILITY_TABLE.json',feas);dump(EXP/'SCORING_RECEIPT.json',{'state':'COMPLETED','n':len(records),'configurations':CONFIGS,'all_task_rows_preserved':True,'no_case_based_feedback_selection':True,'no_method_selected_by_this_tool':True,'matched_denominator_control_reported':True,'source_and_R30_controls':True,'training_seal_sha256':sha(out/'SEALED.json'),'CPU_audit_sha256':sha(EXP/'CPU_AUDIT.json'),'no_U_FINAL_read':True});print('R35_ADEV_SUMMARY',summary);print('R35_FEASIBILITY',feas)
if __name__=='__main__':main()
