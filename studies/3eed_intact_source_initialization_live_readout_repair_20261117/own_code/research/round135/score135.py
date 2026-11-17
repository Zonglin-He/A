"""Post-seal fixed-box GT evaluation; never selects an inference output."""
from common135 import *
import numpy as np,collections
O135=H135/'scores_attempt001'
def metric(rows,k,weighted):
 y=np.array([r['IoUs'][k] for r in rows]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rows]);return {'n':len(rows),'mIoU_percent':float(np.average(y,weights=w)*100),'Acc25_percent':float(np.average(y>.25,weights=w)*100),'Acc50_percent':float(np.average(y>.5,weights=w)*100)}
def main():
 assert rd(H135/'NATIVE_AUDIT.json')['status']=='PASS';mat=rd(B135/'formal/MATRIX_SEALED.json');assert mat['n']==1798;reg=rd(H135/'INPUT_REGISTRY_PRIVATE.json');O135.mkdir(exist_ok=False);allrows=[];methods=[];scenes=[];paired=[];tails=[];stages=[];arms=mat['arms']+['original_source'];comparisons=[('tta_A',c) for c in ['intact_live_A','AB_mean','direct_A','wrong_A','legacy_R134','intact_live_F']]+[('tta_F',c) for c in ['intact_live_F','intact_live_A','AB_mean','direct_F','wrong_F','tta_A','legacy_R134']]+[('intact_live_A','legacy_zero_F'),('intact_live_A','legacy_R134'),('intact_live_F','intact_live_A')]
 for name in sorted({e['source']+'_to_'+e['target'] for e in reg}):
  src,_,tgt=name.split('_');D=B135/'formal'/name;rr=[]
  for e in [e for e in reg if e['source']==src and e['target']==tgt]:
   sid=e['row']['id'];a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));old=np.load(R135/'crossplatform/round97'/name/'samples_PRIVATE'/(sid+'.npz'));ps['original_source']=dict(zip(old['arm_names'],old['scores']))['original_source'];y=np.load(R135/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];q={k:int(p.argmax()) for k,p in ps.items()};ious={k:float(y[j]) for k,j in q.items()};ev=np.load(D/'evidence_PRIVATE'/(sid+'.npz'));rec=rd(D/'fit_records_PRIVATE'/(sid+'.json'));ass=ev['assignment'];good=y>.25;S=ev['S'];C=ev['C'];qa=q['intact_live_A'];g0=ass[qa]
   row={k:e[k] for k in ['source','target','frame_PRIVATE','content_group_PRIVATE','sampling_weight']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'has_B':rec['has_B'],'active':rec['active'],'partial_active':rec['partial_active'],'pair_count':rec['pair_count'],'S_groups':int(S.sum()),'C_groups':int(C.sum()),'A_top_in_C':bool(C[g0]),'A_top_challenged_by_pair':bool((ev['lose']==g0).any()),'recoverable_A_Acc25_error':bool(y[qa]<=.25 and good.any()),'S_has_good':bool((good&S[ass]).any()),'C_has_good':bool((good&C[ass]).any()),'A256_oracle_IoU':float(y.max()),'IoUs':ious,'queries_PRIVATE':q};rr.append(row)
   for method,control in comparisons:
    dv=ious[method]-ious[control]
    if abs(dv)>1e-12:tails.append({k:row[k] for k in ['source','target','id_PRIVATE','frame_PRIVATE','content_group_PRIVATE','scene_PRIVATE','has_B','active','partial_active','pair_count','S_groups','C_groups','sampling_weight','C_has_good']}|{'method':method,'control':control,'method_IoU':ious[method],'control_IoU':ious[control],'delta_IoU':dv,'same_geometric_group':bool(ass[q[method]]==ass[q[control]]),'method_in_C':bool(C[ass[q[method]]]),'control_in_C':bool(C[ass[q[control]]])})
  dump(O135/(name+'_ALL_ROWS_PRIVATE.json'),rr);allrows+=rr
  for weighted in [False,True]:
   agg='design_weighted' if weighted else 'raw';w=np.array([r['sampling_weight'] if weighted else 1. for r in rr])
   for k in arms:
    methods.append({'source':src,'target':tgt,'arm':k,'aggregation':agg,**metric(rr,k,weighted)})
    for sc in sorted({r['scene_PRIVATE'] for r in rr}):scenes.append({'source':src,'target':tgt,'scene_PRIVATE':sc,'arm':k,'aggregation':agg,**metric([r for r in rr if r['scene_PRIVATE']==sc],k,weighted)})
   for m,c in comparisons:
    dv=np.array([r['IoUs'][m]-r['IoUs'][c] for r in rr]);qc=np.array([r['queries_PRIVATE'][m]!=r['queries_PRIVATE'][c] for r in rr]);paired.append({'source':src,'target':tgt,'method':m,'control':c,'aggregation':agg,'gain_pp':float(np.average(dv,weights=w)*100),'better':int((dv>1e-12).sum()),'worse':int((dv< -1e-12).sum()),'same':int((abs(dv)<=1e-12).sum()),'query_changed':int(qc.sum()),'largest_positive_contribution_pp':float(max(0.,(dv*w).max())/w.sum()*100/6),'largest_negative_contribution_pp':float(min(0.,(dv*w).min())/w.sum()*100/6)})
   for subgroup in ['no_B','has_B']:
    mask=np.array([r['has_B']==(subgroup=='has_B') for r in rr]);den=w.sum()
    for m,c in [('legacy_zero_F','intact_live_A'),('legacy_R134','intact_live_A'),('intact_live_F','intact_live_A'),('tta_A','intact_live_A'),('tta_F','intact_live_A')]:
     dv=np.array([r['IoUs'][m]-r['IoUs'][c] for r in rr]);stages.append({'source':src,'target':tgt,'subgroup':subgroup,'aggregation':agg,'method':m,'control':c,'n':int(mask.sum()),'whole_six_mean_contribution_pp':float((dv[mask]*w[mask]).sum()/den*100/6)})
 aggregates=[]
 for agg in ['raw','design_weighted']:
  for group in ['all_six_equal','waymo','drone','quad']:
   for k in arms:
    pp=[r for r in methods if r['aggregation']==agg and r['arm']==k and (group=='all_six_equal' or r['target']==group)];aggregates.append({'aggregation':agg,'group':group,'arm':k,**{m:float(np.mean([r[m] for r in pp])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}})
 decision={'primary':'tta_A','engineering_repair_accepted':True,'TTA_development_candidate_retained':None,'comparisons':{},'target_GT_selects_configuration':False,'independent_validation':False,'FINAL_access':False}
 for agg in ['raw','design_weighted']:
  v={r['arm']:r['mIoU_percent'] for r in aggregates if r['aggregation']==agg and r['group']=='all_six_equal'};decision['comparisons'][agg]={'primary_mIoU_percent':v['tta_A'],'primary_gains_pp':{c:v['tta_A']-v[c] for c in ['intact_live_A','AB_mean','direct_A','wrong_A']},'alternative_F_gains_pp':{c:v['tta_F']-v[c] for c in ['intact_live_A','intact_live_F','AB_mean','direct_F','wrong_F']}}
 decision['TTA_development_candidate_retained']=all(x>0 for d in decision['comparisons'].values() for x in d['primary_gains_pp'].values());decision['current_working_version']='R135_intact_live_A_R134_TTA_development' if decision['TTA_development_candidate_retained'] else 'R135_intact_complete_Frozen_A_engineering_baseline';decision['alternative_F_not_promoted_by_target_argmax']=True
 for n,obj in [('ALL_METHOD_ROWS',methods),('ALL_PAIRED_ROWS',paired),('ALL_SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',aggregates),('SUBGROUP_CONTRIBUTIONS',stages),('ALL_CHANGED_CASES_PRIVATE',tails)]:dump(O135/(n+'.json'),obj)
 dump(H135/'DECISION.json',decision);print(decision,flush=True)
if __name__=='__main__':main()
