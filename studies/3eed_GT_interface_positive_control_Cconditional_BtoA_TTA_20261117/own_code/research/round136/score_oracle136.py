"""Diagnostic-only GT positive-control gate. No method choices by target score."""
from common136 import *
import numpy as np
def main():
 check();assert rd(H136/'ORACLE_AUDIT.json')['status']=='PASS';root=B136/'oracle';reg=rd(H136/'INPUT_REGISTRY_PRIVATE.json');rows=[];methods=[];margin=[]
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];a=np.load(root/name/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(a['arm_names'],a['scores']));rec=rd(root/name/'records_PRIVATE'/(sid+'.json'));y=np.load(R136/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];r={k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'eligible':rec['eligible'],'IoUs':{k:float(y[v.argmax()]) for k,v in p.items()}}
  if rec['eligible']:
   pos=y>=y.max()-1e-12;qa=int(p['Frozen_A'].argmax());lp0=np.log(np.maximum(p['Frozen_A'],1e-300));lp1=np.log(np.maximum(p['GT3_diagnostic'],1e-300));nontrivial=bool(not pos[qa]);d=float(lp1[pos].max()-lp1[qa]-lp0[pos].max()+lp0[qa]);r.update(nontrivial_GT_target=nontrivial,GT_margin_change=d,initial_loss=rec['initial_loss'],final_loss=rec['fit']['trace'][-1]['task_after'],actual_GT_best_selected=bool(pos[int(p['GT3_diagnostic'].argmax())]));margin.append(r)
  rows.append(r)
 for name in sorted({r['source']+'_to_'+r['target'] for r in rows}):
  rr=[r for r in rows if r['source']+'_to_'+r['target']==name]
  for weighted in [False,True]:
   for arm in ['Frozen_A','GT_direct_diagnostic','GT3_diagnostic']:
    y=np.array([r['IoUs'][arm] for r in rr]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rr]);methods.append({'source':rr[0]['source'],'target':rr[0]['target'],'aggregation':'design_weighted' if weighted else 'raw','arm':arm,'mIoU':float(np.average(y,weights=w)*100),'Acc25':float(np.average(y>.25,weights=w)*100),'Acc50':float(np.average(y>.5,weights=w)*100)})
 agg=[]
 for ag in ['raw','design_weighted']:
  for arm in ['Frozen_A','GT_direct_diagnostic','GT3_diagnostic']:
   mm=[r for r in methods if r['aggregation']==ag and r['arm']==arm];agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([r[k] for r in mm])) for k in ['mIoU','Acc25','Acc50']}})
 rr=[r for r in margin if r['nontrivial_GT_target']];improved=sum(r['GT_margin_change']>1e-10 for r in rr);gains={ag:next(r['mIoU'] for r in agg if r['aggregation']==ag and r['arm']=='GT3_diagnostic')-next(r['mIoU'] for r in agg if r['aggregation']==ag and r['arm']=='Frozen_A') for ag in ['raw','design_weighted']};dv=np.array([r['IoUs']['GT3_diagnostic']-r['IoUs']['Frozen_A'] for r in rows]);decision={'interface_positive_control_passed':all(v>0 for v in gains.values()) and improved>len(rr)/2,'same_budget3step':True,'diagnostic_not_TTA':True,'gain_pp':gains,'eligible':len(margin),'nontrivial_eligible':len(rr),'best_GT_margin_improves':improved,'better':int((dv>1e-12).sum()),'worse':int((dv< -1e-12).sum()),'same':int((abs(dv)<=1e-12).sum()),'mean_initial_loss':float(np.mean([r['initial_loss'] for r in margin])),'mean_final_loss':float(np.mean([r['final_loss'] for r in margin])),'target_GT_does_not_select_unlabeled_threshold_or_sample':True,'unlabeled_method_design_already_locked':True,'FINAL_access':False}
 dump(H136/'ORACLE_ALL_ROWS_PRIVATE.json',rows);dump(H136/'ORACLE_METHOD_ROWS.json',methods);dump(H136/'ORACLE_AGGREGATE.json',agg);dump(H136/'ORACLE_DECISION.json',decision);print(decision,flush=True)
if __name__=='__main__':main()
