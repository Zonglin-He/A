"""All factorial levels reported after complete sealed native-audited predictions."""
from common140 import *
import numpy as np
LEVELS=['oldreader_ownsource','fullreader_pooled','fullreader_ownsource']
def main():
 check();assert rd(H/'AUDIT.json')['status']=='PASS' and rd(B/'formal/SEALED.json')['n']==1798;rows=[];metrics=[];paired=[];scenes=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'formal'/name/'samples_PRIVATE'/(sid+'.npz'));qs=dict(zip(a['arm_names'],a['queries']));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ev=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));groups=[np.flatnonzero(ev['assignment']==g) for g in range(int(ev['assignment'].max()+1))];best=int(y.argmax());S=ev['S'];r={k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[q]) for k,q in qs.items()},'queries_PRIVATE':{k:int(q) for k,q in qs.items()},'A256_best_IoU':float(y[best]),'scope_best_in_S':bool(S[ev['assignment'][best]]),'levels':{}}
  for lev in LEVELS:
   C=ev[lev+'_C'];good=np.array([y[g].max()>.25 for g in groups]);r['levels'][lev]={'eligible':bool(ev[lev+'_eligible']),'S_groups':int(S.sum()),'C_groups':int(C.sum()),'best_in_C':bool(C[ev['assignment'][best]]),'Acc25_good_any_C':bool((good&C).any()),'old_A_leader_in_C':bool(C[ev['assignment'][qs['Frozen_A']]]),'correct_delta_IoU':float(y[qs[lev+'_correct']]-y[qs['Frozen_A']]),'classification':'zero_best_already' if qs['Frozen_A']==best else 'best_outside_S' if not S[ev['assignment'][best]] else 'best_outside_C' if not C[ev['assignment'][best]] else 'C_internal_order'}
  rows.append(r)
 arms=list(rows[0]['IoUs'])
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1 for r in rr],float)
   for arm in arms:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'arm':arm,'aggregation':ag,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
    for scene in sorted({r['scene_PRIVATE'] for r in rr}):
     ix=np.array([r['scene_PRIVATE']==scene for r in rr]);scenes.append({'source':src,'target':tgt,'scene_PRIVATE':scene,'arm':arm,'aggregation':ag,'n':int(ix.sum()),'mIoU':float(np.average(y[ix],weights=w[ix])*100)})
   for lev in LEVELS:
    method=lev+'_correct'
    for ctrl in ['Frozen_A','Frozen_F','R135_F','C_B_A',lev+'_direct',lev+'_wrong']:
     dy=np.array([r['IoUs'][method]-r['IoUs'][ctrl] for r in rr]);paired.append({'source':src,'target':tgt,'aggregation':ag,'method':method,'control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE'][method]!=r['queries_PRIVATE'][ctrl] for r in rr)})
 agg=[]
 for ag in ['raw','design_weighted']:
  for arm in arms:
   mm=[m for m in metrics if m['aggregation']==ag and m['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 decision={'primary':'fullreader_ownsource_correct','all_levels_reported_no_target_best_selection':True,'strict_source_only_calibration_primary':True,'calibration_is_source_label_assisted_no_weight_training':True,'comparisons':{},'FINAL_access':False}
 for ag in ['raw','design_weighted']:
  m={a['arm']:a['mIoU'] for a in agg if a['aggregation']==ag};decision['comparisons'][ag]={c:m[decision['primary']]-m[c] for c in ['Frozen_A','Frozen_F','C_B_A','R135_F','fullreader_ownsource_direct','fullreader_ownsource_wrong']}
 decision['adopted']=all(decision['comparisons'][ag][c]>0 for ag in decision['comparisons'] for c in ['Frozen_A','Frozen_F','fullreader_ownsource_direct','fullreader_ownsource_wrong']);decision['disposition']='retain_source_calibrated_development_candidate' if decision['adopted'] else 'matched_calibration_repair_not_sufficient_for_TTA_task_gain'
 for name,x in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('AGGREGATE_ROWS',agg),('PAIRED_ROWS',paired),('SCENE_ROWS_PRIVATE',scenes),('DECISION',decision)]:dump(H/(name+'.json'),x)
 print(decision,flush=True)
if __name__=='__main__':main()
