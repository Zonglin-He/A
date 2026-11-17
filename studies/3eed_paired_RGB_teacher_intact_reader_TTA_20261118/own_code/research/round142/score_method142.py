"""Task scoring after all fixed teacher/Up updates and native audits seal."""
from common142 import *
import numpy as np
def main():
 check();assert rd(H/'TTA_AUDIT.json')['status']=='PASS' and rd(B/'method/SEALED.json')['n']==1798;rows=[];metrics=[];pairs=[];scenes=[];agg=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'method'/name/'samples_PRIVATE'/(sid+'.npz'));q=dict(zip(a['arm_names'],a['queries']));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ev=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];best=int(ass[y.argmax()]);good=np.array([y[g].max()>.25 for g in groups]);cls='zero_best_already' if y[q['F_RGBB_only']]>=y.max()-1e-12 else 'best_outside_S' if not ev['S'][best] else 'best_outside_C' if not ev['C'][best] else 'C_internal_order';rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE','has_B']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[v]) for k,v in q.items()},'queries_PRIVATE':{k:int(v) for k,v in q.items()},'diagnostic':{'eligible':bool(ev['eligible']),'S_groups':int(ev['S'].sum()),'C_groups':int(ev['C'].sum()),'best_in_C':bool(ev['C'][best]),'good25_any_C':bool((good&ev['C']).any()),'classification':cls}})
 controls=['Frozen_A','Frozen_F','R135_F','F_RGBB_only','direct_image_B','wrong_image_B_TTA','F_RGBB_only_wrong','F_Bimage_only_teacher'];arms=list(rows[0]['IoUs'])
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1 for r in rr],float)
   for arm in arms:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
    for scene in sorted({r['scene_PRIVATE'] for r in rr}):
     ix=np.array([r['scene_PRIVATE']==scene for r in rr]);scenes.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'scene_PRIVATE':scene,'n':int(ix.sum()),'mIoU':float(np.average(y[ix],weights=w[ix])*100)})
   for ctrl in controls:
    dy=np.array([r['IoUs']['image_B_TTA']-r['IoUs'][ctrl] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'method':'image_B_TTA','control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE']['image_B_TTA']!=r['queries_PRIVATE'][ctrl] for r in rr)})
 for ag in ['raw','design_weighted']:
  for arm in arms:
   mm=[m for m in metrics if m['aggregation']==ag and m['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 comp={ag:{c:next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']=='image_B_TTA')-next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==c) for c in controls} for ag in ['raw','design_weighted']};passed=all(comp[ag][c]>0 for ag in comp for c in ['F_RGBB_only','direct_image_B','wrong_image_B_TTA','F_Bimage_only_teacher']);dec={'primary':'image_B_TTA','comparisons':comp,'passed':passed,'same_information_parameter_increment_established_in_scope':passed,'source_checkpoint_weights_retrained':False,'interface_parameters':4896,'steps':3,'original_rho0':True,'source_only_image_reader_kappa':True,'target_GT_hyperparameter_selection':False,'independent_validation':False,'all1798_negatives_unknowns_and_noB_retained':True,'FINAL_access':False,'disposition':'retain_as_development_candidate_for_separately_locked_expansion' if passed else 'paired_image_teacher_TTA_recipe_not_adopted'}
 for k,v in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('PAIRED_ROWS',pairs),('SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',agg),('DECISION',dec)]:dump(H/(k+'.json'),v)
 print(dec,flush=True)
if __name__=='__main__':main()
