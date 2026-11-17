"""Full fixed-cohort factorial after all predictions and native audits are sealed."""
from common141 import *
import numpy as np
PRIMARY='F_nativeB_RGBB'
def main():
 check();assert rd(H/'AUDIT.json')['status']=='PASS' and rd(B/'formal/SEALED.json')['n']==1798;rows=[];metrics=[];pairs=[];scenes=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'formal'/name/'samples_PRIVATE'/(sid+'.npz'));q=dict(zip(a['arm_names'],a['queries']));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'has_B':e['has_B'],'IoUs':{k:float(y[v]) for k,v in q.items()},'queries_PRIVATE':{k:int(v) for k,v in q.items()}})
 arms=list(rows[0]['IoUs'])
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1 for r in rr],float)
   for arm in arms:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(z,weights=w)*100) for k,z in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
    for scene in sorted({r['scene_PRIVATE'] for r in rr}):
     ix=np.array([r['scene_PRIVATE']==scene for r in rr]);scenes.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'scene_PRIVATE':scene,'n':int(ix.sum()),'mIoU':float(np.average(y[ix],weights=w[ix])*100)})
   for method in ['F_RGBB_only','F_nativeB_only',PRIMARY,'F_extraA_duplicateA']:
    for ctrl in ['Frozen_A','Frozen_F']+(['F_RGBB_only','F_nativeB_only','F_nativeB_RGBB_wrong','F_extraA_duplicateA'] if method==PRIMARY else []):
     dy=np.array([r['IoUs'][method]-r['IoUs'][ctrl] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'method':method,'control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE'][method]!=r['queries_PRIVATE'][ctrl] for r in rr)})
 agg=[];decision={'primary':PRIMARY,'new_parameters_updated':0,'TTA_increment_not_tested_in_frozen_prerequisite':True,'comparisons':{},'factorial_interaction':{},'FINAL_access':False,'all1798_negatives_and_noB_fallbacks_retained':True}
 for ag in ['raw','design_weighted']:
  for arm in arms:
   mm=[r for r in metrics if r['aggregation']==ag and r['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([r[k] for r in mm])) for k in ['mIoU','Acc25','Acc50']}})
  v={r['arm']:r['mIoU'] for r in agg if r['aggregation']==ag};decision['comparisons'][ag]={c:v[PRIMARY]-v[c] for c in ['Frozen_A','Frozen_F','F_RGBB_only','F_nativeB_only','F_nativeB_RGBB_wrong','F_extraA_duplicateA']};decision['factorial_interaction'][ag]=v[PRIMARY]+v['Frozen_F']-v['F_RGBB_only']-v['F_nativeB_only']
 decision['prerequisite_passed']=all(decision['comparisons'][ag][ctrl]>0 for ag in decision['comparisons'] for ctrl in ['Frozen_A','Frozen_F','F_nativeB_RGBB_wrong','F_extraA_duplicateA']);decision['both_components_incremental']=all(decision['comparisons'][ag][ctrl]>0 for ag in decision['comparisons'] for ctrl in ['F_RGBB_only','F_nativeB_only']);decision['disposition']='retain_frozen_evidence_candidate_for_separate_matched_TTA_test' if decision['prerequisite_passed'] else 'native_paired_bank_recipe_not_adopted_no_TTA_loss_added'
 for name,x in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('PAIRED_ROWS',pairs),('SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',agg),('DECISION',decision)]:dump(H/(name+'.json'),x)
 print(decision,flush=True)
if __name__=='__main__':main()
