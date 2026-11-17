"""All component endpoint scores after prediction seal and native audit."""
from common152 import *
import numpy as np
CORRECT=['full_high','full_high_parent_mean','full_low']
def main():
 check();assert rd(H/'FROZEN_AUDIT.json')['status']=='PASS' and rd(B/'frozen/SEALED.json')['n']==1798;rows=[];metrics=[];pairs=[];scenes=[];agg=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'frozen'/name/'samples_PRIVATE'/(sid+'.npz'));old=np.load(R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz'));qs=dict(zip(old['arm_names'],old['queries']));p150=np.load(R/'crossplatform/round150/frozen'/name/(sid+'.npz'));qs.update(dict(zip(p150['arm_names'],p150['queries'])));p151=np.load(R/'crossplatform/round151/frozen'/name/'samples_PRIVATE'/(sid+'.npz'));qs.update(dict(zip(p151['arm_names'],p151['queries'])));qs.update(dict(zip(a['arm_names'],a['queries'])));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ev=np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz'));rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE','has_B']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[q]) for k,q in qs.items()},'queries_PRIVATE':{k:int(q) for k,q in qs.items()},'components':{'eligible':bool(ev['eligible']),'high_image_tokens':12288 if e['has_B'] else 0,'supported_groups':int(ev['S'].sum()),'retained_groups':int(ev['C'].sum())}})
 controls={'full_high':['norm_joint_zero','full_low','full_wrong','norm_TTA','Frozen_A'],'full_high_parent_mean':['norm_TTA','full_high_zero_mean','full_high','full_low_parent_mean','full_wrong_parent_mean','Frozen_A'],'full_low':['norm_joint_zero','full_high','full_wrong','norm_TTA','Frozen_A']}
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1 for r in rr],float)
   for arm in rr[0]['IoUs']:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
    for scene in sorted({r['scene_PRIVATE'] for r in rr}):
     ix=np.array([r['scene_PRIVATE']==scene for r in rr]);scenes.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'scene_PRIVATE':scene,'n':int(ix.sum()),'mIoU':float(np.average(y[ix],weights=w[ix])*100)})
   for method in CORRECT:
    for ctrl in controls[method]:
     dy=np.array([r['IoUs'][method]-r['IoUs'][ctrl] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'method':method,'control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE'][method]!=r['queries_PRIVATE'][ctrl] for r in rr)})
 for ag in ['raw','design_weighted']:
  for arm in rows[0]['IoUs']:
   mm=[m for m in metrics if m['aggregation']==ag and m['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 dec={'phase':'frozen_fullframe_2x_RGB_prerequisite','primary':'full_high','source_parameters_updated':0,'FINAL_access':False,'target_exposed_development':True,'endpoints':{}}
 for method in CORRECT:
  comp={ag:{c:next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==method)-next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==c) for c in controls[method]} for ag in ['raw','design_weighted']};dec['endpoints'][method]={'comparisons':comp}
 passed=all(dec['endpoints']['full_high']['comparisons'][ag][c]>0 for ag in ['raw','design_weighted'] for c in ['norm_joint_zero','full_low','full_wrong']);ens=all(dec['endpoints']['full_high_parent_mean']['comparisons'][ag][c]>0 for ag in ['raw','design_weighted'] for c in ['norm_TTA','full_high_zero_mean','full_high','full_low_parent_mean','full_wrong_parent_mean']);dec.update(proceed_sourcecal_and_TTA=passed,retained_oldTTA_plus_high_ensemble=ens,disposition='proceed_matched_sourcecal_then_Up_TTA' if passed else 'stop_fullframe2x_recipe_no_new_TTA');
 for k,v in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('PAIRED_ROWS',pairs),('SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',agg),('DECISION',dec)]:dump(H/(k+'.json'),v)
 print(dec,flush=True)
if __name__=='__main__':main()
