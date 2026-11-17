"""Phase1 readback, after image-only controls sealed and CPU audited."""
from common142 import *
import numpy as np
def main():
 check();assert rd(H/'IMAGE_AUDIT.json')['status']=='PASS' and rd(B/'image/SEALED.json')['n']==1798;rows=[];metrics=[];pairs=[];agg=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));q=dict(zip(a['arm_names'],a['queries']));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE','has_B']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[v]) for k,v in q.items()},'queries_PRIVATE':{k:int(v) for k,v in q.items()}})
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1 for r in rr],float)
   for arm in rr[0]['IoUs']:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
   for ctrl in ['Frozen_A','Frozen_F','F_RGBB_only_wrong']:
    dy=np.array([r['IoUs']['F_RGBB_only']-r['IoUs'][ctrl] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'method':'F_RGBB_only','control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE']['F_RGBB_only']!=r['queries_PRIVATE'][ctrl] for r in rr)})
 for ag in ['raw','design_weighted']:
  for arm in rows[0]['IoUs']:
   mm=[m for m in metrics if m['aggregation']==ag and m['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 comp={ag:{c:next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']=='F_RGBB_only')-next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==c) for c in ['Frozen_A','Frozen_F','F_RGBB_only_wrong']} for ag in ['raw','design_weighted']};passed=all(v>0 for x in comp.values() for v in x.values());dec={'phase':'frozen_image_only_binding_prerequisite','passed':passed,'comparisons':comp,'parameters_updated':0,'exposed_target_ablation_followup_not_independent_confirmation':True,'TTA_not_yet_tested':True,'next':'separately_pin_source_image_reader_calibration_and_sameUp4896_conditional_TTA' if passed else 'stop_imagebank_supervision_recipe_no_new_loss','FINAL_access':False}
 for k,v in [('IMAGE_ROWS_PRIVATE',rows),('IMAGE_METHOD_ROWS',metrics),('IMAGE_PAIRED_ROWS',pairs),('IMAGE_AGGREGATE_ROWS',agg),('IMAGE_DECISION',dec)]:dump(H/(k+'.json'),v)
 print(dec,flush=True)
if __name__=='__main__':main()
