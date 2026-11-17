"""All component endpoint scores after prediction seal and native audit."""
from common147 import *
import numpy as np
CORRECT=['inc_pa_zero']
def main():
 check();assert rd(H/'IMAGE_AUDIT.json')['status']=='PASS' and rd(B/'image/SEALED.json')['n']==1798;rows=[];metrics=[];pairs=[];scenes=[];agg=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));qs=dict(zip(a['arm_names'],a['queries']));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE','has_B','backfilled','original_has_B']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[q]) for k,q in qs.items()},'queries_PRIVATE':{k:int(q) for k,q in qs.items()},'components':{'new_evidence_entry':e['backfilled'],'parameter_updates':0,'Apoint_on_new_lateB':True}})
 controls={'inc_pa_zero':['norm_TTA','inc_zero','inc_pa_Aimage','inc_pa_wrong_joint','inc_pa_Bimage','inc_pa_FA_FB_mean','inc_pa_FAB_FB_mean','inc_pa_raw_joint','Frozen_A']}
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
 dec={'primary':'inc_pa_zero','phase':'frozen_entry_prerequisite_not_parameter_TTA','FINAL_access':False,'exposed_target_method_development':True,'new_source_weight_training':False,'endpoints':{}}
 for method in CORRECT:
  comp={ag:{c:next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==method)-next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==c) for c in controls[method]} for ag in ['raw','design_weighted']};passed=all(x[c]>0 for x in comp.values() for c in ['norm_TTA','inc_zero','inc_pa_Aimage','inc_pa_wrong_joint']);dec['endpoints'][method]={'comparisons':comp,'retained_development_candidate':passed}
 dec['disposition']='frozen_entry_prerequisite_PASS_source_reader_calibration_authorized_next' if any(x['retained_development_candidate'] for x in dec['endpoints'].values()) else 'frozen_entry_prerequisite_FAIL_stop_entry_keep_R145';dec['normalized_wrong_binding_and_wrong_conditional_order_are_distinct_controls']=True
 for k,v in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('PAIRED_ROWS',pairs),('SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',agg),('DECISION',dec)]:dump(H/('IMAGE_'+k+'.json'),v)
 print(dec,flush=True)
if __name__=='__main__':main()
