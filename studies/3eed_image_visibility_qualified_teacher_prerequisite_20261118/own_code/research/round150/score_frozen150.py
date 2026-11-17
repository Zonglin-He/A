"""All component endpoint scores after prediction seal and native audit."""
from common150 import *
import numpy as np
CORRECT=['vis_equal','vis_equal_Cteacher','vis_count','vis_equal_oldC_direct']
def main():
 check();assert rd(H/'FROZEN_AUDIT.json')['status']=='PASS' and rd(B/'frozen/SEALED.json')['n']==1798;rows=[];metrics=[];pairs=[];scenes=[];agg=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'frozen'/name/(sid+'.npz'));old=np.load(R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz'));qs=dict(zip(old['arm_names'],old['queries']));qs.update(dict(zip(a['arm_names'],a['queries'])));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ev=np.load(B/'visibility_PRIVATE'/name/(sid+'.npz'));rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE','has_B']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[q]) for k,q in qs.items()},'queries_PRIVATE':{k:int(q) for k,q in qs.items()},'components':{'eligible':bool(ev['eligible']),'RGB_supported_groups':int(ev['S_rgb'].sum()),'supported_groups':int(ev['S'].sum()),'retained_groups':int(ev['C'].sum())}})
 controls={'vis_equal':['norm_Bimage','vis_wrong_equal','vis_global_equal','norm_joint_zero','norm_Aimage','norm_FA_FB_mean','norm_TTA','Frozen_A'],'vis_equal_Cteacher':['old_Cteacher','vis_wrong_equal_Cteacher','vis_global_equal_Cteacher'],'vis_count':['norm_Bimage','vis_wrong_count','vis_global_count','norm_joint_zero','norm_TTA','Frozen_A'],'vis_equal_oldC_direct':['norm_direct','vis_wrong_equal_oldC_direct','vis_global_equal_oldC_direct','norm_joint_zero','norm_TTA','Frozen_A']}
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
 dec={'phase':'frozen_evidence_prerequisite','primary':'vis_equal','secondary':'vis_count_diagnostic_only','new_network_forwards':0,'FINAL_access':False,'target_exposed_development':True,'endpoints':{}}
 for method in CORRECT:
  comp={ag:{c:next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==method)-next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==c) for c in controls[method]} for ag in ['raw','design_weighted']};dec['endpoints'][method]={'comparisons':comp}
 full=all(dec['endpoints']['vis_equal']['comparisons'][ag][c]>0 for ag in ['raw','design_weighted'] for c in ['norm_Bimage','vis_wrong_equal','vis_global_equal']);cond=all(v>0 for ag in ['raw','design_weighted'] for v in dec['endpoints']['vis_equal_Cteacher']['comparisons'][ag].values());dec.update(full_teacher_evidence_gate=full,conditional_oldC_evidence_gate=cond,proceed_sourcecal_and_TTA=bool(full or cond),disposition='proceed_source_only_calibration_then_fixed_Up_TTA' if full or cond else 'stop_visibility_routing_evidence_branch_no_TTA');
 for k,v in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('PAIRED_ROWS',pairs),('SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',agg),('DECISION',dec)]:dump(H/(k+'.json'),v)
 print(dec,flush=True)
if __name__=='__main__':main()
