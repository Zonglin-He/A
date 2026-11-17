"""Registered whole-cohort outcomes after native audit and immutable prediction seal."""
from common144 import *
import numpy as np

def main():
 check();assert rd(H/'AUDIT.json')['status']=='PASS' and rd(B/'formal/SEALED.json')['n']==1798;rows=[];metrics=[];scenes=[];agg=[];pairs=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'formal'/name/'samples_PRIVATE'/(sid+'.npz'));qs=dict(zip(a['arm_names'],a['queries']));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE','has_B']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[q]) for k,q in qs.items()},'queries_PRIVATE':{k:int(q) for k,q in qs.items()}})
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1 for r in rr],float)
   for arm in rr[0]['IoUs']:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
    for scene in sorted({r['scene_PRIVATE'] for r in rr}):
     ix=np.array([r['scene_PRIVATE']==scene for r in rr]);scenes.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'scene_PRIVATE':scene,'n':int(ix.sum()),'mIoU':float(np.average(y[ix],weights=w[ix])*100)})
   for ctrl in ['Frozen_A','Frozen_F','F_RGBB_only','opposite_reference','half_zero_aligned','F_Bimage_only_teacher','image_B_TTA']:
    dy=np.array([r['IoUs']['aligned_reference']-r['IoUs'][ctrl] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum())})
 for ag in ['raw','design_weighted']:
  for arm in rows[0]['IoUs']:
   mm=[m for m in metrics if m['aggregation']==ag and m['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 comp={ag:{c:next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']=='aligned_reference')-next(a['mIoU'] for a in agg if a['aggregation']==ag and a['arm']==c) for c in ['F_RGBB_only','opposite_reference','half_zero_aligned','image_B_TTA','Frozen_A']} for ag in ['raw','design_weighted']};passed=all(comp[a][c]>0 for a in comp for c in ['F_RGBB_only','opposite_reference','half_zero_aligned']);dec={'retain_input_state_development_candidate':passed,'parameter_TTA_contribution_claimed':False,'comparisons':comp,'FINAL_access':False,'no_grid_rescue_if_negative':True,'source_q10_not_certified_ground':True,'disposition':'retain_input_state_candidate_not_parameter_TTA' if passed else 'do_not_adopt_height_reference_recipe'}
 for k,v in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('PAIRED_ROWS',pairs),('SCENE_ROWS_PRIVATE',scenes),('AGGREGATE_ROWS',agg),('DECISION',dec)]:dump(H/(k+'.json'),v)
 print(dec,flush=True)
if __name__=='__main__':main()
