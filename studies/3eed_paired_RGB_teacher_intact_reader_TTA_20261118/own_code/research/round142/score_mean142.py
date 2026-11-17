"""Supplemental same-information baseline, scored only after sealing/auditing."""
from common142 import *
import numpy as np
def main():
 check();assert rd(H/'MEAN_AUDIT.json')['status']=='PASS';root=B/'supplementary_mean';oldrows=rd(H/'ALL_ROWS_PRIVATE.json');lookup={(r['source'],r['target'],r['id_PRIVATE']):r for r in oldrows};rows=[];metrics=[];pairs=[];agg=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(root/name/(sid+'.npz'));q=dict(zip(a['arm_names'],a['queries']));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];r=dict(lookup[src,tgt,sid]);r['IoUs']=dict(r['IoUs'])|{k:float(y[v]) for k,v in q.items()};r['queries_PRIVATE']=dict(r['queries_PRIVATE'])|{k:int(v) for k,v in q.items()};rows.append(r)
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1 for r in rr])
   for arm in ['FA_FB_probability_mean','FAB_FB_probability_mean']:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}});dy=np.array([r['IoUs']['image_B_TTA']-r['IoUs'][arm] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'method':'image_B_TTA','control':arm,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE']['image_B_TTA']!=r['queries_PRIVATE'][arm] for r in rr)})
 for ag in ['raw','design_weighted']:
  for arm in ['FA_FB_probability_mean','FAB_FB_probability_mean']:
   mm=[r for r in metrics if r['aggregation']==ag and r['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 mainagg=rd(H/'AGGREGATE_ROWS.json');comparisons={ag:{arm:next(x['mIoU'] for x in mainagg if x['aggregation']==ag and x['arm']=='image_B_TTA')-next(x['mIoU'] for x in agg if x['aggregation']==ag and x['arm']==arm) for arm in ['FA_FB_probability_mean','FAB_FB_probability_mean']} for ag in ['raw','design_weighted']};passed=all(v>0 for x in comparisons.values() for v in x.values());dec={'status':'POSTHOC_CANONICAL_SAME_INFORMATION_STRONG_CONTROL','canonical_equal_probability_mean_no_GT_weight_search':True,'original_preregistered_TTA_gate_preserved':rd(H/'DECISION.json')['passed'],'comparisons':comparisons,'TTA_exceeds_both_means':passed,'parameter_increment_over_supplementary_strong_controls_supported':passed,'source_checkpoint_OR_predictions_changed':False,'FINAL_access':False}
 for k,v in [('MEAN_ROWS_PRIVATE',rows),('MEAN_METHOD_ROWS',metrics),('MEAN_PAIRED_ROWS',pairs),('MEAN_AGGREGATE_ROWS',agg),('MEAN_DECISION',dec)]:dump(H/(k+'.json'),v)
 print(dec,flush=True)
if __name__=='__main__':main()
