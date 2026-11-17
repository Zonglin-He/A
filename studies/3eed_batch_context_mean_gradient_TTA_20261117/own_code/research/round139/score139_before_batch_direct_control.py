"""Score all three fixed partitions only after complete prediction/native audit."""
from common139 import *
import numpy as np
def main():
 check();assert rd(H/'AUDIT.json')['status']=='PASS';assert rd(B/'formal/SEALED.json')['n']==1798;rows=[];metrics=[];pairs=[];scenes=[]
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'formal/samples_PRIVATE'/name/(sid+'.npz'));y=np.load(R/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];q=dict(zip(a['arm_names'],a['queries']));rows.append({k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'IoUs':{k:float(y[v]) for k,v in q.items()},'queries_PRIVATE':{k:int(v) for k,v in q.items()}})
 arms=list(rows[0]['IoUs'])
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)]
  for ag in ['raw','design_weighted']:
   w=np.array([r['sampling_weight'] if ag=='design_weighted' else 1. for r in rr])
   for arm in arms:
    y=np.array([r['IoUs'][arm] for r in rr]);metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'n':len(rr),**{k:float(np.average(v,weights=w)*100) for k,v in [('mIoU',y),('Acc25',y>.25),('Acc50',y>.5)]}})
    for sc in sorted({r['scene_PRIVATE'] for r in rr}):
     ix=np.array([r['scene_PRIVATE']==sc for r in rr]);scenes.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'scene_PRIVATE':sc,'n':int(ix.sum()),'mIoU':float(np.average(y[ix],weights=w[ix])*100)})
   for order in [0,1,2]:
    method='correct_order'+str(order)
    for ctrl in ['Frozen_A','Frozen_F','direct_C_B_A','C_B_A','wrong_order'+str(order)]:
     dy=np.array([r['IoUs'][method]-r['IoUs'][ctrl] for r in rr]);pairs.append({'source':src,'target':tgt,'aggregation':ag,'method':method,'control':ctrl,'gain_pp':float(np.average(dy,weights=w)*100),'better':int((dy>1e-12).sum()),'worse':int((dy< -1e-12).sum()),'query_changed':sum(r['queries_PRIVATE'][method]!=r['queries_PRIVATE'][ctrl] for r in rr)})
   for kind in ['correct','wrong']:
    mm=[m for m in metrics if (m['source'],m['target'],m['aggregation'])==(src,tgt,ag) and m['arm'] in [kind+'_order'+str(k) for k in [0,1,2]]];metrics.append({'source':src,'target':tgt,'aggregation':ag,'arm':kind+'_partition_mean','n':len(rr),**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 agg=[]
 for ag in ['raw','design_weighted']:
  for arm in arms+['correct_partition_mean','wrong_partition_mean']:
   mm=[m for m in metrics if m['aggregation']==ag and m['arm']==arm];assert len(mm)==6;agg.append({'aggregation':ag,'arm':arm,**{k:float(np.mean([m[k] for m in mm])) for k in ['mIoU','Acc25','Acc50']}})
 decision={'primary':'correct_order0','three_orders_not_cherry_picked':True,'partition_mean_is_replicate_mean_not_a_deployed_ensemble':True,'comparisons':{},'FINAL_access':False,'source_training':False,'transductive_batch_context_exceeds_individual_R136':True,'GT_used_only_diagnostic_gate_and_postseal_scoring':True}
 for ag in ['raw','design_weighted']:
  v={m['arm']:m['mIoU'] for m in agg if m['aggregation']==ag};decision['comparisons'][ag]={}
  for method,wrong in [('correct_order0','wrong_order0'),('correct_partition_mean','wrong_partition_mean')]:decision['comparisons'][ag][method]={c:v[method]-v[c] for c in ['Frozen_A','Frozen_F','direct_C_B_A','C_B_A',wrong]}
 decision['retained']=all(x>0 for a in decision['comparisons'].values() for d in a.values() for x in d.values());decision['disposition']='retain_batch8_development_candidate' if decision['retained'] else 'batch8_recipe_not_adopted'
 for name,x in [('ALL_ROWS_PRIVATE',rows),('METHOD_ROWS',metrics),('SCENE_ROWS_PRIVATE',scenes),('PAIRED_ROWS',pairs),('AGGREGATE_ROWS',agg),('METHOD_DECISION',decision)]:dump(H/(name+'.json'),x)
 print(decision,flush=True)
if __name__=='__main__':main()
