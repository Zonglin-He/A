"""Postseal R143 first-gradient interaction descriptions, no new model or deployment gate."""
from common145 import *
import numpy as np,collections

def main():
 check()
 for p,h in rd(H/'experiment/COMPONENT_CONFLICT_METADATA_CODE_LOCK.json')['files'].items():assert sha(R/p)==h,p
 rows=rd(R/'research/round143/ALL_ROWS_PRIVATE.json');gg={(r['source'],r['target'],r['id_PRIVATE']):r for r in rd(R/'research/round143/GRADIENT_ROWS_PRIVATE.json')};den={(s,t):sum(r['sampling_weight'] for r in rows if (r['source'],r['target'])==(s,t)) for s,t in {(r['source'],r['target']) for r in rows}};out=[]
 for label in ['negative_cosine','nonnegative_cosine','one_component_zero']:
  aa=[]
  for r in rows:
   g=gg.get((r['source'],r['target'],r['id_PRIVATE']))
   if g is None:continue
   lab='one_component_zero' if g['cosine'] is None else 'negative_cosine' if g['cosine']<0 else 'nonnegative_cosine'
   if lab==label:aa.append(r)
  for ctrl in ['F_RGBB_only','partial_only','rank_only']:
   ds=np.array([r['IoUs']['image_B_TTA']-r['IoUs'][ctrl] for r in aa]);cc=np.array([d*r['sampling_weight']/den[r['source'],r['target']]*100/6 for d,r in zip(ds,aa)]);out.append({'gradient_bin':label,'old_R143_reference_control':ctrl,'n':len(aa),'full_R142_better':int((ds>1e-12).sum()),'full_R142_worse':int((ds< -1e-12).sum()),'weighted_contribution_with_original_full_denominator_pp':float(cc.sum())})
 x={'status':'POSTSEAL_HISTORICAL_COMPONENT_METADATA_ONLY','rows':out,'firstgradient_only_not_all_later_steps':True,'not_R145_gradients_or_results':True,'no_new_model_or_GT_cache_or_taskscore':True,'not_deployment_gradient_gate_or_new_optimizer':True,'FINAL_access':False};dump(H/'COMPONENT_CONFLICT_METADATA_ONLY.json',x);print(x,flush=True)
if __name__=='__main__':main()
