"""No-new-RGB controls, locked before any current-round GT score.

Interpolate pre-existing per-endpoint normalized native logits onto new view
frame IDs, then reapply the identical two-offset distribution/readout. These
controls diagnose restricted hypothesis support/logit sampling; not candidates.
"""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_temporal_view128_v1 import D,DEV

def main():
 import torch,numpy as np
 from scipy.special import logsumexp
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_temporal_view128_v1 import readout
 p=read(D/'PLAN_LOCK.json');assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==128;files=[]
 for r in p['records']:
  a=r['alias'];old=load(DEV/'priors'/f'{a}.pt');sig=np.zeros((len(old['ids']),2));raw=[]
  for rec,ll in zip(old['records'],old['logits']):
   x=ll[0].numpy().astype(float);x-=logsumexp(x,axis=0,keepdims=True);sig[np.searchsorted(old['ids'],rec['frame_ids'])]=x
  outputs={};observations={}
  for view,grid in r['observation_grids'].items():
   if grid==old['ids']:v=dict(ids=old['ids'],q=old['q'],records=old['records'],logits=old['logits'],exact_Native_reuse=True)
   else:
    x=np.column_stack([np.interp(grid,old['ids'],sig[:,j])for j in range(2)]).astype(np.float32);rr=[dict(offset=k,frame_ids=grid[k::2])for k in [0,1]];ll=[torch.from_numpy(x[k::2][None].copy())for k in [0,1]];ids,q=mass_table(probabilities(rr,ll));assert ids==grid;v=dict(ids=ids,q=q,records=rr,logits=ll,exact_Native_reuse=False)
   observations[view]=v;outputs[view+'NoRGBDirect']=readout(old['ids'],[(1.,v['ids'],v['q'])]);outputs[view+'NoRGBMix']=readout(old['ids'],[(.5,old['ids'],old['q']),(.5,v['ids'],v['q'])])
  files.append(save(D/'no_RGB_controls'/f'{a}.pt',dict(alias=a,outputs=outputs,observations=observations,new_model_forwards=0,new_RGB=0,GT_read=False)))
 put(D/'CONTROL_BARRIER.json',dict(count=128,files=files,all_before_new_GT=True,main_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),new_models=0,new_RGB=0))
if __name__=='__main__':main()
