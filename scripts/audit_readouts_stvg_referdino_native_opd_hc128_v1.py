"""Independent saved decoder-to-E readouts and fixed native-label path losses."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_referdino_native_opd_hc128_v1 import D,P,PRE,verify

def main():
 import torch,numpy as np
 from methods.decota_final_simplified_v1.objectives import generalized_iou
 p=verify(True);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==128;assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';start=time.perf_counter();counts=dict(readout_paths=0,anchor_costs=0,offset_supports=0);errors=dict(mapping=0.,anchor_loss=0.);seeds=p['seeds']
 for r in p['records']:
  a=r['alias'];inp=load(PRE/'inputs'/(a+'.pt'));z=load(D/'predictions'/(a+'.pt'));target=load(PRE/'teacher0'/(a+'.pt'))['native']['boxes'].double();src=inp['Source'].double();E=torch.tensor(inp['E'],dtype=torch.float64);pos=r['positions8']
  for view in range(2):assert inp['records'][view]['frame_ids']==inp['frame_ids'][view::2]and len(inp['spatial_inputs'][view]['query_tgt'])==len(inp['frame_ids'][view::2]);counts['offset_supports']+=1
  for fit in z['fits'].values():
   for step in fit['path']:
    b=step['native'].double();expected=torch.cat((E[:,:2]+b[:,:2]-src[:,:2],E[:,2:]*(b[:,2:].log()-src[:,2:].log()).exp()),-1);err=float((expected-step['boxes']).abs().max());errors['mapping']=max(errors['mapping'],err);assert err<1e-12;counts['readout_paths']+=1
    obs=step['boxes'][pos];c=5*(obs-target).abs().sum(-1)+2*(1-generalized_iou(obs,target));err=abs(float(c.mean())-step['anchor_loss']);errors['anchor_loss']=max(errors['anchor_loss'],err);assert err<1e-12;counts['anchor_costs']+=1
 results=read(P/'RESULTS_PUBLIC.json');seed_variability={}
 for arm in ['Fixed','Live']:
  seed_variability[arm]={}
  for ep in ['v','unobserved_actual_s']:
   vals=[100*results['by_seed'][str(s)]['means'][f'{arm}_{s}'][ep]for s in seeds];seed_variability[arm][ep]=dict(per_seed_percent=dict(zip(map(str,seeds),vals)),std_across3_seeds_pp=float(np.std(vals,ddof=1)),range_percent=[min(vals),max(vals)],only3_seeds_not_population_CI=True)
 put(P/'READOUT_AND_NATIVE_LABEL_AUDIT.json',dict(status='pass',checks=counts,max_errors=errors,seed_variability=seed_variability,CPU_seconds=time.perf_counter()-start,no_new_inference=True,no_GT_candidate_or_parameter_selection=True));assert not torch.cuda.is_initialized();print(counts,errors,flush=True)
if __name__=='__main__':main()
