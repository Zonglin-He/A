"""Two prelocked first-source zero-guidance numerical interface probes; no training."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_observation_utilization_v1 import BASE,OLD,REPAIR,PARENT,read,put,load,condition

def main():
 import torch
 import vg_tta.stvg_joint_observation_utilization_v1 as impl
 from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior
 from scripts.run_decota_paper_main_v1 import gpu
 from main_method.runtime import resource_guard
 p=read(BASE/'PLAN_LOCK.json');assert read(BASE/'PREDICTION_BARRIER.json')['count']==24;prior=read(BASE/'GPU_COMPLETION.json');lease=gpu();beg=time.monotonic();rows=[];original=impl.observation_terms
 def zero(clean,c,obs,variance,fixed=False):return clean.sum((1,2))*0,dict(active_count=0.,responsibility_mean=0.,residual_mean=0.)
 try:
  for ds in ['hc2','vidstg']:
   resource_guard();st=load(REPAIR/'statistics'/(ds+'.pt'));net=PreconditionedJointPrior(st['mean'],st['scale']).cuda().eval().requires_grad_(False);net.load_state_dict(load(REPAIR/'models'/(ds+'.pt'))['state']);r=next(r for r in p['source']if r['dataset']==ds);z=load(OLD/'features/source'/(r['alias']+'.pt'));c=condition(z);cc={k:v.expand((4,)+v.shape[1:])for k,v in c.items()};eps=load(PARENT/'predictions'/(r['alias']+'.pt'))['path']['epsilon'].cuda();obs=load(BASE/'observations'/(r['alias']+'.pt'))['observations']['DINO_all'];oo={k:v.cuda()for k,v in obs.items()if k!='positions'}
   impl.observation_terms=zero
   for clock in ['fixed','free']:
    q=impl.guided_rollout(net,cc,eps,oo,.01,clock=='fixed');base=load(BASE/'predictions'/clock/(r['alias']+'.pt'))['arms'];d=float((q['final']-base['none']['final']).abs().max());actual=float((base['DINO_all']['final']-base['none']['final']).abs().max());assert all(t['delta_norm']==0 for t in q['trace']);rows.append(dict(dataset=ds,alias=r['alias'],clock=clock,zero_guidance_autograd_vs_no_grad_max_abs=d,actual_DINO_vs_no_grad_max_abs=actual,zero_guidance_exact_zero=True))
   del net;torch.cuda.empty_cache()
 finally:
  impl.observation_terms=original;torch.cuda.synchronize();elapsed=time.monotonic()-beg;lease.close()
 assert elapsed<300 and elapsed+prior['new_reserved_seconds']<300 and prior['cumulative_reserved_seconds']+elapsed<14400
 put(BASE/'NUMERIC_INTERFACE_COMPLETION.json',dict(status='pass',count=2,clock_conditions=4,rows=rows,new_reserved_seconds=elapsed,new_total_reserved_seconds=elapsed+prior['new_reserved_seconds'],cumulative_reserved_seconds=elapsed+prior['cumulative_reserved_seconds'],limits_seconds=dict(current=300,cumulative=14400),purpose='autograd kernel numerical floor only, no scientific predictions changed or new scored arm',new_training=0,new_target_inference=0))
if __name__=='__main__':main()
