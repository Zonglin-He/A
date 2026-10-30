"""Synthetic CPU-only checkpoint serialization/RNG replay fixture, not task weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import random,json
import numpy as np
import torch
from checkpointing import save_resume,load_resume,save_model
from factory import R,sha

def main():
 out=R/'CPU_RESUME_NEXTSTEP_FIXTURE';out.mkdir(exist_ok=False);random.seed(31);np.random.seed(41);torch.manual_seed(51);g=torch.Generator().manual_seed(61);m=torch.nn.Linear(3,2);opt=torch.optim.AdamW([{'params':list(m.parameters())},{'params':[]},{'params':[]}],lr=.01);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[2,3],gamma=.1);loss=m(torch.randn(4,3)).square().mean();loss.backward();opt.step();sched.step();provenance={'plan_sha256':'synthetic-CPU-fixture-NOT-scientific-model','fixture_only':True};saved={k:v.clone() for k,v in m.state_dict().items()};get=torch.cuda.get_rng_state_all;setall=torch.cuda.set_rng_state_all
 try:
  torch.cuda.get_rng_state_all=lambda:[];torch.cuda.set_rng_state_all=lambda xs:None;h=save_resume(m,opt,sched,g,out/'toy_resume.pth',1,1,provenance);expected=(random.random(),np.random.normal(size=5),torch.rand(5),torch.randint(0,10000,(5,),generator=g));fresh=torch.nn.Linear(3,2);op2=torch.optim.AdamW([{'params':list(fresh.parameters())},{'params':[]},{'params':[]}],lr=.7);sc2=torch.optim.lr_scheduler.MultiStepLR(op2,[2,3],gamma=.1);g2=torch.Generator().manual_seed(99);epoch,step=load_resume(out/'toy_resume.pth',fresh,op2,sc2,g2,provenance['plan_sha256']);actual=(random.random(),np.random.normal(size=5),torch.rand(5),torch.randint(0,10000,(5,),generator=g2));assert epoch==2 and step==1 and expected[0]==actual[0] and np.array_equal(expected[1],actual[1]) and torch.equal(expected[2],actual[2]) and torch.equal(expected[3],actual[3]);assert all(torch.equal(saved[k],fresh.state_dict()[k]) for k in saved);assert opt.param_groups[0]['lr']==op2.param_groups[0]['lr'];assert sched.state_dict()==sc2.state_dict();
  fixed=torch.tensor([[.1,.2,.3],[-.2,.4,.5]])
  for net,optim,scheduler in [(m,opt,sched),(fresh,op2,sc2)]:
   optim.zero_grad();net(fixed).square().mean().backward();optim.step();scheduler.step()
  assert all(torch.equal(m.state_dict()[k],fresh.state_dict()[k]) for k in m.state_dict());assert [g['lr'] for g in opt.param_groups]==[g['lr'] for g in op2.param_groups]
  assert not torch.cuda.is_initialized();result={'state':'synthetic_CPU_checkpoint_safe_resume_pass','no_task_model_or_data':True,'no_GPU':True,'weights_only_load':True,'Python_numpy_torch_CPU_loader_generator_exact':True,'model_optimizer_scheduler_roundtrip':True,'next_AdamW_update_after_resume_bitexact':True,'fixture_resume_next_epoch':epoch,'fixture_only_not_source_checkpoint':True,'checkpointing_code_sha256':sha(R/'checkpointing.py'),'toy_resume_sha256':h};(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:torch.cuda.get_rng_state_all=get;torch.cuda.set_rng_state_all=setall
if __name__=='__main__':main()
