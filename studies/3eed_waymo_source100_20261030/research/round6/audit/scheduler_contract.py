"""CPU scalar optimizer/scheduler contract; no model, data, target labels or GPU."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import torch
R=Path(__file__).resolve().parent;B=R.parents[2];SOURCE=B/'crossplatform/round3/p0/upstream/utils/lr_scheduler.py'
def main():
 spec=importlib.util.spec_from_file_location('round6_official_lr_scheduler',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);p=torch.nn.Parameter(torch.zeros(1));q=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([{'params':[p]},{'params':[q],'lr':1e-3},{'params':[],'lr':1e-5}],lr=1e-4,weight_decay=.0005);args=argparse.Namespace(lr_scheduler='step',warmup_epoch=-1,lr_decay_epochs=[25,26],lr_decay_rate=.1,max_epoch=100,warmup_multiplier=100);sch=m.get_scheduler(opt,153,args);assert dict(sch.milestones)=={3978:1,4131:1};rows=[];events=[];previous=[g['lr'] for g in opt.param_groups]
 for step in range(1,15301):
  before=[g['lr'] for g in opt.param_groups];opt.step();sch.step();after=[g['lr'] for g in opt.param_groups]
  if before!=after:events.append({'optimizer_step_just_completed':step,'epoch':(step-1)//153+1,'batch1based':(step-1)%153+1,'before':before,'after':after,'first_optimizer_step_using_newlr':step+1,'first_epoch_using_newlr':step//153+1})
  if step in [1,3825,3826,3977,3978,3979,4130,4131,4132,15300]:rows.append({'step':step,'scheduler_last_epoch':sch.last_epoch,'lr_before_optimizer':before,'lr_after_scheduler':after})
 assert [x['optimizer_step_just_completed'] for x in events]==[3978,4131];assert [x['first_epoch_using_newlr'] for x in events]==[27,28]
 result={'state':'passed_CPU_scalar_scheduler_contract','official_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'torch_version':torch.__version__,'train_expression_reference':3682,'batch24_drop_last_steps_per_epoch':153,'epochs':100,'optimizer_steps':15300,'nominal_script_milestones':[25,26],'warmup_epoch':-1,'actual_scheduler_step_milestones':[3978,4131],'scheduler_fires_after_last_optimizer_step_of_epochs':[26,27],'new_rates_used_starting_epochs':[27,28],'optimizer_then_scheduler_order':True,'empty_text_group_retained':True,'decays':events,'probes':rows,'no_model_GPU_dataset_target_read':True};out=R/'SCHEDULER_CONTRACT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
