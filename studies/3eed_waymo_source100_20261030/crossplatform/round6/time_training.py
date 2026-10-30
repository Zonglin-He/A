"""Released only by parent: exactly4warmup+8timed source-only full SGD steps."""
import argparse,json,time,datetime,os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from factory import R,P,U,ROBE,sha,configure,dataset,order_contract,loader,model,optimizer,scheduler,criterion,to_cuda,state_fingerprint,compute

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert sha(a.plan)==a.plan_sha256;plan=json.loads(a.plan.read_text());assert plan['mode']=='timing12' and plan['GPU_execution_authorized'] is True
 assert sha(P.parent/'research/round6/AUTHORIZATION.json')==plan['authorization_sha256']
 for f,h in plan['code_sha256'].items():assert sha(R/f)==h
 for f,h in plan['official_source_sha256'].items():assert sha(U/f)==h
 for f,h in plan['roberta_sha256'].items():assert sha(ROBE/f)==h
 assert plan['batch_size']==24 and plan['steps']==12 and plan['warmup_steps']==4 and plan['precision']=='FP32';a.output.mkdir(parents=True,exist_ok=False);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes());configure(plan);started=time.perf_counter();events=[]
 def event(name,**kw):
  x={'event':name,'elapsed_seconds':time.perf_counter()-started,**kw};events.append(x);print(json.dumps(x),flush=True)
 try:
  torch.cuda.set_device(0);dist.init_process_group('nccl',init_method='file://'+str((a.output/'nccl_rendezvous').resolve()),rank=0,world_size=1,timeout=datetime.timedelta(seconds=300));event('actual_NCCL_world1_initialized');ds=dataset('train');contract=order_contract(ds,'train');g=torch.Generator().manual_seed(0);dl=loader(ds,'train',g,plan);dl.sampler.set_epoch(1);assert len(dl)==153;event('source_loader_ready',n=len(ds),steps_per_epoch=len(dl));event('CPU_model_constructor_begin');m=model();initial=state_fingerprint(m);visual_initial=state_fingerprint(m,'backbone_net.');assert initial==plan['expected_model_initial_sha256'];event('CPU_model_initialized_from_generic_text_and_random_visual',initial_sha256=initial);m.cuda().train();opt=optimizer(m);sched=scheduler(opt,len(dl));crit=criterion().cuda();m=DDP(m,device_ids=[0],broadcast_buffers=False);assert m.module.text_encoder.training and all(not p.requires_grad for p in m.module.text_encoder.parameters());event('DDP_and_training_state_ready',trainable=sum(p.numel() for p in m.parameters() if p.requires_grad));it=iter(dl);event('spawn_workers_iterator_started');rows=[];torch.cuda.reset_peak_memory_stats()
  for step in range(12):
   before=time.perf_counter();event('next_batch_begin',step=step+1) if step==0 else None;batch=next(it);batch=to_cuda(batch);torch.cuda.synchronize();gpu_begin=time.perf_counter();event('batch_on_cuda_forward_begin',step=step+1) if step==0 else None
   # Same model+criterion as full training; do not spoof distributed normalizer.
   out=m({'point_clouds':batch['point_clouds'].float(),'text':batch['utterances']});event('forward_complete',step=step+1) if step==0 else None
   for k,v in batch.items():assert k not in out;out[k]=v
   from factory import compute_hungarian_loss
   loss,out=compute_hungarian_loss(out,6,crit,query_points_obj_topk=4);assert torch.isfinite(loss);event('loss_complete',step=step+1,loss=float(loss.detach())) if step==0 else None;opt.zero_grad();loss.backward();event('backward_complete',step=step+1) if step==0 else None;grad=torch.nn.utils.clip_grad_norm_(m.parameters(),.1);assert torch.isfinite(grad);assert all(p.grad is None for p in m.module.text_encoder.parameters());nonzero=sum(p.grad is not None and torch.count_nonzero(p.grad).item()>0 for p in m.parameters());opt.step();sched.step();torch.cuda.synchronize();row={'step':step+1,'warmup':step<4,'loss':float(loss.detach()),'grad_norm_before_clip':float(grad),'nonzero_gradient_parameter_count':int(nonzero),'forward_loss_backward_update_seconds':time.perf_counter()-gpu_begin,'wall_including_loader_seconds':time.perf_counter()-before,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'lr':[p['lr'] for p in opt.param_groups]};rows.append(row);print(json.dumps(row),flush=True)
  final=state_fingerprint(m.module);assert final!=initial;assert state_fingerprint(m.module,'text_encoder.')==plan['expected_text_encoder_initial_sha256'];result={'state':'engineering12_completed_weights_discarded_not_scientific_checkpoint','n_steps':12,'warmup_steps':4,'timed_steps':8,'batch_size':24,'precision':'FP32','TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'DDP_world_size':dist.get_world_size(),'broadcast_buffers':False,'source_dataset_contract':contract,'model_initial_sha256':initial,'visual_initial_sha256':visual_initial,'final_model_sha256':final,'frozen_text_encoder_unchanged':True,'training_text_dropout_active':m.module.text_encoder.training,'BN_training':all(z.training for z in m.module.modules() if isinstance(z,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d))),'optimizer_group_sizes':[sum(p.numel() for p in z['params']) for z in opt.param_groups],'timed_mean_seconds':float(np.mean([x['wall_including_loader_seconds'] for x in rows[4:]])),'timed_gpu_mean_seconds':float(np.mean([x['forward_loss_backward_update_seconds'] for x in rows[4:]])),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'source100epoch15300step_extrapolation_seconds':float(np.mean([x['wall_including_loader_seconds'] for x in rows[4:]]))*15300,'scope':'12 engineering steps only; fresh formal training must restart identical initialization/RNG, not reuse these weights. No Drone/Quad data or joint checkpoint.','events':events,'rows':rows};(a.output/'TIMING.json').write_text(json.dumps(result,indent=2)+'\n');event('TIMING_COMPLETE')
 except BaseException as exc:
  (a.output/'FAILED.json').write_text(json.dumps({'state':'engineering_failed','exception_type':type(exc).__name__,'message':str(exc),'events':events,'weights_not_scientific_checkpoint':True},indent=2)+'\n');raise
 finally:
  if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
