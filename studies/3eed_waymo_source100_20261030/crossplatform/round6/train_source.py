"""Genuine Waymo-only100epoch training. Parent-registered release required.
No jointtask checkpoint initialization, no target datasets, no TTA, no VAL selection
of the primary final100epoch model. Resume is exact at completed epoch boundaries.
"""
from pathlib import Path
import argparse,json,time,datetime,math,shutil
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from factory import R,P,U,ROBE,sha,configure,dataset,order_contract,loader,model,optimizer,scheduler,criterion,to_cuda,state_fingerprint,compute_hungarian_loss
from checkpointing import save_model,save_resume,load_resume
from source_validation import predict_source_val

def append(path,row):
 with open(path,'a') as f:f.write(json.dumps(row)+'\n');f.flush()
def recover_completed_epoch100(output,core,plan_sha):
 """Finalize a completed epoch100 resume without rerunning train or validation."""
 ledger=Path(output)/'MILESTONES.jsonl';assert ledger.is_file();current=state_fingerprint(core);matches=[]
 for line in ledger.read_text().splitlines():
  entry=json.loads(line)
  if entry.get('epoch')!=100:continue
  checkpoint=Path(entry['checkpoint_path']);vdir=Path(entry['val_prediction_dir']);assert checkpoint.resolve().is_relative_to(Path(output).resolve()) and vdir.resolve().is_relative_to(Path(output).resolve())
  assert checkpoint.is_file() and (vdir/'sealed.json').is_file();assert sha(checkpoint)==entry['checkpoint_sha256'];assert sha(vdir/'sealed.json')==entry['val_prediction_seal_sha256'];seal=json.loads((vdir/'sealed.json').read_text());assert seal['epoch']==100 and seal['n']==3782 and seal['checkpoint_sha256']==entry['checkpoint_sha256']
  if seal['model_state_fingerprint']!=current:continue
  payload=torch.load(checkpoint,map_location='cpu',weights_only=True);assert payload['epoch']==100 and payload['trained_platforms']==['waymo'] and payload['source_provenance']['plan_sha256']==plan_sha;del payload;matches.append(entry)
 assert matches,'No existing epoch100 checkpoint+VALseal matches the actual restored model; do not retrain or select a different model.'
 return matches[-1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',type=Path);a=ap.parse_args();assert sha(a.plan)==a.plan_sha256;plan=json.loads(a.plan.read_text());assert plan['mode']=='waymo_source_train100' and plan['GPU_execution_authorized'] is True;assert plan['batch_size']==24 and plan['max_epochs']==100 and plan['precision']=='FP32' and plan['source_validation_every']==10 and plan['save_model_every']==10
 assert sha(P.parent/'research/round6/AUTHORIZATION.json')==plan['authorization_sha256']
 for f,h in plan['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in plan['official_source_sha256'].items():assert sha(U/f)==h,f
 for f,h in plan['roberta_sha256'].items():assert sha(ROBE/f)==h,f
 if a.resume:
  assert a.output.exists() and a.resume.resolve().is_relative_to(a.output.resolve());assert sha(a.output/'registered_plan.json')==a.plan_sha256
 else:a.output.mkdir(parents=True,exist_ok=False);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
 attempts=a.output/'attempts';attempts.mkdir(exist_ok=True);attempt=1
 while (attempts/f'{attempt:03d}').exists():attempt+=1
 logdir=attempts/f'{attempt:03d}';logdir.mkdir();configure(plan);started=time.perf_counter();latest=a.output/'resume_latest.pth';last_completed=0;global_step=0;milestones=[];epoch_times=[];val_times=[]
 def event(name,**kw):
  row={'event':name,'attempt':attempt,'elapsed_seconds':time.perf_counter()-started,**kw};append(logdir/'events.jsonl',row);print(json.dumps(row),flush=True)
 try:
  torch.cuda.set_device(0);dist.init_process_group('nccl',init_method='file://'+str((logdir/'nccl_rendezvous').resolve()),rank=0,world_size=1,timeout=datetime.timedelta(seconds=300));event('NCCL_world1_ready');train_ds=dataset('train');val_ds=dataset('val');train_contract=order_contract(train_ds,'train');val_contract=order_contract(val_ds,'val');assert train_contract==plan['source_train_contract'];assert val_contract==plan['source_val_contract'];g=torch.Generator().manual_seed(0);train_dl=loader(train_ds,'train',g,plan);val_dl=loader(val_ds,'val',g,plan);assert len(train_dl)==153 and len(val_dl)==158;event('sourceonly_loaders_ready',train=train_contract,val=val_contract,shared_loader_generator=True);core=model();initial=state_fingerprint(core);assert initial==plan['expected_model_initial_sha256'];assert state_fingerprint(core,'text_encoder.')==plan['expected_text_encoder_initial_sha256'];core.cuda().train();opt=optimizer(core);sched=scheduler(opt,153);crit=criterion().cuda();m=DDP(core,device_ids=[0],broadcast_buffers=False);assert len(opt.param_groups)==3 and len(opt.param_groups[2]['params'])==0;event('model_train_ready',initial_sha256=initial,trainable=sum(p.numel() for p in core.parameters() if p.requires_grad),text_dropout_train=core.text_encoder.training,optimizer_groups=[len(x['params']) for x in opt.param_groups]);provenance={'plan_sha256':a.plan_sha256,'authorization_sha256':plan['authorization_sha256'],'initial_model_sha256':initial,'initial_text_sha256':plan['expected_text_encoder_initial_sha256'],'source_train_contract':train_contract,'source_val_contract':val_contract,'seed':plan['seed'],'trained_platforms':['waymo'],'joint_task_checkpoint_initialization':False,'primary_selection':'fixed epoch100; sourceVAL metrics never alter loop or primary epoch'}
  start_epoch=1
  if a.resume:start_epoch,global_step=load_resume(a.resume,core,opt,sched,g,a.plan_sha256);last_completed=start_epoch-1;event('source_only_resume_restored',resume_path=str(a.resume),sha256=sha(a.resume),next_epoch=start_epoch,global_step=global_step)
  (logdir/'INITIALIZATION.json').write_text(json.dumps({**provenance,'FP32':True,'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'DDP_broadcast_buffers':False,'global_seed_explicit_choice_not_effective_original_flag':True,'loader_context':'spawn4 for NCCL fork safety, no persistent workers','checkpoint_resume':'epochboundary afterscheduledsourceVAL, optimizer/scheduler/global RNG/sharedgenerator allrestored'},indent=2)+'\n')
  for epoch in range(start_epoch,101):
   m.train();assert core.text_encoder.training and all(not p.requires_grad for p in core.text_encoder.parameters());assert all(z.training and z.momentum==.1 for z in core.modules() if isinstance(z,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)));train_dl.sampler.set_epoch(epoch);order=list(iter(train_dl.sampler));order_sha=__import__('hashlib').sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest();epoch_start=time.perf_counter();loss_values=[];torch.cuda.reset_peak_memory_stats();event('epoch_start',epoch=epoch,global_step=global_step,sampler_order_sha256=order_sha,first_batch_anno_indices=order[:24],lr=[z['lr'] for z in opt.param_groups])
   for batch_index,batch in enumerate(train_dl):
    tick=time.perf_counter();batch=to_cuda(batch);out=m({'point_clouds':batch['point_clouds'].float(),'text':batch['utterances']})
    for k,v in batch.items():assert k not in out;out[k]=v
    loss,out=compute_hungarian_loss(out,6,crit,query_points_obj_topk=4)
    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss epoch{epoch} batch{batch_index}')
    opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),.1)
    if not torch.isfinite(grad):raise FloatingPointError(f'Nonfinite gradient epoch{epoch} batch{batch_index}')
    assert all(p.grad is None for p in core.text_encoder.parameters());opt.step();sched.step();global_step+=1;torch.cuda.synchronize();lv=float(loss.detach());loss_values.append(lv);row={'epoch':epoch,'batch':batch_index,'global_step':global_step,'loss':lv,'grad_norm_before_clip':float(grad),'lr':[z['lr'] for z in opt.param_groups],'forward_backward_update_seconds':time.perf_counter()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated()};append(logdir/'steps.jsonl',row)
    if batch_index in [0,49,99,152]:
     elapsed=time.perf_counter()-epoch_start;rate=elapsed/(batch_index+1);event('train_progress',**row,elapsed_epoch_seconds=elapsed,estimated_remaining_train_seconds=rate*((100-epoch)*153+153-batch_index-1),estimate_excludes_future_validation_checkpoint_startups=True)
   assert len(loss_values)==153 and global_step==153*epoch;seconds=time.perf_counter()-epoch_start;epoch_times.append(seconds);event('epoch_training_complete',epoch=epoch,global_step=global_step,mean_loss=float(np.mean(loss_values)),train_seconds=seconds,lr=[z['lr'] for z in opt.param_groups])
   if epoch%10==0:
    ckpt=a.output/'checkpoints'/f'epoch{epoch:03d}_model_attempt{attempt:03d}.pth';modelsha=save_model(core,ckpt,epoch,provenance);event('source_checkpoint_sealed_before_val',epoch=epoch,checkpoint_path=str(ckpt),checkpoint_sha256=modelsha);vstart=time.perf_counter();valout=a.output/'source_val'/f'epoch{epoch:03d}_attempt{attempt:03d}';vseal=predict_source_val(m,val_dl,valout,epoch,ckpt,val_contract);val_times.append(time.perf_counter()-vstart);entry={'epoch':epoch,'checkpoint_path':str(ckpt),'checkpoint_sha256':modelsha,'val_prediction_dir':str(valout),'val_prediction_seal_sha256':sha(valout/'sealed.json'),'source_val_seconds':val_times[-1]};milestones.append(entry);append(a.output/'MILESTONES.jsonl',entry);event('source_validation_sealed',**entry)
   resume_sha=save_resume(core,opt,sched,g,latest,epoch,global_step,provenance);last_completed=epoch;event('resume_latest_atomic_saved',epoch=epoch,path=str(latest),sha256=resume_sha,resume_at_next_epoch=epoch+1);(a.output/'PROGRESS.json').write_text(json.dumps({'state':'running' if epoch<100 else 'finalizing','last_completed_epoch':epoch,'global_step':global_step,'resume_latest_sha256':resume_sha,'attempt':attempt,'loss_mean':float(np.mean(loss_values))},indent=2)+'\n')
  assert last_completed==100 and global_step==15300;assert state_fingerprint(core,'text_encoder.')==plan['expected_text_encoder_initial_sha256'];final=next((x for x in reversed(milestones) if x['epoch']==100),None)
  if final is None:final=recover_completed_epoch100(a.output,core,a.plan_sha256)
  final_resume=a.output/'checkpoints'/f'epoch100_resume_attempt{attempt:03d}.pth';assert not final_resume.exists();shutil.copyfile(a.resume if start_epoch==101 else latest,final_resume);complete={'state':'training_completed_source_only_epoch100','primary_checkpoint':final['checkpoint_path'],'primary_checkpoint_sha256':final['checkpoint_sha256'],'final_resume_checkpoint':str(final_resume),'final_resume_checkpoint_sha256':sha(final_resume),'primary_epoch':100,'completed_optimizer_steps':15300,'plan_sha256':a.plan_sha256,'source_train_contract':train_contract,'source_val_contract':val_contract,'no_joint_checkpoint_initialization':True,'frozen_generic_text_weights_unchanged':True,'no_Drone_or_Quad_data_access':True,'sourceVAL_secondary_only_not_primary_selection':True,'source_val_epoch100_seal_sha256':final['val_prediction_seal_sha256'],'elapsed_attempt_seconds':time.perf_counter()-started,'observed_train_epoch_seconds':epoch_times,'observed_val_seconds':val_times,'resume_latest_is_mutable_rolling_not_immutable_evidence':True,'finalized_after_completed_epoch100_resume_without_training_or_validation':start_epoch==101};(a.output/'TRAINING_COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n');event('TRAINING_COMPLETE',**complete)
 except BaseException as exc:
  (logdir/'INTERRUPTED_OR_FAILED.json').write_text(json.dumps({'state':'interrupted_or_failed','exception_type':type(exc).__name__,'message':str(exc),'last_completed_epoch':last_completed,'global_step_at_failure':global_step,'resume_available':latest.exists(),'resume_scope':'discard incomplete epoch; restore prior completedepoch RNG/optimizer/scheduler','no_checkpoint_or_log_deleted':True},indent=2)+'\n');raise
 finally:
  if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
