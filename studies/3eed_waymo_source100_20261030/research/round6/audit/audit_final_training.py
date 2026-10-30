"""Independent final CPU receipt. Runs only after true epoch100 completion exists.
Reads source-training checkpoints/logs only; never imports model/runner or target data.
"""
import hashlib,json,math,os,time,traceback
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[3];A=R/'research/round6/audit';D=R/'crossplatform/round6';RUN=D/'waymo_source100';EXPECTED_PLAN='66fabf36898f1bc92124b680435e598e7bca0809e48aeb2da35e272331afe7ce'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def fp(state,prefix=None):
 h=hashlib.sha256()
 for n,v in state.items():
  if prefix is not None and not n.startswith(prefix):continue
  x=v.detach().cpu().contiguous();h.update(n.encode()+b'\0');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
 return h.hexdigest()
def expected_lr(step):
 factor=.1**(int(step>=3978)+int(step>=4131));return [base*factor for base in (1e-4,1e-3,1e-5)]
def lr_equal(actual,expected):return len(actual)==len(expected) and all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-16) for x,y in zip(actual,expected))
def within(p):assert Path(p).resolve().is_relative_to(RUN.resolve());return Path(p)
def main():
 started=time.monotonic();completefile=RUN/'TRAINING_COMPLETE.json';assert completefile.is_file(),'Not complete; do not run final audit early';done=read(completefile);assert done['state']=='training_completed_source_only_epoch100' and done['primary_epoch']==100 and done['completed_optimizer_steps']==15300
 planpath=R/'research/round6/FORMAL_PLAN_LOCK.json';plan=read(planpath);assert sha(planpath)==EXPECTED_PLAN==sha(RUN/'registered_plan.json')==done['plan_sha256'];assert plan['GPU_execution_authorized'] is True and plan['max_epochs']==100 and plan['batch_size']==24 and plan['precision']=='FP32';assert plan['primary_selection']=='fixed epoch100;sourcevalidation doesnotchooseprimary'
 code={n:sha(D/n) for n in plan['code_sha256']};assert code==plan['code_sha256'];official={n:sha(D.parent/'round3/p0/upstream'/n) for n in plan['official_source_sha256']};assert official==plan['official_source_sha256'];cancel=sha(D.parent/'round3/p0/CANCELLED.json');assert cancel==plan['old_P0_cancellation_preserved_sha256'];assert done['source_train_contract']==plan['source_train_contract'] and done['source_val_contract']==plan['source_val_contract']
 # Reconstruct the completed trajectory across explicit epoch-boundary resume attempts.
 canonical={};all_events={};all_steps={};initializations=[];attempt_log_hashes={}
 for adir in sorted((RUN/'attempts').iterdir()):
  if not adir.is_dir():continue
  es=lines(adir/'events.jsonl');ss=lines(adir/'steps.jsonl') if (adir/'steps.jsonl').exists() else [];all_events[adir.name]=es;all_steps[adir.name]=ss
  init=read(adir/'INITIALIZATION.json');assert init['plan_sha256']==EXPECTED_PLAN and init['initial_model_sha256']==plan['expected_model_initial_sha256'] and init['initial_text_sha256']==plan['expected_text_encoder_initial_sha256'];assert init['trained_platforms']==['waymo'] and init['joint_task_checkpoint_initialization'] is False;assert init['source_train_contract']==plan['source_train_contract'] and init['source_val_contract']==plan['source_val_contract'];assert init['FP32'] and init['DDP_broadcast_buffers'] is False;initializations.append({'attempt':adir.name,'sha256':sha(adir/'INITIALIZATION.json')})
  resumed=[e for e in es if e['event']=='source_only_resume_restored'];assert len(resumed)<=1;start=resumed[0]['next_epoch'] if resumed else 1
  for epoch in list(canonical):
   if epoch>=start:del canonical[epoch]
  for e in es:
   if e['event']=='resume_latest_atomic_saved':
    ep=e['epoch'];completed=[z for z in es if z['event']=='epoch_training_complete' and z['epoch']==ep];assert len(completed)==1 and completed[0]['global_step']==153*ep;canonical[ep]={'attempt':adir.name,'complete':completed[0],'resume_sha256':e['sha256']}
  attempt_log_hashes[adir.name]={p.name:sha(p) for p in [adir/'events.jsonl',adir/'steps.jsonl'] if p.exists()}
 assert sorted(canonical)==list(range(1,101));trajectory=[];epoch_rows=[]
 for epoch in range(1,101):
  meta=canonical[epoch];ss=[z for z in all_steps[meta['attempt']] if z['epoch']==epoch];assert len(ss)==153 and [z['batch'] for z in ss]==list(range(153));assert [z['global_step'] for z in ss]==list(range((epoch-1)*153+1,epoch*153+1));assert lr_equal(meta['complete']['lr'],expected_lr(epoch*153))
  for z in ss:
   assert all(math.isfinite(z[k]) for k in ['loss','grad_norm_before_clip','forward_backward_update_seconds']);assert z['grad_norm_before_clip']>0 and z['forward_backward_update_seconds']>0;assert lr_equal(z['lr'],expected_lr(z['global_step']));trajectory.append(z)
  epoch_rows.append({'epoch':epoch,'attempt':meta['attempt'],'mean_loss':meta['complete']['mean_loss'],'train_seconds':meta['complete']['train_seconds'],'global_step':epoch*153,'lr':meta['complete']['lr']})
 assert len(trajectory)==15300
 milestones=lines(RUN/'MILESTONES.jsonl');validated=[]
 for epoch in range(10,101,10):
  attempt=canonical[epoch]['attempt'];entries=[e for e in milestones if e['epoch']==epoch and ('attempt'+attempt) in Path(e['checkpoint_path']).name];assert len(entries)==1;entry=entries[0];ckp=within(entry['checkpoint_path']);vd=within(entry['val_prediction_dir']);assert sha(ckp)==entry['checkpoint_sha256'];sp=vd/'sealed.json';assert sha(sp)==entry['val_prediction_seal_sha256'];seal=read(sp);assert seal['epoch']==epoch and seal['n']==3782 and seal['checkpoint_sha256']==entry['checkpoint_sha256'];assert seal['val_order_sha256']==plan['source_val_contract']['order_sha256'];assert seal['source_only'] and seal['all_inference_no_grad'] and seal['parameters_buffers_unchanged'] and seal['dynamic_padding_preserved'];assert seal['actual_batch_size']==24 and seal['last_batch']==14;assert seal['no_validation_metric_read_for_primary_selection'] is True
  files=seal['files'];assert len(files)==3782 and set(files)=={p.name for p in vd.glob('*.npz')};assert all(sha(vd/n)==h for n,h in files.items());trace=read(vd/'trace.json');assert len(trace)==158 and trace[-1]['completed']==3782 and trace[-1]['B']==14
  # Event order proves checkpoint-before-VAL-before-full-RNG-resume on the winning attempt.
  es=all_events[attempt];positions={name:next(i for i,e in enumerate(es) if e['event']==name and e.get('epoch')==epoch) for name in ['source_checkpoint_sealed_before_val','source_validation_sealed','resume_latest_atomic_saved']};assert list(positions.values())==sorted(positions.values())
  validated.append({'epoch':epoch,'checkpoint_sha256':entry['checkpoint_sha256'],'val_seal_sha256':entry['val_prediction_seal_sha256'],'val_count':3782,'all_prediction_file_hashes_verified':True,'model_state_fingerprint':seal['model_state_fingerprint'],'source_val_seconds':entry['source_val_seconds']})
 primary=within(done['primary_checkpoint']);resume=within(done['final_resume_checkpoint']);assert sha(primary)==done['primary_checkpoint_sha256']==validated[-1]['checkpoint_sha256'];assert sha(resume)==done['final_resume_checkpoint_sha256'];assert done['source_val_epoch100_seal_sha256']==validated[-1]['val_seal_sha256']
 ckm=torch.load(primary,map_location='cpu',weights_only=True);ckr=torch.load(resume,map_location='cpu',weights_only=True);assert ckm['epoch']==ckr['epoch']==100 and ckr['next_epoch']==101 and ckr['global_step']==15300;assert ckm['trained_platforms']==ckr['trained_platforms']==['waymo'];assert ckr['joint_checkpoint_initialization'] is False;assert ckr['resume_scope']=='epoch_boundary_after_scheduled_source_validation';assert ckm['source_provenance']==ckr['source_provenance'];assert ckm['source_provenance']['plan_sha256']==EXPECTED_PLAN and ckm['source_provenance']['joint_task_checkpoint_initialization'] is False
 state=ckm['model'];model_fp=fp(state);resume_fp=fp(ckr['model']);assert model_fp==resume_fp==validated[-1]['model_state_fingerprint'];assert all(torch.isfinite(v).all() for v in state.values() if v.dtype.is_floating_point);textfp=fp(state,'text_encoder.');assert textfp==plan['expected_text_encoder_initial_sha256'];assert model_fp!=plan['expected_model_initial_sha256'];ln_deltas={'text_projector.1.weight':float((state['text_projector.1.weight']-1).abs().max()),'text_projector.1.bias':float(state['text_projector.1.bias'].abs().max())};assert all(v>0 for v in ln_deltas.values())
 opt=ckr['optimizer'];groups=opt['param_groups'];assert len(groups)==3 and len(groups[2]['params'])==0 and lr_equal([g['lr'] for g in groups],expected_lr(15300));assert all(g['weight_decay']==.0005 for g in groups);steps=sorted(set(int(v['step'].item()) for v in opt['state'].values()));assert steps==[15300];assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in opt['state'].values());nonzero=sum(int(torch.count_nonzero(v['exp_avg']))>0 for v in opt['state'].values());assert nonzero>0;assert ckr['scheduler']['last_epoch']==15300 and lr_equal(ckr['scheduler']['_last_lr'],expected_lr(15300));rng=ckr['rng'];assert set(rng)=={'python','numpy','torch_cpu','torch_cuda','loader_shared_generator'} and len(rng['torch_cuda'])==1
 probes={str(k):{'lr_after_update':trajectory[k-1]['lr'],'epoch':trajectory[k-1]['epoch'],'batch':trajectory[k-1]['batch']} for k in [3977,3978,3979,4130,4131,4132,15300]}
 receipt={'state':'passed_actual_final_epoch100_source_checkpoint_audit','auditor_source_sha256':sha(Path(__file__)),'formal_plan_sha256':EXPECTED_PLAN,'training_complete_sha256':sha(completefile),'primary_checkpoint_sha256':sha(primary),'final_resume_sha256':sha(resume),'actual_completed_epochs':100,'actual_optimizer_updates':15300,'canonical_attempts':sorted(set(v['attempt'] for v in canonical.values())),'initialization_records':initializations,'step_log_hashes':attempt_log_hashes,'all_15300_completed_steps_finite_nonzero_grad_and_registered_LR':True,'actual_LR_transition_probes':probes,'scheduler_final_last_epoch':15300,'optimizer_state_step_counts':steps,'optimizer_state_tensors':len(opt['state']),'optimizer_nonzero_first_moments':nonzero,'final_param_groups_lr':[g['lr'] for g in groups],'frozen_text_fingerprint':textfp,'final_model_and_resume_identical_fingerprint':model_fp,'actual_non_BN_parameter_change_from_official_LN_default':ln_deltas,'all_model_tensors_finite':True,'all_10_milestone_checkpoint_and_VAL_prediction_hashes_verified':validated,'all_checkpoint_VAL_resume_events_ordered':True,'full_final_RNG_payload_present':'Python/NumPy/TorchCPU/allCUDA/sharedDataLoaderGenerator; loaded on CPU, no CUDA call','source_train_val_contracts_bound':True,'source_only_provenance_and_fixed_epoch100_no_metric_selection':True,'source_only_claim_boundary':'Bound source-Waymo dataset constructors/order contracts, frozen code, checkpoint provenance and no VAL metric selection in runner. This is not a syscall-level proof that no other process accessed target data.','old_P0_cancel_sha256_unchanged':cancel,'source_code_hashes':code,'official_source_hashes':official,'epoch_training_scalar_records':epoch_rows,'cost_seconds':{'auditor':time.monotonic()-started,'canonical_train_total':sum(e['train_seconds'] for e in epoch_rows),'scheduled_source_val_total':sum(e['source_val_seconds'] for e in validated)},'no_model_GPU_targetlabel_or_newscoring_call_by_auditor':True,'interpretation':'Actual source-only supervised baseline completed. This receipt does not claim successful cross-platform generalization or any TTA benefit.'}
 out=A/'FINAL_TRAINING_INDEPENDENT_AUDIT.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'state':receipt['state'],'receipt_sha256':sha(out),'primary_sha256':receipt['primary_checkpoint_sha256'],'auditor_seconds':receipt['cost_seconds']['auditor']}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  fail=A/('FINAL_AUDIT_FAILURE_'+str(time.time_ns())+'.json');fail.write_text(json.dumps({'state':'failed_not_hidden','type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'auditor_sha256':sha(Path(__file__))},indent=2)+'\n');raise
