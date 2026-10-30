"""One-shot CPU audit of an atomically replaced rolling source-only checkpoint."""
import hashlib,json,os,time
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[3];A=R/'research/round6/audit';D=R/'crossplatform/round6';RUN=D/'waymo_source100';planpath=R/'research/round6/FORMAL_PLAN_LOCK.json'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def fp(state,prefix=None):
 h=hashlib.sha256()
 for n,v in state.items():
  if prefix is not None and not n.startswith(prefix):continue
  x=v.detach().cpu().contiguous();h.update(n.encode()+b'\0');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
 return h.hexdigest()
plan=json.loads(planpath.read_text());psha=sha(planpath);assert psha=='66fabf36898f1bc92124b680435e598e7bca0809e48aeb2da35e272331afe7ce';assert sha(RUN/'registered_plan.json')==psha
codechecks={n:sha(D/n)==h for n,h in plan['code_sha256'].items()};assert all(codechecks.values());assert '15300' in plan['scope'] and 'weightsdiscarded' not in plan['scope']
init=json.loads((RUN/'attempts/001/INITIALIZATION.json').read_text());assert init['initial_model_sha256']==plan['expected_model_initial_sha256'];assert init['trained_platforms']==['waymo'] and init['joint_task_checkpoint_initialization'] is False;assert init['plan_sha256']==psha
snapshot=A/'EARLY_RESUME_SNAPSHOT_PRIVATE.pth';assert not snapshot.exists();h=hashlib.sha256()
# Hold an open descriptor to one inode. Atomic replacement of the pathname cannot change these bytes.
with open(RUN/'resume_latest.pth','rb') as src,open(snapshot,'xb') as dst:
 before=os.fstat(src.fileno())
 for block in iter(lambda:src.read(8<<20),b''):h.update(block);dst.write(block)
 after=os.fstat(src.fileno());dst.flush();os.fsync(dst.fileno())
assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);snapsha=h.hexdigest();assert sha(snapshot)==snapsha
ck=torch.load(snapshot,map_location='cpu',weights_only=True);epoch=ck['epoch'];steps=ck['global_step'];assert steps==153*epoch and ck['next_epoch']==epoch+1;assert ck['trained_platforms']==['waymo'] and ck['joint_checkpoint_initialization'] is False;assert ck['source_provenance']['plan_sha256']==psha;assert ck['resume_scope']=='epoch_boundary_after_scheduled_source_validation';assert ck['source_provenance']['source_train_contract']==plan['source_train_contract'];assert ck['source_provenance']['source_val_contract']==plan['source_val_contract']
state=ck['model'];textfp=fp(state,'text_encoder.');assert textfp==plan['expected_text_encoder_initial_sha256'];fullfp=fp(state);assert fullfp!=plan['expected_model_initial_sha256'];visualfp=fp(state,'backbone_net.');pre=json.loads((D/'CPU_PREFLIGHT.json').read_text());assert visualfp!=pre['CPU_visual_initial_sha256']
# LayerNorm affine parameters start at ones/zeros by the untouched official nn.LayerNorm constructor.
ln={'text_projector.1.weight':float((state['text_projector.1.weight']-1).abs().max()),'text_projector.1.bias':float(state['text_projector.1.bias'].abs().max())};assert all(v>0 for v in ln.values())
opt=ck['optimizer'];groups=opt['param_groups'];assert len(groups)==3 and len(groups[2]['params'])==0;assert [g['lr'] for g in groups]==[.0001,.001,.00001];assert all(g['weight_decay']==.0005 for g in groups);osteps=sorted(set(int(v['step'].item()) for v in opt['state'].values()));assert osteps==[steps];assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in opt['state'].values());nonzero_moments=sum(int(torch.count_nonzero(v['exp_avg']))>0 for v in opt['state'].values());assert nonzero_moments>0
sched=ck['scheduler'];assert sched['last_epoch']==steps;assert sched['_last_lr']==[.0001,.001,.00001]
rng=ck['rng'];assert set(rng)=={'python','numpy','torch_cpu','torch_cuda','loader_shared_generator'};assert len(rng['torch_cuda'])==1;assert all(isinstance(rng[k],torch.Tensor) for k in ['torch_cpu','loader_shared_generator'])
events=[json.loads(x) for x in (RUN/'attempts/001/events.jsonl').read_text().splitlines()];save=[e for e in events if e['event']=='resume_latest_atomic_saved' and e['sha256']==snapsha];assert len(save)==1 and save[0]['epoch']==epoch
first=[e for e in events if e['event']=='epoch_training_complete' and e['epoch']==1];assert len(first)==1 and first[0]['global_step']==153 and first[0]['lr']==[.0001,.001,.00001]
step_rows=[json.loads(x) for x in (RUN/'attempts/001/steps.jsonl').read_text().splitlines() if x.strip()];firststeps=[x for x in step_rows if x['epoch']==1];assert len(firststeps)==153 and [x['global_step'] for x in firststeps]==list(range(1,154));assert all(x['lr']==[.0001,.001,.00001] for x in firststeps);assert all(x['grad_norm_before_clip']>0 for x in firststeps)
r={'state':'pass_actual_early_run_source_only_provenance','formal_plan_sha256':psha,'initialization_sha256':sha(RUN/'attempts/001/INITIALIZATION.json'),'source_code_matches_registered_plan':codechecks,'snapshot':{'private_file':snapshot.name,'sha256':snapsha,'bytes':before.st_size,'inode_stable_during_copy':True,'epoch':epoch,'global_step':steps,'matches_atomic_save_event':True},'cold_initial_model_sha256':init['initial_model_sha256'],'text_fingerprint_matches_original_frozen_generic_text':textfp,'full_state_changed':fullfp,'visual_state_changed':visualfp,'actual_parameter_only_change_proof':{'definition':'Official text_projector LayerNorm affine weight default ones/bias default zeros; actual trained tensors differ. These are parameters, not BN buffers.','max_abs_delta_from_known_default':ln},'optimizer':{'groups':3,'frozen_text_group_empty':True,'lr':[g['lr'] for g in groups],'weight_decay':[g['weight_decay'] for g in groups],'all_state_step_counts':osteps,'state_tensors':len(opt['state']),'states_with_nonzero_first_moment':nonzero_moments,'moments_finite':True},'scheduler_last_epoch_equals_optimizer_updates':sched['last_epoch'],'sourceonly_and_sameplan_checkpoint_provenance':True,'full_rng_payload_present':'Python,NumPy,TorchCPU,oneCUDAdevice,sharedDataLoaderGenerator; CUDA tensors inspected on CPU, no CUDA init','epoch1':{'steps':153,'mean_loss':first[0]['mean_loss'],'train_seconds':first[0]['train_seconds'],'constant_registered_lrs':True,'nonzero_finite_grad_norms':True,'observed_peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in firststeps)},'checkpoint_after_scheduled_sourceVAL':'This early epoch is before first scheduled sourceVAL; code order reviewed. Actual sourceVAL state/order audit deferred to immutable milestone/final receipts.','source_train_val_contracts_bound':True,'formal_plan_stale_timing_scope_corrected':True,'no_new_task_model_GPU_or_targetlabel_access_by_auditor':True,'boundary':'This proves actual early updates and provenance. It does not assert final100epoch completion or target generalization.'}
(A/'EARLY_FORMAL_RUN_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'receipt_sha256':sha(A/'EARLY_FORMAL_RUN_REVIEW.json'),'epoch':epoch,'steps':steps,'snapshot_sha256':snapsha,'text_unchanged':True,'parameter_only_LN_deltas':ln,'state_tensors':len(opt['state'])}))
