"""Finite dense/sparse local motion probe; frozen spatial and global time grid."""
from pathlib import Path
import sys,os,time,gc,fcntl,signal
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_temporal_registered_motion128_v1 import D as OLD,EARLY,PRE,FIN,DEV,STAGE
D=R/'artifacts/stvg_temporal_local_motion_fixedbatch128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_local_motion_fixedbatch128';R30=R/'artifacts/stvg_temporal_local_motion128_v1';AUD=R/'artifacts/stvg_temporal_clip_independence_v1'
ARMS=['Native','HiResAuxMix','DenseLocal','SparseLocal','DenseLocalMix','SparseLocalMix']

def prepare():
 from vg_tta.stvg_temporal_local_motion128_recovery001 import properties
 old=read(R30/'PLAN_LOCK_RECOVERY001.json');audit=read(AUD/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in audit['pins'].items());assert all(audit['pins'][f]==h for f,h in old['pins'].items());assert read(AUD/'ENGINEERING_RESULTS.json')['independence_pass'];records=old['records']
 own=list(audit['pins'])+['artifacts/stvg_temporal_clip_independence_v1/PLAN_LOCK.json','artifacts/stvg_temporal_clip_independence_v1/ENGINEERING_RESULTS.json','artifacts/stvg_temporal_clip_independence_v1/COST.json','protocols/stvg_temporal_local_motion_fixedbatch128_v1.md','scripts/run_stvg_temporal_local_motion_fixedbatch128_v1.py','scripts/score_stvg_temporal_local_motion_fixedbatch128_v1.py','scripts/publish_stvg_temporal_local_motion_fixedbatch128_v1.py','results/stvg_error_state/2026-10-20/temporal_clip_independence/RESULTS_PUBLIC.json','results/stvg_error_state/2026-10-20/temporal_registered_motion128/RESULTS_PUBLIC.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,primary='DenseLocalMix',pure_teacher='DenseLocal',source_checkpoint=old['source_checkpoint'],source_checkpoint_sha256=old['source_checkpoint_sha256'],clip_frames=8,center_index=3,dense_physical_stride=2,batch_clips=8,tail_padding='repeat last real clip to B8; retain only real output centers',expected_calls=dict(motion=2320,projection=2320,fusion=768,decoder=1536,head=768),expected_RGB_slots=148130,base_RGB_slots=147456,GPU_seconds_cap=900,CPU_seconds_cap=900,prior_engineering_GPU_seconds=read(R30/'COST_RECOVERY001.json')['GPU_seconds']+read(AUD/'COST.json')['GPU_seconds'],created_epoch=time.time(),deadline_epoch=time.time()+14400,pins={f:sha(R/f)for f in sorted(set(own))},spatial_frozen=True,not_OPD=True,new_experts=0,new_training=0,new_student_updates=0,original_R30_qualification_passed=False)
 assert len(records)==len({r['parent']for r in records})==128;put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked fixed B8 local motion128',flush=True)

def measure():
 import torch,numpy as np
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1._source_forward import transient_native_masks
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_temporal_local_motion128_recovery001 import clip_indices,decode_all,normalize
 from vg_tta.stvg_temporal_registered_motion128_v1 import fuse_motion,decode_prefix,tensor_sha
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup()
 def reject(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('local motion forbids GT')
 sys.addaudithook(reject);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();hooks=[];files=[];partial=None;success=False;times=[];controls_seconds=0.;counts={k+'_'+s:0 for k in ['motion','projection','fusion','decoder','head']for s in ['attempts','completed']};limits=p['expected_calls'];RGB_slots=0;qualification_files=[]
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>5*2**30 and time.time()<p['deadline_epoch']and time.perf_counter()-beg+40<p['GPU_seconds_cap']
 def enter(k):
  def f(m,a):counts[k+'_attempts']+=1;assert counts[k+'_attempts']<=limits[k]
  return f
 def leave(k):
  def f(m,a,o):
   counts[k+'_completed']+=1
   if k=='head'and partial is not None and partial.get('inflight')is not None:partial['inflight'].setdefault('returned_head_logits',[]).append(o.detach().cpu())
  return f
 try:
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>4*2**30;fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);check();m=load_source();mh=state_hash(m.state_dict());flags={n:v.training for n,v in m.named_modules()};assert not any(flags.values());loading=time.perf_counter()-beg
  for module,k in [(m.vid,'motion'),(m.input_proj2,'projection'),(m.ground_encoder.encoder,'fusion'),(m.ground_decoder,'decoder'),(m.temp_embed,'head')]:hooks.extend([module.register_forward_pre_hook(enter(k)),module.register_forward_hook(leave(k))])
  def encode(x,num_frames):
   nonlocal RGB_slots
   partial['inflight']=dict(stage='Swin_entered',input_sha256=tensor_sha(x),num_frames=num_frames);check();RGB_slots+=len(x);assert RGB_slots<=p['expected_RGB_slots']
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16),transient_native_masks(m):f=m.vid(x.to('cuda'),num_frames)
   rr=f['3'].detach().cpu();partial['inflight'].update(stage='Swin_returned',raw=rr);check()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):v=m.input_proj2(f['3'])
   v=v.detach().cpu();partial['inflight'].update(stage='projection_returned',projection=v);del f;check();return dict(raw=rr,projected=v,input_sha256=tensor_sha(x),num_frames=num_frames)
  def restore_order(v):
   return v.reshape(8,8,*v.shape[1:])[torch.arange(7,-1,-1)].flatten(0,1)
  for r in p['records']:
   check();tick=time.perf_counter();ctrl=0.;a=r['alias'];is_qual=len(files)<2
   for path,h in [(EARLY/'early_inputs'/f'{a}.pt',r['early_sha256']),(OLD/'predictions'/f'{a}.pt',r['registered_sha256']),(R/r['subject_path'],r['subject_sha256'])]:assert sha(path)==h
   e=load(EARLY/'early_inputs'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');meta=raw[r['query_ordinal']]['input'];ids=meta['frame_ids'];frames=decode_all(meta,R);indices=clip_indices(ids,len(frames));x=normalize(frames,meta);partial=dict(alias=a,indices=indices,features={},batches={},metadata={},qualification_controls={},GT_read=False);features={};batches={};qualification=dict(original_R30_qualification_passed=False);controls=partial['qualification_controls']
   if is_qual:
    ct=time.perf_counter()
    for off in (0,1):
     vv=encode(x[ids[off::2]],len(ids[off::2]));controls['Native'+str(off)]=vv;hh,ww=e['original_views'][off]['info']['fea_map_size'];want=e['captured_encoder_inputs'][off]['src'][-hh*ww:].permute(1,2,0).reshape(vv['projected'].shape);assert torch.equal(vv['projected'],want)
    ctrl+=time.perf_counter()-ct
   for arm in ['DenseLocal','SparseLocal']:
    vals=[];br=[];partial['features'][arm]=vals;partial['batches'][arm]=br
    for start in range(0,65,8):
     check();real=indices[arm][start:start+8];count=len(real);packed=np.concatenate([real,np.repeat(real[-1:],8-count,axis=0)],axis=0);assert packed.shape==(8,8);pixels=frames[packed].copy();xx=x[packed].reshape(-1,3,*x.shape[-2:]);ret=encode(xx,8);out=ret['projected'];centers=out[3::8][:count].clone();vals.append(centers);br.append(dict(start=start,count=count,padded_slots=8-count,packed_clip_indices=packed,RGB_sha256=sha_bytes(pixels),input_sha256=tensor_sha(xx),projected_center_sha256=tensor_sha(centers)))
     if is_qual and start==0:
      ct=time.perf_counter();qc={};controls[arm]=qc;qc['Batch']=ret
      rep=encode(xx,8);qc['Repeat']=rep
      rev=encode(xx.reshape(8,8,*xx.shape[1:])[torch.arange(7,-1,-1)].flatten(0,1),8);qc['Reverse']=rev
      single=encode(xx[:8],8);qc['Single']=single
      # Retain controls to disk BEFORE any new exact numerical assertion.
      save(D/'qualification_stages'/f'{a}_{arm}_first.pt',qc)
      for k in ['raw','projected']:assert torch.equal(rep[k],ret[k])and torch.equal(restore_order(rev[k]),ret[k])
      relative=float(((single['projected'].float()-out[:8].float()).abs()/single['projected'].float().abs().clamp_min(1)).max());qualification[arm]=dict(same_shape_repeat_exact=True,full_clip_permutation_equivariant_exact=True,single_fullclip_max_relative_error=relative,original_bound=8*torch.finfo(torch.float16).eps,comparison_with_original_bound_passed=relative<8*torch.finfo(torch.float16).eps);ctrl+=time.perf_counter()-ct
     if count<8:
      if is_qual:controls[arm]['Tail']=ret;save(D/'qualification_stages'/f'{a}_{arm}_tail.pt',ret)
      for k in ['raw','projected']:
       clips=ret[k].reshape(8,8,*ret[k].shape[1:]);assert all(torch.equal(clips[0],clips[j])for j in range(1,8))
     partial['inflight']=None;del out,xx,pixels,ret
    features[arm]=torch.cat(vals);batches[arm]=br;assert features[arm].shape[0]==65 and torch.isfinite(features[arm]).all()
   allviews={k:[]for k in ['Native','DenseLocal','SparseLocal']};metadata={};newlogits={};partial['logits']=newlogits
   for off in (0,1):
    cap=e['captured_encoder_inputs'][off];ov=e['original_views'][off];hh,ww=ov['info']['fea_map_size'];src=cap['src'];native=src[-hh*ww:].permute(1,2,0).reshape(len(ids[off::2]),256,hh,ww);md={}
    for arm,motion in [('Native',native),('DenseLocal',features['DenseLocal'][off::2]),('SparseLocal',features['SparseLocal'][off::2])]:
     check();partial['inflight']=dict(stage='fusion_entered',offset=off,arm=arm);v,md[arm]=fuse_motion(m,cap,ov,motion,partial['inflight']);allviews[arm].append(v);partial['views']=allviews;partial['inflight']=None
     if arm=='Native':assert torch.equal(v['prefix'],ov['prefix'])
    metadata[str(off)]=md
   for arm,v in allviews.items():
    check();partial['inflight']=dict(stage='decode_entered',arm=arm);v=decode_prefix(m,v,65);partial['inflight'].update(stage='decode_returned',values=v);newlogits[arm]=v['logits'];check();partial['inflight']=None
    if arm=='Native':assert all(torch.equal(xx,yy)for xx,yy in zip(v['logits'],old['logits']['Native']))
   qs={k:old['outputs'][k]['q']for k in ['Native','HiResAuxMix']}
   for arm,ll in newlogits.items():
    grid,q=mass_table(probabilities(old['records'],ll));assert grid==ids
    if arm=='Native':assert np.array_equal(q,qs[arm])
    qs[arm]=q
   for arm in ['DenseLocal','SparseLocal']:qs[arm+'Mix']=.5*qs['Native']+.5*qs[arm]
   outputs={k:dict(q=q,interval=final_mbr(ids,q)['interval'])for k,q in qs.items()};assert set(outputs)==set(ARMS)
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,records=old['records'],outputs=outputs,trajectories=old['trajectories'],motion_features=features,clip_indices=indices,batches=batches,metadata=metadata,logits=newlogits,total_frames=len(frames),GT_read=False,spatial_updated=False,new_student_updates=0)))
   if is_qual:qualification_files.append(save(D/'qualification'/f'{a}.pt',dict(controls=controls,diagnostics=qualification)))
   times.append(time.perf_counter()-tick-ctrl);controls_seconds+=ctrl;put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time(),actual_calls=counts),True);print('local_motion',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+controls_seconds+128*max(times)+60;put(D/'PILOT2.json',dict(case_seconds=times,loading=loading,controls_seconds=controls_seconds,estimated_GPU_seconds=estimate,qualification=[load(R/f['path'])['diagnostics']for f in qualification_files],GT_read=False));assert estimate<p['GPU_seconds_cap']
   partial=None;del frames,x,e,old,features,batches,allviews,newlogits,metadata,qs,outputs,src,native,v;gc.collect();torch.cuda.empty_cache()
  check();assert all(sha(R/f)==h for f,h in p['pins'].items())and state_hash(m.state_dict())==mh and {n:v.training for n,v in m.named_modules()}==flags and all(not v.requires_grad and v.grad is None for v in m.parameters());assert all(counts[k+'_attempts']==counts[k+'_completed']==v for k,v in limits.items())and RGB_slots==p['expected_RGB_slots'];check();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_and_flags_unchanged=True,qualification_files=qualification_files,RGB_slots=RGB_slots));success=True
 finally:
  for h in hooks:h.remove()
  if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",dict(partial,actual_calls=counts))
  elapsed=time.perf_counter()-beg;put(D/'COST.json',dict(GPU_seconds=elapsed+3,prior_engineering_GPU_seconds=p['prior_engineering_GPU_seconds'],RGB_slots=RGB_slots,measured_lease_seconds=elapsed,cleanup_reserve_seconds=3,actual_calls=counts,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()if torch.cuda.is_initialized()else 0,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()if torch.cuda.is_initialized()else 0,new_DINO=0,new_ReferDINO=0,new_network_training=0,new_student_updates=0,spatial_updates=0,success=success));lock.close()

def sha_bytes(x):
 import hashlib
 return hashlib.sha256(x.tobytes()).hexdigest()

if __name__=='__main__':
 def terminated(signum,frame):raise TimeoutError('owned RuntimeMax SIGTERM')
 signal.signal(signal.SIGTERM,terminated)
 D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=bool(list((D/'partials').glob('*.pt'))or list((D/'predictions').glob('*.pt')))));raise
 finally:lock.close()
