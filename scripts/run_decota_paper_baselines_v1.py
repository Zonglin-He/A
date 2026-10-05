"""Isolated serial Table1 public ports/reference and official-source Fisher.

This runner is not GPU qualified merely by existing. The first controller's
handoff must run its live smoke before any formal baseline predictions.
"""
import os,sys,time,gc,json,traceback,zipfile,shutil,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
OUT=BASE/'baselines'

def lock():
 from vg_tta.decota_paper_baselines_20261005_v1 import CONFIGS
 verify();lf=OUT/'RUNTIME_LOCK.json'
 if not lf.exists():
  code=['scripts/run_decota_paper_baselines_v1.py','vg_tta/decota_paper_baselines_20261005_v1.py','vg_tta/decota_paper_baseline_math_v1.py','vg_tta/native_baselines_paper_v1.py','vg_tta/native_probability_interface_v1.py','vg_tta/tastvg_baseline_expansion.py','protocols/decota_paper_baselines_20261005_v1.md']
  write(lf,dict(version='native_paper_ports_20261005',configs=CONFIGS,pins={f:sha(ROOT/f) for f in code},main_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),source_roster_lock_sha256=sha(BASE/'source_Fisher/ROSTER_LOCK.json'),GT_read=False,time=time.time()))
 p=read(lf);assert p['main_runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json') and p['source_roster_lock_sha256']==sha(BASE/'source_Fisher/ROSTER_LOCK.json')
 for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
 return p

def flat(params):
 import numpy as np
 return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in params]).astype(np.float32)

def native_input(model,ds,row,cache):
 from scripts.run_decota_paper_main_v1 import frames_for
 from methods.decota_final_simplified_v1.backbone import make_batch
 frames,ids,reuse=frames_for(ds,row,cache);batch=make_batch(frames,ids,row['input'],model)
 return batch,ids,row['parses']['subject'],reuse

def source_input(ds,r,at,cache):
 from scripts.prepare_official_dense_pool_v2 import frame_ids_from_segment
 from vg_tta import exact_frame_decode_audit_v2 as binding
 if ds=='hc2':
  from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
  q=read(BASE/'source_Fisher/hc2/media'/f'{r["filename"]}.json')['input']
  assert sha(q['video_path'])==q['video_sha256']
 else:
  decode=binding.decode;media=r['media'];archive=Path(media['path']);assert archive.stat().st_size==media['archive_bytes'] and archive.stat().st_mtime_ns==media['archive_mtime_ns']
  folder=BASE/'source_Fisher/vid_media_scratch';folder.mkdir(parents=True,exist_ok=True);dest=folder/f'{r["source"]}.mp4'
  if not dest.exists():
   for f in folder.glob('*.mp4'):
    if f.stat().st_size+sum(x.stat().st_size for x in folder.glob('*.mp4'))>512*2**20:f.unlink() # reproducible source extraction scratch only
   with zipfile.ZipFile(archive) as z:
    info=z.getinfo(media['member']);assert info.CRC==media['crc32'] and info.file_size==media['bytes']
    with z.open(info) as src,dest.open('xb') as out:shutil.copyfileobj(src,out,4<<20)
  assert dest.stat().st_size==media['bytes']
  ids=frame_ids_from_segment(frame_count=r['frame_count'],fps=r['fps'],start_frame=r['start_frame'],end_frame=r['end_frame'],max_frames=200)
  q={k:r[k] for k in ['source','original_video_id','caption','width','height','fps','frame_count','start_frame','end_frame']}
  q.update(index=at,video_path=str(dest),video_sha256=sha(dest),frame_ids=ids,duration=r['frame_count']/r['fps'],kind='vidstg')
 key=digest([q['video_sha256'],q['frame_ids']])
 if key in cache:frames,ids=cache.pop(key);cache[key]=(frames,ids)
 else:
  frames,ids=decode(q);cache[key]=(frames,ids)
  while sum(x[0].nbytes for x in cache.values())>256*2**20 and len(cache)>1:cache.popitem(last=False)
 return q,frames,ids

def fisher(ds):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from vg_tta.tastvg_baseline_expansion import decoder_layernorm_scope
 from vg_tta.native_baselines_paper_v1 import live_output
 from vg_tta.decota_paper_baselines_20261005_v1 import pseudo_native_loss
 from methods.decota_final_simplified_v1.observations import QuerySubjectParser
 from methods.decota_final_simplified_v1.backbone import make_batch
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.decota_matrix_common_v1 import save,load
 lock();source=read(BASE/'source_Fisher'/ds/'SOURCE_ROSTER.json');assert len(source['rows'])==2000
 if ds=='hc2':assert read(BASE/'source_Fisher/hc2/MEDIA_BARRIER.json')['status']=='sealed'
 dest=OUT/'Fisher'/f'{ds}.pt'
 if dest.with_suffix('.json').exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha256'];return
 lease=gpu();model=model_for('vidstg' if ds=='vidstg' else 'hcstvg2');names,params,scope=decoder_layernorm_scope(model);initial=[p.detach().clone() for p in params];modelhash=state_hash(model.state_dict())
 parser=QuerySubjectParser(ROOT/'.cache/stanza');sums=[torch.zeros_like(p,dtype=torch.float64) for p in params];cache=collections.OrderedDict();records=[];tick=time.time()
 try:
  for at,r in enumerate(source['rows']):
   budget();q,frames,ids=source_input(ds,r,at,cache);batch=make_batch(frames,ids,q,model);subject=parser(q['caption'])['subject']
   for p in params:p.requires_grad_(True)
   native,_=live_output(model,batch,ids,subject);loss=pseudo_native_loss(native);gs=torch.autograd.grad(loss,params,allow_unused=True)
   assert any(g is not None for g in gs) and all(g is None or torch.isfinite(g).all() for g in gs)
   for acc,p,g in zip(sums,params,gs):
    if g is not None:acc.add_(g.detach().double().square())
    p.requires_grad_(False);p.grad=None
   records.append(dict(key=r['key'],input_sha256=digest(q),video_sha256=q['video_sha256'],loss=float(loss.detach()),unused_parameters=sum(g is None for g in gs),GT_read=False))
   del frames,batch,native,loss,gs
   if (at+1)%20==0:status(OUT/'Fisher'/f'{ds}_STATUS.json',dict(status='running',pid=os.getpid(),done=at+1,total=2000,seconds=time.time()-tick,GT_read=False,time=time.time()));print('SOURCE_FISHER_PROGRESS',ds,at+1,2000,flush=True)
  assert state_hash(model.state_dict())==modelhash
  values=[(s/2000).float().cpu() for s in sums];assert any(v.count_nonzero() for v in values) and all(torch.isfinite(v).all() and (v>=0).all() for v in values)
  result=dict(names=names,values=values,source_parameters=[p.cpu() for p in initial],source_inputs_only=True,source_queries=2000,target_labels_used=False,source_state_sha256=modelhash,source_roster_sha256=sha(BASE/'source_Fisher'/ds/'SOURCE_ROSTER.json'),records=records,scope=scope)
  save(dest,result);write(dest.with_suffix('.json'),dict(status='source_Fisher_sealed',sha256=sha(dest),source_queries=2000,parameter_count=sum(p.numel() for p in params),runtime_lock_sha256=sha(OUT/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
  print('SOURCE_FISHER_SEALED',ds,2000,flush=True)
 finally:lease.close()

def smoke(method=None,ds=None):
 import torch
 from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
 from vg_tta.tastvg_baseline_expansion import decoder_layernorm_scope
 from vg_tta.native_baselines_paper_v1 import live_output
 from vg_tta.decota_paper_baselines_20261005_v1 import OnlineBaseline
 from vg_tta.decota_paper_baseline_math_v1 import audit_updates
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.decota_matrix_common_v1 import load,save
 cfg=lock();lease=gpu();records=[]
 try:
  jobs=[j for j in JOBS if ds is None or j[1]==ds]
  methods=['TENT','EATA','SAR'] if method is None else [method]
  assert jobs and all(m in ['TENT','EATA','SAR'] for m in methods)
  for job,ds,source in jobs:
   for method in methods:
    accept=OUT/'smoke'/f'{method}_{ds}'/'ROOT_ACCEPTANCE.json'
    if accept.exists():
     receipt=read(accept);assert receipt['runtime_lock_sha256']==sha(OUT/'RUNTIME_LOCK.json') and receipt['status']=='pass'
     records.append(receipt);continue
    model=model_for(source);names,params,scope=decoder_layernorm_scope(model);mh=state_hash(model.state_dict());cache=collections.OrderedDict()
    fi=load(OUT/'Fisher'/f'{"vidstg" if source=="vidstg" else "hc2"}.pt') if method=='EATA' else None
    opt=OnlineBaseline(names,params,method,fi);opt.audit_updates=True;cs=[]
    for parent in [0,1]:
     row=read_row(ds,parent);batch,ids,subject,_=native_input(model,ds,row,cache)
     def closure():return live_output(model,batch,ids,subject)[0]
     def infer():return live_output(model,batch,ids,subject)[1]
     # Zero lr performs the actual native protocol, including selection and SAM
     # perturbations, while requiring exact restoration to original output.
     with torch.no_grad():zero=infer()
     saved=opt.state_dict();saved_lr=opt.optimizer.param_groups[0]['lr'];opt.optimizer.param_groups[0]['lr']=0.
     _,lr0,a0=opt.arrive(closure,infer);assert torch.equal(lr0['boxes'],zero['boxes']) and lr0['indices']==zero['indices']
     opt.load_state_dict(saved);opt.optimizer.param_groups[0]['lr']=saved_lr
     pre,post,a=opt.arrive(closure,infer);assert a['arrival']==parent+1 and torch.isfinite(post['boxes']).all();mathcheck=audit_updates(a)
     cs.append(dict(parent=parent,lr0_exact_native_parity=True,scope_parameters=sum(p.numel() for p in params),audit={k:v for k,v in a.items() if k!='update_evidence'},math_audit=mathcheck,GT_read=False))
     sample=OUT/'smoke'/job/method/f'{parent:05}.pt';save(sample,dict(before=pre,after=post,audit=a,state=opt.state_dict()))
    opt.reset();assert state_hash(model.state_dict())==mh
    receipt=dict(status='pass',job=job,method=method,records=cs,actual_queries=2,source_restored_after_stream=True,runtime_lock_sha256=sha(OUT/'RUNTIME_LOCK.json'),GT_read=False,time=time.time())
    write(accept,receipt);records.append(receipt)
    del model,opt,fi,params;cache.clear();gc.collect();torch.cuda.empty_cache()
  if len(records)==6 and not (OUT/'SMOKE_ROOT_ACCEPTANCE.json').exists():
   write(OUT/'SMOKE_ROOT_ACCEPTANCE.json',dict(status='pass',records=records,actual_queries=12,runtime_lock_sha256=sha(OUT/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
  print('PAPER_BASELINE_LIVE_SMOKE_PASS',sum(r['actual_queries'] for r in records),flush=True)
 finally:lease.close()

def execute(method,ds):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
 from vg_tta.tastvg_baseline_expansion import decoder_layernorm_scope
 from vg_tta.native_baselines_paper_v1 import live_output
 from vg_tta.decota_paper_baselines_20261005_v1 import OnlineBaseline
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.decota_matrix_common_v1 import load,save
 lock();accept=read(OUT/'smoke'/f'{method}_{ds}'/'ROOT_ACCEPTANCE.json');assert accept['status']=='pass' and accept['runtime_lock_sha256']==sha(OUT/'RUNTIME_LOCK.json');source='vidstg' if ds=='hc2' else 'hcstvg2';job=method+'_'+ds;dest=OUT/job
 assert not (dest/'PREDICTION_BARRIER.json').exists();lease=gpu();model=model_for(source);mh=state_hash(model.state_dict());names,params,scope=decoder_layernorm_scope(model)
 fi=load(OUT/'Fisher'/f'{"vidstg" if source=="vidstg" else "hc2"}.pt') if method=='EATA' else None
 opt=OnlineBaseline(names,params,method,fi);plan=read(BASE/ds/'PLAN.json');cache=collections.OrderedDict();files={};count=0;tick=time.time()
 try:
  write(dest/'PARAMETER_SCOPE.json',dict(names=names,shapes=[list(p.shape) for p in params],count=sum(p.numel() for p in params),scope=scope))
  for order,seq in plan['orders'].items():
   opt.reset();previous=None
   for at,parent in enumerate(seq):
    budget();f=dest/order/f'{at:05}.npz';assert not f.exists(),'Preserve partial stream; explicit engineering recovery checkpoint needed'
    row=read_row(ds,parent);batch,ids,subject,reuse=native_input(model,ds,row,cache);before=flat(params);prehash=digest(before.tolist())
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    pre,post,audit=opt.arrive(lambda:live_output(model,batch,ids,subject)[0],lambda:live_output(model,batch,ids,subject)[1]);torch.cuda.synchronize();seconds=time.perf_counter()-begin
    after=flat(params);md=dict(method=method,dataset=ds,source_checkpoint=source,parent=parent,order=order,arrival=at,frame_ids=ids,
     interval_before=pre['physical_interval'],interval_after=post['physical_interval'],indices_before=pre['indices'],indices_after=post['indices'],
     previous_payload_sha256=previous,parameter_before_sha256=prehash,parameter_after_sha256=digest(after.tolist()),input_metadata_sha256=digest(row['input']),
     port_lock_sha256=sha(OUT/'RUNTIME_LOCK.json'),audit={k:v for k,v in audit.items() if k!='update_evidence'},compute=dict(model_seconds=seconds,CUDA_peak_allocated=torch.cuda.max_memory_allocated(),CUDA_peak_reserved=torch.cuda.max_memory_reserved(),new_DINO_calls=0,new_backbone_forwards=2*(2+audit['counts']['forward_closures']),decoded_frame_reuse=reuse),GT_read=False)
    if at%100==0:
     sample=dest/'state_samples'/order/f'{at:05}.pt';save(sample,opt.state_dict());md['optimizer_sample']=dict(path=str(sample.relative_to(BASE)),sha256=sha(sample))
    rc=save_npz(f,dict(Before=pre['boxes'].cpu().numpy(),After=post['boxes'].cpu().numpy(),state_before=before,state_after=after),md)
    previous=rc['sha256'];files[str(f.relative_to(BASE))]=previous;count+=1;del batch,pre,post,before,after
    if count%10==0:status(dest/'STATUS.json',dict(status='online_running',method=method,pid=os.getpid(),done=count,total=plan['queries']*3,order=order,seconds=time.time()-tick,GT_read=False,time=time.time()));print('BASELINE_PROGRESS',job,count,plan['queries']*3,flush=True)
  opt.reset();assert state_hash(model.state_dict())==mh and count==plan['queries']*3
  write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',method=method,dataset=ds,cells=count,files=files,GT_read=False,time=time.time()))
  status(dest/'STATUS.json',dict(status='completed_sealed_pending_all_Table1',done=count,total=count,GT_read=False,time=time.time()))
 finally:lease.close()

if __name__=='__main__':
 action=sys.argv[1]
 try:
  if action=='lock':lock()
  elif action=='smoke':smoke()
  elif action.startswith('smoke_'):
   _,method,ds=action.split('_');smoke(method,ds)
  elif action.startswith('fisher_'):fisher(action.split('_',1)[1])
  else:method,ds=action.split('_',1);execute(method,ds)
 except BaseException:
  fd=OUT/'failures'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True);(fd/'traceback.txt').write_text(traceback.format_exc())
  status(OUT/'WORKER_FAILURE.json',dict(status='failed',action=action,pid=os.getpid(),failure=str(fd),GT_read=False,time=time.time()));raise
