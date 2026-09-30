"""Real uncached serialized deployment on one32GB GPU; cold loads included, no reuse."""
import sys,time,subprocess,os,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_paper48_common_v1 import BASE,read,write,load,save,sha,status,budget,guard,verify
OUT=BASE/'P4'

def expert_child(stage,parent):
 # One freshly generated input, no old receipts/cache paths, no existing output allowed.
 from scripts import run_tastvg_paper48_experts_v1 as worker
 p=verify();dst=OUT/'uncached_experts'/f'{parent:05}'/stage;assert not dst.exists()
 worker.OUT=dst;worker.OLD=OUT/'forbidden_old_cache';worker.reuse_receipts=lambda *args:None
 one={**p,'expert_needed':[parent],'conditions_by_parent':{str(parent):['clean']},'total':1}
 worker.verify=lambda:one
 worker.run(stage)
 rr=read(dst/stage/'clean'/f'{parent:05}.json');assert rr['new_inference'] is True


def native_child(mode,index):
 import torch,numpy as np
 from scripts.run_final_simplification_v1 import lease as gpu_lease
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
 from scripts.run_spatial_regression_alignment_v1 import model_load
 from methods.decota_final_simplified_v1.tensors import state_hash,detached
 from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,offset_batch
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.tastvg_dual_evidence_j0_v1.method import OnlineMethod
 from vg_tta.exact_frame_decode_audit_v2 import decode
 # Importing runner only defines source_capture; PANEL argv is irrelevant to that helper.
 from scripts.run_tastvg_paper48_online_v1 import source_capture
 p=verify();plan=read(BASE/'P4_PLAN.json');parent=plan['parents'][index];row=p['rows'][parent];budget();sys.addaudithook(guard);install_clean_loader();lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 started=time.perf_counter();model=model_load('hcstvg1_test');torch.cuda.synchronize();load_sec=time.perf_counter()-started;mh=state_hash(model.state_dict());assert mh=='fbb1ed8871d6c2aa093879efefc2ee500bb7de5e0fe1c25b809c5393d010f3c7'
 tick=time.perf_counter();frames,ids=decode(row['input']);subjectfile=BASE/'P1/subjects'/f'{parent:05}.json';subject=read(subjectfile)['parses']['subject'];decode_sec=time.perf_counter()-tick;tick=time.perf_counter();adapter_sec=0.;initialization_sec=0.;scheduled=mode=='Ours' and index%4==0
 if mode=='Frozen':
  batch=make_batch(frames,ids,row['input'],model);boxes=[];logits=[];records=[]
  with torch.no_grad(),query_subject(model,batch,subject):
   for offset in [0,1]:
    view=offset_batch(batch,offset)
    with torch.autocast('cuda',dtype=torch.float16):z=model(view['videos'],view['texts'],view['targets'],iteration_rate=-1)
    boxes.append(z['pred_boxes']);logits.append(z['pred_sted']);records.append(dict(frame_ids=view['targets'][0]['frame_ids']))
  native=prediction(logits,torch.stack([boxes[i%2][i//2] for i in range(len(ids))]).float(),records,ids);torch.cuda.synchronize();native_sec=time.perf_counter()-tick;post=None;previous=None;updated=False
 else:
  data=source_capture(model,frames,row,subject);torch.cuda.synchronize();native_sec=time.perf_counter()-tick;tick=time.perf_counter()
  support=load(ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1/PARAMETER_SUPPORT.pt');deltas=[{n:(v-support['center'][n]).cuda() for n,v in x.items()} for x in support['states']];policy=OnlineMethod(model,deltas);assert sum(v.numel() for _,v in policy.actor.named)==1792
  previous=state_hash(policy.actor.initial)
  if index:
   prev=load(OUT/'Ours'/f'{index-1:03}.pt');previous=prev['post_sha'];policy.actor.restore(device_tree(prev['post_state'],'cuda'))
  data=device_tree(data,'cuda');from scripts.c1_controlled_corruption_v1 import pixelhash
  pixel=pixelhash(frames);data['pixel_sha256']=pixel;torch.cuda.synchronize();initialization_sec=time.perf_counter()-tick;tick=time.perf_counter();calls=[]
  def expert(stage):
   assert scheduled;dst=OUT/'uncached_experts'/f'{parent:05}'/stage;rr=read(dst/stage/'clean'/f'{parent:05}.json');cf=dst/rr['cache'];assert rr['new_inference'] and sha(cf)==rr['cache_sha256'] and rr['pixel_sha256']==pixel;calls.append(stage);return {**load(cf),'pixel_sha256':pixel}
  x,_=policy.arrive(data,scheduled,lambda:expert('temporal'),lambda:expert('spatial'));torch.cuda.synchronize();adapter_sec=time.perf_counter()-tick;assert x['pre_state_sha256']==previous and calls==(['temporal','spatial'] if scheduled else []);native=x['output_prediction'];post=x['post_state'];updated=x['updated'];policy.close();assert state_hash(model.state_dict())==mh
 f=OUT/mode/f'{index:03}.pt';save(f,dict(parent=parent,index=index,mode=mode,prediction=detached(native,'cpu'),post_state=post,post_sha=state_hash(post) if post is not None else None,pre_sha=previous,updated=updated,GT_read=False))
 write(f.with_suffix('.json'),dict(sha256=sha(f),parent=parent,index=index,mode=mode,expert_scheduled=scheduled,load_seconds=load_sec,decode_seconds=decode_sec,native_seconds=native_sec,initialization_seconds=initialization_sec,adapter_seconds=adapter_sec,peak_vram_bytes=torch.cuda.max_memory_allocated(),uncached_native=True,cached_previous_expert_calls_used=False,GT_read=False));lease.close()


def child(script_args,log,python='.conda/tubedetr/bin/python'):
 budget();left=read(BASE/'TIME_BUDGET.json')['deadline_unix']-time.time()-60;assert left>0
 start=time.perf_counter()
 with log.open('a') as out:
  env=os.environ.copy()
  if script_args[:2]==['expert','spatial']:env['PYTHONPATH']=str(ROOT/'.runtime/sa2va_deps')+os.pathsep+env.get('PYTHONPATH','')
  p=subprocess.Popen(['bash','scripts/with_local_cuda.sh',str(ROOT/python),'-B',__file__,*map(str,script_args)],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=min(600,left))
  except subprocess.TimeoutExpired:
   import signal
   os.killpg(p.pid,signal.SIGINT)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30)
   raise TimeoutError('P4 child budget exhausted; keep all outputs')
 assert code==0,(script_args,code);return time.perf_counter()-start


def run():
 import numpy as np
 p=verify();plan=read(BASE/'P4_PLAN.json');OUT.mkdir(exist_ok=True);rows=[]
 for mode in ['Frozen','Ours']:
  for index,parent in enumerate(plan['parents']):
   budget();receipt=OUT/mode/f'{index:03}.timing.json'
   if receipt.exists():rows.append(read(receipt));continue
   start=time.perf_counter();times={};peaks=[]
   if mode=='Ours' and index%4==0:
    for stage in ['spatial','temporal']:
     dst=OUT/'uncached_experts'/f'{parent:05}'/stage
     if dst.exists():raise RuntimeError('Unfinished P4 arrival preserved; do not reuse and claim uncached')
     times[stage]=child(['expert',stage,parent],OUT/(stage+'.log'),'.venv-exost/bin/python' if stage=='temporal' else '.conda/tubedetr/bin/python')
     stat=read(dst/(stage.upper()+'_STATUS.json'));assert stat['status']=='completed' and stat['new_inferences']==1;peaks.append(stat['peak_vram_bytes'])
   native=child(['native',mode,index],OUT/(mode+'.log'));r=read(OUT/mode/f'{index:03}.json');z=dict(**r,end_to_end_cold_seconds=time.perf_counter()-start,native_process_seconds=native,spatial_process_seconds=times.get('spatial',0.),temporal_process_seconds=times.get('temporal',0.),peak_vram_including_experts=max([r['peak_vram_bytes']]+peaks),logical_expert_calls=2*int(r['expert_scheduled']))
   write(receipt,z);rows.append(z);status(OUT/'STATUS.json',dict(status='running',done=len(rows),total=200,mode=mode,index=index));print('P4',mode,index,z['end_to_end_cold_seconds'],flush=True)
 summaries={}
 for mode in ['Frozen','Ours']:
  rr=[r for r in rows if r['mode']==mode];summaries[mode]={}
  for group in ['all','expert','nonexpert']:
   seq=[r for r in rr if group=='all' or r['expert_scheduled']==(group=='expert')]
   if not seq:continue
   fields=['end_to_end_cold_seconds','native_process_seconds','load_seconds','decode_seconds','native_seconds','initialization_seconds','adapter_seconds','spatial_process_seconds','temporal_process_seconds','logical_expert_calls']
   summaries[mode][group]={k:dict(mean=float(np.mean([r[k] for r in seq])),p50=float(np.median([r[k] for r in seq])),p95=float(np.quantile([r[k] for r in seq],.95))) for k in fields};summaries[mode][group]['peak_vram_bytes']=max(r['peak_vram_including_experts'] for r in seq);summaries[mode][group]['queries']=len(seq)
 write(OUT/'SUMMARY.json',dict(timing_model='real serialized cold-start processes on single32GB GPU; each query loads native model; each expert call freshly loads its specialist. Includes Python launch/checkpoint/model loading and all uncached inference. This is not warm resident service latency.',persistent_parameters=1792,queries=100,new_native_runs=200,new_spatial_calls=25,new_temporal_calls=25,results=summaries))
 assert len(rows)==200;write(OUT/'AUDIT.json',dict(status='pass',uncached_native_runs=200,uncached_specialist_calls=50,previous_call_cache_reuse=False,persistent_state_chain=True,GT_read=False));write(OUT/'COMPLETION.json',dict(status='completed',publication='pending',time=time.time()));status(OUT/'STATUS.json',dict(status='completed',done=200,total=200))
if __name__=='__main__':
 if len(sys.argv)==1:run()
 elif sys.argv[1]=='expert':expert_child(sys.argv[2],int(sys.argv[3]))
 else:native_child(sys.argv[2],int(sys.argv[3]))
