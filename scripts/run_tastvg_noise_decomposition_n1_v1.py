"""Authorized dev-only GT mechanism oracle. Paper48 stays paused."""
import sys,time,gc,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,save,sha,status
from scripts.run_tastvg_spatial_expansion_s0_v1 import OUT as S0,OLD,CONDS,observation
from scripts.run_tastvg_paper48_raw_v1 import full_temporal_check
OUT=ROOT/'artifacts/tastvg_noise_decomposition_n1_v1';J01=ROOT/'artifacts/tastvg_schedule_j01_v1';TEMP=ROOT/'artifacts/tastvg_temporal_fourarm_v1';NATIVE=ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1';METHOD=ROOT/'methods/tastvg_dual_evidence_j0_v1'
PRIMARY=['all','useful','noisy','useful_positive','useful_negative'];MATCHED=['useful_matched','noisy_matched'];ARMS=PRIMARY+MATCHED

def prepare():
 p=read(J01/'LOCK.json');freeze=read(METHOD/'FREEZE_J01.json');assert sha(METHOD/'method.py')==freeze['method_code_sha256'] and sha(METHOD/'config.json')==freeze['config_sha256']
 files=['protocols/tastvg_noise_decomposition_n1_v1.md','scripts/run_tastvg_noise_decomposition_n1_v1.py','vg_tta/tastvg_noise_decomposition_n1_v1.py','tests/test_tastvg_noise_decomposition_n1_v1.py']
 lp='artifacts/tastvg_corruption_c0c1_v1/GT_SUBSET.json'
 write(OUT/'LOCK.json',dict(rows=p['rows'],orders=p['orders'],conditions=CONDS,arms=ARMS,expert_indices=p['expert_indices'],pins={**p['pins'],**{f:sha(ROOT/f) for f in files}},inputs=p['inputs'],GT_container=lp,GT_sha256=sha(ROOT/lp),freeze_sha256=sha(METHOD/'FREEZE_J01.json'),J01_barrier_sha256=sha(J01/'PREDICTION_BARRIER.json'),cap_seconds=3600,total_arrivals=3360,GT_for_analysis_filter=True,deployable=False,time=time.time()))
 write(OUT/'ORDERS.json',read(J01/'ORDERS.json'))

def verify():
 p=read(OUT/'LOCK.json')
 for f,h in {**p['pins'],**p['inputs']}.items():assert sha(ROOT/f)==h,f
 assert sha(METHOD/'FREEZE_J01.json')==p['freeze_sha256'] and sha(J01/'PREDICTION_BARRIER.json')==p['J01_barrier_sha256'];assert sha(ROOT/p['GT_container'])==p['GT_sha256']
 return p

def run():
 p=verify();tick=time.monotonic();done=updates=0;failure=None;state='failed';lease=None;policy=None
 prior=sum(read(f)['seconds'] for f in OUT.glob('**/allocations/*.json'))
 try:
  import numpy as np,torch,ijson
  from methods.decota_final_simplified_v1.tensors import state_hash,detached
  from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
  from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
  from scripts.run_final_simplification_v1 import lease as gpu_lease
  from methods.tastvg_dual_evidence_j0_v1.method import fast_rerank
  from vg_tta.tastvg_noise_decomposition_n1_v1 import NoiseMethod,eligible,hash_subset
  from vg_tta.tastvg_native_spatial_rollout_s05_v1 import reinsert
  from vg_tta.exact_frame_decode_audit_v2 import decode
  procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs;lease=gpu_lease()
  torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  keys={r['key'] for r in p['rows']}
  with (ROOT/p['GT_container']).open('rb') as f:gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
  assert set(gt)==keys
  write(OUT/'GT_FILTER_EXPOSURE.json',dict(time=time.time(),sources=16,queries=16,old_exposed_only=True,GT_read_before_updates=True,GT_purpose='center-to-probe correctness filter, not teacher direction; offline oracle mechanism',GT_container_sha256=p['GT_sha256'],deployable=False))
  def guard(event,args):
   if event=='open' and args and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ['/ROWS.json','/SUMMARY.json']):raise PermissionError('N1 inference forbids past outcome tables; GT filtering explicitly authorized')
  sys.addaudithook(guard);install_clean_loader()
  from scripts.run_spatial_regression_alignment_v1 import model_load
  model=model_load('hcstvg1_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==read(OLD/'CAPTURE_BARRIER.json')['model_state_sha256']
  support=load(NATIVE/'PARAMETER_SUPPORT.pt');deltas=[{n:(v-support['center'][n]).cuda() for n,v in x.items()} for x in support['states']];policy=NoiseMethod(model,deltas)
  assert all(torch.equal(v.cpu(),support['center'][n]) for n,v in policy.actor.initial.items())
  hb=read(S0/'H_BARRIER.json');eb=read(S0/'EXPERT_BARRIER.json');tb=read(TEMP/'C2_BARRIER.json');ob=read(OLD/'CAPTURE_BARRIER.json');rein=[];trein=[];streams=[];match_records=[]
  def prepare_one(row,order,cond,arrival):
   assert prior+time.monotonic()-tick<p['cap_seconds']-20 and shutil.disk_usage(ROOT).free>8*2**30
   name=f"{row['ordinal']:03}.pt";hr=f'capture/{cond}/{name}';assert sha(S0/hr)==hb['files'][hr];data=device_tree(load(S0/hr),'cuda');scheduled=arrival in p['expert_indices'];called=[];teachers=[]
   def temporal_provider():
    assert scheduled;rel=f'c2/{cond}/{name}';assert sha(TEMP/rel)==tb['files'][rel];called.append('temporal');x=load(TEMP/rel);teachers.append(x);return x
   def spatial_provider():
    assert scheduled;rel=f'expert/{cond}/{name}';assert sha(S0/rel)==eb['files'][rel];called.append('spatial');return load(S0/rel)
   prepared=policy.prepare(data,scheduled,gt[row['key']],temporal_provider,spatial_provider);assert called==(['temporal','spatial'] if scheduled else [])
   return data,prepared,teachers,called
  def finalize(row,order,cond,arrival,arm,data,result,post_ev,teachers,called,local_updates):
   nonlocal done,updates
   name=f"{row['ordinal']:03}.pt";hr=f'capture/{cond}/{name}';scheduled=arrival in p['expert_indices']
   result.update(ablation=arm,order=order,parent=row['ordinal'],condition=cond,arrival=arrival,expert_scheduled=scheduled,pixel_sha256=data['pixel_sha256'],expert_reads=called)
   assert sha(OLD/hr)==ob['files'][hr];old=load(OLD/hr);assert old['pixel_sha']==data['pixel_sha256'] and torch.equal(old['native']['boxes'],data['prediction']['boxes'].cpu())
   if arrival==0:
    assert torch.equal(result['prediction']['boxes'],old['native']['boxes']) and result['prediction']['indices']==old['native']['indices']
    assert all(torch.equal(a,b) for a,b in zip(result['prediction']['logits'],old['native']['logits']))
   if result['updated']:updates+=1;local_updates+=1
   if scheduled:
    fast,fd=fast_rerank(old['native'],old['candidates']['temporal'],teachers[0]);assert fd['selected']==teachers[0]['selected'] and fd['scores']==teachers[0]['scores'];result['fast_control']=fd
   if (order,cond) in [('order1','clean'),('order5','frame_drop_5')] and ((result['updated'] and local_updates==1) or arrival==15):
    frames,ids=decode(row['input']);assert ids==row['frame_ids'];shifted,_=observation(row,cond,frames);audit=reinsert(model,shifted,row,policy.actor.state(),post_ev);rein.append(dict(arm=arm,order=order,parent=row['ordinal'],condition=cond,arrival=arrival,**audit))
   if (order,cond) in [('order1','clean'),('order5','frame_drop_5')] and arrival==8:
    frames,_=decode(row['input']);shifted,_=observation(row,cond,frames);audit=full_temporal_check(model,shifted,row,device_tree(result['pre_state'],'cuda'),result['temporal_layers']);trein.append(dict(arm=arm,order=order,parent=row['ordinal'],condition=cond,**audit))
   assert state_hash(policy.actor.state())==result['post_state_sha256'];save(OUT/'online'/arm/order/cond/name,result);done+=1
   status(OUT/'STATUS.json',dict(status='running',done=done,total=3360,updates=updates,arm=arm,order=order,condition=cond));print('N1',done,3360,arm,order,cond,arrival,'updated',result['updated'],'signals',result.get('selected_count',0),flush=True)
   return local_updates
  for arm in PRIMARY:
   for order in p['orders']:
    for cond in CONDS:
     policy.reset();previous=state_hash(policy.actor.state());start=previous;local_updates=0
     for arrival,parent in enumerate(p['orders'][order]):
      row=next(r for r in p['rows'] if r['ordinal']==parent);data,prep,teachers,called=prepare_one(row,order,cond,arrival);assert prep[0]['pre_state_sha256']==previous
      result,post_ev=policy.finish(data,prep,arm);local_updates=finalize(row,order,cond,arrival,arm,data,result,post_ev,teachers,called,local_updates);previous=result['post_state_sha256']
      del data,prep,result,post_ev,teachers;gc.collect();torch.cuda.empty_cache()
     streams.append(dict(arm=arm,order=order,condition=cond,start_sha256=start,final_sha256=previous,updates=local_updates));save(OUT/'final_states'/arm/order/f'{cond}.pt',detached(policy.actor.state(),'cpu'))
  for order in p['orders']:
   for cond in CONDS:
    policy.reset();states={a:detached(policy.actor.state(),'cpu') for a in MATCHED};start=state_hash(policy.actor.state());local={a:0 for a in MATCHED}
    for arrival,parent in enumerate(p['orders'][order]):
     row=next(r for r in p['rows'] if r['ordinal']==parent);prepared={};pools={}
     for arm in MATCHED:
      policy.actor.restore(device_tree(states[arm],'cuda'));data,prep,teachers,called=prepare_one(row,order,cond,arrival);prepared[arm]=(data,prep,teachers,called)
      pools[arm]=eligible(arm,prep[0]['teacher_labels'],prep[0]['gt_labels']) if arrival in p['expert_indices'] else []
     m=min(map(len,pools.values()));key=f'{order}|{parent}'
     if arrival in p['expert_indices']:match_records.append(dict(order=order,condition=cond,parent=parent,arrival=arrival,useful_available=len(pools[MATCHED[0]]),noisy_available=len(pools[MATCHED[1]]),matched_count=m))
     for arm in MATCHED:
      policy.actor.restore(device_tree(states[arm],'cuda'));data,prep,teachers,called=prepared.pop(arm);assert state_hash(states[arm])==prep[0]['pre_state_sha256']
      chosen=hash_subset(pools[arm],m,key);result,post_ev=policy.finish(data,prep,arm,chosen,match_count=m,hash_key=key);local[arm]=finalize(row,order,cond,arrival,arm,data,result,post_ev,teachers,called,local[arm]);states[arm]=result['post_state']
      del data,prep,teachers,result,post_ev;gc.collect();torch.cuda.empty_cache()
    for arm in MATCHED:
     streams.append(dict(arm=arm,order=order,condition=cond,start_sha256=start,final_sha256=state_hash(states[arm]),updates=local[arm]));save(OUT/'final_states'/arm/order/f'{cond}.pt',states[arm])
  policy.close();policy=None;assert state_hash(model.state_dict())==mh;verify()
  assert len(trein)==14 and all(sum(r['arm']==a for r in rein)>=2 for a in ARMS)
  write(OUT/'REINSERTION_AUDIT.json',dict(status='pass',spatial=rein,temporal_all_layers=trein));write(OUT/'STATE_CHAIN.json',dict(status='pass',streams=streams));write(OUT/'COUNT_MATCH.json',match_records)
  write(OUT/'PREDICTION_BARRIER.json',dict(cells=done,arms=7,updates=updates,files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'online').rglob('*.pt')},final_states={str(f.relative_to(OUT)):sha(f) for f in (OUT/'final_states').rglob('*.pt')},GT_read=True,GT_for_signal_filter=True,deployable=False,model_restored=True,time=time.time()));state='completed'
 except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if policy:policy.close()
  r=dict(status=state,done=done,updates=updates,seconds=time.monotonic()-tick,failure=failure,time=time.time());write(OUT/'allocations'/f'{time.time_ns()}.json',r);status(OUT/'STATUS.json',r)
  if lease:lease.close()
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
