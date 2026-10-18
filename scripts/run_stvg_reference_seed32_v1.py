"""Native-content paired reference seed32: global seal then independent CPU score."""
from pathlib import Path
import os,sys,time,argparse,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_reference_seed32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/reference_seed32';PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';PREFIX=ROOT/'artifacts/stvg_shared_text_query32_v1/prefix'
CODE=['vg_tta/stvg_reference_seed32_v1.py','scripts/run_stvg_reference_seed32_v1.py','scripts/continue_stvg_reference_seed32_v1.py','protocols/stvg_reference_seed32_v1.md'];VERIFIER=None

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
 global VERIFIER
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 from main_method.runtime import PinVerifier,resource_guard
 if VERIFIER is None:VERIFIER=PinVerifier(r['pins'])
 VERIFIER.verify(full=full);resource_guard()
 if live:assert time.time()<p['deadline_epoch']
 return p

def prepare():
 from vg_tta.stvg_reference_seed32_v1 import ARMS,properties
 old=read(ROOT/'artifacts/stvg_object_condition32_v1/PLAN_LOCK.json');night=read(ROOT/'artifacts/stvg_opd_overnight_20261018_v1/STATE.json');assert read(ROOT/'artifacts/stvg_bidirectional_region_opd32_v1/FINAL_COMPLETION.json')['status']=='complete'
 files=CODE+['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
 for r in old['records']:files+=[str((PRE/k/(r['alias']+'.pt')).relative_to(ROOT))for k in ['source','E','predictions']]+[str((PREFIX/(r['alias']+'.pt')).relative_to(ROOT))]
 put(D/'CPU_PROPERTIES.json',properties());put(D/'PLAN_LOCK.json',dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,new_Source_captures=0,new_DINO_calls=0,new_training=0,new_OPD=0,native_content_and_memory_fixed=True,only_second_pass_initial_reference_changed=True,absolute_output_no_E_retaining=True,no_reference_outputs_keep_E=True,all32_seal_before_NEW_GT=True,development_exposed=True,independent_confirmation=False,GPU_seconds_cap=300,created_epoch=time.time(),deadline_epoch=min(time.time()+1800,night['deadline_epoch']),decision='Both-domain E_reference-minus-E and E_reference-minus-Source_reference complete-v and unobserved actual-s means must be positive; otherwise stop exact reseeding interface without parameter or blend search'))
 pins={**read(ROOT/'artifacts/stvg_object_condition32_v1/RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in files}};put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));verify(full=True);status('prepared',0,GT_read=False)

def measure():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.replay import SpatialReplay
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
 from vg_tta.stvg_reference_seed32_v1 import readout
 p=verify(True);lease=None
 while lease is None:
  verify(True)
  try:lease=gpu()
  except BlockingIOError:status('waiting_GPU_lease',0);time.sleep(15)
 start=time.perf_counter();model=None;domain=None;files=[]
 try:
  put(D/'GPU_RUNTIME.json',dict(pid=os.getpid(),start_epoch=time.time(),shared_lease=True))
  for ix,r in enumerate(p['records']):
   verify(True);assert time.perf_counter()-start+3<p['GPU_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30;tick=time.perf_counter();a=r['alias'];ds=r['dataset'];status('raw_reference_pair',ix,alias=a,GT_read=False)
   if domain!=ds:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;mh=state_hash(model.state_dict())
   s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));old=load(PRE/'predictions'/(a+'.pt'));c=load(PREFIX/(a+'.pt'));assert torch.equal(c['source'],s['Source'])and c['records']==s['records'];base=SpatialReplay(model,transfer(c['views'],'cuda'),len(s['frame_ids']));base.restore(base.initial)
   for _,param in base.named:param.requires_grad_(False)
   assert torch.equal(base.zero['boxes'].cpu(),s['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],s['logits']))
   S=s['Source'].cuda();E=torch.as_tensor(e['E']).double().cuda();identity=readout(base);assert torch.equal(identity['boxes'],s['Source']);rt=time.perf_counter();ss=readout(base,S);ee=readout(base,E);equal_control=readout(base,S.clone());assert torch.equal(ss['boxes'],equal_control['boxes'])
   # Spatial-only decoder never invokes temporal head; original time is fixed.
   torch.cuda.synchronize();active=e['estimate']['support_frames']>0
   if not active:assert torch.equal(S.double(),E)and torch.equal(ss['boxes'],ee['boxes'])
   outputs=dict(Source_MBR=s['Source'].numpy(),E_MBR=E.cpu().numpy(),Source_reference=ss['boxes'].numpy()if active else E.cpu().numpy(),E_reference=ee['boxes'].numpy()if active else E.cpu().numpy())
   assert state_hash(model.state_dict())==mh and not any(v.requires_grad or v.grad is not None for v in model.parameters())
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=old['frame_ids'],planned_positions=old['planned_positions'],interval=old['interval'],active=active,outputs=outputs,values=dict(native=identity,Source_reference=ss,E_reference=ee),model_hash=mh,zero_native_exact=True,equal_reference_exact=True,time_logits_exact=True,GT_read=False,compute=dict(new_GPU_readout_seconds=time.perf_counter()-rt,case_seconds=time.perf_counter()-tick,new_Source_captures=0,new_DINO_calls=0))))
   if ix==1:put(D/'PILOT2_TIMING.json',dict(count=2,elapsed_seconds=time.perf_counter()-start,not_a_guarantee=True))
   print(a,ix+1,round(time.perf_counter()-start,3),flush=True);del base,s,e,old,c,S,E,ss,ee,identity,equal_control;gc.collect();torch.cuda.empty_cache()
  verify(full=True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_outputs_before_NEW_GT=True,GPU_allocation_seconds=time.perf_counter()-start+3,actual_worker_GPU_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_Source_captures=0,new_DINO_calls=0));status('all32_sealed',32,GT_read=False)
 finally:lease.close()

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 torch.set_num_threads(4)
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_reference_seed32_v1 import ARMS
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_outputs_before_NEW_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 old={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/bidirectional_region_opd32/RESULTS_PUBLIC.json')['rows']}
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q])
  for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
   x,y=m['values']['E_MBR'][k],old[a]['values']['E_MBR'][k];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12
  S=o['outputs']['Source_MBR'];E=o['outputs']['E_MBR'];diagnostics={arm:dict(mean_center_shift_from_E=float(np.linalg.norm(o['outputs'][arm][:,:2]-E[:,:2],axis=-1).mean()),raw_all_nodes_changed=int(np.any(o['values'][arm]['boxes'].numpy()!=S,axis=1).sum()))for arm in ARMS[2:]}
  rows.append(dict(alias=a,dataset=ds,active=o['active'],diagnostics=diagnostics,compute=o['compute'],**m))
 groups={};contrasts=[(a,'E_MBR')for a in ARMS if a!='E_MBR']+[('E_reference','Source_reference')]
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,active=sum(r['active']for r in rr),metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in ['v','t','s','observed_actual_s','unobserved_actual_s']}for a in ARMS},paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in ['v','s','observed_actual_s','unobserved_actual_s']}for a,b in contrasts})
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,configuration={k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']},development_exposed=True,GT_offline_after_seal=True))
 passed=all(groups[ds]['paired'][c][ep]['mean']>0 for ds in gt for c in ['E_reference_minus_E_MBR','E_reference_minus_Source_reference']for ep in ['v','unobserved_actual_s']);put(P/'GATE.json',dict(status='evaluated',passed=passed,rule=p['decision'],OPD_not_run=True));put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print('gate',passed,{ds:groups[ds]['paired']['E_reference_minus_E_MBR']['v']['mean']for ds in gt},flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
