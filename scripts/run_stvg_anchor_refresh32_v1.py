"""Bounded frozen-model reference-conditioning qualification, then postseal CPU score."""
from pathlib import Path
import os,sys,time,argparse,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_anchor_refresh32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/anchor_refresh32';PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';CLOSED=ROOT/'artifacts/stvg_dino_query_closedloop32_v1';FEAT=ROOT/'artifacts/stvg_discriminative_dense32_v1';VERIFIER=None
CODE=['vg_tta/stvg_anchor_refresh32_v1.py','scripts/run_stvg_anchor_refresh32_v1.py','scripts/continue_stvg_anchor_refresh32_v1.py','protocols/stvg_anchor_refresh32_v1.md']
def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
 global VERIFIER
 p=read(D/'PLAN_LOCK.json');rt=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==rt['plan_sha256']
 from main_method.runtime import PinVerifier,resource_guard
 if VERIFIER is None:VERIFIER=PinVerifier(rt['pins'])
 VERIFIER.verify(full=full);resource_guard()
 if live:assert time.time()<p['deadline_epoch']
 return p

def prepare():
 from vg_tta.stvg_anchor_refresh32_v1 import properties,ARMS
 old=read(ROOT/'artifacts/stvg_realizable_query_opd32_v1/PLAN_LOCK.json');assert read(ROOT/'artifacts/stvg_realizable_query_opd32_v1/FINAL_COMPLETION.json')['status']=='complete';night=read(ROOT/'artifacts/stvg_opd_overnight_20261018_v1/STATE.json');D.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True)
 put(D/'CPU_PROPERTIES.json',properties());p=dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,new_Source_captures=0,new_DINO_calls=0,new_training_runs=0,new_heads=0,new_OPD=0,pretrained_weights_frozen=True,second_pass_only=True,content_fullmemory_native=True,reference_schedules=['initial1','all6'],readout='zE+zDINOcondition-zEcondition',main_updates_only_unobserved_actual_nodes=True,pilot_aliases=['NH001','NV001'],GPU_seconds_cap=900,overnight_GPU_remaining_seconds=7200-night['new_GPU_spent_seconds'],created_epoch=time.time(),deadline_epoch=min(time.time()+1800,night['deadline_epoch']),MBR_fixed=True,E_retained=True,GT_after_full32_seal=True,development_exposed=True,independent_confirmation=False,no_parameter_search=True,decision='Both-domain Persistent_delta minus E complete-v and unobserved actual-s positive, Persistent_delta minus Initial_delta and Persistent_shuffle complete-v positive; retain CI and tails, no automatic OPD')
 put(D/'PLAN_LOCK.json',p);pins=dict(read(ROOT/'artifacts/stvg_realizable_query_opd32_v1/RUNTIME_LOCK.json')['pins'])
 for f in CODE:pins[f]=sha(ROOT/f)
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));verify(full=True);status('prepared',0,GT_read=False)

def measure():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
 from vg_tta.stvg_anchor_refresh32_v1 import run,paired_readout
 p=verify(True);lease=None
 while lease is None:
  verify(True)
  try:lease=gpu()
  except BlockingIOError:status('waiting_shared_GPU_lease',0);time.sleep(15)
 start=time.perf_counter();model=None;domain=None;files=[];pilot=[];modelhash=None;complete=False;partial={};a=None
 def check():
  verify(True);assert time.perf_counter()-start+3<min(p['GPU_seconds_cap'],p['overnight_GPU_remaining_seconds']);assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  put(D/'GPU_RUNTIME.json',dict(pid=os.getpid(),start_epoch=time.time(),shared_lease=True));records=[next(r for r in p['records']if r['alias']==a)for a in p['pilot_aliases']]+[r for r in p['records']if r['alias']not in p['pilot_aliases']]
  for ix,r in enumerate(records):
   check();a=r['alias'];ds=r['dataset'];tick=time.perf_counter();status('actual_anchor_conditioning',ix,alias=a,GT_read=False)
   if domain!=ds:
    if model is not None:
     assert state_hash(model.state_dict())==modelhash;del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;modelhash=state_hash(model.state_dict())
   old=load(CLOSED/'predictions'/(a+'.pt'));rc=load(CLOSED/'replay_inputs'/(a+'.pt'));src=load(PRE/'source'/(a+'.pt'));x=load(FEAT/'features'/(a+'.pt'));E=x['E'].double();S=src['Source'];assert torch.equal(rc['Source'],S)and np.array_equal(old['outputs']['E_MBR'],E.numpy());inputs=transfer(rc['spatial_inputs'],'cuda');decoder=model.ground_decoder.decoder;n=len(S)
   inputhash=state_hash({str(v)+'/'+k:t for v,kw in enumerate(inputs)for k,t in kw.items()if torch.is_tensor(t)})
   positions=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];targets=np.array([f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0]).reshape(-1,4);active=bool(positions);raw={};calls=0;partial={}
   def call(key,pos,boxes,persistent):
    nonlocal calls
    check();out,tr=run(decoder,inputs,n,pos,boxes,persistent);calls+=1;partial[key]=dict(boxes=out,traces=tr);check();return out
   native=call('native_copy',[],np.empty((0,4)),False);assert torch.equal(native,S)
   check();original=[]
   with torch.no_grad():
    for view,kw in enumerate(inputs):
     direct=decoder(**kw);copied=torch.stack([v['raw_output']for v in partial['native_copy']['traces'][view]['layers']]).transpose(1,2);assert torch.equal(direct.cpu(),copied);original.append(direct.cpu())
   calls+=1;partial['native_original']=dict(all_six_outputs=original);check()
   if active:
    for schedule,persistent in [('initial',False),('persistent',True)]:
     e=call(schedule+'_E',positions,E.numpy()[positions],persistent);t=call(schedule+'_DINO',positions,targets,persistent);raw[schedule]=dict(E=e,DINO=t)
    shuffle=call('persistent_shuffle',positions,np.roll(targets,1,axis=0)if len(targets)>1 else targets,True)
    if ix<2:repeat=call('persistent_E_repeat',positions,E.numpy()[positions],True);assert torch.equal(repeat,raw['persistent']['E'])
    initial=paired_readout(raw['initial']['DINO'],raw['initial']['E'],E,old['planned_positions']);persistent=paired_readout(raw['persistent']['DINO'],raw['persistent']['E'],E,old['planned_positions']);sh=paired_readout(shuffle,raw['persistent']['E'],E,old['planned_positions']);allnodes=paired_readout(raw['persistent']['DINO'],raw['persistent']['E'],E,old['planned_positions'],False)
   else:initial=persistent=sh=allnodes=E
   if active:
    for view in range(2):
     if not any(p%2==view for p in positions):
      assert torch.equal(raw['persistent']['DINO'][view::2],raw['persistent']['E'][view::2])and torch.equal(persistent[view::2],E[view::2])
   assert state_hash({str(v)+'/'+k:t for v,kw in enumerate(inputs)for k,t in kw.items()if torch.is_tensor(t)})==inputhash
   local=E.clone()
   if active:local[positions]=torch.as_tensor(targets,dtype=torch.float64)
   outputs=dict(Source_MBR=S.double().numpy(),E_MBR=E.numpy(),Anchor_local=local.numpy(),Initial_delta=initial.numpy(),Persistent_delta=persistent.numpy(),Persistent_shuffle=sh.numpy(),Persistent_allnodes=allnodes.numpy());assert all(v.shape==E.shape and np.isfinite(v).all()and(v[:,2:]>0).all()for v in outputs.values());assert not any(v.requires_grad or v.grad is not None for v in model.parameters())
   cost=dict(case_seconds=time.perf_counter()-tick,actual_trajectory_decoder_calls=calls,actual_PosDecoder_calls=2*calls,new_Source_captures=0,new_DINO_calls=0,backward_calls=0)
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=old['frame_ids'],planned_positions=old['planned_positions'],interval=old['interval'],active=active,positions=positions,targets=targets,binding_degenerate=len(positions)<2,Source=S,E=E,outputs=outputs,raw_calls=partial,zero_native_exact=True,pair_equal_input_E_exact=torch.equal(paired_readout(native,native,E,old['planned_positions']),E),GT_read=False,frozen_model_hash=modelhash,compute=cost)))
   if ix<2:pilot.append(dict(alias=a,case_seconds=cost['case_seconds'],active=active,nonzero_unobserved_persistent_delta=not torch.equal(persistent,E),zero_native_exact=True))
   if ix==1:
    estimate=time.perf_counter()-start+15*sum(r['case_seconds']for r in pilot)+60;qualified=any(r['active']and r['nonzero_unobserved_persistent_delta']for r in pilot)and estimate<p['GPU_seconds_cap'];put(D/'PILOT2_TIMING.json',dict(status='measured',cases=pilot,elapsed_seconds=time.perf_counter()-start,estimated_total_upper_seconds=estimate,not_a_guarantee=True,qualified=qualified))
    if not qualified:put(D/'PILOT_STOP.json',dict(status='stopped',reason='no unobserved response in supported pilots or budget infeasible',no_NEW_GT_scoring=True,preserved_count=len(files)));return
   print(a,ix+1,round(time.perf_counter()-start,3),calls,flush=True);partial={};del inputs,old,rc,src,x,S,E,native,raw;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.state_dict())==modelhash;verify(full=True);check();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_outputs_before_NEW_GT=True,GPU_allocation_seconds=time.perf_counter()-start+3,new_Source_captures=0,new_DINO_calls=0,no_network_backward=True));status('all32_sealed',32,GT_read=False);complete=True
 finally:
  try:
   if not complete and partial:save(D/'partial'/(str(a)+'.pt'),dict(alias=a,raw_calls=partial,GT_read=False,complete=False))
  finally:
   try:put(D/'GPU_COST_RECEIPT.json',dict(status='released',GPU_allocation_seconds=time.perf_counter()-start+3,complete32=complete))
   finally:lease.close()

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 torch.set_num_threads(4)
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_anchor_refresh32_v1 import ARMS
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_outputs_before_NEW_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/realizable_query_opd32/RESULTS_PUBLIC.json')['rows']};rows=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q])
  for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
   xx,yy=m['values']['E_MBR'][k],prior[a]['values']['E_MBR'][k];assert xx is None and yy is None or xx is not None and yy is not None and abs(xx-yy)<1e-12
  E=o['outputs']['E_MBR'];diagnostics={key:dict(mean_per_node_center_L2=float(np.linalg.norm(v[:,:2]-E[:,:2],axis=1).mean()),whole_tube_L2=float(np.linalg.norm(v-E)),max_absolute_coordinate_shift=float(np.max(abs(v-E))),changed_actual_nodes=int(np.any(v!=E,axis=1).sum()))for key,v in o['outputs'].items()if key!='Source_MBR'}
  rows.append(dict(alias=a,dataset=ds,active=o['active'],retained_anchors=len(o['positions']),shuffle_degenerate=o['binding_degenerate'],diagnostics=diagnostics,compute=o['compute'],**m))
 contrasts=[(a,'E_MBR')for a in ARMS if a!='E_MBR']+[('Persistent_delta',a)for a in ['Initial_delta','Persistent_shuffle','Anchor_local','Persistent_allnodes']];groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,active=sum(r['active']for r in rr),metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in ['v','t','s','observed_actual_s','unobserved_actual_s']}for a in ARMS},paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in ['v','s','observed_actual_s','unobserved_actual_s']}for a,b in contrasts})
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,GT_offline_after_seal=True,development_exposed=True,independent_confirmation=False,new_Source_captures=0,new_DINO_calls=0,new_OPD=0,pretrained_weights_frozen=True));passed=all(groups[ds]['paired']['Persistent_delta_minus_E_MBR'][e]['mean']>0 for ds in gt for e in ['v','unobserved_actual_s'])and all(groups[ds]['paired']['Persistent_delta_minus_'+a]['v']['mean']>0 for ds in gt for a in ['Initial_delta','Persistent_shuffle']);put(P/'DECISION.json',dict(status='evaluated',passed=passed,rule=p['decision'],OPD_not_run=True));put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print('gate',passed,{ds:groups[ds]['paired']['Persistent_delta_minus_E_MBR']['v']['mean']for ds in gt},flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),partials_preserved=True));raise
