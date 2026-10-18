"""Bounded actual latent decoder rollouts; zero expert/capture/model training."""
from pathlib import Path
import os,sys,time,argparse,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_realizable_query_opd32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/realizable_query_opd32';PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';CLOSED=ROOT/'artifacts/stvg_dino_query_closedloop32_v1';FEAT=ROOT/'artifacts/stvg_discriminative_dense32_v1';VERIFIER=None
CODE=['vg_tta/stvg_realizable_query_opd32_v1.py','scripts/run_stvg_realizable_query_opd32_v1.py','scripts/continue_stvg_realizable_query_opd32_v1.py','protocols/stvg_realizable_query_opd32_v1.md']
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
 from vg_tta.stvg_realizable_query_opd32_v1 import properties,ARMS
 old=read(ROOT/'artifacts/stvg_bijection_layout_opd32_v1/PLAN_LOCK.json');assert read(ROOT/'artifacts/stvg_bijection_layout_opd32_v1/FINAL_COMPLETION.json')['status']=='complete';night=read(ROOT/'artifacts/stvg_opd_overnight_20261018_v1/STATE.json');D.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True)
 put(D/'CPU_PROPERTIES.json',properties());p=dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,mechanism='actual frozen decoder QR256 rollouts; standardized latent Gaussian mean control-variate importance update; no pathwise decoder derivative',new_Source_captures=0,new_DINO_calls=0,new_training_runs=0,new_heads=0,pretrained_weights_frozen=True,external_mean_only256=True,rounds=2,Gaussian_draws_per_round=16,zero_diagnostic_not_integrated=True,temperature=.1,step=.5,scale='native original query_tgt channelRMS/sqrt256',pilot_aliases=['NH001','NV001'],GPU_seconds_cap=900,overnight_GPU_remaining_seconds=7200-night['new_GPU_spent_seconds'],created_epoch=time.time(),deadline_epoch=min(time.time()+1800,night['deadline_epoch']),MBR_fixed=True,E_retained=True,GT_after_full32_seal=True,development_exposed=True,independent_confirmation=False,no_parameter_search=True,decision='Both-domain Latent_live minus E complete-v and unobserved actual-s means positive, and Live minus Fixed/Search/MeanControl/Shuffle complete-v positive; CIs retain uncertainty')
 put(D/'PLAN_LOCK.json',p);pins=dict(read(ROOT/'artifacts/stvg_bijection_layout_opd32_v1/RUNTIME_LOCK.json')['pins'])
 for f in CODE+['vg_tta/stvg_dino_query_closedloop32_v1.py','vg_tta/stvg_opd_frozen_capture_cpu007.py','scripts/run_stvg_dino_local_residual32_v1.py','external/TA-STVG/models/grounding_model/query_decoder.py','scripts/run_decota_paper_main_v1.py']:pins[f]=sha(ROOT/f)
 for r in p['records']:
  for root,sub in [(CLOSED,'replay_inputs'),(CLOSED,'predictions'),(PRE,'source'),(PRE,'E'),(PRE,'predictions')]:
   f=str((root/sub/(r['alias']+'.pt')).relative_to(ROOT));pins[f]=sha(ROOT/f)
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));status('prepared',0,GT_read=False)

def measure():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
 from vg_tta.stvg_realizable_query_opd32_v1 import Decoder,noise,fit,reward,weights,FIT_ARMS
 p=verify(True);lease=None
 while lease is None:
  verify(True)
  try:lease=gpu()
  except BlockingIOError:status('waiting_shared_GPU_lease',0);time.sleep(15)
 start=time.perf_counter();model=None;domain=None;files=[];pilot=[];modelhash=None;done=False;dec=None;a=None
 def check():
  verify(True);assert time.perf_counter()-start+3<min(p['GPU_seconds_cap'],p['overnight_GPU_remaining_seconds']);assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  put(D/'GPU_RUNTIME.json',dict(pid=os.getpid(),start_epoch=time.time(),shared_lease=True));records=[next(r for r in p['records']if r['alias']==a)for a in p['pilot_aliases']]+[r for r in p['records']if r['alias']not in p['pilot_aliases']]
  for ix,r in enumerate(records):
   check();a=r['alias'];ds=r['dataset'];tick=time.perf_counter();status('actual_realizable_latent_rollouts',ix,alias=a,GT_read=False)
   if domain!=ds:
    if model is not None:
     assert state_hash(model.state_dict())==modelhash;del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;modelhash=state_hash(model.state_dict())
   old=load(CLOSED/'predictions'/(a+'.pt'));rc=load(CLOSED/'replay_inputs'/(a+'.pt'));src=load(PRE/'source'/(a+'.pt'));x=load(FEAT/'features'/(a+'.pt'));E=x['E'].double();S=src['Source'];assert torch.equal(rc['Source'],S)and np.array_equal(old['outputs']['E_MBR'],E.numpy());inputs=transfer(rc['spatial_inputs'],'cuda');dec=Decoder(model.ground_decoder.decoder,inputs,S,E,check=check);zero=dec(np.zeros(256));assert zero['valid']and torch.equal(zero['native'],S)and np.array_equal(zero['boxes'],E.numpy())
   positions=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];targets=np.array([f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0]).reshape(-1,4);eps=noise(a);fits={};ft=time.perf_counter()
   for mode in FIT_ARMS:fits[mode]=fit(dec,eps,positions,targets,mode,check)
   fitsecs=time.perf_counter()-ft;fixed=fits['Latent_fixed'];live=fits['Latent_live'];assert np.array_equal(fixed['path'][1]['state'],live['path'][1]['state'])and np.array_equal(fixed['path'][1]['boxes'],live['path'][1]['boxes'])
   acts=np.concatenate([t['actions']for t in fixed['rounds']]);rr=np.concatenate([t['rewards']for t in fixed['rounds']]);valid=np.concatenate([t['valid']for t in fixed['rounds']]);baseReward=reward(E.numpy(),positions,targets);scores=np.r_[baseReward,np.where(valid,rr,-np.inf)];best=int(scores.argmax());search=zero if best==0 else dec(acts[best-1]);w=weights(acts,np.zeros(256),np.zeros(256),rr,valid)if positions else weights(acts,np.zeros(256),np.zeros(256),np.zeros(len(acts)),valid);meanstate=w['delta'];mc=dec(meanstate);meanrejected=not mc['valid']
   if meanrejected:meanstate=np.zeros(256);mc=zero
   outputs=dict(Source_MBR=S.double().numpy(),E_MBR=E.numpy(),Endpoint_local=old['outputs']['Center_full_local'],Latent_search=search['boxes'],Latent_mean_control=mc['boxes'],**{k:f['final']for k,f in fits.items()});assert all(v.shape==E.shape and np.isfinite(v).all()for v in outputs.values());assert not any(v.requires_grad or v.grad is not None for v in model.parameters());dh=state_hash(model.ground_decoder.state_dict());active=bool(positions)
   cost=dict(case_seconds=time.perf_counter()-tick,three_fit_seconds=fitsecs,actual_decoder_trajectory_calls=dec.calls,actual_PosDecoder_calls=2*dec.calls,actual_decoder_seconds=dec.model_seconds,new_Source_captures=0,new_DINO_calls=0,backward_calls=0)
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=old['frame_ids'],planned_positions=old['planned_positions'],interval=old['interval'],active=active,positions=positions,targets=targets,binding_degenerate=len(positions)<2,sigma=dec.sigma,query_channel_RMS=dec.query_rms,Source=S,E=E,noise=eps,outputs=outputs,fits=fits,cache=list(dec.cache.values()),search=dict(best_index=best,scores=scores,actions=acts,base_reward=baseReward),mean_control=dict(state=meanstate,weights=w,invalid_mean_rejected=meanrejected),zero_native_exact=True,zero_E_exact=True,GT_read=False,frozen_model_hash=modelhash,frozen_decoder_hash=dh,compute=cost)))
   if ix<2:
    trial=fixed['rounds'][0];response=any(v['valid']and not np.array_equal(v['boxes'],E.numpy())for v in trial['actual_candidates']);informative=bool(trial['weights']['informative']);pilot.append(dict(alias=a,case_seconds=cost['case_seconds'],sigma=dec.sigma,active=active,nonzero_decoded_response=response,nonconstant_feedback=informative))
   if ix==1:
    estimate=time.perf_counter()-start+15*sum(r['case_seconds']for r in pilot)+60;qualified=not(all(not r['nonzero_decoded_response']or not r['nonconstant_feedback']for r in pilot if r['active']))and estimate<p['GPU_seconds_cap'];put(D/'PILOT2_TIMING.json',dict(status='measured',cases=pilot,elapsed_seconds=time.perf_counter()-start,estimated_total_upper_seconds=estimate,not_a_guarantee=True,qualified=qualified))
    if not qualified:
     put(D/'PILOT_STOP.json',dict(status='stopped',reason='fixed-scale decoder/feedback response or budget qualification failed; no sigma rescue',no_NEW_GT_scoring=True,preserved_count=len(files)));return
   print(a,ix+1,round(time.perf_counter()-start,3),cost['actual_decoder_trajectory_calls'],flush=True);del inputs,old,rc,src,x,S,E,fits,zero,fixed,live,search,mc;dec=None;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.state_dict())==modelhash;verify(full=True);check();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_outputs_before_NEW_GT=True,GPU_allocation_seconds=time.perf_counter()-start+3,actual_worker_GPU_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_Source_captures=0,new_DINO_calls=0,no_network_backward=True));status('all32_sealed',32,GT_read=False);done=True
 finally:
  if not done and dec is not None and dec.cache:save(D/'partial'/(str(a)+'.pt'),dict(alias=a,cache=list(dec.cache.values()),sigma=dec.sigma,GT_read=False,complete=False))
  put(D/'GPU_COST_RECEIPT.json',dict(status='released',GPU_allocation_seconds=time.perf_counter()-start+3,complete32=done));lease.close()


def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 torch.set_num_threads(4)
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_realizable_query_opd32_v1 import ARMS
 p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_outputs_before_NEW_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/roi_layout_opd32/RESULTS_PUBLIC.json')['rows']};rows=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q])
  for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
   xx,yy=m['values']['E_MBR'][k],prior[a]['values']['E_MBR'][k];assert xx is None and yy is None or xx is not None and yy is not None and abs(xx-yy)<1e-12
  fits={key:dict(state_standardized_L2=float(np.linalg.norm(f['path'][-1]['state'])),raw_u_L2=float(o['sigma']*np.linalg.norm(f['path'][-1]['state'])),informative_steps=sum(t['weights']['informative']for t in f['rounds']),invalid_mean_rejected=sum(t['invalid_mean_rejected']for t in f['rounds']),output_from_E_L2=float(np.linalg.norm(f['final']-o['outputs']['E_MBR'])),teacher_ESS=[float(1/np.sum(t['weights']['teacher']**2))if np.sum(t['weights']['teacher']**2)>0 else None for t in f['rounds']],base_ESS=[float(1/np.sum(t['weights']['base']**2))if np.sum(t['weights']['base']**2)>0 else None for t in f['rounds']])for key,f in o['fits'].items()}
  rows.append(dict(alias=a,dataset=ds,active=o['active'],retained_anchors=len(o['positions']),shuffle_degenerate=o['binding_degenerate'],query_channel_RMS=o['query_channel_RMS'],sigma=o['sigma'],fits=fits,compute=o['compute'],**m))
 contrasts=[(a,'E_MBR')for a in ARMS if a!='E_MBR']+[('Latent_live',a)for a in ['Latent_fixed','Latent_shuffle','Latent_search','Latent_mean_control','Endpoint_local']];groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=16,active=sum(r['active']for r in rr),metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in ['v','t','s','observed_actual_s','unobserved_actual_s']}for a in ARMS},paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in ['v','s','observed_actual_s','unobserved_actual_s']}for a,b in contrasts})
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,GT_offline_after_seal=True,development_exposed=True,independent_confirmation=False,new_Source_captures=0,new_DINO_calls=0,pretrained_weights_frozen=True));needed=['Latent_live_minus_'+a for a in ['E_MBR','Latent_fixed','Latent_search','Latent_mean_control','Latent_shuffle']];passed=all(groups[ds]['paired'][c]['v']['mean']>0 for ds in gt for c in needed)and all(groups[ds]['paired'][needed[0]]['unobserved_actual_s']['mean']>0 for ds in gt);put(P/'DECISION.json',dict(status='evaluated',passed=passed,rule=p['decision'],no_network_backward=True));put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print('gate',passed,{ds:groups[ds]['paired']['Latent_live_minus_E_MBR']['v']['mean']for ds in gt},flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),partials_preserved=True));raise
