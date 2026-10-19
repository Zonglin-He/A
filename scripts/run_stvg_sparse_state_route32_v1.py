"""Bounded native-state route experiment with cached sparse DINO labels."""
import os,sys,time,gc,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_sparse_state_route32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/sparse_state_route32';CLOSED=ROOT/'artifacts/stvg_dino_query_closedloop32_v1';FEAT=ROOT/'artifacts/stvg_discriminative_dense32_v1';SRC=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1/source'
CODE=['vg_tta/stvg_sparse_state_route32_v1.py','scripts/run_stvg_sparse_state_route32_v1.py','scripts/continue_stvg_sparse_state_route32_v1.py','protocols/stvg_sparse_state_route32_v1.md'];VER=None

def verify(live=False,full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 pins=dict(r['pins'])
 if (D/'PREFLIGHT_RECOVERY001.json').exists():
  fix=read(D/'PREFLIGHT_RECOVERY001.json');assert sha(D/'RUNTIME_LOCK.json')==fix['original_runtime_sha256'];assert sha(D/'PLAN_LOCK.json')==fix['original_plan_sha256'];pins.update(fix['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard()
 if live:assert time.time()<p['deadline_epoch']
 return p

def prepare():
 from vg_tta.stvg_sparse_state_route32_v1 import properties,ARMS
 assert not D.exists();D.mkdir();P.mkdir();old=read(ROOT/'artifacts/stvg_realizable_query_opd32_v1/PLAN_LOCK.json');records=old['records'];assert len(records)==len({r['parent']for r in records})==32
 files=CODE+['vg_tta/stvg_dino_query_closedloop32_v1.py','vg_tta/stvg_opd_frozen_capture_cpu007.py','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_opd_p1_v1.py','vg_tta/tastvg_oracle_event5_v1.py','scripts/score_stvg_dino_mask_align32_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py','configs/stvg_fine_center_dataset_candidate_v1.json','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','results/stvg_error_state/2026-10-19/fine_dataset_formal64/RESULTS_PUBLIC.json','artifacts/stvg_fine_dataset_formal64_v1/GLOBAL_PREDICTION_BARRIER.json']
 # Pin actual model-loading code/checkpoints and decoder dependencies, no new expert use.
 oldpins=read(ROOT/'artifacts/stvg_realizable_query_opd32_v1/RUNTIME_LOCK.json')['pins'];files+= [f for f in oldpins if f.startswith(('external/TA-STVG/','methods/decota_final_simplified_v1/','checkpoints/'))or f in ['scripts/run_decota_paper_main_v1.py','scripts/run_spatial_regression_alignment_v1.py','scripts/run_spatial_ssl_gpu_v1.py','scripts/run_final_simplification_v1.py']]
 for r in records:
  a=r['alias'];ds=r['dataset'];files +=[str((root/sub/(a+'.pt')).relative_to(ROOT))for root,sub in [(CLOSED,'replay_inputs'),(FEAT,'features')]]+[str((SRC/(a+'.pt')).relative_to(ROOT))]
  r['fine_path']=f'artifacts/'+('stvg_fine_roi_sensitivity32_v1'if ds=='hc2'else'stvg_fine_roi_scoped_opd32_v1')+f'/predictions/{a}.pt';r['fine_arm']='Step_high'if ds=='hc2'else'Fine_live';r['fine_result']='results/stvg_error_state/2026-10-19/'+('fine_roi_sensitivity32'if ds=='hc2'else'fine_roi_scoped_opd32')+'/RESULTS_PUBLIC.json';files +=[r['fine_path'],r['fine_result']]
 put(D/'PLAN_LOCK.json',dict(status='locked',count=32,per_domain=16,records=records,arms=ARMS,state='final layer native ca_qcontent_proj transient shared bias256',scale='native final qcontent channel RMS, sigma*m',teacher='raw cached E-selected DINO boxes, equal retained anchors',loss='5*L1sum4+2*(1-GIoU)',OPD='same loss tilt exp(-loss), current/proposal Gaussian RN, control variate score gradient',rounds=2,draws=16,final_step=2,Adam=dict(hc2=.01,vidstg=.03,betas=[.9,.999],eps=1e-8),GPU_cap_seconds=900,deadline_epoch=time.time()+3600,pilot_aliases=['NH001','NV001'],new_Source_DINO_experts_training_head=0,all_outputs_before_GT=True,formal64_excluded=True,development_exposed=True,no_parameter_search=True))
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in set(files)}));put(D/'CPU_PROPERTIES.json',properties());baseline=[f for f in set(files)if f.startswith(('configs/stvg_fine_center','results/stvg_error_state/2026-10-19/fine_'))or f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']];put(D/'FINE_BASELINE_FREEZE.json',dict(status='archived_research_baseline',files={f:sha(ROOT/f)for f in baseline},unchanged=True,no_production_promotion=True));print('locked32',flush=True)

def measure():
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_frozen_capture_cpu007 import transfer
 from vg_tta.stvg_sparse_state_route32_v1 import Decoder,noise,fit,FIT_ARMS
 p=verify(True,True);lease=gpu();start=time.perf_counter();model=None;domain=None;dec=None;done=False;files=[];pilots=[];a=None
 def check():
  verify(True);assert time.perf_counter()-start+3<p['GPU_cap_seconds'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  records=[next(r for r in p['records']if r['alias']==a)for a in p['pilot_aliases']]+[r for r in p['records']if r['alias']not in p['pilot_aliases']]
  for ix,r in enumerate(records):
   check();a=r['alias'];ds=r['dataset'];tick=time.perf_counter()
   if domain!=ds:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;modelhash=state_hash(model.state_dict())
   rc=load(CLOSED/'replay_inputs'/(a+'.pt'));src=load(SRC/(a+'.pt'));ft=load(FEAT/'features'/(a+'.pt'));fine=load(ROOT/r['fine_path']);S=src['Source'];E=ft['E'].double();assert torch.equal(rc['Source'],S)and np.array_equal(fine['outputs']['E_MBR'],E.numpy());inputs=transfer(rc['spatial_inputs'],'cuda');dec=Decoder(model.ground_decoder.decoder,inputs,S,E,check);positions=[f['position']for f,k in zip(ft['frames'],ft['selected'])if k>=0];targets=np.asarray([f['boxes'][k].tolist()for f,k in zip(ft['frames'],ft['selected'])if k>=0]).reshape(-1,4)
   probe=dec.gradient_check(a)if ix<2 else None;eps=noise(a);fits={arm:fit(dec,eps,positions,targets,arm,p['Adam'][ds],check)for arm in FIT_ARMS};f,l=fits['State_fixed'],fits['State_live'];assert np.array_equal(f['path'][1]['state'],l['path'][1]['state'])and np.array_equal(f['path'][1]['boxes'],l['path'][1]['boxes']);assert np.array_equal(f['rounds'][0]['gradient'],l['rounds'][0]['gradient'])
   outputs=dict(E_MBR=E.numpy(),Fine_center=fine['outputs'][r['fine_arm']],**{k:v['final']for k,v in fits.items()});assert all(np.isfinite(v).all()and(v[:,2:]>0).all()for v in outputs.values());assert all(not v.requires_grad and v.grad is None for v in model.parameters());assert state_hash(model.state_dict())==modelhash
   cost=dict(case_elapsed_seconds=time.perf_counter()-tick,actual_full_trajectory_decoder_calls=dec.calls,actual_PosDecoder_calls=2*dec.calls,actual_decoder_seconds=dec.model_seconds,backward_calls=dec.backward_calls,probe_backward_calls=int(probe is not None),prefix_native_checks=dec.prefix_checks,new_Source_DINO_experts_training=0)
   files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,active=bool(positions),positions=positions,targets=targets,frame_ids=ft['frame_ids'],planned_positions=ft['planned_positions'],interval=ft['interval'],Source=S,E=E,outputs=outputs,fits=fits,sigma=dec.scale,projection_native_RMS=dec.scale,noise=eps,gradient_probe=probe,model_hash=modelhash,compute=cost,GT_read=False,zero_Source_E_exact=True)))
   if ix<2:pilots.append(dict(alias=a,case_seconds=cost['case_elapsed_seconds'],gradient_probe=probe,native_scale=dec.scale,actual_response_L2=float(np.linalg.norm(l['final']-E.numpy()))))
   dec.close();dec=None;del inputs,rc,src,ft,fine,fits,f,l;gc.collect();torch.cuda.empty_cache();print(a,ix+1,round(time.perf_counter()-start,3),cost['actual_full_trajectory_decoder_calls'],flush=True)
   if ix==1:
    estimate=time.perf_counter()-start+15*sum(v['case_seconds']for v in pilots)+60;put(D/'PILOT2_TIMING.json',dict(cases=pilots,estimated_total_seconds=estimate,not_guaranteed=True,within_GPU_cap=estimate<p['GPU_cap_seconds']))
    if estimate>=p['GPU_cap_seconds']:raise RuntimeError('prelocked total-cost estimate exceeds900s, no new scale/parameter attempt')
  assert state_hash(model.state_dict())==modelhash;verify(full=True);check();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_actions_states_before_GT=True,GPU_allocation_seconds=time.perf_counter()-start+3,actual_worker_GPU_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_Source_DINO_experts_training=0));done=True
 finally:
  if dec is not None:dec.close()
  put(D/'GPU_COST_RECEIPT.json',dict(status='released',complete32=done,GPU_allocation_seconds=time.perf_counter()-start+3,partial_count=len(files)));lease.close()

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_sparse_state_route32_v1 import ARMS
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_actions_states_before_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 for r in p['records']:
  z=load(D/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];m=metrics(z,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);old=next(v for v in read(ROOT/r['fine_result'])['rows']if v['alias']==r['alias'])
  for a,b in [('E_MBR','E_MBR'),('Fine_center',r['fine_arm'])]:
   for e in ['v','t','s','unobserved_actual_s','observed_actual_s']:assert m['values'][a][e]is None and old['values'][b][e]is None or abs(m['values'][a][e]-old['values'][b][e])<1e-10
  di={k:dict(loss_before=[t['loss_before']for t in f['rounds']],loss_after=[t['loss_after']for t in f['rounds']],updated_steps=sum(t['updated']for t in f['rounds']),state_L2=float(np.linalg.norm(f['path'][-1]['state'])),output_L2_from_E=float(np.linalg.norm(f['final']-z['outputs']['E_MBR'])),changed_size_nodes=int(np.any(f['final'][:,2:]!=z['outputs']['E_MBR'][:,2:],1).sum()),teacher_ESS=[float(1/np.sum(t['weights']['teacher']**2))if t['weights']is not None else None for t in f['rounds']])for k,f in z['fits'].items()}
  rows.append(dict(alias=r['alias'],dataset=ds,active=z['active'],retained_references=len(z['positions']),sigma=z['sigma'],fits=di,compute=z['compute'],gradient_probe=z['gradient_probe'],**m))
 contrasts=[(a,'E_MBR')for a in ARMS[1:]]+[('State_live',b)for b in ['Fine_center','Direct_anchor','State_fixed']]+[('Direct_anchor','Fine_center')];ep=['v','t','s','unobserved_actual_s','observed_actual_s','benchmark_interpolated_s'];groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(parents=len(rr),active=sum(r['active']for r in rr),means={a:{e:mean([r['values'][a][e]for r in rr])for e in ep}for a in ARMS},paired={a+'_minus_'+b:{e:paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rr])for e in ep}for a,b in contrasts})
 verify(full=True);put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,all_outputs_before_GT=True,formal64_excluded=True,new_Source_DINO_experts_training=0,development_exposed=True,independent_confirmation=False));put(D/'SCORE_COMPLETION.json',dict(status='complete',count=32));print('score complete',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,error_type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
