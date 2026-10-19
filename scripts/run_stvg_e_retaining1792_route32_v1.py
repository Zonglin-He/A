"""Finite E-retaining original1792 experiment. Real barrier before GT control."""
import os,sys,time,gc,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_e_retaining1792_route32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/e_retaining1792_route32'
PRE=ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1';CLOSED=ROOT/'artifacts/stvg_dino_query_closedloop32_v1';FEAT=ROOT/'artifacts/stvg_discriminative_dense32_v1';VER=None
ARMS=['Direct','Fixed','Live'];STAGES=['real','oracle'];CODE=['vg_tta/stvg_e_retaining1792_v1.py','scripts/run_stvg_e_retaining1792_route32_v1.py','protocols/stvg_e_retaining1792_route32_v1.md']

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
 pins=dict(l['pins'])
 for amendment in sorted((D/'amendments').glob('*.json')):
  a=read(amendment);assert a['original_runtime_sha256']==sha(D/'RUNTIME_LOCK.json');pins.update(a['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return p

def prepare():
 import numpy as np,torch
 assert not D.exists();D.mkdir();P.mkdir();rs=read(PRE/'PLAN_LOCK.json')['records'];cfg={ds:read(ROOT/'methods/decota_spatial_opd_v1/configs.json')['datasets'][ds]['config']for ds in ['hc2','vidstg']};files=list(CODE)+['methods/decota_spatial_opd_v1/configs.json','vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/stvg_dino_query_closedloop32_v1.py','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py','results/stvg_error_state/2026-10-19/fine_dataset_formal64/RESULTS_PUBLIC.json'];oldpins=read(ROOT/'artifacts/stvg_sparse_state_route32_v1/RUNTIME_LOCK.json')['pins'];files +=[f for f in oldpins if f.startswith(('external/TA-STVG/','methods/decota_final_simplified_v1/','checkpoints/'))or f in ['scripts/run_decota_paper_main_v1.py','scripts/run_spatial_regression_alignment_v1.py','scripts/run_spatial_ssl_gpu_v1.py','scripts/run_final_simplification_v1.py']];support=[]
 for r in rs:
  a=r['alias'];ft=load(FEAT/'features'/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));rc=load(CLOSED/'replay_inputs'/(a+'.pt'));assert torch.equal(s['Source'],rc['Source'])and np.array_equal(ft['E'].numpy(),e['E']);anchors=[]
  for f,k in zip(ft['frames'],ft['selected']):
   if k>=0:anchors.append(dict(position=f['position'],box=f['boxes'][k].tolist(),weight=1.))
  assert len(anchors)==e['estimate']['support_frames'];save(D/'inputs_real'/(a+'.pt'),dict(alias=a,anchors=anchors,GT_read=False,reference='existing E-selected raw original DINO'))
  support.append(dict(alias=a,dataset=r['dataset'],references=len(anchors)))
  files +=[str(f.relative_to(ROOT))for f in [FEAT/'features'/(a+'.pt'),PRE/'source'/(a+'.pt'),PRE/'E'/(a+'.pt'),CLOSED/'replay_inputs'/(a+'.pt'),D/'inputs_real'/(a+'.pt')]]
 put(D/'PLAN_LOCK.json',dict(status='locked_before_real_updates_or_GT_control',records=rs,count=32,arms=ARMS,configurations=cfg,pilot_aliases=['NH001','NV001'],GPU_cap_seconds=900,wall_deadline_epoch=time.time()+3600,new_Source_DINO_experts_training=0,reference='original E-selected, no repaired what outputs',real_support_not_GT_filtered=True,oracle_support='same E-selected positions with exact known GT only; unknown omitted only in oracle',no_automatic_followup=True,created_epoch=time.time()))
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in sorted(set(files))}));put(P/'CONFIG_PUBLIC.json',dict(count=32,per_domain=16,arms=ARMS,configurations=cfg,parameter_count=1792,temporary_state='QR256+last spatial block norm1/norm3/norm4 affine1536',reference_support=support,GT_real_support_selection=False,oracle_known_support_caveat=True,sampling='original native logit Gaussian, candidates mapped to E before IoU',real32_seal_before_GT_control=True,original_E_time_and_production_unchanged=True,independent_confirmation=False,no_parameter_scan=True));print('LOCKED',32,'real refs',sum(r['references']for r in support),flush=True)

def oracle_inputs():
 import torch
 p=verify(True);bar=read(D/'REAL_PREDICTION_BARRIER.json');assert bar['count']==32;[load(ROOT/f['path'])for f in bar['files']]
 from scripts.score_stvg_opd_p1_v1 import truths
 g={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in g};files=[];support=[]
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];real=load(D/'inputs_real'/(a+'.pt'));src=load(PRE/'source'/(a+'.pt'));row=plans[ds][q];gt=g[ds][0][q];anchors=[]
  for old in real['anchors']:
   pos=old['position'];fid=src['frame_ids'][pos]
   if fid not in gt:continue
   x,y,x2,y2=gt[fid];W,H=row['input']['width'],row['input']['height'];anchors.append(dict(position=pos,box=[(x+x2)/(2*W),(y+y2)/(2*H),(x2-x)/W,(y2-y)/H],weight=1.))
  files.append(save(D/'inputs_oracle'/(a+'.pt'),dict(alias=a,anchors=anchors,GT_used=True,only_known_exact_GT=True)));support.append(dict(alias=a,dataset=ds,real_references=len(real['anchors']),oracle_known_references=len(anchors),omitted_unknown=len(real['anchors'])-len(anchors)))
 put(D/'ORACLE_INPUT_BARRIER.json',dict(status='complete',count=32,files=files,GT_assisted_diagnostic=True));put(P/'ORACLE_SUPPORT_PUBLIC.json',dict(count=32,rows=support,real_prediction_barrier_sha256=sha(D/'REAL_PREDICTION_BARRIER.json'),GT_not_used_in_real_support=True,coordinate_quality_and_support_omission_not_purely_separated=True));print('oracle inputs sealed',flush=True)

def measure(stage):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash,detached
 from vg_tta.stvg_e_retaining1792_v1 import Replay1792,fit
 p=verify(True);assert stage in STAGES
 if stage=='oracle':assert read(D/'REAL_PREDICTION_BARRIER.json')['count']==read(D/'ORACLE_INPUT_BARRIER.json')['count']==32
 assert not (D/(stage.upper()+'_PREDICTION_BARRIER.json')).exists();previous=sum(read(f)['GPU_allocation_seconds']for f in (D/'allocations').glob('*.json'));lease=gpu();start=time.perf_counter();domain=None;model=None;rp=None;files=[];pilots=[];done=False
 def check():
  verify();assert time.time()<p['wall_deadline_epoch'];assert previous+time.perf_counter()-start+3<900;assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  records=[next(r for r in p['records']if r['alias']==a)for a in p['pilot_aliases']]+[r for r in p['records']if r['alias']not in p['pilot_aliases']]
  for ix,r in enumerate(records):
   check();tick=time.perf_counter();a=r['alias'];ds=r['dataset']
   existing=D/stage/(a+'.pt')
   if existing.exists():
    z=load(existing);assert z['alias']==a and z['GT_used']==(stage=='oracle');files.append(read(existing.with_suffix('.receipt.json')));print(stage,a,'reuse sealed',flush=True);continue
   inp=load(D/('inputs_'+stage)/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));rc=load(CLOSED/'replay_inputs'/(a+'.pt'))
   if domain!=ds:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;mh=state_hash(model.state_dict())
   rp=Replay1792(model,rc,s['Source'],e['E'],check);fits={arm:fit(rp,inp['anchors'],a,arm,p['configurations'][ds])for arm in ARMS};f,l=fits['Fixed'],fits['Live'];assert all(torch.equal(f['initial'][n],l['initial'][n])for n in f['initial']);assert torch.equal(f['path'][min(1,len(f['path'])-1)]['boxes'],l['path'][min(1,len(l['path'])-1)]['boxes'])
   if f['rounds']:
    for k in ['samples','rewards','weights']:assert torch.equal(f['rounds'][0]['rollout'][k],l['rounds'][0]['rollout'][k])
    assert f['rounds'][0]['updated']==l['rounds'][0]['updated']
   outputs=dict(E_MBR=e['E'],**{k:v['final'].numpy()for k,v in fits.items()});assert all(np.isfinite(b).all()and(b[:,2:]>0).all()for b in outputs.values());assert all(not v.requires_grad and v.grad is None for v in model.parameters())and state_hash(model.state_dict())==mh
   cost=dict(seconds=time.perf_counter()-tick,full_actual_trajectory_decoder_calls=rp.calls,PosDecoder_calls=2*rp.calls,decoder_seconds=rp.seconds,backward_calls=rp.backward_calls,new_Source_DINO_experts_training=0)
   files.append(save(D/stage/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=e['frame_ids'],planned_positions=e['planned_positions'],interval=e['interval'],Source=s['Source'],E=e['E'],outputs=outputs,fits=fits,positions=[x['position']for x in inp['anchors']],targets=inp['anchors'],GT_used=stage=='oracle',GT_free_support=stage=='real',real_barrier_prehoc=stage=='oracle',model_hash=mh,zero_exact_E=True,compute=cost)));print(stage,a,ix+1,'refs',len(inp['anchors']),'seconds',round(cost['seconds'],3),'total',round(previous+time.perf_counter()-start,2),flush=True)
   if ix<2:pilots.append(cost['seconds'])
   del rp,rc,inp,s,e,fits,f,l,outputs;rp=None;gc.collect();torch.cuda.empty_cache()
   if ix==1:
    estimate=previous+time.perf_counter()-start+15*sum(pilots)+30;put(D/('PILOT2_'+stage+'.json'),dict(case_seconds=pilots,estimated_stage_finish_total_GPU_seconds=estimate,not_guaranteed=True,within_cap=estimate<900));assert estimate<900,'original-budget pilot exceeds cap; preserve partials, no scan'
  verify(True);check();put(D/(stage.upper()+'_PREDICTION_BARRIER.json'),dict(status='complete',count=32,logical_outputs=96,files=files,GT_used=stage=='oracle',real_support_not_GT_filtered=stage=='real',GPU_allocation_seconds=time.perf_counter()-start+3,peak_allocated_bytes=torch.cuda.max_memory_allocated()));done=True
 finally:
  put(D/'allocations'/(stage+'_'+str(time.time_ns())+'.json'),dict(stage=stage,complete=done,count=len(files),GPU_allocation_seconds=time.perf_counter()-start+3));lease.close()

def score():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify(True);bars={s:read(D/(s.upper()+'_PREDICTION_BARRIER.json'))for s in STAGES};assert all(b['count']==32 for b in bars.values());zs={s:[load(ROOT/f['path'])for f in bars[s]['files']]for s in STAGES};gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 for stage in STAGES:
  for z in zs[stage]:
   a=z['alias'];ds=z['dataset'];r=next(r for r in p['records']if r['alias']==a);q=r['query_ordinal'];m=metrics(z,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);di={}
   for arm,f in z['fits'].items():
    rr=f['rounds'];di[arm]=dict(active_parameters=f['active_parameters'],actual_rounds=f['actual_rounds'],gradient_calls=f['gradient_calls'],anchor_loss_initial=f['path'][0]['anchor_loss'],anchor_loss_final=f['path'][-1]['anchor_loss'],state_L2=float(torch.cat([(f['state'][n]-f['initial'][n]).flatten()for n in f['initial']]).norm()),changed_nodes=int(np.any(f['final'].numpy()!=z['E'],axis=1).sum()),seconds=f['seconds'],no_information_rounds=sum(not t['updated']for t in rr),all_zero_reward_rounds=sum(t['rollout']is not None and not torch.count_nonzero(t['rollout']['rewards'])for t in rr),mean_reward_initial=None if not rr or rr[0]['rollout']is None else float(rr[0]['rollout']['rewards'].mean()),mean_reward_final=None if not rr or rr[-1]['rollout']is None else float(rr[-1]['rollout']['rewards'].mean()))
   rows.append(dict(alias=a,dataset=ds,stage=stage,references=len(z['positions']),GT_used=stage=='oracle',fit_diagnostics=di,compute=z['compute'],**m))
 groups={};ep=['v','t','s','unobserved_actual_s','observed_actual_s'];pairs=[(a,'E_MBR')for a in ARMS]+[('Live','Direct'),('Live','Fixed')]
 for stage in STAGES:
  groups[stage]={}
  for ds in ['hc2','vidstg']:
   rr=[r for r in rows if r['stage']==stage and r['dataset']==ds];groups[stage][ds]=dict(parents=len(rr),support_queries=sum(r['references']>0 for r in rr),references=sum(r['references']for r in rr),means={a:{k:mean([r['values'][a][k]for r in rr])for k in ep}for a in ['E_MBR']+ARMS},paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in ep}for a,b in pairs})
 put(P/'RESULTS_PUBLIC.json',dict(status='completed',count=32,logical_updated_outputs=192,rows=rows,groups=groups,real_support_GT_free=True,correct_reference_separate_diagnostic=True,development_exposed=True,independent_confirmation=False));put(P/'COST_BREAKDOWN.json',dict(GPU_allocation_seconds=sum(read(f)['GPU_allocation_seconds']for f in (D/'allocations').glob('*.json')),decoder_trajectory_calls=sum(r['compute']['full_actual_trajectory_decoder_calls']for r in rows),PosDecoder_calls=sum(r['compute']['PosDecoder_calls']for r in rows),backward_calls=sum(r['compute']['backward_calls']for r in rows),total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_model_training_experts=0));print('SCORED',32,'real/oracle',flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','real','oracle_inputs','oracle','score']);a=ap.parse_args()
 try:measure(a.stage)if a.stage in STAGES else globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(e).__name__,error=str(e),partial_preserved=True));raise
