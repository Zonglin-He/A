"""Prospective frozen dataset-specific Fine Center E/Fixed/Live evaluation64."""
from pathlib import Path
import os,sys,time,argparse,collections,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_fine_dataset_formal64_v1';P=ROOT/'results/stvg_error_state/2026-10-19/fine_dataset_formal64'
VERIFIER=None
CODE=['scripts/lock_stvg_fine_dataset_formal64_v1.py','scripts/run_stvg_fine_dataset_formal64_v1.py','protocols/stvg_fine_dataset_formal64_v1.md']
def verify(full=False):
 global VERIFIER
 from main_method.runtime import resource_guard
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 from main_method.runtime import PinVerifier
 if VERIFIER is None:VERIFIER=PinVerifier(r['pins'])
 VERIFIER.verify(full=full);resource_guard();return p

def prepare():
 p=read(D/'ROSTER_LOCK.json');assert len(p['records'])==64;P.mkdir(parents=True);files=CODE+['vg_tta/stvg_fine_roi_scoped_opd32_v1.py','vg_tta/stvg_roi_layout_opd32_v1.py','vg_tta/stvg_discriminative_dense32_v1.py','scripts/run_stvg_dino_supported_expanded256_recovery002.py','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
 for r in p['records']:files +=[r['source_path'],r['prediction_path']]
 # New explicit finite formal evaluation; no other queue restart or parameter change.
 deps=read(ROOT/'artifacts/stvg_fine_step_confirm32_v1/INPUT_DEPENDENCY_AUDIT.json')['dependencies'];files+=list(deps)
 files+=['configs/stvg_fine_center_dataset_candidate_v1.json','checkpoints/TASTVG_VidSTG.pth','checkpoints/TASTVG_HCSTVG2.pth']
 cfg=read(ROOT/p['frozen_configuration']);assert sha(ROOT/p['frozen_configuration'])==p['frozen_configuration_sha256'] and cfg['step_size_by_target_dataset']==dict(hc2=1.,vidstg=.5)
 plan=dict(p,status='locked',arms=['E_MBR','Fixed','Live'],configuration=dict(temperature=.1,center_scale=.25,rounds=2,candidates_per_round=17,step_size_by_target_dataset=cfg['step_size_by_target_dataset']),initial='same frozenSource/E/finalMBR no history replay',noise_seed='formal64-v1|alias, same noise Fixed/Live',new_training=0,new_experts=0,new_DINO=0,max_Source=64,CPU_cap_seconds=900,deadline_epoch=time.time()+3600,created_epoch=time.time(),GPU_cap_seconds=900,old_heartbeat_remains_paused=True,decision='Report Fixed−E,Live−E,Live−Fixed per dataset, macrofullv/ALLunobsactuals, paired95CI and tails; frozen config no search, no new gate/production promotion')
 put(D/'PLAN_LOCK.json',plan);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),roster_sha256=sha(D/'ROSTER_LOCK.json'),pins={f:sha(ROOT/f)for f in files}));print('locked64',flush=True)

def capture():
 import torch,numpy as np
 import torch.nn.functional as F
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from scripts.run_stvg_dino_supported_expanded256_recovery002 import capture as forward
 from methods.decota_final_simplified_v1.tensors import state_hash
 p=verify();lease=gpu();tick=time.perf_counter();domain=None;model=None;cache=collections.OrderedDict();files=[]
 def guard():
  verify();assert time.time()<p['deadline_epoch'];assert time.perf_counter()-tick<p['GPU_cap_seconds'],'900sec GPU cap';assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  for ix,r in enumerate(p['records']):
   verify();assert time.time()<p['deadline_epoch'];assert time.perf_counter()-tick<p['GPU_cap_seconds'];assert torch.cuda.mem_get_info()[0]>4*2**30;ds=r['dataset'];a=r['alias']
   if domain!=ds:
    if model is not None:assert state_hash(model.state_dict())==mh;del model;gc.collect();torch.cuda.empty_cache()
    guard();model=model_for('vidstg'if ds=='hc2'else'hcstvg2');domain=ds;mh=state_hash(model.state_dict());cache.clear();guard()
   fine=[];mask=[];h1=model.vis_encoder[0].body.layer3.register_forward_hook(lambda m,args,out:fine.append(out.detach().float().cpu()));h2=model.vis_encoder.register_forward_pre_hook(lambda m,args:mask.append(args[0].mask.detach().cpu()))
   try:
    guard();row,recs,base,native,expert,inp,seconds=forward(model,ds,r['query_ordinal'],cache);guard()
   finally:h1.remove();h2.remove()
   old=load(ROOT/r['source_path']);e=load(ROOT/r['prediction_path']);assert len(base.zero['logits'])==len(old['logits'])==2;assert torch.equal(native['boxes'].cpu(),old['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],old['logits']));assert list(row['frame_ids'])==e['frame_ids'];n=len(e['frame_ids']);assert len(fine)==len(mask)==2;maps=torch.empty(n,*fine[0].shape[1:]);valid=torch.empty(n,*fine[0].shape[-2:],dtype=torch.bool)
   for off in [0,1]:
    assert recs[off]['offset']==off and recs[off]['frame_ids']==e['frame_ids'][off::2]
    maps[off::2]=fine[off];valid[off::2]=~F.interpolate(mask[off][:,None].float(),size=maps.shape[-2:],mode='nearest')[:,0].bool()
   frames=e['frames'];selected=e['estimate']['selected'];assert len(frames)==len(selected)
   for f,k in zip(frames,selected):assert 0<=f['position']<len(e['frame_ids']) and f['position']in e['planned_positions'] and (k==-1 or 0<=k<len(f['boxes']))
   positions=[f['position']for f,k in zip(frames,selected)if k>=0];boxes=[torch.as_tensor(f['boxes'][k]).tolist()for f,k in zip(frames,selected)if k>=0]
   files.append(save(D/'features'/(a+'.pt'),dict(alias=a,dataset=ds,maps=maps,valid=valid,E=torch.as_tensor(e['outputs']['E_DINO_support']),frame_ids=e['frame_ids'],interval=e['interval'],planned_positions=e['planned_positions'],positions=positions,boxes=boxes,model_hash=mh,Source_and_time_logits_exact=True,GT_read=False,new_DINO=0,capture_seconds=seconds)));print(a,ix+1,time.perf_counter()-tick,flush=True)
   del base,native,expert,old,e,maps,valid,fine,mask;gc.collect();torch.cuda.empty_cache()
   if ix==1:
    elapsed=time.perf_counter()-tick;put(D/'PILOT2_TIMING.json',dict(count=2,elapsed_seconds=elapsed,rough64_seconds=32*elapsed,not_guaranteed=True));assert 32*elapsed<p['GPU_cap_seconds'],'pilot extrapolation exceeds cap'
  assert state_hash(model.state_dict())==mh;verify(full=True);guard();put(D/'FEATURE_BARRIER.json',dict(status='complete',count=64,files=files,GPU_seconds=time.perf_counter()-tick,GT_read=False))
 finally:lease.close();put(D/'GPU_STOP.json',dict(GPU_seconds=time.perf_counter()-tick,all_partials_preserved=True))

def measure():
 os.environ['CUDA_VISIBLE_DEVICES']='';import torch,numpy as np
 from vg_tta.stvg_roi_layout_opd32_v1 import build
 from vg_tta.stvg_fine_roi_scoped_opd32_v1 import scope,scoped_fit
 from vg_tta.stvg_discriminative_dense32_v1 import noise
 p=verify();torch.set_num_threads(4);bar=read(D/'FEATURE_BARRIER.json');assert bar['count']==64
 for receipt in bar['files']:assert sha(ROOT/receipt['path'])==receipt['sha256']
 files=[];start=time.perf_counter()
 for r in p['records']:
  verify();assert time.perf_counter()-start<p['CPU_cap_seconds'];a=r['alias'];x=load(D/'features'/(a+'.pt'));E=x['E'].double();ctx=build(x['maps'],x['valid'],x['positions'],x['boxes']);_,allowed=scope(x['frame_ids'],x['planned_positions'],x['interval']);eps=noise('formal64-v1|'+a,len(E));fits={}
  for arm,mode in [('Fixed','Fixed_dense'),('Live','On_policy_dense')]:
   step=p['configuration']['step_size_by_target_dataset'][r['dataset']]
   cfg=dict(p['configuration'],step_size=step);fits[arm]=scoped_fit(E,ctx,eps,cfg,x['planned_positions'],mode,list(range(len(E))),allowed)
  assert torch.equal(fits['Fixed']['path'][1]['boxes'],fits['Live']['path'][1]['boxes']);outputs=dict(E_MBR=E.numpy(),**{a:f['final'].numpy()for a,f in fits.items()});files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],allowed=allowed,noise=eps,fits=fits,outputs=outputs,GT_read=False)));print(a,'two_fits_sealed',flush=True)
 verify(full=True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,files=files,all_outputs_before_NEW_GT=True,actual_fits=128,round_slots=256,CPU_seconds=time.perf_counter()-start))

def score():
 os.environ['CUDA_VISIBLE_DEVICES']='';import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 for r in p['records']:
  o=load(D/'predictions'/(r['alias']+'.pt'));v=metrics(o,plans[r['dataset']][r['query_ordinal']],gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']]);rows.append(dict(alias=r['alias'],dataset=r['dataset'],**v))
 ep=['v','t','unobserved_actual_s','observed_actual_s'];pairs=[(a,'E_MBR')for a in p['arms'][1:]]+[('Live','Fixed')];groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=len(rr),metrics={a:{e:mean([r['values'][a][e]for r in rr])for e in ep}for a in p['arms']},paired={a+'_minus_'+b:{e:paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rr])for e in ep if e!='t'}for a,b in pairs})
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=64,groups=groups,rows=rows,all_outputs_before_NEW_GT=True,prospective_frozen_parameter_evaluation=True,new_parents_excluded_listed_method_and_parameter_selection=True,historical_Source_E_full_expanded_exposure=True,not_globally_unexposed=True));put(D/'SCORE_COMPLETION.json',dict(status='complete',count=64));print('score complete',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
