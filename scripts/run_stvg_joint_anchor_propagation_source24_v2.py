"""Strong sparse-anchor oracle diagnosis; source only, no optimizers/training."""
from pathlib import Path
import argparse,sys,time,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_observation_utilization_v1 import BASE as PREVIOUS,PARENT,OLD,REPAIR,read,put,load,save,sha,condition
BASE=ROOT/'artifacts/stvg_joint_anchor_propagation_source24_v2'
OUT=ROOT/'results/stvg_joint_posterior/2026-10-18/anchor_propagation_source24_v2'

def prepare():
 import torch,numpy as np
 from vg_tta.stvg_joint_anchor_projection_v2 import interpolation_matrix,project_anchors
 from scripts.run_stvg_joint_posterior_v1 import decode_state
 assert not(BASE/'PLAN_LOCK.json').exists();rr=read(PREVIOUS/'PLAN_LOCK.json')['source'];pins={};rows=[];files=[]
 for f in ['vg_tta/stvg_joint_anchor_projection_v2.py','scripts/run_stvg_joint_anchor_propagation_source24_v2.py','vg_tta/stvg_joint_posterior_repair_v1.py','vg_tta/stvg_joint_posterior_v1.py','scripts/run_stvg_joint_posterior_v1.py','protocols/stvg_joint_anchor_propagation_source24_v2.md','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']:pins[f]=sha(ROOT/f)
 for ds in ['hc2','vidstg']:
  for f in [REPAIR/'models'/(ds+'.pt'),REPAIR/'statistics'/(ds+'.pt')]:load(f);pins[str(f.relative_to(ROOT))]=sha(f)
 for r in rr:
  z=load(OLD/'features/source'/(r['alias']+'.pt'));c=condition(z,'cpu');obs=load(PREVIOUS/'observations'/(r['alias']+'.pt'))['observations']['GT_planned'];W=interpolation_matrix(c['native'][0],z,obs['positions']);x,pr=project_anchors(c['native'][0],W,obs['boxes'].numpy());d=decode_state(x,z);assert d['interval']==z['native_interval']
  if len(W):assert np.max(np.abs(d['boxes'][obs['positions']]-obs['boxes'].numpy()))<2e-6
  pp=save(BASE/'inputs'/(r['alias']+'.pt'),dict(W=W,target=obs['boxes'],positions=obs['positions'],native=c['native'][0],anchor_corrected=x,projection=pr));files.append(pp);rows.append(dict(alias=r['alias'],dataset=r['dataset'],references=len(W),**{k:v for k,v in pr.items()if k!='references'}))
  for f in [OLD/'features/source'/(r['alias']+'.pt'),OLD/'source_supervision'/(r['alias']+'.pt'),PREVIOUS/'observations'/(r['alias']+'.pt')]:pins[str(f.relative_to(ROOT))]=sha(f)
 assert not torch.cuda.is_initialized()
 put(BASE/'PLAN_LOCK.json',dict(status='locked_before_frozen_model_inference',source=rr,count=24,arms=['N','A','P','AP'],source_GT_sparse_oracle=True,new_training=0,new_target_inference=0,sigma_start=.213,sigma_end=.03,levels=16,added_noise=False,deterministic_one_output_per_query=True,checkpoints=dict(hc2=30,vidstg=20),primary='AP-A unobserved actual decoder sIoU',secondary='AP-A outside anchor interpolation footprint actual decoder sIoU',qualification='max anchor coordinate error<2e-6 for A and AP; time exact',DINO_if_all='positive AP-A mean in both domains on both actual unobserved supports, otherwise stop',previous_GPU_reserved_seconds=read(ROOT/'artifacts/stvg_joint_anchor_propagation_source24_v1/dino_followup/GPU_COMPLETION.json')['cumulative_reserved_seconds'],new_GPU_limit_seconds=300-read(ROOT/'artifacts/stvg_joint_anchor_propagation_source24_v1/dino_followup/GPU_COMPLETION.json')['current_diagnosis_total_reserved_seconds'],cumulative_GPU_limit_seconds=14400,no_automatic_training_target_OPD=True))
 put(BASE/'RUNTIME_LOCK.json',dict(pins=pins));put(BASE/'CPU_CONTRACT_COMPLETION.json',dict(status='pass',count=24,rows=rows,all_anchors_representable=True,exact_decoder_anchor_match=True,other_knots_unchanged=True,GPU_initialized=False));put(BASE/'INPUT_BARRIER.json',dict(status='complete',count=24,files=files));put(BASE/'STATUS.json',dict(status='prepared',stage='frozen_model_inference_pending'))
 print('CPU projection pass',len(rows),'maxerror',max(r['error']for r in rows))

def infer():
 import torch,numpy as np
 from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior
 from vg_tta.stvg_joint_anchor_projection_v2 import propagate
 from scripts.run_decota_paper_main_v1 import gpu
 from main_method.runtime import resource_guard
 p=read(BASE/'PLAN_LOCK.json');assert read(BASE/'CPU_CONTRACT_COMPLETION.json')['status']=='pass'
 for f,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f
 assert not(BASE/'INFERENCE_STARTED.json').exists();put(BASE/'INFERENCE_STARTED.json',dict(status='started',pid=os.getpid()));lease=gpu();beg=time.monotonic();files=[];calls=0
 try:
  for ds in ['hc2','vidstg']:
   st=load(REPAIR/'statistics'/(ds+'.pt'));net=PreconditionedJointPrior(st['mean'],st['scale']).cuda().eval().requires_grad_(False);net.load_state_dict(load(REPAIR/'models'/(ds+'.pt'))['state'])
   for r in [r for r in p['source']if r['dataset']==ds]:
    resource_guard();torch.cuda.synchronize();assert time.monotonic()-beg<p['new_GPU_limit_seconds'];z=load(OLD/'features/source'/(r['alias']+'.pt'));q=load(BASE/'inputs'/(r['alias']+'.pt'));c=condition(z);none=load(ROOT/'artifacts/stvg_joint_anchor_propagation_source24_v1/predictions'/(r['alias']+'.pt'))['P'];assert torch.equal(none['final'][0],q['native'][0]);calls+=0
    if len(q['W']):ap=propagate(net,c,z,q['W'],q['target'].numpy());calls+=16
    else:ap=none
    assert all(v.grad is None for v in net.parameters());assert torch.equal(ap['final'][0],q['native'][0]);files.append(save(BASE/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=ds,N=q['native'],A=q['anchor_corrected'],P=none,AP=ap,no_anchor_AP_exact_P=not len(q['W']),new_training=0,new_target_inference=0)));put(BASE/'STATUS.json',dict(status='running',stage='frozen_propagation',done=len(files),total=24),True)
   del net;torch.cuda.empty_cache()
  put(BASE/'PREDICTION_BARRIER.json',dict(status='complete',count=24,logical_predictions=96,files=files,model_forward_calls=calls,unanchored_P_exact_v1_reuses=24,all_sealed_before_full_GT_score=True))
 finally:
  torch.cuda.synchronize();sec=time.monotonic()-beg;assert sec<p['new_GPU_limit_seconds'] and sec+p['previous_GPU_reserved_seconds']<14400;put(BASE/'GPU_COMPLETION.json',dict(status='complete',new_reserved_seconds=sec,cumulative_reserved_seconds=sec+p['previous_GPU_reserved_seconds'],limit_seconds=14400,new_training=0));lease.close()

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','infer']);arg=a.parse_args();globals()[arg.command]()
