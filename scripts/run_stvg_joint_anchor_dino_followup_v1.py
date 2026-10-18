"""Conditional cached-DINO branch, only after source-oracle propagation gate."""
from pathlib import Path
import sys,argparse,time,os,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_anchor_propagation_source24_v1 import BASE,OUT,OLD,PREVIOUS,REPAIR,read,put,load,save,sha,condition
D=BASE/'dino_followup';SETS=['DINO_all','DINO_common','GT_common']

def prepare():
 import torch,numpy as np
 from vg_tta.stvg_joint_anchor_projection_v1 import interpolation_matrix,project_anchors
 assert read(BASE/'PROPAGATION_DECISION.json')['DINO_gate']is True;assert not(D/'PLAN_LOCK.json').exists();p=read(BASE/'PLAN_LOCK.json');files=[];rows=[];pins={}
 for f in ['scripts/run_stvg_joint_anchor_dino_followup_v1.py','vg_tta/stvg_joint_anchor_projection_v1.py','protocols/stvg_joint_anchor_dino_followup_v1.md']:pins[f]=sha(ROOT/f)
 for r in p['source']:
  z=load(OLD/'features/source'/(r['alias']+'.pt'));v=load(BASE/'inputs'/(r['alias']+'.pt'));obs=load(PREVIOUS/'observations'/(r['alias']+'.pt'))['observations'];ss={}
  for name in SETS:
   o=obs[name];W=interpolation_matrix(v['native'],z,o['positions']);x,pr=project_anchors(v['native'],W,o['boxes'].numpy());ss[name]=dict(W=W,target=o['boxes'],positions=o['positions'],anchor_corrected=x,projection=pr);rows.append(dict(alias=r['alias'],dataset=r['dataset'],arm=name,references=len(W),error=pr['error']))
  assert ss['DINO_common']['positions']==ss['GT_common']['positions'];files.append(save(D/'inputs'/(r['alias']+'.pt'),ss))
 put(D/'PLAN_LOCK.json',dict(status='locked_before_followup_inference',sets=SETS,count=24,full_cohort_retained=True,common_reference_positions_identical=True,no_correctness_filter=True,full_DINO_vs_GT_planned_not_pure_quality=True,new_expert_calls=0,new_training=0,new_target_inference=0,previous_GPU_reserved_seconds=read(BASE/'GPU_COMPLETION.json')['cumulative_reserved_seconds'],new_GPU_limit_seconds=300-read(BASE/'GPU_COMPLETION.json')['new_reserved_seconds'],cumulative_limit_seconds=14400));put(D/'RUNTIME_LOCK.json',dict(pins=pins));put(D/'CPU_CONTRACT_COMPLETION.json',dict(status='pass',count=24,rows=rows,all_actual_DINO_and_common_GT_anchors_representable=True));put(D/'INPUT_BARRIER.json',dict(status='complete',count=24,files=files));print('DINO input pass',len(rows))

def infer():
 import torch,numpy as np
 from vg_tta.stvg_joint_posterior_repair_v1 import PreconditionedJointPrior
 from vg_tta.stvg_joint_anchor_projection_v1 import propagate
 from scripts.run_decota_paper_main_v1 import gpu
 from main_method.runtime import resource_guard
 p=read(BASE/'PLAN_LOCK.json');dp=read(D/'PLAN_LOCK.json');assert read(BASE/'PROPAGATION_DECISION.json')['DINO_gate'];assert not(D/'INFERENCE_STARTED.json').exists()
 for f,h in {**read(BASE/'RUNTIME_LOCK.json')['pins'],**read(D/'RUNTIME_LOCK.json')['pins']}.items():assert sha(ROOT/f)==h
 put(D/'INFERENCE_STARTED.json',dict(status='started',pid=os.getpid()));lease=gpu();beg=time.monotonic();files=[];new=0;reuses=0
 try:
  for ds in ['hc2','vidstg']:
   st=load(REPAIR/'statistics'/(ds+'.pt'));net=PreconditionedJointPrior(st['mean'],st['scale']).cuda().eval().requires_grad_(False);net.load_state_dict(load(REPAIR/'models'/(ds+'.pt'))['state'])
   for r in [r for r in p['source']if r['dataset']==ds]:
    resource_guard();torch.cuda.synchronize();assert time.monotonic()-beg<dp['new_GPU_limit_seconds'];z=load(OLD/'features/source'/(r['alias']+'.pt'));q=load(D/'inputs'/(r['alias']+'.pt'));oldq=load(BASE/'inputs'/(r['alias']+'.pt'));v=load(BASE/'predictions'/(r['alias']+'.pt'));c=condition(z);cache={};arms={};reuse={}
    def key(W,t):return hashlib.sha256(W.tobytes()+t.numpy().tobytes()).hexdigest()if len(W)else 'empty'
    cache['empty']=('old_P',v['P']);cache[key(oldq['W'],oldq['target'])]=('old_GT_planned',v['AP'])
    for name in SETS:
     o=q[name];k=key(o['W'],o['target'])
     if k in cache:reuse[name]=cache[k][0];a=cache[k][1];reuses+=1
     else:a=propagate(net,c,z,o['W'],o['target'].numpy());cache[k]=(name,a);new+=1
     arms[name]=dict(A=o['anchor_corrected'],AP=a)
    files.append(save(D/'predictions'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=ds,arms=arms,reused_from=reuse,new_training=0)));put(BASE/'STATUS.json',dict(status='running',stage='qualified_cached_DINO_comparison',done=len(files),total=24),True)
   del net;torch.cuda.empty_cache()
  put(D/'PREDICTION_BARRIER.json',dict(status='complete',count=24,logical_new_conditions=144,new_frozen_rollouts=new,exact_reuses=reuses,files=files,all_sealed_before_new_full_GT_scoring=True,new_expert_calls=0))
 finally:
  torch.cuda.synchronize();s=time.monotonic()-beg;assert s<dp['new_GPU_limit_seconds']and s+dp['previous_GPU_reserved_seconds']<14400;put(D/'GPU_COMPLETION.json',dict(status='complete',new_reserved_seconds=s,current_diagnosis_total_reserved_seconds=s+read(BASE/'GPU_COMPLETION.json')['new_reserved_seconds'],cumulative_reserved_seconds=s+dp['previous_GPU_reserved_seconds'],new_training=0));lease.close()

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','infer']);arg=a.parse_args();globals()[arg.command]()
