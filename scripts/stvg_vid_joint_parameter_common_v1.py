"""Paths and immutable plans; importing this module does not select a device."""
from pathlib import Path
import sys,json,hashlib,time,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
D=ROOT/'artifacts/stvg_vid_joint_parameter_search128_v1'
P=ROOT/'results/stvg_error_state/2026-10-20/vid_joint_parameter_search128'
OLD=ROOT/'artifacts/stvg_spatial_strong_bidirectional256_v1'
def read(p):return json.loads(Path(p).read_text())
def put(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 import torch
 return torch.load(p,map_location='cpu',weights_only=False)
def save(p,x):
 import torch,os
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p);return sha(p)
def verify():
 p=read(D/'PLAN_LOCK.json')
 for f,h in {**p['pins'],**p['protected']}.items():assert sha(ROOT/f)==h,f
 return p
def prepare():
 assert not(D/'PLAN_LOCK.json').exists()
 previous=read(OLD/'PLAN_LOCK.json');records=[r for r in previous['records']if r['dataset']=='vidstg'];cfg=dict(previous['configs']['vidstg'],seeds=[41]);configs={}
 for i,(lr,tau)in enumerate(itertools.product([.003,.01,.03],[.2,.5])):configs[f'C{i:02}']=dict(cfg,lr=lr,tau=tau)
 baseline=next(k for k,v in configs.items()if v['lr']==.01 and v['tau']==.2)
 files=['protocols/stvg_vid_joint_parameter_search128_v1.md','scripts/stvg_vid_joint_parameter_common_v1.py','scripts/run_stvg_vid_joint_parameter_search128_v1.py',
 'scripts/run_stvg_spatial_strong_bidirectional256_v1.py','scripts/run_decota_paper_main_v1.py','artifacts/decota_paper_experiments_v1/vidstg/PLAN.json']
 files += [str(f.relative_to(ROOT))for f in (ROOT/'methods/stvg_referdino_opd_v1').glob('*.py')]
 protected=read(ROOT/'artifacts/stvg_vid_spatial_failure_cached_v1/SCOPE.json')['pins']
 p=dict(status='locked_before_new_forward',dataset='vidstg',parents=128,queries=128,seed=41,records=records,configs=configs,baseline=baseline,
  coarse_steps=[10,20,30,40],coarse_alphas=[.5,1.,1.5,2.],fine_rule=dict(optimizer_multipliers=[2**-.5,2**.5],step_offsets=[-5,0,5],alpha_offsets=[-.25,0,.25],step_bounds=[1,40],minimum_alpha=.05),
  cumulative_GPU_worker_wall_limit_seconds=5400,maximum_new_primary_updates=46080,compatibility_updates=40,
  existing_experts_reused=True,new_Source224_feature_capture_allowed=True,new_network_training=False,GT_read=False,
  previous_saved_parameter_grid_sha256=sha(ROOT/'results/stvg_error_state/2026-10-20/vid_saved_parameters128/RESULTS_PUBLIC.json'),
  pins={f:sha(ROOT/f)for f in files},protected=protected,checkpoint={cfg['source_checkpoint']:previous['checkpoints'][cfg['source_checkpoint']]},epoch=time.time())
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','epoch']});put(D/'STATE.json',dict(status='locked_pending_GPU',phase='capture_coarse',new_expert_forwards=0));print('LOCKED',baseline,len(configs),flush=True)
if __name__=='__main__':prepare()
