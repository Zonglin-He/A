"""Finite dataset-specific joint tuning of the single-expert spatial method."""
from pathlib import Path
import sys,json,time,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_bidirectional256_common_v1 import read,put,sha,load,save
D=ROOT/'artifacts/stvg_single_referdino_parameter_search256_v1'
P=ROOT/'results/stvg_error_state/2026-10-20/single_referdino_parameter_search256'
BASE=ROOT/'artifacts/stvg_single_referdino_bidirectional256_v1'
def verify():
 p=read(D/'PLAN_LOCK.json')
 for f,h in {**p['pins'],**p['protected']}.items():assert sha(ROOT/f)==h,f
 return p

def prepare():
 assert not(D/'PLAN_LOCK.json').exists();old=read(BASE/'PLAN_LOCK.json');configs={};sources={}
 for ds in ['hc2','vidstg']:
  configs[ds]={f'C{i:02}':dict(old['configs'][ds],lr=lr,tau=tau,steps=40)for i,(lr,tau)in enumerate(itertools.product([.003,.01,.03],[.2,.5]))}
  for rec in [r for r in old['records']if r['dataset']==ds]:
   for sub in ['inputs','predictions','traces']:
    f=BASE/f"{sub}/{ds}/{rec['alias']}.pt";sources[str(f.relative_to(ROOT))]=sha(f)
 own=['protocols/stvg_single_referdino_parameter_search256_v1.md','scripts/stvg_single_referdino_parameter_search256_common_v1.py','scripts/run_stvg_single_referdino_parameter_search256_v1.py','scripts/score_stvg_single_referdino_parameter_search256_v1.py','scripts/audit_stvg_single_referdino_parameter_search256_v1.py']
 own += [str(f.relative_to(ROOT))for pkg in ['stvg_single_referdino_opd_v1','stvg_referdino_opd_v1']for f in (ROOT/'methods'/pkg).glob('*.py')]
 oldconfig='methods/stvg_single_referdino_opd_v1/config.json';protected={f:sha(ROOT/f)for f in [oldconfig,'methods/CURRENT_METHOD.json','methods/RESEARCH_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']}
 p=dict(status='locked_before_new_joint_tuning',records=old['records'],configs=configs,baseline='C02',baseline_steps={'hc2':40,'vidstg':30},coarse_steps=[10,20,30,40],coarse_alphas=[.25,.5,1.,1.5,2.],fine_rule=dict(lr_tau_multipliers=[2**-.5,2**.5],step_offsets=[-5,0,5],step_bounds=[1,40],alpha_offsets=[-.25,0,.25],minimum_alpha=.05),
 cumulative_GPU_worker_wall_limit_seconds=7200,maximum_new_primary_fits=2432,maximum_new_primary_updates=97280,compatibility_updates=40,seed=41,parents_per_dataset=128,
 expert_count=1,initialization='frozen E_Ref from same ReferDINO8; no initialization or expert selection change',Strong_unchanged=True,network_training=False,dataset_specific_selection=True,
 selection='maximum all128 mean complete vIoU; ties alpha closest1 then fewer steps then smaller lr then larger tau; retain original baseline among candidates; no per-video choice; exposed development only',
 pins={f:sha(ROOT/f)for f in own},protected=protected,source_files=sources,checkpoints={**old['checkpoints'],read(BASE/'EXPERT_WEIGHT_LOCK.json')['checkpoint']:read(BASE/'EXPERT_WEIGHT_LOCK.json')['sha256']},epoch=time.time())
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','source_files','epoch']});put(D/'STATE.json',dict(status='locked_pending_coarse',datasets=['hc2','vidstg'],new_expert_forwards=0));print('JOINT_TUNING_LOCKED',flush=True)
if __name__=='__main__':prepare()
