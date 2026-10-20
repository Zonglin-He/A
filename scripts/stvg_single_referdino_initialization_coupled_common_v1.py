"""Finite initialization/optimizer coupling supplement under the same GPU cap."""
import sys,time,math,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D as MAIN_D,P as MAIN_P,BASE,read,put,sha,load,save,verify as verify_main
D=MAIN_D/'initialization_coupled';P=MAIN_P/'initialization_coupled'
def initial(S,E,gamma):
 S=np.asarray(S,dtype=np.float64);E=np.asarray(E,dtype=np.float64)
 if gamma==0:return S.copy()
 if gamma==1:return E.copy()
 return np.concatenate((S[:,:2]+gamma*(E[:,:2]-S[:,:2]),S[:,2:]*np.exp(gamma*np.log(E[:,2:]/S[:,2:]))),-1)
def verify():
 master=verify_main();p=read(D/'PLAN_LOCK.json');assert p['master_plan_sha256']==sha(MAIN_D/'PLAN_LOCK.json')
 for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
 return p

def prepare():
 master=verify_main();screen=read(MAIN_P/'INITIALIZATION_STRENGTH_RESULTS_PUBLIC.json');assert not(D/'PLAN_LOCK.json').exists();datasets={}
 for ds,g in screen['datasets'].items():
  if g['best_gamma']==1:continue
  candidates=[]
  for phase in ['coarse','fine']:
   r=read(MAIN_P/ds/phase/'RESULTS_PUBLIC.json');assert read(MAIN_P/ds/phase/'ROOT_AUDIT_PUBLIC.json')['status']=='pass'
   candidates +=[(r['means'][a]['v'],phase,a,r['definitions'][a])for a in r['definitions']]
  candidates.sort(key=lambda v:(-v[0],abs(v[3]['alpha']-1),v[3]['steps'],v[3]['lr'],-v[3]['tau'],v[1]));mean,phase,arm,c=candidates[0];base=dict(master['configs'][ds][master['baseline']],lr=c['lr'],tau=c['tau'],steps=40);gammas=[int(a[1:])/100 for a in g['ranking']if int(a[1:])/100 not in [0.,1.]][:2];configs={};oldbase=dict(master['configs'][ds][master['baseline']],steps=40)
  for gamma in gammas:
   for tag,center in [('T',base),('B',oldbase)]:configs['G'+str(int(gamma*100)).zfill(3)+tag]=dict(center,initialization_gamma=gamma)
  datasets[ds]=dict(reference_phase=phase,reference_arm=arm,reference_config=c,reference_v=mean,gammas=gammas,coarse_configs=configs,coarse_steps=sorted(set([10,20,30,40,c['steps']])),coarse_alphas=sorted(set([.25,.5,1.,1.5,2.,c['alpha']])))
 own=['protocols/stvg_single_referdino_initialization_coupled256_v1.md','scripts/stvg_single_referdino_initialization_coupled_common_v1.py','scripts/run_stvg_single_referdino_initialization_coupled256_v1.py','scripts/score_stvg_single_referdino_initialization_coupled256_v1.py','scripts/audit_stvg_single_referdino_initialization_coupled256_v1.py']
 p=dict(status='locked_before_new_gamma_OPD',datasets=datasets,records=[r for r in master['records']if r['dataset']in datasets],master_plan_sha256=sha(MAIN_D/'PLAN_LOCK.json'),screen_results_sha256=sha(MAIN_P/'INITIALIZATION_STRENGTH_RESULTS_PUBLIC.json'),coarse_fit_count=sum(len(d['coarse_configs'])*128 for d in datasets.values()),fine_fit_count=512*len(datasets),new_update_limit=40*sum((len(d['coarse_configs'])+4)*128 for d in datasets.values()),overall_GPU_worker_wall_limit_seconds=7200,seed=41,initialization_rule='same eight-reference consensus; scalar gamma in center/log-size; candidate generation follows actual initialization; at each gamma compare original baseline optimizer and original gamma1 full-panel tuned center',pins={f:sha(ROOT/f)for f in own},epoch=time.time())
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','epoch']});print('GAMMA_COUPLING_LOCKED',datasets.keys(),flush=True)
if __name__=='__main__':prepare()
