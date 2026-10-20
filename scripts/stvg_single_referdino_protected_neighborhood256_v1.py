"""Finite harm-informed neighborhood with original, audited OPD engines."""
import sys,time,math,itertools,json,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D as MD,P as MP,read,put,sha,verify as vm
from scripts.stvg_single_referdino_initialization_coupled_common_v1 import D as GD,P as GP,verify as vg
D=MD/'protected_neighborhood';P=MP/'protected_neighborhood'
def verify():
 vm();vg();p=read(D/'PLAN_LOCK.json');assert p['master_plan_sha256']==sha(MD/'PLAN_LOCK.json')and p['gamma_plan_sha256']==sha(GD/'PLAN_LOCK.json')
 for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
 return p

def prepare():
 master=vm();gamma=vg();r=read(GP/'hc2/coarse/RESULTS_PUBLIC.json');assert read(GP/'hc2/coarse/ROOT_AUDIT_PUBLIC.json')['status']=='pass';old=read(MP/'hc2/coarse/RESULTS_PUBLIC.json');baseline=master['baseline']+f"_S{master['baseline_steps']['hc2']:02}_A100";cap=old['paired'][baseline+'_minus_E_Strong']['tail']['below_minus5'];valid=[]
 for a,c in r['definitions'].items():
  tail=r['paired'][a+'_minus_E_G'+str(int(c['initialization_gamma']*100)).zfill(3)]['tail']['below_minus5']
  if tail<=cap:valid.append(a)
 valid.sort(key=lambda a:(-r['means'][a]['v'],abs(r['definitions'][a]['alpha']-1),r['definitions'][a]['steps'],r['definitions'][a]['lr'],-r['definitions'][a]['tau']));a=valid[0];c=r['definitions'][a];cfg=gamma['datasets']['hc2']['coarse_configs'][c['config_id']];configs={f'R{i:02}':dict(cfg,lr=lr,tau=tau)for i,(lr,tau)in enumerate(itertools.product([cfg['lr']/math.sqrt(2),cfg['lr']*math.sqrt(2)],[cfg['tau']/math.sqrt(2),cfg['tau']*math.sqrt(2)]))};fine=dict(center_arm=a,center_id=c['config_id'],center_config=cfg,configs=configs,steps=sorted(set([max(1,c['steps']-5),c['steps'],min(40,c['steps']+5)])),alphas=sorted(set([max(.05,c['alpha']-.25),c['alpha'],c['alpha']+.25])),severe_loss_limit=cap)
 files=['protocols/stvg_single_referdino_protected_selection256_revision002.md','scripts/stvg_single_referdino_protected_neighborhood256_v1.py'];p=dict(status='locked_before_protected_neighborhood_fits',records=gamma['records'],datasets=gamma['datasets'],master_plan_sha256=sha(MD/'PLAN_LOCK.json'),gamma_plan_sha256=sha(GD/'PLAN_LOCK.json'),source_coarse_results_sha256=sha(GP/'hc2/coarse/RESULTS_PUBLIC.json'),pins={f:sha(ROOT/f)for f in files},new_fits=512,new_updates=20480,same_GPU_worker_cap_seconds=7200,seed=41,adaptive_offline_development_selection=True)
 assert not(D/'PLAN_LOCK.json').exists();put(D/'PLAN_LOCK.json',p);put(D/'hc2/FINE_PLAN_LOCK.json',fine);put(D/'hc2/COARSE_FIT_BARRIER.json',read(GD/'hc2/COARSE_FIT_BARRIER.json'));put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records']});put(P/'hc2/FINE_PLAN_PUBLIC.json',fine);print('PROTECTED_NEIGHBORHOOD_LOCKED',a,cap,flush=True)

def bind(m):m.D=D;m.P=P;m.verify=verify

def execute(kind):
 if kind=='worker':
  import scripts.run_stvg_single_referdino_initialization_coupled256_v1 as m
  bind(m);m.main('hc2','fine');l=read(MD/'GPU_LEDGER.json');v=l['invocations'][-1];assert v['dataset']=='hc2'and v['phase']=='initialization_fine';v['engine_phase_name']=v['phase'];v['phase']='protected_neighborhood_fine';v['namespace']='protected_neighborhood';put(MD/'GPU_LEDGER.json',l)
 elif kind in ['seal','score']:
  import scripts.score_stvg_single_referdino_initialization_coupled256_v1 as m
  bind(m);getattr(m,kind)('hc2','fine')
 else:
  import scripts.audit_stvg_single_referdino_initialization_coupled256_v1 as m
  bind(m);getattr(m,'pregt'if kind=='pre-gt'else'scored')('hc2','fine')

def control():
 import subprocess
 prepare();beg=time.time();py=str(ROOT/'.conda/tubedetr/bin/python')
 while not(GD/'hc2/FINE_FIT_BARRIER.json').exists():
  assert time.time()-beg<2400
  assert not list(MD.glob('COUPLING_RESUME_FAILURE_*.json'))
  time.sleep(3)
 while True:
  with (ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:
    assert time.time()-beg<2500;time.sleep(2);continue
  break
 for mode in ['worker','seal','pre-gt','score','scored']:
  print('PROTECTED_CONTROL',mode,flush=True)
  with (D/(mode+'.log')).open('a')as f:q=subprocess.run([py,'-B',__file__,mode],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  assert q.returncode==0,(mode,q.returncode)
 put(MD/'PROTECTED_NEIGHBORHOOD_COMPLETION.json',dict(status='completed_additional_finite_harm_informed_neighborhood',new_fits=512,new_updates=20480,wall=time.time()-beg));print('PROTECTED_NEIGHBORHOOD_COMPLETE',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='control':
  try:control()
  except BaseException as e:put(MD/('PROTECTED_CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(e)));raise
 else:execute(sys.argv[1])
