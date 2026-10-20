"""One bounded local execution of the prelocked four phases; no scheduler."""
import subprocess,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D,put
PYTHON=str(ROOT/'.conda/tubedetr/bin/python');started=time.time();active=None

def cmd(file,args,log):
 f=(D/log).open('a');p=subprocess.Popen([PYTHON,'-B','scripts/'+file,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);f.close();return p

def wait(p):
 rc=p.wait();assert rc==0,('owned phase process failed',p.pid,rc)

def cpu(ds,phase):
 cmds=[('score_stvg_single_referdino_parameter_search256_v1.py',['seal',ds,phase]),('audit_stvg_single_referdino_parameter_search256_v1.py',['pre-gt',ds,phase]),('audit_stvg_single_referdino_parameter_readouts256_v1.py',[ds,phase]),('score_stvg_single_referdino_parameter_search256_v1.py',['score',ds,phase]),('audit_stvg_single_referdino_parameter_search256_v1.py',['scored',ds,phase])]
 if phase=='coarse':cmds.append(('score_stvg_single_referdino_parameter_search256_v1.py',['fine-plan',ds]))
 for script,args in cmds:print('CONTROL',script,*args,flush=True);wait(cmd(script,args,f'{ds}_{phase}_cpu.log'))

def gpu(ds,phase):return cmd('run_stvg_single_referdino_parameter_search256_v1.py',[ds,phase],f'{ds}_{phase}.log')
try:
 while not(D/'hc2/COARSE_FIT_BARRIER.json').exists():
  assert time.time()-started<7400,'coarse wait timeout'
  if list(D.glob('FAILURE_hc2_coarse_*.json')):raise RuntimeError('HC coarse failed; inspect saved receipt')
  time.sleep(3)
 active=gpu('vidstg','coarse');cpu('hc2','coarse');wait(active)
 active=gpu('hc2','fine');cpu('vidstg','coarse');wait(active)
 active=gpu('vidstg','fine');cpu('hc2','fine');wait(active)
 active=None;cpu('vidstg','fine');put(D/'FOUR_PHASE_COMPLETION.json',dict(status='completed_all_four_phases_and_independent_scoring',epoch=time.time(),controller_wall_seconds=time.time()-started));print('ALL_SEARCH_PHASES_COMPLETE',flush=True)
except BaseException as e:
 put(D/f'CONTROLLER_FAILURE_{time.time_ns()}.json',dict(error=repr(e),active_owned_pid=active.pid if active and active.poll()is None else None));raise
