"""Bounded actual HC initialization/optimizer coupling; no scheduler."""
import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D,put
PY=str(ROOT/'.conda/tubedetr/bin/python');beg=time.time()
def run(script,args):
 print('COUPLING_CONTROL',script,*args,flush=True)
 log=D/'initialization_coupled'/('control_'+('_'.join(args))+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a')as f:r=subprocess.run([PY,'-B','scripts/'+script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(script,args,r.returncode)
try:
 run('stvg_single_referdino_initialization_coupled_common_v1.py',[])
 for phase in ['coarse','fine']:
  run('run_stvg_single_referdino_initialization_coupled256_v1.py',['hc2',phase])
  run('score_stvg_single_referdino_initialization_coupled256_v1.py',['seal','hc2',phase])
  run('audit_stvg_single_referdino_initialization_coupled256_v1.py',['pre-gt','hc2',phase])
  run('score_stvg_single_referdino_initialization_coupled256_v1.py',['score','hc2',phase])
  run('audit_stvg_single_referdino_initialization_coupled256_v1.py',['scored','hc2',phase])
  if phase=='coarse':run('score_stvg_single_referdino_initialization_coupled256_v1.py',['fine-plan','hc2'])
 put(D/'INITIALIZATION_COUPLING_COMPLETION.json',dict(status='completed_both_phases_and_independent_audits',wall=time.time()-beg))
except BaseException as e:
 put(D/('COUPLING_CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(e)));raise
