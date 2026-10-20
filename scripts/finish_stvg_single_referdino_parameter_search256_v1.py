"""Finite selected evaluation, runtime verification and reporting after locked searches."""
import sys,time,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D,P,put,read
PY=str(ROOT/'.conda/tubedetr/bin/python');beg=time.time();active=None

def run(script,args):
 global active
 print('FINALIZE_CONTROL',script,*args,flush=True)
 with (D/('final_'+script.removesuffix('.py')+'_'+('_'.join(args))+'.log')).open('a')as f:
  active=subprocess.Popen([PY,'-B','scripts/'+script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);rc=active.wait()
 assert rc==0,(script,args,rc)
try:
 for marker in ['FOUR_PHASE_COMPLETION.json','INITIALIZATION_COUPLING_COMPLETION.json']:
  while not(D/marker).exists():
   assert time.time()-beg<6500
   assert not list(D.glob('CONTROLLER_FAILURE_*.json'))and not list(D.glob('COUPLING_RESUME_FAILURE_*.json'))
   time.sleep(3)
 run('finalize_stvg_single_referdino_parameter_search256_v2.py',['select'])
 run('finalize_stvg_single_referdino_parameter_search256_v2.py',['score'])
 run('audit_stvg_single_referdino_selected256_v2.py',[])
 run('document_stvg_single_referdino_parameter_search256_v1.py',[])
 run('analyze_stvg_single_referdino_selected_failures256_v1.py',[])
 for ds in ['hc2','vidstg']:run('profile_stvg_single_referdino_tuned_gamma256_v1.py',[ds])
 run('report_stvg_single_referdino_parameter_search256_v2.py',[])
 put(D/'SELECTED_EVALUATION_AND_PROFILES_COMPLETION.json',dict(status='completed_selected_five_rows_independent_audit_two_complete_input_profiles_report_pending_publication',wall=time.time()-beg,scientific_search_stopped=True,automatic_new_grid=False))
 print('SELECTED_SCIENCE_AND_RUNTIME_COMPLETE',flush=True)
except BaseException as e:
 put(D/('FINISHER_FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(e),owned_active_pid=active.pid if active and active.poll()is None else None));raise
