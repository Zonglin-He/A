"""Finite dependent CPU continuation, with explicit GPU handoff evidence."""
import os,sys,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_three_scope_common_v1 import BASE,PUB,read,write,status,sha
PY=ROOT/'.conda/tubedetr/bin/python'

def wait_for(files,controller):
    while not all(f.exists() for f in files):
        assert Path(f'/proc/{controller}').exists(),'GPU controller disappeared without required handoff'
        time.sleep(2)

def run(script,action):
    status(BASE/'CPU_STAGE.json',dict(status='running',stage=action,pid=os.getpid(),time=time.time()))
    with (BASE/(action+'_CPU.log')).open('a') as out:subprocess.run([str(PY),'-B',script,action],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)

def main():
    with (BASE/'CPU_controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(BASE/'CPU_LAUNCH.json',dict(pid=os.getpid(),time=time.time(),finite=True,code_sha256=sha(Path(__file__))))
        gpu=read(BASE/'LAUNCH.json')['pid']
        try:
            status(BASE/'CPU_STAGE.json',dict(status='waiting_native_key_barrier',pid=os.getpid(),gpu_pid=gpu,time=time.time()))
            wait_for([BASE/ds/'FEATURE_BARRIER.json' for ds in ['vidstg','hc2']],gpu)
            if not (BASE/'MEMORY_KEY_BARRIER.json').exists():run('scripts/score_decota_three_scope_v1.py','memory_prepare')
            if not (BASE/'MEMORY_DECISION.json').exists():run('scripts/score_decota_three_scope_v1.py','memory_score')
            status(BASE/'CPU_STAGE.json',dict(status='waiting_new_prediction_barrier',pid=os.getpid(),gpu_pid=gpu,time=time.time()))
            wait_for([BASE/'GLOBAL_PREDICTION_BARRIER.json'],gpu)
            if not (BASE/'SCOPE_DECISION.json').exists():run('scripts/score_decota_three_scope_v1.py','scope_score')
            if not (PUB/'ROOT_AUDIT.json').exists():run('scripts/audit_decota_three_scope_v1.py','root')
            run('scripts/report_decota_three_scope_v1.py','report')
            if not (PUB/'PUBLIC_AUDIT.json').exists():run('scripts/audit_decota_three_scope_v1.py',str(PUB))
            status(BASE/'CPU_STAGE.json',dict(status='completed_pending_root_visual_conditional_publication',pid=os.getpid(),time=time.time()))
        except Exception as e:
            write(BASE/'CPU_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
            status(BASE/'CPU_STAGE.json',dict(status='engineering_failed_pending_root',pid=os.getpid(),time=time.time()));raise

if __name__=='__main__':main()
