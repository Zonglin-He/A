"""Finite CPU handoff for the already authorized correction-scope predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_correction_scope_common_v1 import *
from scripts.tastvg_cpu_handoff_v1 import verify_cpu

def run():
    verify_cpu(BASE)
    write(BASE/'CPU_LAUNCH.json',dict(pid=os.getpid(),time=time.time(),new_GPU_calls=0))
    try:
        status(BASE/'CPU_STAGE.json',dict(status='waiting_for_existing_global_barrier',pid=os.getpid(),time=time.time()))
        while not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
            st=read(BASE/'STATUS.json');assert not st['status'].startswith('failed'),st
            os.kill(read(BASE/'LAUNCH.json')['controller_pid'],0);time.sleep(10)
        seal()
        commands=[('score_tastvg_correction_scope_v1.py',[]),('audit_tastvg_correction_scope_v1.py',[]),
            ('report_tastvg_correction_scope_v1.py',[]),('audit_tastvg_correction_scope_v1.py',[str(PUB)])]
        for script,args in commands:
            verify_cpu(BASE)
            status(BASE/'CPU_STAGE.json',dict(status='running',script=script,args=args,pid=os.getpid(),time=time.time()))
            log=BASE/'logs'/('CPU_'+script+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:
                subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts'/script),*args],
                    cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        write(BASE/'CPU_COMPLETION.json',dict(status='completed_pending_root_visual_publication',time=time.time(),
            root_audit_sha256=sha(BASE/'ROOT_AUDIT.json'),public_audit_sha256=sha(BASE/'PUBLIC_EXPORT_AUDIT.json')))
        status(BASE/'CPU_STAGE.json',dict(status='completed_pending_root_visual_publication',time=time.time()))
    except BaseException as e:
        status(BASE/'CPU_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        status(BASE/'CPU_STAGE.json',dict(status='failed_pending_root',time=time.time()));raise

if __name__=='__main__':run()
