"""One finite CPU successor; waits on the existing controller, never restarts GPU."""
import sys,os,time,select,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_transform_common_v1 import *

def run():
    try:
        if not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
            launch=read(BASE/'LAUNCH_REVISION001.json');pid=launch['pid']
            status(BASE/'CPU_STAGE.json',dict(status='waiting_existing_GPU_controller',pid=os.getpid(),waiting_pid=pid,time=time.time(),GT_read=False))
            if hasattr(os,'pidfd_open'):
                try:fd=os.pidfd_open(pid)
                except ProcessLookupError:fd=None
                if fd is not None:
                    try:select.select([fd],[],[])
                    finally:os.close(fd)
            else:
                # This Conda Python lacks the optional Linux binding. Only wait
                # for this exact existing finite process; never spawn a GPU job.
                proc=Path('/proc')/str(pid)
                while proc.exists():
                    try:
                        stat=(proc/'stat').read_text();state=stat.rsplit(')',1)[1].split()[0]
                    except FileNotFoundError:break
                    if state=='Z':break
                    time.sleep(15)
        b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and not b['GT_read'] and b['unique_inputs']==576
        write(BASE/'CPU_SUCCESSOR_CODE_LOCK.json',dict(time=time.time(),pins={f:sha(ROOT/f) for f in [
            'scripts/continue_decota_transform_cpu_v1.py','scripts/score_decota_transform_p0_v1.py',
            'scripts/audit_decota_transform_p0_v1.py','scripts/report_decota_transform_p0_v1.py']}))
        stages=[('score','scripts/score_decota_transform_p0_v1.py'),('audit','scripts/audit_decota_transform_p0_v1.py'),('report','scripts/report_decota_transform_p0_v1.py')]
        for name,entry in stages:
            status(BASE/'CPU_STAGE.json',dict(status='running_'+name,pid=os.getpid(),time=time.time()))
            subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',entry],cwd=ROOT,check=True)
        write(BASE/'CPU_COMPLETION.json',dict(status='completed_pending_visual_publication',time=time.time(),decision=read(PUB/'DECISION.json')))
        status(BASE/'CPU_STAGE.json',dict(status='completed_pending_visual_publication',pid=os.getpid(),time=time.time()))
        status(BASE/'STATUS.json',dict(status='completed_pending_visual_publication',time=time.time()))
    except BaseException as e:
        status(BASE/'CPU_FAILURE.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()))
        raise

if __name__=='__main__':run()
