"""Finite R1 controller: calibration, factorial, global seal, independent CPU."""
import sys,os,time,subprocess,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_optimizer_posterior_common_v1 import *

def run():
    handle=(BASE/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    write(BASE/'LAUNCH.json',dict(pid=os.getpid(),argv=sys.argv,time=time.time(),finite=True))
    py=str(ROOT/'.conda/tubedetr/bin/python');worker='scripts/run_decota_optimizer_posterior_v1.py'
    commands=[('temporal','CPU_posterior',None,[py,'-B',worker,'temporal'])]
    for ds in DATASETS:
        for action in ['calibrate','predict']:
            commands.append((ds+'_'+action,action,ds,['bash','scripts/with_local_cuda.sh',py,'-B',worker,action,ds]))
    commands += [('seal','seal',None,[py,'-B',worker,'seal']),('score','CPU_audit',None,[py,'-B','scripts/score_decota_optimizer_posterior_v1.py']),
                 ('report','report',None,[py,'-B','scripts/report_decota_optimizer_posterior_v1.py'])]
    try:
        for name,stage,ds,cmd in commands:
            log=BASE/(name+'.log');assert not log.exists()
            with log.open('w') as out:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
                status(BASE/'STATUS.json',dict(status='running',stage=stage,dataset=ds,worker_pid=p.pid,controller_pid=os.getpid(),log=str(log),GT_read=stage in ['CPU_audit','report']))
                assert p.wait()==0,(name,p.returncode)
        status(BASE/'STATUS.json',dict(status='R1_completed_pending_root_continuation',all_route_complete=False,time=time.time()))
    except BaseException as e:
        write(BASE/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        status(BASE/'STATUS.json',dict(status='failed_pending_root',error=repr(e)));raise
    finally:handle.close()

if __name__=='__main__':run()
