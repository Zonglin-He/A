"""Finite P1 chain, global barrier, then CPU audit and report."""
import sys,os,time,subprocess,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_critic_ln_common_v1 import *


def run():
    handle=(BASE/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
    write(BASE/'LAUNCH.json',dict(pid=os.getpid(),time=time.time(),argv=sys.argv,finite=True))
    commands=[]
    for ds in DATASETS:
        commands.append((f'{ds}_predict','predict',ds,['bash','scripts/with_local_cuda.sh',
            str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_tastvg_decota_critic_ln_p1_v1.py','predict',ds]))
    commands += [('seal','seal',None,[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_tastvg_decota_critic_ln_p1_v1.py','seal']),
        ('CPU','CPU_scoring_audit',None,[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/score_audit_tastvg_decota_critic_ln_p1_v1.py','run']),
        ('REPORT','reporting',None,[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/report_tastvg_decota_critic_ln_p1_v1.py'])]
    try:
        for name,stage,ds,cmd in commands:
            log=BASE/f'{name}.log';assert not log.exists()
            with log.open('w') as out:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
                status(BASE/'STATUS.json',dict(status='running',stage=stage,dataset=ds,
                    controller_pid=os.getpid(),worker_pid=p.pid,log=str(log),GT_read=stage in ['CPU_scoring_audit','reporting']))
                assert p.wait()==0,(stage,ds,p.returncode)
        status(BASE/'STATUS.json',dict(status='completed_pending_root_visual_publication',arrivals=1152,time=time.time()))
    except BaseException as e:
        write(BASE/'CONTROLLER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        status(BASE/'STATUS.json',dict(status='failed_pending_root',error=repr(e),time=time.time()));raise
    finally:handle.close()


if __name__=='__main__':run()
