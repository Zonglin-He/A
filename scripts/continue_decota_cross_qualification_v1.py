"""Finite newly authorized R6 worker; no legacy controller is resumed."""
import sys,os,time,fcntl,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_cross_qualification_common_v1 import *

def run():
    handle=(CROSS/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    write(CROSS/f'LAUNCH_{os.getpid()}.json',dict(pid=os.getpid(),time=time.time(),finite=True,argv=sys.argv))
    py=str(ROOT/'.conda/tubedetr/bin/python');worker='scripts/run_decota_cross_qualification_v1.py'
    def execute(name,cmd,complete,GT=False):
        if complete.exists():print('REUSE_SEALED_CROSS_STAGE',name,flush=True);return
        log=CROSS/f'{name}_{os.getpid()}.log'
        with log.open('w') as out:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
            status(CROSS/'STATUS.json',dict(status='running',stage=name,controller_pid=os.getpid(),worker_pid=proc.pid,log=str(log),GT_read=GT))
            status(BASE/'STATUS.json',dict(status='R6_cross_running',stage=name,controller_pid=os.getpid(),worker_pid=proc.pid,GT_read=GT))
            assert proc.wait()==0,(name,proc.returncode)
    try:
        execute('smoke',['bash','scripts/with_local_cuda.sh',py,'-B',worker,'smoke'],CROSS/'SMOKE_ROOT_ACCEPTANCE.json')
        for ds in DATASETS:
            execute('capture_'+ds,['bash','scripts/with_local_cuda.sh',py,'-B',worker,'capture',ds],CROSS/ds/'CAPTURE_BARRIER.json')
            execute('predict_'+ds,['bash','scripts/with_local_cuda.sh',py,'-B',worker,'predict',ds],CROSS/ds/'PREDICTION_BARRIER.json')
        execute('global_seal',[py,'-B',worker,'seal'],CROSS/'GLOBAL_PREDICTION_BARRIER.json')
        execute('root_score',[py,'-B','scripts/score_decota_cross_qualification_v1.py'],PUB/'cross_domain'/'PUBLIC_AUDIT.json',True)
        status(CROSS/'STATUS.json',dict(status='completed_pending_root_visual_publication',time=time.time(),cross_domain_measured=True))
        status(BASE/'STATUS.json',dict(status='all_executed_stages_completed_pending_root_visual_publication',cross_domain_complete=True,time=time.time()))
        archive('R6新跨域576输入与2304读出已全部封存并CPU独立根审计；待总报告图/代码结果公开核验，未晋升方法')
    except BaseException as e:
        write(CROSS/f'FAILURE_{os.getpid()}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        status(CROSS/'STATUS.json',dict(status='failed_pending_root',error=repr(e),time=time.time()));raise
    finally:handle.close()

if __name__=='__main__':run()
