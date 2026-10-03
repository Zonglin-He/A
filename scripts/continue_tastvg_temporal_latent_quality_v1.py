"""One finite serial source-fit/target-readout queue; never resumes other tasks."""
import os,sys,time,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status
BASE=ROOT/'artifacts/tastvg_temporal_latent_quality_v1'
STAGES=[['capture_source','vidstg'],['capture_source','hc2'],['fit'],
        ['capture_target','vidstg'],['capture_target','hc2'],['seal'],['diagnose']]

def main():
    assert not (BASE/'QUEUE_STATUS.json').exists(),'Do not duplicate controller'
    write(BASE/'LAUNCH.json',dict(controller_pid=os.getpid(),stages=STAGES,time=time.time()))
    for i,args in enumerate(STAGES):
        name='_'.join(args);log=BASE/(name+'.log')
        with log.open('x') as f:
            p=subprocess.Popen([sys.executable,'-B','scripts/run_tastvg_temporal_latent_quality_v1.py',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            status(BASE/'QUEUE_STATUS.json',dict(status='running',stage=args,index=i,
                worker_pid=p.pid,controller_pid=os.getpid(),time=time.time(),log=str(log.relative_to(ROOT))))
            code=p.wait()
        if code:
            status(BASE/'QUEUE_STATUS.json',dict(status='failed',stage=args,exit_code=code,time=time.time()));return code
    status(BASE/'QUEUE_STATUS.json',dict(status='completed_pending_root_audit_report_publication',time=time.time()))
    return 0

if __name__=='__main__':sys.exit(main())
