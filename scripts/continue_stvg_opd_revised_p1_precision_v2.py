"""Root-authorized fixed P1; original controller plus pinned audit supplement."""
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,PYTHON,read,sha,status,verify
from scripts.run_stvg_opd_gradient_recovery_001 import activate_revision

def run():
    verify();activate_revision()
    runtime=read(BASE/'P1_PRECISION_RUNTIME.json')
    for f,h in runtime['pins'].items():assert sha(ROOT/f)==h
    assert read(BASE/'P0_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    assert read(BASE/'P1_LAUNCH_AUTHORIZATION.json')['root_decision_after_actual_revised_P0_readback']
    import scripts.continue_stvg_opd_paper_hc2_revision_v2 as original
    def child(job,*args,cpu=False):
        verify();cmd=[str(PYTHON),'-B','scripts/run_stvg_opd_gradient_recovery_001.py',job,*args]
        log=BASE/'logs'/('P1_precision_'+ '_'.join([job,*args])+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as out:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''} if cpu else None)
            status(BASE/'STAGE.json',dict(status='running',controller_pid=os.getpid(),worker_pid=p.pid,
                job=job,args=list(args),command=cmd,log=str(log.relative_to(ROOT)),CPU=cpu,
                precision_runtime_sha256=sha(BASE/'P1_PRECISION_RUNTIME.json'),time=time.time()))
            assert p.wait()==0,'Preserve failure before further engineering diagnosis'
    original.child=child;original.run('P1')

if __name__=='__main__':run()
