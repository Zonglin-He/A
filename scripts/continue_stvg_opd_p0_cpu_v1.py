"""Bounded CPU continuation: labels are inaccessible until the real global P0 seal."""
import os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *

def run():
    pins=read(BASE/'CPU_RUNTIME_LOCK.json')
    for f,h in pins['pins'].items():assert sha(ROOT/f)==h,f
    start=time.time();controller=read(BASE/'LAUNCH.json')['controller_pid']
    while not (BASE/'P0_GPU_COMPLETION.json').exists():
        assert time.time()-start<4*3600,'Finite P0 wait expired; root must inspect preserved state.'
        assert Path(f'/proc/{controller}').exists(),'P0 controller disappeared before global completion.'
        time.sleep(20)
    assert read(BASE/'P0_PREDICTION_BARRIER.json')['all_deployment_arms_both_directions']
    for ds in DATASETS:
        command=[str(PYTHON),'-B','scripts/score_stvg_opd_paper_v1.py','P0_'+ds]
        logfile=BASE/'logs'/('P0_'+ds+'_CPU.log')
        with logfile.open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            status(BASE/'CPU_STAGE.json',dict(status='postseal_scoring',pid=os.getpid(),worker_pid=child.pid,
                stage='P0_'+ds,command=command,log=str(logfile.relative_to(ROOT)),time=time.time()))
            assert child.wait()==0,'P0 postseal CPU worker failed; preserve evidence, no GPU rerun.'
    command=[str(PYTHON),'-B','scripts/finalize_stvg_opd_p0_v1.py']
    with (BASE/'P0_root_CPU.log').open('a') as log:
        assert subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode==0

if __name__=='__main__':
    try:run()
    except BaseException:
        d=BASE/'CPU_controller_failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True)
        (d/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'CPU_STAGE.json',dict(status='failed_preserved',evidence=str(d),time=time.time()))
        raise
