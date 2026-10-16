"""Single pidfd wait for P2, then finite native qualification8 if budget remains."""
from pathlib import Path
import os,select,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
from scripts.stvg_overnight_tubedetr_v1 import BASE,verify
def main(pid):
    verify();c.write(BASE/'LAUNCH.json',dict(status='waiting',stage='confirmation_controller_exit',pid=os.getpid(),count=8,GT_read=False));fd=os.pidfd_open(pid);p=select.poll();p.register(fd,select.POLLIN);p.poll(max(1,int((c.deadline()-time.time())*1000)));os.close(fd)
    if not(c.BASE/'confirmation128/MEASUREMENT_COMPLETION.json').exists():c.write(BASE/'STATUS.json',dict(status='requires_root',stage='confirmation_predecessor_not_complete',count=8,GT_read=False),True);return
    if c.deadline()-time.time()<1800:c.write(BASE/'STATUS.json',dict(status='window_checkpoint',stage='qualification_not_started_last30min',count=8,GT_read=False),True);return
    c.reserve('TubeDETR_engineering_qualification8',fits=8);out=BASE/'execution';out.mkdir(exist_ok=False)
    with (out/'worker.log').open('xb') as f:
        t=time.perf_counter();child=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/stvg_overnight_tubedetr_v1.py')],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT);c.write(out/'LAUNCH.json',dict(status='running',pid=child.pid,count=8,GT_read=False));rc=child.wait(timeout=max(1,c.deadline()-time.time()))
    c.write(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));c.write(BASE/'STATUS.json',dict(status='root_handoff' if rc==0 else 'technical_unqualified',stage='native_interface_root_effect_adapter_publication_required',count=8,GT_read=False),True)
if __name__=='__main__':main(int(sys.argv[1]))
