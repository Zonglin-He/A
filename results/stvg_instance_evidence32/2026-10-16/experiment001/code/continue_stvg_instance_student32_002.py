"""Continue after saved E0 output, without repeating any GPU fit."""
from pathlib import Path
import json,os,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
UNIT='stvg-instance-evidence32-student-002.service'
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    from scripts.run_stvg_instance_student32_v1 import check
    b=BASE/'recovery/reader_precision002';rt=read(b/'RUNTIME.json')
    for rel,h in rt['pins'].items():assert sha(ROOT/rel)==h,rel
    assert read(BASE/'student_execution/E0/worker/WORKER_EXIT.json')['exit_code']==0
    assert read(BASE/'trials/E0/STUDENT_GLOBAL_BARRIER.json')['count']==32
    oldstate=subprocess.check_output(['systemctl','--user','show','stvg-instance-evidence32-student-001.service','--property=ActiveState','--value'],text=True).strip();assert oldstate in ('inactive','failed')
    resource.BASE=b;resource.UNIT=UNIT;resource.STAGE='student_continuation';g=1024**3;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
    stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for tid,stage,script,args in [('E0','root','review_stvg_instance_student32_002.py',['E0']),('E1','worker','run_stvg_instance_student32_v1.py',['worker','E1']),('E1','root','review_stvg_instance_student32_002.py',['E1'])]:
            p=b/'execution'/tid/stage;p.mkdir(parents=True,exist_ok=False)
            with(p/'worker.log').open('xb')as log:
                start=time.perf_counter();c=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT);put(p/'LAUNCH.json',dict(status='running',pid=c.pid,GT_read=False));code=c.wait()
            put(p/'WORKER_EXIT.json',dict(status='exited',exit_code=code,wall_seconds=time.perf_counter()-start,GT_read=False));assert code==0,(tid,stage)
        put(BASE/'EXPERIMENT_GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,trials={t:sha(BASE/'trials'/t/'GLOBAL_PREDICTION_BARRIER.json')for t in ('E0','E1')},GT_read=False,reader_recovery_sha256=sha(b/'RUNTIME.json')))
    finally:stop.set()
if __name__=='__main__':main()
