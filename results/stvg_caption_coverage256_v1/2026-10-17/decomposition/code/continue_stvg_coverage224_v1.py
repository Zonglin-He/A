"""Finite complementary224 queue; fixed noCoverage, no branching search."""
from pathlib import Path
import json,os,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_coverage224_v1 import BASE,EXP,verify,read,sha,put
UNIT='stvg-coverage224-001.service'
def main():
    verify();assert read(BASE/'CAPTION_DECOMPOSITION_COMPLETION.json')['status']=='complete'
    from scripts import launch_stvg_instance_identity_v1 as resource
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='fixed_coverage224';v=resource.sample();g=1024**3;assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for stage in ('jobs','qualification','teacher','teacher_root','student_prepare','student_worker','student_root','score'):
            verify();p=EXP/'execution'/stage
            if (p/'WORKER_EXIT.json').exists():assert read(p/'WORKER_EXIT.json')['exit_code']==0;continue
            p.mkdir(parents=True,exist_ok=False);script='score_stvg_coverage256_v1.py' if stage=='score' else 'run_stvg_coverage224_v1.py';interpreter='.venv/stvg_native_distill_v1/bin/python' if stage in ('qualification','teacher') else '.conda/tubedetr/bin/python';args=[] if stage=='score' else [stage];put(BASE/'STATUS.json',dict(status='running',stage=stage,count=224,GT_read=stage=='score'),replace=True)
            with (p/'worker.log').open('xb') as log:
                tick=time.perf_counter();w=subprocess.Popen([str(ROOT/interpreter),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(p/'LAUNCH.json',dict(status='running',pid=w.pid,GT_read=stage=='score'));rc=w.wait()
            put(p/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
        put(BASE/'STATUS.json',dict(status='root_handoff',stage='all256_initial32_complement224_sealed_scored',count=256,GT_read=False),replace=True)
    except BaseException as e:
        put(BASE/'FAILURE.json',dict(status='technical_failure',type=type(e).__name__,error=str(e),all_partial_preserved=True,scientific_zero_not_imputed=True,GT_read=False));raise
    finally:stop.set()
if __name__=='__main__':main()
