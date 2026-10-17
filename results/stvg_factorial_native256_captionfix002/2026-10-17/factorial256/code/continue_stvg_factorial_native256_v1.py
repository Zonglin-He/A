"""Native CPU preparation, then serial GPU after current fixed224 queue."""
from pathlib import Path
import json,os,select,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_factorial_native256_v1 import BASE,verify,read,sha,put
UNIT='stvg-factorial-native256-001.service'
def main(pid):
    from scripts import launch_stvg_instance_identity_v1 as resource
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='native_factorial256';verify();stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    def run(stage,arm=None):
        verify();dest=BASE/'execution'/(arm or 'shared')/stage;dest.mkdir(parents=True,exist_ok=False);put(BASE/'STATUS.json',dict(status='running',stage=stage,arm=arm,count=256,GT_read=stage=='score'),replace=True);gpu=stage in ('features','qualification','teacher');interp='.venv/stvg_native_distill_v1/bin/python' if gpu else '.conda/tubedetr/bin/python';script='score_stvg_factorial_native256_v1.py' if stage=='score' else 'stvg_factorial_native256_v1.py';args=[] if stage=='score' else [stage]+([arm] if arm else [])
        with (dest/'worker.log').open('xb') as log:
            tick=time.perf_counter();p=subprocess.Popen([str(ROOT/interp),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(dest/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
        put(dest/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,(stage,arm)
    try:
        run('prepare');run('input_root');put(BASE/'STATUS.json',dict(status='waiting',stage='current_fixed224_exit_and_complete_roots',count=256,GT_read=False),replace=True)
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:fd=None
        if fd is not None:poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
        assert read(ROOT/'artifacts/stvg_caption_coverage256_v1/CPU_COMPLETION.json')['count']==256
        v=resource.sample();g=1024**3;assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
        run('features')
        for arm in ('A','B'):
            for stage in ('jobs','qualification','teacher','teacher_root','student_prepare','student_worker','student_root'):run(stage,arm)
        run('score');put(BASE/'STATUS.json',dict(status='root_handoff',stage='true_native_spatial_A_B_factorial_all_sealed_and_scored',count=256,GT_read=False),replace=True)
    except BaseException as e:
        put(BASE/'FAILURE.json',dict(status='technical_failure',type=type(e).__name__,error=str(e),all_partial_preserved=True,scientific_zero_not_imputed=True,GT_read=False));raise
    finally:stop.set()
if __name__=='__main__':main(int(sys.argv[1]))
