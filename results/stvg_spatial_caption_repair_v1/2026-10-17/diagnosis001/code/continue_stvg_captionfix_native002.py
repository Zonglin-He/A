"""Adopt existing feature work to normal exit, then decoupled caption repair."""
from pathlib import Path
import json,os,select,signal,subprocess,sys,time,threading,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_spatial_caption_repair_v1 import BASE,NATIVE,FIXED,read,put,sha,verify
from scripts.continue_stvg_controls256_speed004 import identity
UNIT='stvg-captionfix-native-002.service';OLD_UNIT='stvg-factorial-native256-001.service'
def adopt():
    verify();lock=read(BASE/'LIVE_HANDOFF_LOCK.json');controller=lock['controller']['pid'];worker=lock['feature']['pid']
    for kind,pid in [('controller',controller),('feature',worker)]:
        now=identity(pid)
        for k in ('pid','ppid','start_ticks','cmdline_sha256','cgroup'):assert now[k]==lock[kind][k]
    assert identity(worker)['ppid']==controller;os.kill(controller,signal.SIGSTOP);time.sleep(.05);assert identity(controller)['state'] in ('T','t');put(BASE/'CONTROLLER_HANDOFF_STATUS.json',dict(status='waiting_original_features_normal_exit',pid=worker,controller_pid=controller,live_GPU_worker_signalled=False,GT_read=False))
    fd=os.pidfd_open(worker);poll=select.poll();poll.register(fd,select.POLLIN)
    from scripts import launch_stvg_instance_identity_v1 as resource
    cg=Path('/sys/fs/cgroup')/lock['feature']['cgroup'].splitlines()[0][3:].lstrip('/');errors=0
    try:
        while not poll.poll(1000):
            try:
                stats={k:int(v) for k,v in (x.split() for x in (cg/'memory.stat').read_text().splitlines())};available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'));free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0])*2**20;why=resource.trigger(dict(current=int((cg/'memory.current').read_text()),anon=stats.get('anon',0)+stats.get('shmem',0),available=available,gpu_free=free,own_gpu_active=True,disk_free=shutil.disk_usage(ROOT).free));errors=0
            except Exception:
                errors+=1
                if errors<3:continue
                why='three_resource_sampling_failures'
            if why:
                put(BASE/'HANDOFF_RESOURCE_STOP_RECEIPT.json',dict(status='resource_stop',reason=why,GT_read=False));subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=SIGTERM',OLD_UNIT],check=True);raise RuntimeError(why)
        end=identity(worker);assert end['state']=='Z' and end['start_ticks']==lock['feature']['start_ticks'];code=os.waitstatus_to_exitcode(end['exit_status']);assert code==0;bar=read(NATIVE/'FEATURE_BARRIER.json');assert bar['count']==256 and bar['status']=='complete';put(BASE/'ADOPTED_FEATURE_WORKER_COMPLETION.json',dict(status='complete',count=256,exit_code=code,barrier_sha256=sha(NATIVE/'FEATURE_BARRIER.json'),GPU_worker_never_signalled=True,GT_read=False));subprocess.run(['systemctl','--user','stop',OLD_UNIT],check=True,timeout=120);put(BASE/'CONTROLLER_HANDOFF_COMPLETION.json',dict(status='complete',count=256,old_controller_retired_after_GPU_exit=True,old_GPU_features_preserved=True,GT_read=False))
    finally:os.close(fd)
def main():
    adopt()
    from scripts import launch_stvg_instance_identity_v1 as resource
    resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='captionfix_native';stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    def run(stage,arm=None):
        verify();dest=BASE/'execution'/(arm or 'shared')/stage;dest.mkdir(parents=True,exist_ok=False);put(BASE/'STATUS.json',dict(status='running',stage=stage,arm=arm,count=54 if stage in ('prepare','generate') else 256,GT_read=stage=='score'),replace=True)
        if FIXED.exists():put(FIXED/'STATUS.json',dict(status='running',stage=stage,arm=arm,count=256,GT_read=stage=='score'),replace=True)
        script='score_stvg_factorial_captionfix002.py' if stage=='score' else 'stvg_spatial_caption_repair_v1.py';args=[] if stage=='score' else [stage]+([arm] if arm else []);inter='.venv/stvg_native_distill_v1/bin/python' if stage in ('generate','features_fixed','qualification','teacher') else '.conda/tubedetr/bin/python'
        with (dest/'worker.log').open('xb') as log:
            tick=time.perf_counter();p=subprocess.Popen([str(ROOT/inter),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(dest/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'));rc=p.wait()
        put(dest/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,(stage,arm)
    try:
        if not (BASE/'INPUT_BARRIER.json').exists():run('prepare')
        run('generate');run('setup_fixed');run('features_fixed')
        for arm in ('A','B'):
            for stage in ('jobs','qualification','teacher','teacher_root','student_prepare','student_worker','student_root'):run(stage,arm)
        run('score');put(BASE/'STATUS.json',dict(status='root_handoff',stage='caption_repair_and_matched_Native_factorial_complete',count=256,GT_read=False),replace=True)
    except BaseException as e:
        put(BASE/'FAILURE.json',dict(status='technical_failure',type=type(e).__name__,error=str(e),all_old_and_partial_outputs_preserved=True,GT_read=False));raise
    finally:stop.set()
if __name__=='__main__':main()
