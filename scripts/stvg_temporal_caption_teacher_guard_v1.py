"""Own-job RAM/GPU headroom guard; never signals another experiment."""
from pathlib import Path
import subprocess,time,json
GiB=1024**3

def reason(current,anon,available,gpu_free,own_gpu_active):
    if current>=18*GiB:return 'own_cgroup_RAM_reached_18GiB_soft_stop'
    if anon>=14*GiB:return 'own_nonreclaimable_RAM_reached_14GiB_soft_stop'
    if available<6*GiB:return 'host_MemAvailable_below_6GiB_soft_stop'
    if own_gpu_active and gpu_free is not None and gpu_free<4*GiB:return 'GPU_headroom_below_4GiB_soft_stop'
    return None

def cgroup():
    text=Path('/proc/self/cgroup').read_text();line=next(x for x in text.splitlines() if x.startswith('0::'))
    name=line[3:]
    if not name.endswith('/stvg-temporal-caption-teacher-001.service'):raise RuntimeError('guard only authorized own transient job')
    return Path('/sys/fs/cgroup')/name.lstrip('/')

def sample(cg,gpu=False):
    stats={k:int(v) for k,v in (line.split() for line in (cg/'memory.stat').read_text().splitlines())}
    mem={line.split(':',1)[0]:int(line.split(':',1)[1].split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':',1)[0]=='MemAvailable'}
    result={'current':int((cg/'memory.current').read_text()),'anon':stats.get('anon',0)+stats.get('shmem',0),'available':mem['MemAvailable'],'gpu_free':None,'own_gpu_active':False}
    if gpu:
        own={int(x) for p in cg.rglob('cgroup.procs') for x in p.read_text().split()}
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=3)
        result['own_gpu_active']=bool(own & {int(x.strip()) for x in rows.splitlines() if x.strip().isdigit()})
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=3)
        result['gpu_free']=int(free.splitlines()[0].strip())*1024**2
    return result

def monitor(stop,base):
    cg=cgroup();last_gpu=0
    while not stop.wait(.25):
        try:
            gpu=time.monotonic()-last_gpu>=1
            values=sample(cg,gpu=gpu)
            if gpu:last_gpu=time.monotonic()
            trigger=reason(**values)
            if trigger:
                data={'status':'proactive_resource_guard_stop_before_kernel_OOM','reason':trigger,'sample':values,'GT_read':False,'only_owned_transient_job_signalled':True,'time':time.time()}
                path=base/'RESOURCE_GUARD_STOP001.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
                subprocess.run(['systemctl','--user','kill','--signal=SIGTERM','--kill-whom=all','stvg-temporal-caption-teacher-001.service'],check=True,timeout=5)
                return
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            # Sampling failure is reported; it never authorizes another process signal.
            (base/'RESOURCE_GUARD_SAMPLING_ERROR001.json').write_text(json.dumps({'status':'resource_sampling_error','error_type':type(exc).__name__,'GT_read':False,'time':time.time()}))

def contracts():
    assert reason(10*GiB,5*GiB,12*GiB,8*GiB,True) is None
    assert reason(18*GiB,5*GiB,12*GiB,8*GiB,True).startswith('own_cgroup')
    assert reason(10*GiB,14*GiB,12*GiB,8*GiB,True).startswith('own_nonreclaimable')
    assert reason(10*GiB,5*GiB,5*GiB,8*GiB,True).startswith('host_')
    assert reason(10*GiB,5*GiB,12*GiB,3*GiB,True).startswith('GPU_')
    assert reason(10*GiB,5*GiB,12*GiB,3*GiB,False) is None
    return {'status':'pass','CPU_only':True,'GT_read':False,'synthetic_policy_checks':6,'signals_sent':False}
if __name__=='__main__':print(contracts())
