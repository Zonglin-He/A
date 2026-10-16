"""Exact frame-grid equality adapter, retaining the first failed intake."""
from pathlib import Path
import ast,inspect,json,os,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_stvg_spatial_controls256_v1 as old
BASE=old.BASE;UNIT='stvg-spatial-controls256-source-002.service'
def verify():
    old.verify()
    for p,h in old.read(BASE/'SOURCE_RECOVERY_RUNTIME002.json')['pins'].items():assert old.sha(ROOT/p)==h,p
def source():
    verify();s=inspect.getsource(old.source);target="c['frame_ids']==rec['frame_ids']";assert s.count(target)==1;s=s.replace(target,"np.array_equal(c['frame_ids'],rec['frame_ids'])");ast.parse(s);ns=dict(old.__dict__);ns['verify']=verify;exec(compile(s,str(Path(__file__)),'exec'),ns);ns['source']();old.put(BASE/'SOURCE_RECOVERY_COMPLETION002.json',dict(status='complete',count=256,original_runtime_sha256=old.sha(BASE/'SOURCE_RUNTIME.json'),recovery_runtime_sha256=old.sha(BASE/'SOURCE_RECOVERY_RUNTIME002.json'),barrier_sha256=old.sha(BASE/'SOURCE_CAPTURE_BARRIER.json'),GT_read=False))
def supervise():
    from scripts import launch_stvg_instance_identity_v1 as r
    verify();r.BASE=BASE;r.UNIT=UNIT;r.STAGE='source256';g=1024**3;v=r.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=r.monitor,args=(stop,),daemon=True).start();out=BASE/'execution/source002';out.mkdir(parents=True,exist_ok=False)
    try:
        with(out/'worker.log').open('xb')as log:
            t=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__)),'source'],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);old.put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
        old.put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0
    finally:stop.set()
if __name__=='__main__':{'source':source,'supervise':supervise}[sys.argv[1]]()
