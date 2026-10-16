"""Only missing A-BT updates; B-BT is exact R01, never duplicated."""
from pathlib import Path
import ast,copy,inspect,json,os,select,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_referent_abc002 import BASE as ABC,read,put,sha
BASE=ROOT/'artifacts/stvg_AB_backtrack_v1/revision002';UNIT='stvg-AB-backtrack-002.service'
def setup():
    BASE.mkdir(parents=True,exist_ok=False)
    for n in ('ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','TEMPORAL_LOCK.json'):(BASE/n).write_bytes((ABC/n).read_bytes())
    b=BASE/'trials/ABT';b.mkdir(parents=True)
    for n in ('REFERENCE_LOCK.json','TEMPORAL_LOCK.json'):(b/n).write_bytes((ABC/'trials/A'/n).read_bytes())
    d=copy.deepcopy(read(ABC/'trials/A/DESIGN_LOCK.json'));d.update(trial_id='ABT',recipe='lite_backtrack',scope='fixed A references; original four scales and unchanged rejection');put(b/'DESIGN_LOCK.json',d)
    put(BASE/'DESIGN_LOCK.json',dict(status='locked',count=16,arms=['A_Full','B_Full','A_BT','B_BT'],A_BT_only_new=True,B_BT='exact R01 reuse',scales=[1,.5,.25,.125],unchanged_acceptance=True,unchanged_source_Temporal_and_hyperparameters=True,GT_read=False,all_development=True,bootstrap=False,CURRENT_change=False))
    files=[Path(__file__),BASE/'DESIGN_LOCK.json']+[BASE/n for n in ('ROSTER_LOCK.json','SOURCE_CAPTURE_BARRIER.json','TEMPORAL_LOCK.json')]+[b/n for n in ('DESIGN_LOCK.json','REFERENCE_LOCK.json','TEMPORAL_LOCK.json')]+[ABC/'trials/A/GLOBAL_PREDICTION_BARRIER.json',ABC/'trials/B/GLOBAL_PREDICTION_BARRIER.json',ROOT/'artifacts/stvg_B_transfer_v1/trials/R01/ROOT_READBACK.json',ROOT/'scripts/run_stvg_instance_student32_v1.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py']
    pins={str(p.relative_to(ROOT)):sha(p)for p in files};put(BASE/'RUNTIME.json',dict(status='locked',pins=pins,GT_read=False));rt=copy.deepcopy(read(ABC/'trials/A/RUNTIME.json'));rt.update(trial_id='ABT');rt['pins'].update(pins);rt['pins'][str((BASE/'RUNTIME.json').relative_to(ROOT))]=sha(BASE/'RUNTIME.json');put(b/'RUNTIME.json',rt);print('A_BT_LOCKED')
def check(arm):
    for rel,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/rel)==h,rel
    rt=read(BASE/'trials'/arm/'RUNTIME.json')
    for rel,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/rel)==h,rel
def run(mode):
    from scripts import run_stvg_B_transfer_v1 as old
    old.BASE=BASE;old.check=check;old.recipe=lambda _: 'lite_backtrack';old.execute(mode,'ABT')
def supervise():
    from scripts import run_stvg_B_transfer_v1 as old
    old.BASE=BASE;old.UNIT=UNIT;old.ARMS=('ABT',)
    # Its subprocess entrypoint is pinned and still addresses its own BASE;
    # run our own sequential worker/root processes under the same guards.
    from scripts import launch_stvg_instance_identity_v1 as res
    import threading
    res.BASE=BASE;res.UNIT=UNIT;res.STAGE='A_BT';v=res.sample();g=1024**3;assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=res.monitor,args=(stop,),daemon=True).start()
    try:
        for mode in ('worker','root'):
            out=BASE/'execution/ABT'/mode;out.mkdir(parents=True,exist_ok=False)
            with(out/'worker.log').open('xb')as log:
                t=time.perf_counter();p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__)),mode],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=False));rc=p.wait()
            put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=False));assert rc==0
        put(BASE/'CPU_HANDOFF.json',dict(status='measurements_complete_pending_score_public',count=16,GT_read=False))
    finally:stop.set()
def after_prompt(pid):
    fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
    for a in ('FULL_ORDER','A_ORDER'):
        b=ROOT/'artifacts/stvg_spatial_who_order_v1/trials'/a;assert read(b/'ROOT_READBACK.json')['status']=='pass'and read(b/'GLOBAL_PREDICTION_BARRIER.json')['count']==16
    supervise()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='setup':setup()
    elif mode=='after_prompt':after_prompt(int(sys.argv[2]))
    elif mode=='supervise':supervise()
    else:run(mode)
