"""Same32 recipe, corrected no-reuse descriptor and qualified frozen-prefix path."""
from pathlib import Path
import ast,concurrent.futures,inspect,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_global32_student_v1 as old
from scripts import stvg_pipeline_speed_v1 as speed
from scripts.run_stvg_controls256_speed_student_v1 import source_patch
BASE,read,put,sha=old.BASE,old.read,old.put,old.sha
def verify():
    old.verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME002.json')['pins'].items():assert sha(ROOT/p)==h,p
def prepare_source():
    s=inspect.getsource(old.prepare);assert s.count('family_reuse_from=None')==1
    s=s.replace('family_reuse_from=None',"family_reuse_from={ds:{m:None for m in d['configs'][ds]} for ds in d['configs']}")
    needle='for p in [Path(__file__),';assert s.count(needle)==1
    return s.replace(needle,"for p in [Path(__file__), BASE/'EXECUTION_RUNTIME002.json', ROOT/'artifacts/stvg_pipeline_speed_v1/QUALIFICATION_COMPLETION.json',")
def prepare():
    verify();ns=dict(old.__dict__);ns.update(verify=verify,__file__=str(Path(__file__)));exec(compile(prepare_source(),str(Path(__file__)),'exec'),ns);ns['prepare']()
def check(tid):verify();old.check(tid)
def worker():
    from scripts import run_stvg_instance_student32_v1 as original
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    q=read(speed.BASE/'QUALIFICATION_COMPLETION.json');enabled=q['status']=='pass' and q['engineering_adopted']
    engine=speed.Engine('G32',enabled);pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);pending=[];seconds=[]
    def enqueue(save,dest,payload):
        p=dict(payload);p['path']=list(payload['path'])
        def task():
            t=time.perf_counter();r=save(dest,p);seconds.append(time.perf_counter()-t);return r
        pending.append(pool.submit(task))
    def flush():
        while pending:pending.pop(0).result()
    def install(w):w.__dict__.update(speed_engine=engine,speed_enqueue=enqueue,speed_flush=flush)
    s=inspect.getsource(original.worker);needle='tree=ast.parse(Path(original.__file__).read_text())';assert s.count(needle)==1
    s=s.replace(needle,'tree=ast.parse(speed_source_patch(Path(original.__file__).read_text()))');assert s.count('w.main()')==1;s=s.replace('w.main()','speed_install(w);w.main()')
    ns=dict(original.__dict__);ns.update(BASE=BASE,check=check,modes=lambda _:('M2',),speed_source_patch=lambda t:source_patch(t,enabled),speed_install=install)
    capture,readout=h._capture_source,h._readout;h._capture_source=engine.capture;h._readout=engine.readout
    try:
        exec(compile(s,str(Path(__file__)),'exec'),ns);ns['worker']('G32');flush();engine.receipt()
        put(speed.BASE/'student_profiles/G32_IO.json',dict(status='complete',count=32,checkpoint_writes=len(seconds),checkpoint_writer_seconds=sum(seconds),full_original_step_checkpoints_preserved=True,GT_read=False))
    finally:flush();pool.shutdown(wait=True);h._capture_source=capture;h._readout=readout
def root():
    ns=dict(old.__dict__);ns.update(check=check,__file__=str(Path(__file__)));exec(compile(inspect.getsource(old.execute),str(Path(__file__)),'exec'),ns);ns['execute']('root')
def contracts():
    ast.parse(prepare_source());assert 'family_reuse_from=None' not in prepare_source()
    print('GLOBAL32_SPEED_DESCRIPTOR_AND_SOURCE_ADAPTER_CPU_CONTRACTS_PASS_NO_MODEL_NO_GT')
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else {'prepare':prepare,'worker':worker,'root':root}[sys.argv[1]]()
