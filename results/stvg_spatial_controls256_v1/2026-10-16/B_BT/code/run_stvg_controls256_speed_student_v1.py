"""Versioned descriptor correction and qualified engineering-only Student path."""
from pathlib import Path
import ast,concurrent.futures,copy,gc,inspect,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_spatial_controls256_student_v1 as old
from scripts import stvg_pipeline_speed_v1 as speed
BASE,read,put,sha=old.BASE,old.read,old.put,old.sha
def verify():
    from scripts.continue_stvg_spatial_controls256_native003 import verify as original
    original()
    for p,h in read(BASE/'NATIVE_DISPATCH_RUNTIME004.json')['pins'].items():assert sha(ROOT/p)==h,p
def prepare_source():
    src=inspect.getsource(old.prepare)
    assert src.count('family_reuse_from=None')==1
    src=src.replace('family_reuse_from=None',"family_reuse_from={ds:{m:None for m in d['configs'][ds]} for ds in d['configs']}")
    needle='for p in [Path(__file__),';assert src.count(needle)==1
    return src.replace(needle,"for p in [Path(__file__), BASE/'NATIVE_DISPATCH_RUNTIME004.json', ROOT/'artifacts/stvg_pipeline_speed_v1/EXECUTION_RUNTIME.json', ROOT/'artifacts/stvg_pipeline_speed_v1/QUALIFICATION_COMPLETION.json',")
def prepare(arm):
    verify();assert arm in ('Full_E1','A','B','C','A_BT','B_BT')
    ns=dict(old.__dict__);ns.update(verify=verify,__file__=str(Path(__file__)));exec(compile(prepare_source(),str(Path(__file__)),'exec'),ns);ns['prepare'](arm)
def check(arm):
    verify();old.check(arm)
def source_patch(text,enabled):
    if not enabled:return text
    changes={
        'base = model_batch = native = raw = records = None':'base = model_batch = native = raw = records = frames = None',
        '            source_capture_count += 1':'            source_capture_count += int(not speed_engine.current["cache_hit"])',
        '                    save_pt(destination, payload)':'                    speed_enqueue(save_pt, destination, payload)',
        '        checkpoints = {}':'        speed_flush()\n        checkpoints = {}',
        'source_capture_reused=False if source_cold is not None else True':'source_capture_reused=(bool(speed_engine.current["cache_hit"]) if source_cold is not None else True)',
        '        del case\n        gc.collect()\n        if torch.cuda.is_initialized():\n            torch.cuda.empty_cache()':'        del case, base, model_batch, native, raw, records, frames, initial\n        speed_engine.release()',
    }
    for before,after in changes.items():assert text.count(before)==1,(before,text.count(before));text=text.replace(before,after)
    return text
def worker(arm):
    import torch
    from scripts import run_stvg_instance_student32_v1 as original
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    q=read(speed.BASE/'QUALIFICATION_COMPLETION.json');enabled=q['status']=='pass' and q['engineering_adopted']
    engine=speed.Engine(arm,enabled);pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);pending=[];save_seconds=[]
    def enqueue(save,dest,payload):
        # Rows already contain detached CPU snapshots. Copy the growing path
        # container before handing immutable snapshots to the one CPU writer.
        p=dict(payload);p['path']=list(payload['path'])
        def task():
            t=time.perf_counter();z=save(dest,p);save_seconds.append(time.perf_counter()-t);return z
        pending.append(pool.submit(task))
    def flush():
        while pending:pending.pop(0).result()
    def install(w):w.__dict__.update(speed_engine=engine,speed_enqueue=enqueue,speed_flush=flush)
    src=inspect.getsource(original.worker)
    before='tree=ast.parse(Path(original.__file__).read_text())';assert src.count(before)==1
    src=src.replace(before,'tree=ast.parse(speed_source_patch(Path(original.__file__).read_text()))')
    assert src.count('w.main()')==1;src=src.replace('w.main()','speed_install(w);w.main()')
    tree=ast.parse(src)
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(256),n) if type(n.value)is int and n.value==32 else n
    src=ast.unparse(ast.fix_missing_locations(Count().visit(tree)));assert src.count("'lite_fullstep'")==1;src=src.replace("'lite_fullstep'",repr(old.recipe(arm)))
    ns=dict(original.__dict__);ns.update(BASE=BASE,check=check,modes=lambda _:('M2',),speed_source_patch=lambda t:source_patch(t,enabled),speed_install=install)
    capture,readout=h._capture_source,h._readout
    h._capture_source=engine.capture;h._readout=engine.readout
    try:
        exec(compile(src,str(Path(__file__)),'exec'),ns);ns['worker'](arm);flush();engine.receipt()
        put(speed.BASE/'student_profiles'/(arm+'_IO.json'),dict(status='complete',arm=arm,checkpoint_writes=len(save_seconds),checkpoint_writer_seconds=sum(save_seconds),CPU_only=True,GT_read=False,full_original_step_checkpoints_preserved=True))
    finally:
        flush();pool.shutdown(wait=True);h._capture_source=capture;h._readout=readout
def root(arm):
    ns=dict(old.__dict__);ns.update(check=check,__file__=str(Path(__file__)));exec(compile(inspect.getsource(old.execute),str(Path(__file__)),'exec'),ns);ns['execute']('root',arm)
def contracts():
    import torch
    from scripts import run_stvg_soft_ep_coordinate_trial003 as original
    s=source_patch(Path(original.__file__).read_text(),True);ast.parse(s);assert source_patch(Path(original.__file__).read_text(),False)==Path(original.__file__).read_text()
    ast.parse(prepare_source())
    d={'configs':{'hc2':{'M2':{}},'vidstg':{'M2':{}}}};fixed={ds:{m:None for m in d['configs'][ds]} for ds in d['configs']}
    assert all(fixed[ds]['M2'] is None for ds in fixed)
    a={'x':torch.tensor([1.,2.]),'path':[{'gradient':torch.tensor([.1])}],'wall_seconds':1.};b=copy.deepcopy(a);b['wall_seconds']=4.;speed.compare_tree(a,b)
    b['path'][0]['gradient'][0]=.2
    try:speed.compare_tree(a,b)
    except AssertionError:pass
    else:raise AssertionError('different gradient accepted')
    print('SPEED_STUDENT_CPU_CONTRACTS_PASS_SCHEMA_PATCH_TIMING_EXCLUSION_GRADIENT_NEGATIVE_NO_GPU_NO_GT')
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='contracts':contracts()
    else:{'prepare':prepare,'worker':worker,'root':root}[mode](sys.argv[2])
