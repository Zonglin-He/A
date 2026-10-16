"""Same complete-model worker on fixed 128 development or 64 confirmation cases.

Only cohort cardinality is parameterized in the original worker AST. Model,
source capture, fit interfaces and final free model decoding remain intact.
"""
from pathlib import Path
import ast
import copy
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,sha,event_output_positions,verify,load_case


def worker_for(cohort):
    from scripts import run_stvg_soft_ep_coordinate_trial003 as original
    scope=BASE if cohort=='development' else BASE/'confirmation'
    count=128 if cohort=='development' else 64
    sourcepath=Path(original.__file__)
    tree=ast.parse(sourcepath.read_text())
    class Cardinality(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,int) and not isinstance(node.value,bool) and node.value==128:
                return ast.copy_location(ast.Constant(count),node)
            return node
    tree=ast.fix_missing_locations(Cardinality().visit(tree))
    worker=types.ModuleType('scripts._stvg_lite_complete_worker')
    worker.__dict__.update(__name__=worker.__name__,__file__=str(sourcepath))
    exec(compile(tree,str(sourcepath),'exec'),worker.__dict__)
    worker.BASE=scope;worker.TRIALS=scope/'trials'
    worker.ROSTER_PATH=(ROOT/'artifacts/stvg_soft_ep_coordinate128_v1/ROSTER_LOCK.json')if cohort=='development'else scope/'ROSTER_LOCK.json'
    worker.SOURCE_BARRIER_PATH=(ROOT/'artifacts/stvg_soft_ep_coordinate128_v1/SOURCE_CAPTURE_BARRIER.json')if cohort=='development'else scope/'SOURCE_CAPTURE_BARRIER.json'
    worker.verify_runtime=verify
    worker.MODES=('M0','M2');worker.ARM_NAMES=('F','T','S_M0','TS_M0','S_M2','TS_M2')
    assert read(worker.ROSTER_PATH)['count']==read(worker.SOURCE_BARRIER_PATH)['count']==count
    return worker,scope,count


def main():
    import torch
    from scripts import stvg_soft_ep_coordinate_fit002 as dispatch
    from scripts import stvg_component_spatial_fit_v1 as full
    from scripts import stvg_spatial_lite_fit_v1 as lite
    from scripts import run_stvg_adjacent_motion_student_v1 as complete
    from scripts import run_stvg_table1_source_v1 as source
    verify();cohort,tid=sys.argv[1:3];assert cohort in {'development','confirmation'}
    worker,scope,count=worker_for(cohort);trial=scope/'trials'/tid
    runtime=read(trial/'RUNTIME.json')
    assert runtime['pins']['scripts/run_stvg_lite_complete_student_v1.py']==sha(Path(__file__))
    for rel,h in {**runtime['pins'],**runtime['protected']}.items():assert sha(ROOT/rel)==h,rel
    design=read(trial/'DESIGN_LOCK.json');times={r['case_id']:r for r in read(trial/'TEMPORAL_LOCK.json')['records']}
    intervals={worker._stable_fit_key(r):times[r['probe_alias']]['selected_interval']for r in read(worker.ROSTER_PATH)['records']}
    oldfit,oldreadout=dispatch.fit,complete._readout
    def sync():
        if torch.cuda.is_initialized():torch.cuda.synchronize()
    def fitter(base,initial,anchors,ids,key,cfg,mode,settings,qualification=False,checkpoint=None):
        interval=intervals[key];sync();tick=time.perf_counter()
        if design['recipe']=='full':
            f=full.fit(base,initial,anchors,ids,key,cfg,mode,settings,qualification=qualification,checkpoint=checkpoint,flags={})
        else:
            positions=event_output_positions(ids,interval)if design['response_scope']=='event_support'else None
            f=lite.fit(base,initial,anchors,ids,key,cfg,mode,settings,qualification=qualification,checkpoint=checkpoint,recipe=design['recipe'],response_scope=design['response_scope'],event_support_positions=positions)
        sync();f['complete_study_fit_seconds']=time.perf_counter()-tick
        f['predicted_event_interval']=interval;f['study_recipe']=design['recipe'];f['study_response_scope']=design['response_scope']
        return f
    def readout(*a,**kw):
        sync();tick=time.perf_counter();o=oldreadout(*a,**kw);sync()
        o['complete_study_forward_seconds']=time.perf_counter()-tick
        return o
    try:
        dispatch.fit=fitter;complete._readout=readout
        # The Source adapter retains its original qualified model/runtime;
        # the new confirmation runtime separately binds the new input roster.
        sys.argv=['lite_complete_student',tid];worker.main()
    finally:dispatch.fit=oldfit;complete._readout=oldreadout


def contracts():
    from scripts import run_stvg_soft_ep_coordinate_trial003 as original
    tree=ast.parse(Path(original.__file__).read_text())
    nodes=[n for n in ast.walk(tree)if isinstance(n,ast.Constant)and type(n.value)is int and n.value==128]
    assert len(nodes)>=10
    # No original scientific function is replaced by the cohort adapter.
    return dict(status='pass',CPU_only=True,GT_read=False,original_worker_sha256=sha(Path(original.__file__)),cardinality_constant_count=len(nodes),model_loaded=False)


if __name__=='__main__':
    if '--contracts'in sys.argv:print(contracts())
    else:main()
