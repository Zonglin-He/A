"""GT-blind corrected Native WHEN budget and isolated namespace bindings."""
import inspect,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import ROOT,read,write,status,sha,load
MAIN=ROOT/'artifacts/stvg_opd_query_observability_q0_v1'
BASE=MAIN/'Native_support_scope003'
PUB=ROOT/'results/stvg_opd_query_observability_q0/2026-10-12/Native_support_scope003'


def context():
    import scripts.stvg_opd_query_q0_common_v1 as common
    common.BASE=BASE;common.PUB=PUB
    return common


def prepare():
    import copy
    d=copy.deepcopy(read(MAIN/'DESIGN_LOCK.json'));values=[];counts=[]
    for r in d['records']:
        assert sha(ROOT/r['original_Full']['path'])==r['original_Full']['sha256']
        assert sha(ROOT/r['input']['path'])==r['input']['sha256']
        old=load(ROOT/r['original_Full']['path']);inp=load(ROOT/r['input']['path'])
        ids=np.asarray(inp['frame_ids']);a,b=old['interval'];positions=np.flatnonzero((ids>=a)&(ids<b)).tolist()
        assert positions, 'Empty original Native support must not silently disappear'
        x=old['fit']['path'][0]['boxes'].numpy().astype(float)[positions]
        y=old['fit']['path'][1]['boxes'].numpy().astype(float)[positions]
        value=float(np.sqrt(np.mean((y-x)**2)));values.append(value);counts.append(len(positions))
        r['native_positions']=positions;r['first_step_output_RMS']=value
    d.update(status='corrected_fixed_Native_WHEN_support_before_new_GT',pooled_output_RMS_budget=float(np.median(values)),
             budget_definition='median over64 old Full first-step RMS on fixed Native interval sample centers, including zeros',
             old_scope_qualification_not_eligible=True,old_scope_design_sha256=sha(MAIN/'DESIGN_LOCK.json'),time=time.time())
    BASE.mkdir(parents=True,exist_ok=True)
    write(BASE/'DESIGN_LOCK.json',d)
    common=context();write(BASE/'CPU_CONTRACTS.json',common.contracts())
    pins=dict(read(MAIN/'RUNTIME.json')['pins'])
    own=['scripts/stvg_opd_query_q0_native_scope003.py','scripts/run_stvg_opd_query_q0_native003.py',
         'scripts/continue_stvg_opd_query_q0_native003.py','scripts/continue_stvg_opd_query_q0_native_root003.py',
         'scripts/review_stvg_opd_query_q0_v1.py','scripts/report_stvg_opd_query_q0_v1.py',
         'scripts/continue_stvg_opd_query_q0_root_v1.py','protocols/stvg_opd_query_q0_native_scope003.md']
    pins.update({p:sha(ROOT/p) for p in own})
    write(BASE/'RUNTIME.json',dict(status='locked_Native_support_scope_before_actual_GPU_and_GT',pins=pins,
          design_sha256=sha(BASE/'DESIGN_LOCK.json'),CPU_contract_sha256=sha(BASE/'CPU_CONTRACTS.json'),
          original_Q0_runtime_sha256=sha(MAIN/'RUNTIME.json'),science_scope_correction_not_parameter_search=True,GT_read=False,time=time.time()))
    write(BASE/'SOURCE_SCOPE_READBACK.json',dict(status='pass',scope='all64 fixed original clean Before/input/native intervals; no GT',
          cells=64,empty_Native_support=0,min_Native_samples=min(counts),max_Native_samples=max(counts),
          old_complete_HC_qualification_scope_only=True,new_scope_qualification_queries=0,new_scope_formal_queries=0,CPU_only=True,GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_native_scope_pending_serial_O_GPU_exit',done=0,total=64,GT_read=False,CPU_only=True,time=time.time()))


def bind_geometry():
    import vg_tta.stvg_opd_query_q0_v1 as g
    original=g.episode;original_rms=g.rms
    def episode(base,old,expert,dataset,key,budget):
        from scripts.run_decota_paper_main_v1 import read_row
        ids=np.asarray(read_row(dataset,old['parent'])['frame_ids']);a,b=old['interval'];positions=np.flatnonzero((ids>=a)&(ids<b)).tolist()
        assert positions
        g.rms=lambda x,y:original_rms(x[positions],y[positions])
        try:result=original(base,old,expert,dataset,key,budget)
        finally:g.rms=original_rms
        result['native_positions']=positions;result['output_RMS_scope']='fixed_Native_WHEN_interval_sample_centers'
        return result
    g.episode=episode


def bind_runner(original):
    original_verify=original.verify
    def verify():
        result=original_verify();assert read(BASE/'DESIGN_LOCK.json')['old_scope_qualification_not_eligible']
        return result
    original.verify=verify
    # Root readback must use the same declared support as actual GPU calibration.
    s=inspect.getsource(original.root)
    old="before=ep['before'].numpy().astype(float);boxes=c['boxes'].numpy().astype(float)"
    new="before=ep['before'].numpy().astype(float)[ep['native_positions']];boxes=c['boxes'].numpy().astype(float)[ep['native_positions']]"
    assert s.count(old)==1;s=s.replace(old,new)
    s=s.replace("t['boxes'].numpy().astype(float)-before","t['boxes'].numpy().astype(float)[ep['native_positions']]-before")
    ns=dict(original.__dict__);exec(compile(s,__file__,'exec'),ns);original.root=ns['root']


if __name__=='__main__':prepare()
