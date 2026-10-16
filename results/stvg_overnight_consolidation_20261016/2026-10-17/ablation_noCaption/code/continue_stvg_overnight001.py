"""Finite P1 continuation, fixed gate, one attribution control, then root handoff."""
from pathlib import Path
import copy,json,math,os,select,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
read,sha,put=c.read,c.sha,c.write
UNIT='stvg-overnight-p1-001.service'
def run(scope,stage,script,mode,gpu=False):
    c.verify();base=c.BASE/scope;out=base/'execution'/stage
    if (out/'WORKER_EXIT.json').exists():assert read(out/'WORKER_EXIT.json')['exit_code']==0;return
    assert time.time()<read(c.BASE/'PLAN_LOCK.json')['last_new_noncheckpointable_stage_epoch'],'last30min_no_new_stage'
    c.status(scope+'/'+stage,count=read(base/'ROSTER_LOCK.json')['count']);out.mkdir(parents=True,exist_ok=False)
    interpreter='.venv/stvg_native_distill_v1/bin/python' if gpu else '.conda/tubedetr/bin/python'
    with (out/'worker.log').open('xb') as f:
        t=time.perf_counter();p=subprocess.Popen([str(ROOT/interpreter),'-B',str(ROOT/'scripts'/script),scope,mode],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT)
        put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'))
        try:rc=p.wait(timeout=max(1,c.deadline()-time.time()))
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            put(out/'WINDOW_CHECKPOINT.json',dict(status='window_exhausted',count=read(base/'ROSTER_LOCK.json')['count'],partial_outputs_retained=True,GT_read=False));raise TimeoutError('bounded night deadline')
    put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=stage=='score'));assert rc==0,(scope,stage)
def execute(scope):
    cfg=read(c.BASE/scope/'SCOPE_CONFIG.json');n=read(c.BASE/scope/'ROSTER_LOCK.json')['count'];c.reserve(scope,WHO=4*n+6,fits=n)
    for mode in ('jobs','qualification','teacher','teacher_root'):run(scope,mode,'stvg_overnight_global_v1.py',mode,mode in ('qualification','teacher'))
    for mode in ('prepare','worker','root'):run(scope,'student_'+mode,'stvg_overnight_student_v1.py',mode)
    # Scorer uses scope as its first argument; second mode is intentionally unused.
    run(scope,'score','stvg_overnight_score_v1.py','score');put(c.BASE/scope/'MEASUREMENT_COMPLETION.json',dict(status='complete_pending_independent_view_public_archive',count=n,GT_read=True))
def clone(scope,frames,keep):
    b=c.BASE/scope;b.mkdir(parents=True,exist_ok=False)
    for name in ('ROSTER_LOCK.json','GLOBAL_RGB_BARRIER.json','ROOT_RGB_READBACK.json'): (b/name).write_bytes((c.BASE/'global_B'/name).read_bytes())
    cfg=copy.deepcopy(read(c.BASE/'global_B/SCOPE_CONFIG.json'));cfg.update(global_frames=frames,global_keep=keep,vision_reuse_scope=str((c.BASE/'global_B').relative_to(ROOT)),label='one matched-budget context attribution '+scope);put(b/'SCOPE_CONFIG.json',cfg)
def choose():
    p=c.BASE/'global_B/POSTSEAL_RESULTS.json';d=read(p);cpu=read(p.with_name('CPU_COMPLETION.json'));assert cpu['result_sha256']==sha(p) and cpu['status']=='complete'
    deltas=[next(x['mean_delta_pp'] for x in d['paired'] if x['comparison']=='Candidate_minus_LocalB' and x['dataset']==ds and x['metric']=='m_vIoU') for ds in ('hc2','vidstg')];M=math.fsum(deltas)/2;promote=M>.10 and min(deltas)>=-.20
    put(c.BASE/'GLOBAL_DEVELOPMENT_DECISION.json',dict(status='locked',count=32,domain_delta_pp=deltas,M_mean_pp=M,accuracy_gate_pass=promote,cost_alternative_pass=False,cost_alternative_status='not_established_by_matched_end_to_end_measurement; reuse costs not treated as zero',GT_read=True))
    controls=[]
    if M>0:
        import numpy as np
        from scripts.stvg_global32_native_v1 import global_geometry
        dims=read(c.CORE/'SOURCE_GEOMETRY.json')['records'];rows=read(c.BASE/'global_B/ROSTER_LOCK.json')['records'];capacity=[int(global_geometry(**dims[r['probe_alias']])[0][:2].sum()) for r in rows];K=min(512,min(capacity));assert K>=32
        put(c.BASE/'MATCHED_CONTEXT_BUDGET_LOCK.json',dict(status='locked',count=32,global_keep=K,short_capacity_min=min(capacity),no_padding_or_duplicate_tokens=True,one_control_protocol_only=True,GT_read=False))
        if K!=512:clone('attribution_long',32,K);execute('attribution_long');controls.append('attribution_long')
        clone('attribution_short',4,K);execute('attribution_short');controls.append('attribution_short')
    core=read(c.BASE/'CORE_DECISION.json');selected='global_B' if promote else core['selected_arm'];put(c.BASE/'CANDIDATE_LOCK.json',dict(status='locked',count=128,selected_scope=selected,recipe=core['recipe'],physical_clock=promote,global_frames=32 if promote else 4,global_keep=512 if promote else 64,local_keep=96,policy='B',design_template=str((c.CORE/'trials'/core['selected_arm']/'DESIGN_LOCK.json').relative_to(ROOT)),selection_sources_count=256,additional_global_selection_parents_are_core_subset=True,confirmation_roster_sha256=sha(c.BASE/'confirmation128/ROSTER_LOCK.json'),development_decision_sha256=sha(c.BASE/'GLOBAL_DEVELOPMENT_DECISION.json'),method_source_runtime_sha256=sha(c.BASE/'EXECUTION_RUNTIME001.json'),GT_read=False,production_promotion=False,fullset_authorized=False,attribution_scopes=controls));c.log('P1','candidate_locked',selected_scope=selected)
def main(pid):
    c.verify();c.status('waiting_original_G32_A',count=32)
    try:
        if pid>0:
            try:fd=os.pidfd_open(pid)
            except ProcessLookupError:fd=None
            if fd is not None:p=select.poll();p.register(fd,select.POLLIN);p.poll(max(1,int((c.deadline()-time.time())*1000)));os.close(fd)
        assert read(c.OLDGLOBAL/'MEASUREMENT_COMPLETION.json')['status'].startswith('complete') and read(c.OLDGLOBAL/'CPU_COMPLETION.json')['status']=='complete'
        from scripts import launch_stvg_instance_identity_v1 as resource
        resource.BASE=c.BASE;resource.UNIT=UNIT;resource.STAGE='night_P1';g=1024**3;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g
        stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
        try:execute('global_B');choose()
        finally:stop.set()
        put(c.BASE/'P1_COMPLETION.json',dict(status='complete_pending_independent_view_public_archive_and_confirmation',count=32,GT_read=True));c.write(c.BASE/'STATUS.json',dict(status='root_handoff',stage='candidate_frozen_confirmation_required',count=128,GT_read=False),True)
    except BaseException as e:
        put(c.BASE/'P1_FAILURE.json',dict(status='technical_unqualified_or_window_checkpoint',error_type=type(e).__name__,error=str(e),scientific_zero_not_imputed=True,original_outputs_retained=True,GT_read=False));c.write(c.BASE/'STATUS.json',dict(status='requires_root',stage='bounded_engineering_or_window_handoff',GT_read=False),True);raise
if __name__=='__main__':main(int(sys.argv[1]))
