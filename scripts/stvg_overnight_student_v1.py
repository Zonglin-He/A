"""Unchanged qualified episodic Student with variable locked cohort and evidence scope."""
from pathlib import Path
import ast,concurrent.futures,copy,gc,inspect,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as common
from scripts import stvg_pipeline_speed_v1 as speed
from scripts.run_stvg_controls256_speed_student_v1 import source_patch
read,sha,put=common.read,common.sha,common.write
BASE=CORE=CFG=None;N=None
def configure(scope):
    global BASE,CORE,CFG,N
    BASE=(common.BASE/scope).resolve();assert BASE.is_relative_to(common.BASE);CFG=read(BASE/'SCOPE_CONFIG.json');CORE=ROOT/CFG['core_scope'];N=read(BASE/'ROSTER_LOCK.json')['count'];assert N in (32,128)
def verify():common.verify();assert read(BASE/'teacher/G32/ROOT_READBACK.json')['status']=='pass'
def prepare():
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.prepare_stvg_lite_reclocked_inputs002 import _anchor_from_teacher_selection
    verify();jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];rr=read(BASE/'ROSTER_LOCK.json')['records'];original={r['case_id']:r for r in read(CORE/'SOURCE_CAPTURE_BARRIER.json')['files']};source=[];refs=[]
    for rec in rr:
        cid=rec['probe_alias'];c,p=load_case(rec);f=original[cid];assert sha(p)==f['sha256'];source.append(f);anchors=[];decisions=[];jj=[j for j in jobs if j['case_id']==cid];pool=read(CORE/'spatial/E1/proposals'/(cid+'.json')) if jj else None
        for j in jj:
            rp=BASE/'teacher/G32'/(j['job_id']+'.json');d=read(rp);w=next(x for x in pool['windows'] if x['slot']==j['center_slot']);a,why=_anchor_from_teacher_selection(j,d,c['frame_ids'],w['source_grid_position'],teacher_output_sha256=sha(rp))
            if a is not None:anchors.append(a)
            decisions.append(dict(slot=j['center_slot'],reason=why,selected_candidate_id=d['parsed'].get('selected_candidate_id'),teacher_sha256=sha(rp),GT_read=False))
        refs.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=c['source_case_sha256'],anchors=anchors,decisions=decisions,GT_read=False));del c;gc.collect()
    put(BASE/'SOURCE_CAPTURE_BARRIER.json',dict(status='complete',count=N,files=source,exact_source_reuse=True,GT_read=False));b=BASE/'trials/G32';b.mkdir(parents=True,exist_ok=False);put(b/'REFERENCE_LOCK.json',dict(status='locked',count=N,records=refs,source_candidate_forbidden=True,presence_threshold=.6,GT_read=False));ids={r['probe_alias'] for r in rr};tt=copy.deepcopy(read(CORE/'TEMPORAL_LOCK.json'));tt.update(count=N,records=[r for r in tt['records'] if r['case_id'] in ids]);assert len(tt['records'])==N;put(b/'TEMPORAL_LOCK.json',tt)
    template=ROOT/CFG['design_template'];d=copy.deepcopy(read(template));d.update(trial_id='G32',count=N,cohort='fixed_overnight_roster',scope=CFG['label'],recipe=CFG['recipe'],family_reuse_from={ds:{m:None for m in v} for ds,v in d['configs'].items()});put(b/'DESIGN_LOCK.json',d)
    rt=copy.deepcopy(read(template.with_name('RUNTIME.json')));rt.update(trial_id='G32',scope=d['scope'])
    for p in (Path(__file__),BASE/'SCOPE_CONFIG.json',common.BASE/'EXECUTION_RUNTIME001.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'ROSTER_LOCK.json',BASE/'teacher/G32/ROOT_READBACK.json',BASE/'teacher/G32/TEACHER_GLOBAL_BARRIER.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',b/'DESIGN_LOCK.json'):rt['pins'][str(p.relative_to(ROOT))]=sha(p)
    put(b/'RUNTIME.json',rt)
def check(tid):
    assert tid=='G32';verify();r=read(BASE/'trials/G32/RUNTIME.json')
    for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
def resized(src):
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(N),n) if type(n.value)is int and n.value==32 else n
    return ast.unparse(ast.fix_missing_locations(Count().visit(ast.parse(src))))
def worker():
    from scripts import run_stvg_instance_student32_v1 as original
    from scripts import run_stvg_adjacent_motion_student_v1 as h
    q=read(speed.BASE/'QUALIFICATION_COMPLETION.json');enabled=q['status']=='pass' and q['engineering_adopted'];label='night_'+str(BASE.relative_to(common.BASE)).replace('/','_');engine=speed.Engine(label,enabled);pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);pending=[];seconds=[]
    def enqueue(save,dest,payload):
        p=dict(payload);p['path']=list(payload['path'])
        def task():
            t=time.perf_counter();x=save(dest,p);seconds.append(time.perf_counter()-t);return x
        pending.append(pool.submit(task))
    def flush():
        while pending:pending.pop(0).result()
    def install(w):w.__dict__.update(speed_engine=engine,speed_enqueue=enqueue,speed_flush=flush)
    src=inspect.getsource(original.worker);needle='tree=ast.parse(Path(original.__file__).read_text())';assert src.count(needle)==1;src=src.replace(needle,'tree=ast.parse(speed_source_patch(Path(original.__file__).read_text()))');assert src.count('w.main()')==1;src=src.replace('w.main()','speed_install(w);w.main()');src=resized(src);assert src.count("'lite_fullstep'")==1;src=src.replace("'lite_fullstep'",repr(CFG['recipe']))
    ns=dict(original.__dict__);ns.update(BASE=BASE,check=check,modes=lambda _:('M2',),speed_source_patch=lambda t:source_patch(t,enabled),speed_install=install)
    capture,readout=h._capture_source,h._readout;h._capture_source=engine.capture;h._readout=engine.readout
    try:
        exec(compile(src,str(Path(__file__)),'exec'),ns);ns['worker']('G32');flush();engine.receipt();put(BASE/'STUDENT_IO_COMPLETION.json',dict(status='complete',count=N,checkpoint_writes=len(seconds),checkpoint_writer_seconds=sum(seconds),all_original_step_checkpoints_preserved=True,GT_read=False))
    finally:flush();pool.shutdown(wait=True);h._capture_source=capture;h._readout=readout
def root():
    from scripts import run_stvg_instance_student32_v1 as original
    src=resized(inspect.getsource(original.root));needle="bb = arr(o['boxes'])";assert src.count(needle)==1;src=src.replace(needle,needle+'.astype(float)');assert src.count("'lite_fullstep'")==1;src=src.replace("'lite_fullstep'",repr(CFG['recipe']));ns=dict(original.__dict__);ns.update(BASE=BASE,check=check,modes=lambda _:('M2',));exec(compile(src,str(Path(__file__)),'exec'),ns);ns['root']('G32')
if __name__=='__main__':
    configure(sys.argv[1]);{'prepare':prepare,'worker':worker,'root':root}[sys.argv[2]]()
