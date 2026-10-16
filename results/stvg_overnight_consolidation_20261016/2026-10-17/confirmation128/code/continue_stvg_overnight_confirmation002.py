"""One finite confirmation queue after P1; no score-driven retuning."""
from pathlib import Path
import copy,os,select,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
from scripts.stvg_overnight_confirmation_v1 import BASE,verify
from scripts.continue_stvg_overnight001 import execute
read,sha,put=c.read,c.sha,c.write
UNIT='stvg-overnight-confirmation-002.service'
def run(stage,mode=None):
    verify();out=BASE/'execution'/stage
    if (out/'WORKER_EXIT.json').exists():assert read(out/'WORKER_EXIT.json')['exit_code']==0;return
    assert time.time()<read(c.BASE/'PLAN_LOCK.json')['last_new_noncheckpointable_stage_epoch'],'last30min_no_new_noncheckpointable_stage'
    stage_mode=mode or stage;gpu=stage in ('source','temporal_teacher','features');interp='.venv/stvg_native_distill_v1/bin/python' if stage in ('temporal_teacher','features') else '.conda/tubedetr/bin/python';script='score_stvg_overnight_confirmation_v1.py' if stage=='score' else 'stvg_overnight_confirmation_v1.py';out.mkdir(parents=True,exist_ok=False);c.status('confirmation128/'+stage,count=128)
    with (out/'worker.log').open('xb') as f:
        tick=time.perf_counter();p=subprocess.Popen([str(ROOT/interp),'-B',str(ROOT/'scripts'/script),stage_mode],cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdout=f,stderr=subprocess.STDOUT);put(out/'LAUNCH.json',dict(status='running',pid=p.pid,GT_read=stage=='score'))
        try:rc=p.wait(timeout=max(1,c.deadline()-time.time()))
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            raise TimeoutError('eight-hour finite window; partial checkpoints retained')
    put(out/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-tick,GT_read=stage=='score'));assert rc==0,stage
def fallback():
    # Technical failure never becomes a numerical zero. Keep the registered
    # P0 core, disclose this technical branch, and do not re-search a recipe.
    core=read(c.BASE/'CORE_DECISION.json');put(c.BASE/'CANDIDATE_LOCK.json',dict(status='locked',count=128,selected_scope=core['selected_arm'],recipe=core['recipe'],policy='B',physical_clock=False,global_frames=4,global_keep=64,local_keep=96,design_template=str((c.CORE/'trials'/core['selected_arm']/'DESIGN_LOCK.json').relative_to(ROOT)),confirmation_roster_sha256=sha(BASE/'ROSTER_LOCK.json'),P1_technical_unqualified=True,production_promotion=False,fullset_authorized=False,GT_read=False));c.log('P1','technical_fallback_to_frozen_core_not_scientific_zero',failure_sha256=sha(c.BASE/'P1_FAILURE.json'))
def ablations():
    for kind in ('noCaption','noCoverage'):
        scope='ablation_'+kind;b=c.BASE/scope;b.mkdir(parents=True,exist_ok=False);(b/'ROSTER_LOCK.json').write_bytes((c.BASE/'global_B/ROSTER_LOCK.json').read_bytes());cfg=copy.deepcopy(read(c.BASE/'global_B/SCOPE_CONFIG.json'));cfg.update(label='fixed32 frozen core B mechanism '+kind,global_frames=4,global_keep=64,physical_clock=False,ablation=kind,vision_reuse_scope=None);put(b/'SCOPE_CONFIG.json',cfg);execute(scope)
def main(pid):
    verify();put(BASE/'LAUNCH.json',dict(status='waiting',stage='original_P1_controller_exit',pid=os.getpid(),count=128,GT_read=False))
    try:
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:fd=None
        if fd is not None:p=select.poll();p.register(fd,select.POLLIN);p.poll(max(1,int((c.deadline()-time.time())*1000)));os.close(fd)
        if not(c.BASE/'CANDIDATE_LOCK.json').exists():assert (c.BASE/'P1_FAILURE.json').exists();fallback()
        from scripts import launch_stvg_instance_identity_v1 as res
        res.BASE=c.BASE;res.UNIT=UNIT;res.STAGE='night_confirmation';g=1024**3;v=res.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;stop=threading.Event();threading.Thread(target=res.monitor,args=(stop,),daemon=True).start()
        try:
            for stage in ('setup','source','temporal_prepare','temporal_teacher','temporal_fit','temporal_readback','spatial_prepare','spatial_root','features'):run(stage)
            if read(c.BASE/'CANDIDATE_LOCK.json')['selected_scope']=='global_B':run('rgb')
            run('scopes')
            for arm in read(BASE/'ARM_LOCK.json')['arms']:
                scope='confirmation128/arms/'+arm;n=128;c.reserve(scope,WHO=4*n+6,fits=n)
                # Same execution chain, but no per-arm GT score: all arms seal first.
                from scripts.continue_stvg_overnight001 import run as arm_run
                for mode in ('jobs','qualification','teacher','teacher_root'):arm_run(scope,mode,'stvg_overnight_global_v1.py',mode,mode in ('qualification','teacher'))
                for mode in ('prepare','worker','root'):arm_run(scope,'student_'+mode,'stvg_overnight_student_v1.py',mode)
            run('score');put(BASE/'MEASUREMENT_COMPLETION.json',dict(status='complete_pending_independent_view_public_archive',count=128,GT_read=True));c.log('P2','confirmation_complete_requires_group_publication')
            # These are two predeclared deletions; no new parameter choice follows.
            if c.deadline()-time.time()>3600:ablations()
        finally:stop.set()
        put(c.BASE/'FINITE_MEASUREMENT_HANDOFF.json',dict(status='requires_root_cost_TubeDETR_qualification8_view_public_archive_closing',count=128,GT_read=False,fullset_execution_authorized=False));c.write(c.BASE/'STATUS.json',dict(status='root_handoff',stage='finite_measurements_complete_remaining_authorized_qualification_and_closing',count=128,GT_read=False),True)
    except BaseException as e:
        put(BASE/'FAILURE.json',dict(status='technical_or_window_handoff',error_type=type(e).__name__,error=str(e),scientific_zero_not_imputed=True,all_partial_outputs_retained=True,GT_read=False));c.write(c.BASE/'STATUS.json',dict(status='requires_root',stage='confirmation_engineering_or_window_handoff',count=128,GT_read=False),True);raise
if __name__=='__main__':main(int(sys.argv[1]))
