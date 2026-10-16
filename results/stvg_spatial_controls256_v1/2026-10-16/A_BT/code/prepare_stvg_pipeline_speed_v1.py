"""Pin engineering-only acceleration, qualification, and the exact live handoff."""
from pathlib import Path
import datetime,inspect,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_pipeline_speed_v1 as speed
from scripts.prepare_stvg_spatial_controls256_v1 import BASE as CORE,read,put,sha
FILES=('stvg_pipeline_speed_v1.py','run_stvg_controls256_speed_student_v1.py','qualify_stvg_pipeline_speed_v1.py','run_stvg_controls256_speed_teacher_v1.py','continue_stvg_controls256_speed004.py','run_stvg_global32_speed_student002.py','continue_stvg_global32_speed002.py','prepare_stvg_pipeline_speed_v1.py')
def main(controller,teacher,waiter):
    from scripts import continue_stvg_controls256_speed004 as dispatch
    from scripts.continue_stvg_spatial_controls256_native003 import verify
    from scripts.run_stvg_controls256_speed_student_v1 import contracts as student
    from scripts.run_stvg_controls256_speed_teacher_v1 import contracts as teacher_contract
    from scripts.run_stvg_global32_speed_student002 import contracts as global_student
    from scripts.continue_stvg_global32_speed002 import contracts as global_wait
    verify();student();teacher_contract();dispatch.contracts();global_student();global_wait()
    scope=speed.BASE;scope.mkdir(parents=True,exist_ok=False)
    old_status=read(CORE/'CORE6_STATUS003.json');assert old_status['stage']=='teacher' and old_status['arm']=='Full_E1'
    assert not (CORE/'trials/Full_E1').exists() and not (CORE/'teacher/Full_E1/TEACHER_GLOBAL_BARRIER.json').exists()
    identities={k:dispatch.identity(v) for k,v in [('controller',controller),('teacher',teacher),('global_waiter',waiter)]}
    assert identities['teacher']['ppid']==controller and identities['teacher']['state']!='Z'
    put(scope/'LIVE_HANDOFF_LOCK.json',dict(status='locked',**identities,GT_read=False,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    files=[ROOT/'scripts'/n for n in FILES]+[CORE/'NATIVE_DISPATCH_RUNTIME003.json',CORE/'EXECUTION_SCOPE_LOCK003.json',CORE/'SOURCE_CAPTURE_BARRIER.json',CORE/'ROSTER_LOCK.json',ROOT/'methods/CURRENT_METHOD.json']
    put(scope/'EXECUTION_RUNTIME.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p) for p in files},GT_read=False,scientific_parameters_or_acceptance_changed=False,qualification_failure='original unaccelerated compute with corrected no-reuse descriptor; no experiment cancellation',cache_boundary='Source views before trainable LNs; final readout recomputes both decoder passes and every output frame',current_research_step_audits_preserved=True,forbid_precision_or_steps_change=True))
    put(CORE/'NATIVE_DISPATCH_RUNTIME004.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p) for p in files+[scope/'EXECUTION_RUNTIME.json',scope/'LIVE_HANDOFF_LOCK.json']},GT_read=False,scope='six original scientific recipes, safe original Teacher adoption, exactness-qualified speed only'))
    global_base=ROOT/'artifacts/stvg_global32_context_v1'
    gf=[ROOT/'scripts/continue_stvg_global32_speed002.py',ROOT/'scripts/run_stvg_global32_speed_student002.py',global_base/'EXECUTION_RUNTIME.json',CORE/'NATIVE_DISPATCH_RUNTIME004.json',scope/'EXECUTION_RUNTIME.json']
    put(global_base/'EXECUTION_RUNTIME002.json',dict(status='locked',count=32,pins={str(p.relative_to(ROOT)):sha(p) for p in gf},GT_read=False,waits_for_six_core256_measurements=True,scientific_recipe_unchanged=True,original_waiter_replaced_before_original_controller_retired=True))
    put(scope/'CPU_CONTRACT_COMPLETION.json',dict(status='pass',count=5,CPU_only=True,GT_read=False,scope='Student descriptor/AST/fallback/gradient negative; Teacher lossless storage/protocol; six-group dispatch; global descriptor; global dependency',new_model_calls=0,new_GT_reads=0))
    print('SPEED_RUNTIME_LOCKED',len(files))
if __name__=='__main__':main(*map(int,sys.argv[1:4]))
