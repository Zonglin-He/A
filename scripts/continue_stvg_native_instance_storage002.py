"""Finite current-scope phase handoff; never dispatches a historical queue."""
import sys,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import BASE,read,write,sha
PY=ROOT/'.conda/tubedetr/bin/python'
NATIVE=ROOT/'.venv/stvg_native_distill_v1/bin/python'

def verify_plan():
    d=read(BASE/'SERIAL_STORAGE002_LOCK.json')
    assert d['status']=='locked' and d['GT_read'] is False
    for p,h in {**d['pins'],**d['protected']}.items():assert sha(ROOT/p)==h,p
    return d

def unit(stage):return 'stvg-native-instance-'+stage+('-002.service' if stage=='teacher' else '-001.service')
def properties(stage):
    out=subprocess.check_output(['systemctl','--user','show',unit(stage),'-p','ActiveState,SubState,Result,MainPID'],text=True)
    return dict(x.split('=',1)for x in out.splitlines()if '='in x)

def barrier(stage):
    p=BASE/(stage.upper()+'_GLOBAL_BARRIER.json');d=read(p)
    assert d['status']=='complete' and d['count']==8 and d['GT_read'] is False
    assert d['runtime_sha256']==sha(BASE/(stage.upper()+'_RUNTIME.json'))
    ids=[r['case_id']for r in read(BASE/'ROSTER_LOCK.json')['records']]
    assert [r['case_id']for r in d['files']]==ids
    for r in d['files']:assert sha(ROOT/r['path'])==r['sha256']
    return d

def wait_phase(stage):
    while True:
        props=properties(stage)
        if props['ActiveState'] not in ('active','activating','deactivating'):
            assert props['Result']=='success',props
            barrier(stage)
            write(BASE/(stage.upper()+'_ACTUAL_PHASE_EXIT.json'),dict(status='pass',unit=unit(stage),properties=props,GT_read=False,time=time.time()))
            return
        time.sleep(15)

def lock_runtime(stage,extra):
    plan=verify_plan();pins=dict(plan['stage_pins'][stage])
    for p in extra:pins[str(p.relative_to(ROOT))]=sha(p)
    for p,h in pins.items():assert sha(ROOT/p)==h,p
    write(BASE/(stage.upper()+'_RUNTIME.json'),dict(status='locked',scope='native_instance_'+stage,count=8,GT_read=False,pins=pins,protected=plan['protected'],time=time.time()))

def launch(stage):
    verify_plan()
    subprocess.run(['systemd-run','--user','--unit='+unit(stage),'--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--property=OOMPolicy=stop','--property=KillMode=control-group','--working-directory='+str(ROOT),str(PY),'-B',str(ROOT/('scripts/launch_stvg_native_instance_teacher_storage002.py' if stage=='teacher' else 'scripts/launch_stvg_native_instance_v1.py')),stage],check=True)
    wait_phase(stage)

def main():
    verify_plan()
    write(BASE/'SERIAL_STORAGE002_LAUNCH.json',dict(status='waiting_tracking_actual_exit',GT_read=False,time=time.time()))
    assert properties('tracking')['ActiveState']=='inactive'
    barrier('tracking')
    lock_runtime('teacher',[BASE/'TRACKING_GLOBAL_BARRIER.json',BASE/'TRACKING_ACTUAL_PHASE_EXIT.json'])
    launch('teacher')
    verify_plan()
    subprocess.run([str(NATIVE),'-B',str(ROOT/'scripts/prepare_stvg_native_instance_references_v1.py')],check=True)
    # All references are locked before any student update. The preparer only
    # consumes official masks, R1 outputs and the original unlabeled QA centers.
    refs=read(BASE/'REFERENCE_LOCK.json')
    assert refs['status']=='locked' and refs['GT_read'] is False and len(refs['records'])==8
    lock_runtime('student',[BASE/'TRACKING_GLOBAL_BARRIER.json',BASE/'TEACHER_GLOBAL_BARRIER.json',BASE/'TEACHER_ACTUAL_PHASE_EXIT.json',BASE/'REFERENCE_LOCK.json',BASE/'DIRECT_TEACHER_READOUT.json'])
    launch('student')
    verify_plan()
    paths=[BASE/(s+'_GLOBAL_BARRIER.json')for s in ('TRACKING','TEACHER','STUDENT')]+[BASE/'REFERENCE_LOCK.json',BASE/'DIRECT_TEACHER_READOUT.json']
    # Bind payloads, not only receipts, into the final pre-label seal.
    for r in read(BASE/'ROSTER_LOCK.json')['records']:
        for ext in ('json','receipt.json'):paths.append(BASE/'teacher'/(r['case_id']+'.'+ext))
        for ext in ('pt','receipt.json'):paths.append(BASE/'student'/(r['case_id']+'.'+ext))
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=8,GT_read=False,files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)for p in paths],time=time.time()))
    write(BASE/'STATUS.json',dict(status='predictions_sealed_pending_independent_root_and_postseal_scoring',stage='root_handoff',done=8,total=8,GT_read=False,time=time.time()),mutable=True)
    write(BASE/'ACTUAL_ROOT_HANDOFF.json',dict(status='pending_actual_root_scoring_visual_public_archive',GT_read=False,all_predictions_sealed=True,experiment_complete=False,time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(BASE/'SERIAL_STORAGE002_FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()))
        raise
