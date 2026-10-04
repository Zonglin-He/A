"""Finite conditional route: no historical queue resumed, no unbounded polling."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import *

def run():
    handle=(BASE/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    launch=BASE/'LAUNCH.json'
    if launch.exists():launch=BASE/f'RESUME_LAUNCH_{os.getpid()}.json'
    write(launch,dict(pid=os.getpid(),time=time.time(),finite=True,argv=sys.argv))
    py=str(ROOT/'.conda/tubedetr/bin/python');worker='scripts/run_decota_actuation_scope_v1.py'
    def execute(name,cmd,GT=False):
        complete=BASE/'SMOKE_ROOT_ACCEPTANCE.json' if name=='smoke' else BASE/'ONLINE_LOCK.json' if name=='online_prepare' else None
        if name.endswith('_seal'):
            stage=name[:-5];complete=BASE/stage/'GLOBAL_PREDICTION_BARRIER.json'
        elif name.endswith('_score'):
            stage=name[:-6];complete=PUB/stage/'PUBLIC_AUDIT.json'
        elif any(name.endswith('_'+ds) for ds in DATASETS):
            stage,ds=name.rsplit('_',1);complete=BASE/stage/ds/'PREDICTION_BARRIER.json'
        if complete is not None and complete.exists():
            print('REUSE_COMPLETED_STAGE',name,str(complete.relative_to(ROOT)),flush=True);return
        log=BASE/(name+'.log')
        if log.exists():log=BASE/(name+f'.resume_{os.getpid()}.log')
        with log.open('w') as out:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
            status(BASE/'STATUS.json',dict(status='running',stage=name,worker_pid=p.pid,controller_pid=os.getpid(),log=str(log),GT_read=GT))
            assert p.wait()==0,(name,p.returncode)
    def stage(s):
        inputs=[BASE/'R2_SELECTION.json'] if s=='R3' else [BASE/'R3_FINAL_SELECTION.json'] if s=='R4' else [BASE/'R3_SELECTION.json']
        lock=BASE/s/'STAGE_LOCK.json';expected={str(p.relative_to(ROOT)):sha(p) for p in inputs}
        if lock.exists():assert read(lock)['inputs']==expected
        else:write(lock,dict(time=time.time(),inputs=expected,GT_prediction=False))
        for ds in DATASETS:execute(s+'_'+ds,['bash','scripts/with_local_cuda.sh',py,'-B',worker,'predict',s,ds])
        execute(s+'_seal',[py,'-B',worker,'seal',s])
        execute(s+'_score',[py,'-B','scripts/score_decota_actuation_scope_v1.py',s],True)
        archive(s+'全部1152匹配到达封存后完成独立CPU审计与开发选择，待公开收尾')
    try:
        execute('smoke',['bash','scripts/with_local_cuda.sh',py,'-B',worker,'smoke'])
        stage('R3')
        if read(BASE/'R3_SELECTION.json')['GIoU']=='eligible':stage('R3G')
        stage('R4')
        if read(r1.PUB/'DECISION.json')['T1']=='eligible' and not (BASE/'T1_COMPLETION.json').exists():
            status(BASE/'STATUS.json',dict(status='qualified_temporal_pending_root_T1_occupancy',R3_R4_complete=True,time=time.time()))
            archive('R3/R4已审计；T0资格成立，明确交接根实现已授权T1和occupancy，未提前运行最终组合')
            return
        # R5 and R6 use genuinely independent trajectories, not the local intervention prestates.
        execute('online_prepare',[py,'-B','scripts/run_decota_online_qualification_v1.py','prepare'])
        for ds in DATASETS:execute('online_'+ds,['bash','scripts/with_local_cuda.sh',py,'-B','scripts/run_decota_online_qualification_v1.py','predict',ds])
        execute('online_seal',[py,'-B','scripts/run_decota_online_qualification_v1.py','seal'])
        execute('online_score',[py,'-B','scripts/score_decota_online_qualification_v1.py'],True)
        status(BASE/'STATUS.json',dict(status='R3_R4_R5_R6_same_domain_predictions_complete_pending_root',cross_domain_complete=False,time=time.time()))
    except BaseException as e:
        failure=BASE/'FAILURE.json'
        if failure.exists():failure=BASE/f'FAILURE_{os.getpid()}.json'
        write(failure,dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));status(BASE/'STATUS.json',dict(status='failed_pending_root',error=repr(e)));raise
    finally:handle.close()

if __name__=='__main__':run()
