"""Finite serial controller with an explicit predecessor handoff."""
import sys,os,time,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_cross_domain_common_v1 import *
PY=ROOT/'.conda/tubedetr/bin/python'

def child(script,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        env=os.environ.copy();interpreter=PY
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2')
        if script=='run_tastvg_cross_domain_experts_v1.py':
            if args[-1]=='spatial':env['PYTHONPATH']=str(ROOT/'.runtime/sa2va_deps')+os.pathsep+env.get('PYTHONPATH','')
            else:interpreter=ROOT/'.venv-exost/bin/python'
        command=['bash',str(ROOT/'scripts/with_local_cuda.sh'),str(interpreter),'-B',str(ROOT/'scripts'/script),*args]
        q=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        status(BASE/'CURRENT_WORKER.json',dict(pid=q.pid,script=script,args=args,GT_read=False,time=time.time()))
        assert q.wait()==0,script+' '+str(args)

def run():
    verify();pid=os.getpid();write(BASE/'LAUNCH.json',dict(controller_pid=pid,GT_read=False,time=time.time()))
    try:
        predecessor=ROOT/'artifacts/tastvg_correction_scope_v1'
        status(BASE/'STATUS.json',dict(status='waiting_for_predecessor_gpu',controller_pid=pid,
            predecessor=str(predecessor),GT_read=False,time=time.time()))
        while not (predecessor/'GLOBAL_PREDICTION_BARRIER.json').exists():
            st=read(predecessor/'STATUS.json')
            assert not st['status'].startswith('failed'),'Predecessor needs root engineering recovery'
            os.kill(read(predecessor/'LAUNCH.json')['controller_pid'],0)
            time.sleep(10)
        assert read(predecessor/'GLOBAL_PREDICTION_BARRIER.json')['GT_read'] is False
        # Its GPU controller creates the barrier only after all workers have exited.
        for direction in DIRECTIONS:
            child('prepare_tastvg_cross_domain_v1.py',[direction],BASE/'logs'/f'{direction}_subjects.log')
            for stage in ['spatial','temporal','capture','target_reference','online']:
                folder=BASE/direction
                barrier=folder/'experts'/(stage.upper()+'_BARRIER.json') if stage in ['spatial','temporal'] else folder/('PREDICTION_BARRIER.json' if stage=='online' else stage.upper()+'_BARRIER.json')
                if barrier.exists():continue
                budget();verify(direction)
                status(BASE/'STATUS.json',dict(status='running',direction=direction,stage=stage,
                    controller_pid=pid,GT_read=False,time=time.time()))
                script='run_tastvg_cross_domain_experts_v1.py' if stage in ['spatial','temporal'] else 'run_tastvg_cross_domain_v1.py'
                child(script,[direction,stage],BASE/'logs'/f'{direction}_{stage}.log')
        write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',GT_read=False,
            total=1557,directions={d:sha(BASE/d/'PREDICTION_BARRIER.json') for d in DIRECTIONS},time=time.time()))
        seal()
        status(BASE/'STATUS.json',dict(status='predictions_sealed_pending_cpu_diagnosis_root_publication',
            controller_pid=pid,GT_read=False,total=1557,time=time.time()))
    except BaseException as e:
        status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),
            controller_pid=pid,time=time.time()));raise

if __name__=='__main__':run()
