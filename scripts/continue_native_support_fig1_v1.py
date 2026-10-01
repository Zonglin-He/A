"""Finite new diagnostic, never starts any historical queue or GT scoring."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_support_fig1_common_v1 import *
def run(mode):
 assert mode in ['smoke','capture']
 lock=(OUT/'PROCESS.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 if mode=='capture':
  q=read(OUT/'SMOKE_ROOT_ACCEPTANCE.json');assert q['status']=='pass' and q['GT_read'] is False
 for model in MODELS:
  receipt=OUT/model/('SMOKE.json' if mode=='smoke' else 'PREDICTION_BARRIER.json')
  if receipt.exists():continue
  python='.venv-ptd-audit/bin/python' if model=='ptd' else '.conda/tubedetr/bin/python'
  env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env['OPENBLAS_NUM_THREADS']='4';env['OMP_NUM_THREADS']='4'
  with (OUT/f'{model}_{mode}.log').open('a') as f:
   proc=subprocess.Popen(['bash','scripts/with_local_cuda.sh',str(ROOT/python),'-B','scripts/run_native_support_fig1_v1.py',model,mode],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   set_status(dict(status='running',stage=mode,backbone=model,controller_pid=os.getpid(),worker_pid=proc.pid,GPU_started=True,GT_read=False,time=time.time()));code=proc.wait()
  if code or not receipt.exists():raise RuntimeError(f'{model}/{mode} exit={code}; missing_receipt={not receipt.exists()}')
 if mode=='capture':
  barriers={m:sha(OUT/m/'PREDICTION_BARRIER.json') for m in MODELS};assert all(read(OUT/m/'PREDICTION_BARRIER.json')['cells']==768 for m in MODELS)
  write(OUT/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',cells=2304,barriers=barriers,GT_read=False,time=time.time()))
 set_status(dict(status='smoke_completed_pending_root' if mode=='smoke' else 'all_predictions_sealed_pending_scoring',GPU_started=True,GT_read=False,model_predictions=6 if mode=='smoke' else 2304,time=time.time()))
if __name__=='__main__':
 try:run(sys.argv[1])
 except BaseException as e:set_status(dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
