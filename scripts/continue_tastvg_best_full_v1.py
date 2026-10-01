"""One finite serial controller; full global prediction seal precedes any GT scorer."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_best_full_common_v1 import *
def stage(dataset,name,script,receipt,python='.conda/tubedetr/bin/python',args=None,cpu=False):
 if receipt.exists():return
 verify(dataset);budget();out=BASE/dataset
 with (out/(name+'.log')).open('a') as log:
  env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2')
  if cpu:env['CUDA_VISIBLE_DEVICES']=''
  if name=='spatial':env['PYTHONPATH']=str(ROOT/'.runtime/sa2va_deps')+os.pathsep+env.get('PYTHONPATH','')
  command=['bash','scripts/with_local_cuda.sh',str(ROOT/python),'-B',script,*(args or [dataset])]
  proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status(BASE/'STATUS.json',dict(status='running',dataset=dataset,stage=name,controller_pid=os.getpid(),worker_pid=proc.pid,time=time.time()));code=proc.wait()
 if code or not receipt.exists():raise RuntimeError(f'{dataset}/{name}: exit{code}; missing {receipt.name}')
def run():
 handle=(BASE/'PROCESS.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 # Both selected parameter settings qualify before full stream work.
 for d in DATASETS:stage(d,'smoke','scripts/smoke_tastvg_best_full_v1.py',BASE/d/'SMOKE.json')
 if not (BASE/'SMOKE_ACCEPTANCE.json').exists():
  for d in DATASETS:assert read(BASE/d/'SMOKE.json')['status']=='pass'
  write(BASE/'SMOKE_ACCEPTANCE.json',dict(status='pass',datasets={d:sha(BASE/d/'SMOKE.json') for d in DATASETS},GT_read=False,time=time.time()))
  archive('两数据集日志实现与封存v3逐值一致通过，开始全量准备')
 for d in DATASETS:
  out=BASE/d
  stage(d,'subjects','scripts/prepare_tastvg_best_full_subjects_v1.py',out/'SUBJECT_BARRIER.json',cpu=True)
  stage(d,'spatial','scripts/run_tastvg_best_full_experts_v1.py',out/'experts/SPATIAL_BARRIER.json',args=[d,'spatial'])
  stage(d,'temporal','scripts/run_tastvg_best_full_experts_v1.py',out/'experts/TEMPORAL_BARRIER.json',python='.venv-exost/bin/python',args=[d,'temporal'])
  stage(d,'online','scripts/run_tastvg_best_full_online_v1.py',out/'PREDICTION_BARRIER.json')
  if not (out/'PREDICTION_STAGE_HANDOFF.json').exists():
   write(out/'PREDICTION_STAGE_HANDOFF.json',dict(status='predictions_complete_waiting_other_dataset_global_seal',cells=read(out/'PREDICTION_BARRIER.json')['cells'],GT_read=False,time=time.time()));archive(d+'全量预测完成封存，GT评分等待两数据集全局封存')
 if not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
  for d in DATASETS:assert read(BASE/d/'PREDICTION_BARRIER.json')['cells']==read(BASE/d/'PLAN.json')['total']
  write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(cells=165420,datasets={d:sha(BASE/d/'PREDICTION_BARRIER.json') for d in DATASETS},GT_read=False,time=time.time()))
 for d in DATASETS:stage(d,'score_diagnose','scripts/score_tastvg_best_full_v1.py',BASE/d/'COMPLETION.json',cpu=True)
 status(BASE/'STATUS.json',dict(status='completed_pending_root_audit_publication',time=time.time()));archive('两数据集全量评分及GT pipeline诊断已执行，待根复核报告公开')
if __name__=='__main__':
 try:run()
 except BaseException as e:
  status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));archive('工程失败已保存 '+repr(e));raise
