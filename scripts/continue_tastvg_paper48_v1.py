"""Finite, exclusive serial Paper48 continuation. Never resumes old B1/paper jobs."""
import os,sys,time,subprocess,signal,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_paper48_common_v1 import BASE,read,write,sha,status,verify

def archive(event):
 p=ROOT/'docs/RESEARCH_HISTORY.md';s=p.read_text();e='**Paper48执行事件：'+event+'。** 固定方法和hash名单不变；按最新授权无总截止，P5必做，旧B1/外部基线/Pairwise/动态流不恢复。已完成与待执行分开；见[队列状态](</home/wwww/visual grounding/artifacts/tastvg_paper48_v1/QUEUE_STATUS.json>)。';s=s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1);s+='\n\n### Paper48 execution event\n\n'+e+'\n';p.write_text(s)
 with (BASE/'ARCHIVE.log').open('a') as log:
  for cmd in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def stage(name,script,args,receipt,python='.conda/tubedetr/bin/python',gpu=True):
 if (BASE/receipt).exists():return
 verify();now=time.time()
 status(BASE/'QUEUE_STATUS.json',dict(status='running',stage=name,pid=os.getpid(),time=now,deadline_unix=None,P5_required=True));archive(name+'开始，尚未完成')
 with (BASE/(name+'.log')).open('a') as log:
  env=os.environ.copy()
  if name in ['spatial','P5_spatial']:env['PYTHONPATH']=str(ROOT/'.runtime/sa2va_deps')+os.pathsep+env.get('PYTHONPATH','')
  process=subprocess.Popen(['bash','scripts/with_local_cuda.sh',str(ROOT/python),'-B',script,*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status(BASE/'QUEUE_STATUS.json',dict(status='running',stage=name,pid=os.getpid(),worker_pid=process.pid,time=time.time(),deadline_unix=None,P5_required=True))
  code=process.wait()
 if code or not (BASE/receipt).exists():raise RuntimeError(f'{name} exit={code}, completion_receipt={(BASE/receipt).exists()}')
 archive(name+'完成，根核验与公开同步待执行')

def run():
 lock=(BASE/'QUEUE_PROCESS.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 assert read(BASE/'old_task_stop/STOP_RECEIPT.json')['status']=='paused_by_user_superseded_by_paper48'
 stage('P0_raw','scripts/run_tastvg_paper48_raw_v1.py',['run'],'P0/PREDICTION_BARRIER.json')
 stage('P0_score','scripts/score_tastvg_paper48_raw_v1.py',[],'P0/AUDIT.json',gpu=False)
 stage('P0_report','scripts/report_tastvg_paper48_raw_v1.py',[],'P0/COMPLETION.json',gpu=False)
 stage('spatial','scripts/run_tastvg_paper48_experts_v1.py',['spatial'],'experts/SPATIAL_BARRIER.json')
 stage('temporal','scripts/run_tastvg_paper48_experts_v1.py',['temporal'],'experts/TEMPORAL_BARRIER.json',python='.venv-exost/bin/python')
 for panel in ['P1','P2','P3_b0','P3_b25','P3_b100']:
  stage(panel+'_online','scripts/run_tastvg_paper48_online_v1.py',[panel],panel+'/PREDICTION_BARRIER.json')
  stage(panel+'_score','scripts/score_tastvg_paper48_v1.py',[panel],panel+'/COMPLETION.json',gpu=False)
 stage('P4_efficiency','scripts/run_tastvg_paper48_efficiency_v1.py',[],'P4/COMPLETION.json')
 stage('paper_readouts','scripts/report_tastvg_paper48_v1.py',[],'MANDATORY_COMPLETION.json',gpu=False)
 stage('P5_hc2','scripts/continue_tastvg_paper48_p5_v1.py',[],'P5/COMPLETION.json')
 status(BASE/'QUEUE_STATUS.json',dict(status='P0_P5_completed_pending_root_review',remaining='root verification and public synchronization of P0-P5',pid=os.getpid(),time=time.time()));archive('P0-P5执行完成，根复核/公开同步待执行；不称已公开收尾')
if __name__=='__main__':
 try:run()
 except BaseException as e:
  status(BASE/'QUEUE_STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));archive('中断保留：'+repr(e));raise
