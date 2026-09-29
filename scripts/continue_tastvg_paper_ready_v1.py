"""Finite serial continuation: wait for B1, run only implemented/locked ready stages."""
import os,sys,time,json,subprocess,hashlib,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/tastvg_paper_matrix_v1';B=ROOT/'artifacts/tastvg_full_b1_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**value):
 p=OUT/'QUEUE_STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(**value,time=time.time()),indent=2)+'\n');tmp.replace(p)
def verify():
 for f,h in read(OUT/'READY_LOCK.json')['pins'].items():assert sha(ROOT/f)==h,f

def archive(event):
 p=ROOT/'docs/RESEARCH_HISTORY.md';s=p.read_text();e='**2026-09-30｜论文补证串行队列：'+event+'。** B1冻结配方和全量名单不改；A2为16曝光源/五序六条件960到达；baseline smoke仅两旧fixture，不是完整baseline效果。后续未实现阶段不称已运行。见[队列状态](</home/wwww/visual grounding/artifacts/tastvg_paper_matrix_v1/QUEUE_STATUS.json>)。'
 s=s.replace('## 1. 当前状态：先读这一节\n','## 1. 当前状态：先读这一节\n\n'+e+'\n',1);s+='\n\n### Paper evidence queue event\n\n'+e+'\n';p.write_text(s)
 with (OUT/'QUEUE_ARCHIVE.log').open('a') as log:
  for op in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',op],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def run():
 OUT.mkdir(exist_ok=True);lock=(OUT/'QUEUE_PROCESS.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 status(status='waiting_for_B1',pid=os.getpid(),active_GPU_job_started=False)
 deadline=time.monotonic()+8*86400
 while not (B/'COMPLETION.json').exists():
  z=read(B/'PIPELINE_STATUS.json')
  if z.get('status')=='failed':raise RuntimeError('B1 failed; preserve state and notify root, no research continuation')
  if time.monotonic()>deadline:raise TimeoutError('Finite B1 waiting deadline')
  pid=z.get('attached_pid')
  if z.get('stage')=='spatial' and pid and not Path('/proc',str(pid)).exists():raise RuntimeError('Spatial process absent without B1 completion')
  time.sleep(60)
 assert read(B/'COMPLETION.json')['status']=='completed'
 stages=[('a2_predict','scripts/run_tastvg_matched_ablation_a2_v1.py',['run'], 'artifacts/tastvg_matched_ablation_a2_v1/PREDICTION_BARRIER.json',2700),('a2_score','scripts/score_tastvg_matched_ablation_a2_v1.py',[],'artifacts/tastvg_matched_ablation_a2_v1/AUDIT.json',1800),('a2_report','scripts/report_tastvg_matched_ablation_a2_v1.py',[],'artifacts/tastvg_matched_ablation_a2_v1/COMPLETION.json',600),('baseline_smoke','scripts/validate_tastvg_paper_baselines_v1.py',[],'artifacts/tastvg_paper_matrix_v1/baseline_smoke/AUDIT.json',1200),('b1_readouts','scripts/analyze_tastvg_full_b1_paper_v1.py',[],'artifacts/tastvg_paper_matrix_v1/b1_readouts/READOUTS.json',12*3600)]
 for name,script,args,receipt,timeout in stages:
  verify()
  if (ROOT/receipt).exists():continue
  status(status='running',stage=name,pid=os.getpid());archive(name+'启动')
  with (OUT/(name+'.log')).open('a') as log:
   process=subprocess.Popen(['bash','scripts/with_local_cuda.sh',str(ROOT/'.conda/tubedetr/bin/python'),'-B',script,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=60)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=30)
    raise TimeoutError(name+' finite stage budget exhausted; outputs preserved')
  if code or not (ROOT/receipt).exists():raise RuntimeError(f'{name}: exit={code}, receipt_exists={(ROOT/receipt).exists()}')
  archive(name+'完成，结果等待主任务复核与GitHub同步')
 status(status='ready_stages_completed_pending_root_review',remaining='CoTTA/RoTTA/full baseline integration; budget/dynamic/HC/efficiency',pid=os.getpid());archive('已实现阶段完成；完整论文矩阵仍未完成')
if __name__=='__main__':
 try:run()
 except BaseException as e:
  status(status='failed',error=repr(e),traceback=traceback.format_exc());archive('失败保留，等待根处理：'+repr(e));raise
