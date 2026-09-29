"""One finite full-evaluation pipeline, attaches to the already running spatial stage."""
import os,sys,time,subprocess,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/tastvg_full_b1_v1'
def read(f):return json.loads(f.read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def status(value):
 p=OUT/'PIPELINE_STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(**value,time=time.time()),indent=2)+'\n');tmp.replace(p)
def archive(event):
 p=ROOT/'docs/RESEARCH_HISTORY.md';s=p.read_text();entry=f'**Phase B全量流程状态更新：{event}。** 固定9411query/670源、三序16条件；方法/名单不改，尚未封存的阶段不称科学完成。实时回执见[流程状态](</home/wwww/visual grounding/artifacts/tastvg_full_b1_v1/PIPELINE_STATUS.json>)。'
 h='## 1. 当前状态：先读这一节\n';s=s.replace(h,h+'\n'+entry+'\n',1);s+='\n\n### Phase B pipeline event\n\n'+entry+'\n';p.write_text(s)
 with (OUT/'PIPELINE_ARCHIVE.log').open('a') as log:
  for cmd in ['check','snapshot','check']:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/research_archive.py',cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
def verify():
 p=read(OUT/'POSTPROCESS_LOCK.json')
 for f,h in p['pins'].items():assert sha(ROOT/f)==h,f

def stage(name,script,arguments,barrier,env=None):
 if (OUT/barrier).exists():return
 verify();status(dict(status='running',stage=name));archive(name+'运行')
 with (OUT/f'{name.upper()}_PIPELINE.log').open('a') as log:
  code=subprocess.call(['bash','scripts/with_local_cuda.sh',str(ROOT/'.conda/tubedetr/bin/python'),'-B',script,*arguments],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if code or not (OUT/barrier).exists():raise RuntimeError(f'{name} failed: exit={code}, missing_barrier={not (OUT/barrier).exists()}')
 archive(name+'阶段完成，后续阶段继续')

if __name__=='__main__':
 try:
  verify();pid=int(sys.argv[1]);status(dict(status='running',stage='spatial',attached_pid=pid));archive('空间专家正式运行；后续temporal、online、评分、报告串行接续已配置')
  while not (OUT/'SPATIAL_BARRIER.json').exists():
   try:os.kill(pid,0)
   except ProcessLookupError:raise RuntimeError('spatial process ended before barrier; preserve failed receipts')
   time.sleep(30)
  stage('temporal','scripts/run_tastvg_full_b1_experts_v1.py',['temporal'],'TEMPORAL_BARRIER.json')
  stage('online','scripts/run_tastvg_full_b1_online_v1.py',[],'PREDICTION_BARRIER.json')
  stage('score','scripts/score_tastvg_full_b1_v1.py',[],'AUDIT.json')
  stage('report','scripts/report_tastvg_full_b1_v1.py',[],'COMPLETION.json')
  status(dict(status='completed_pending_remote_publication',stage='complete',report=str(OUT/'REPORT.md')));archive('全量推理/评分/报告完成，GitHub远端核验待执行')
 except BaseException as e:
  status(dict(status='failed',error=repr(e)));archive('工程中断，保留所有完成产物及失败回执：'+str(e));raise
