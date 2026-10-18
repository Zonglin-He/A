"""One bounded serial controller, no retry loop or historical queue restart."""
from pathlib import Path
import json,os,select,subprocess,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/stvg_joint_posterior_v1';PY=ROOT/'.conda/tubedetr/bin/python'
def read(p):return json.loads(Path(p).read_text())
def put(p,x):
 t=Path(p).with_name(Path(p).name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=read(BASE/'PLAN_LOCK.json');old=read(BASE/'EXECUTION_RUNTIME001.json');pid=int(old['pid'])
 if Path(f'/proc/{pid}/stat').exists():
  assert Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]==old['starttick']
  fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN);remaining=max(0,int((p['deadline_epoch']-time.time())*1000));assert poll.poll(remaining),'deadline while waiting capture';os.close(fd)
 assert read(BASE/'FEATURE_BARRIER.json')['count']==128
 report=read(BASE/'REPORT_RUNTIME_LOCK001.json')
 for f,h in report['pins'].items():assert sha(ROOT/f)==h
 stages=[('train','scripts/run_stvg_joint_posterior_v1.py','train'),('source_validation','scripts/report_stvg_joint_posterior_v1.py','source_validation'),('infer','scripts/run_stvg_joint_posterior_v1.py','infer'),('score','scripts/run_stvg_joint_posterior_v1.py','score'),('root_audit','scripts/report_stvg_joint_posterior_v1.py','audit'),('render','scripts/report_stvg_joint_posterior_v1.py','render')]
 for name,script,command in stages:
  remaining=p['deadline_epoch']-time.time();assert remaining>0
  put(BASE/'CONTROLLER_STATUS.json',dict(status='running',stage=name,pid=os.getpid()))
  with open(BASE/('worker_'+name+'001.log'),'a')as log:subprocess.run([str(PY),'-B',script,command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
 put(BASE/'CONTROLLER_COMPLETION.json',dict(status='complete',count=32,pid=os.getpid(),decision=read(BASE/'DECISION.json')['status'],root_pixel_review_and_publication_pending=True))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  put(BASE/'CONTROLLER_FAILURE.json',dict(status='failed',exception=type(e).__name__,message=str(e),pid=os.getpid()));raise
