"""Preserve active GPU worker; replace only the downstream source label binding."""
from pathlib import Path
import os,sys,json,time,select,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/stvg_joint_event_critic_v1'
def read(p):return json.loads(Path(p).read_text())
def put(p,x):
 p=Path(p);assert not p.exists(),p;p.write_text(json.dumps(x,indent=2)+'\n')
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split();return dict(starttick=int(stat[19]),state=stat[0],cgroup=p.joinpath('cgroup').read_text())
 except FileNotFoundError:return None
def main():
 lock=read(BASE/'LABEL_RUNTIME_LOCK002.json')
 for rel,h in lock['pins'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
 rec=read(BASE/'recovery/label_binding002/CPU_HANDOFF.json');p=read(BASE/'PLAN_LOCK.json');worker=rec['source_worker'];old=rec['controller'];now=identity(worker['pid'])
 assert now and now['starttick']==worker['starttick']and now['cgroup']==worker['cgroup']
 original=identity(old['pid']);assert original and original['state']=='T'and original['starttick']==old['starttick']
 put(BASE/'EXECUTION_RUNTIME002.json',dict(status='waiting_original_source_worker_normal_exit',original_worker=worker['pid'],original_controller=old['pid'],deadline_epoch=p['deadline_epoch'],source_model_computations_unchanged=True,source_labels_rebound_before_training=True))
 fd=os.pidfd_open(worker['pid']);poll=select.poll();poll.register(fd,select.POLLIN)
 while not poll.poll(1000):
  if time.time()>=p['deadline_epoch']:raise RuntimeError('Original deadline reached; original partials preserved')
 os.close(fd);bar=read(BASE/'SOURCE_PREDICTION_BARRIER.json');assert bar['status']=='complete'and bar['count']==96
 now=identity(worker['pid']);assert now is None or now['starttick']!=worker['starttick']or now['state']=='Z'
 original=identity(old['pid']);assert original and original['state']=='T'and original['starttick']==old['starttick']
 subprocess.run(['systemctl','--user','stop','stvg-joint-event-critic-001.service'],check=True)
 put(BASE/'CONTROLLER_HANDOFF_COMPLETION002.json',dict(status='complete',count=96,original_GPU_exited_before_old_CPU_unit_retired=True,source_barrier_complete=True,original_worker_not_signalled=True,old_controller_retired=True,deadline_not_reset=True))
 py=ROOT/'.conda/tubedetr/bin/python'
 for stage in ['source_labels','train','infer','score']:
  if time.time()>=p['deadline_epoch']:raise RuntimeError('Original finite deadline reached')
  script=ROOT/'scripts/stvg_joint_event_source_labels002.py'if stage=='source_labels'else ROOT/'scripts/run_stvg_joint_event_critic_v1.py';argument='labels'if stage=='source_labels'else stage
  with (BASE/('worker_'+stage+'002.log')).open('xb')as log:subprocess.run([str(py),'-B',str(script),argument],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 decision=read(BASE/'DECISION.json');put(BASE/'ROOT_HANDOFF.json',dict(status='ready',count=32,scientific_decision=decision['status'],independent_root_pixel_publication_required=True,matched_OPD_required=decision.get('matched_OPD')=='required_next'))
if __name__=='__main__':
 try:main()
 except BaseException as e:put(BASE/('CONTROLLER_FAILURE002_'+str(time.time_ns())+'.json'),dict(status='failed',error_type=type(e).__name__,error=str(e),partial_preserved=True));raise
