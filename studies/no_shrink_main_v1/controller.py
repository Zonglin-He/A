"""Adopt existing paid collector and finite independent kappa0 computations."""
from ns_common import *
import subprocess,traceback,signal,shutil,argparse
JOBS=[(t,d)for t in ['components','CG','budget','corruption','spatial']for d in COUNTS]
def mem():return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
def queue():
 p=validate();assert read(D/'preflight_receipt.json')['status']=='passed';assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who);atomic(D/'launch.json',dict(controller=who,fingerprint=p['fingerprint'],adopted_HC_collector=read(PREV/'spatial/hc2_worker.json'),time=time.time()));live={};failed=[];gpu=None;timed=False;vid_started=False
 def maintain():
  for job,(c,w)in list(live.items()):
   if c.poll()is not None:
    atomic(D/f'{job[0]}_{job[1]}_domain_exit.json',dict(code=c.returncode));live.pop(job)
    if c.returncode:failed.append(job)
  for job in JOBS:
   if job in live or job in failed or(D/f'{job[0]}_{job[1]}_domain_child.json').exists():continue
   if len(live)>=2 or mem()<8*2**30:break
   if job[0]=='spatial'and job[1]=='vidstg'and not vid_started:continue
   task,ds=job;log=(D/f'{task}_{ds}_domain.log').open('ab');args=['taskset','-c','14,15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'run.py'),'domain','--task',task,'--dataset',ds];c=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},start_new_session=True);log.close();w=identity(c.pid);live[job]=(c,w);atomic(D/f'{task}_{ds}_domain_child.json',w)
 def pause_cpu():
  groups=[w for c,w in live.values()if c.poll()is None]
  for w in groups:assert os.getpgid(w['pid'])==w['pid'];os.killpg(w['pid'],signal.SIGSTOP)
  atomic(D/'deployment_CPU_pause.json',dict(groups=groups,time=time.time()));return groups
 def resume_cpu(groups):
  for w in groups:
   if not exited(w):os.killpg(w['pid'],signal.SIGCONT)
  atomic(D/'deployment_CPU_resume.json',dict(groups=groups,time=time.time()))
 try:
  hc=read(PREV/'spatial/hc2_worker.json');assert hc==p['paid_grid_collector']['existing_HC']
  while not exited(hc):maintain();atomic(D/'latest_monitor.json',dict(time=time.time(),HC_collector=hc,HC_progress=read(PREV/'spatial/hc2_progress.json'),available_RAM=mem(),disk_free=shutil.disk_usage(R).free,CPU_jobs={str(k):w for k,(c,w)in live.items()},failed=failed));time.sleep(10)
  assert read(PREV/'spatial/hc2_sealed.json')['queries']==1902,'Preserve incomplete paid collector; do not retry'
  assert not failed,'CPU failure preserved; no further GPU work'
  groups=pause_cpu()
  try:
   assert not(D/'deployment/worker.json').exists();log=(D/'deployment.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(S/'timing.py'),'run'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);atomic(D/'deployment_child.json',identity(c.pid));rc=c.wait();log.close();atomic(D/'deployment_exit.json',dict(code=rc));assert rc==0,'Preserve timed paid attempts; no retry'
  finally:resume_cpu(groups)
  log=(D/'deployment_export.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(S/'timing.py'),'export'],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});assert c.wait()==0;log.close();timed=True
  assert not(PREV/'spatial/vidstg_worker.json').exists(),'No duplicate Vid collector';assert not(D/'Vid_collector_child.json').exists();log=(D/'Vid_collector.log').open('ab');gpu=subprocess.Popen([sys.executable,'-B',str(R/'studies/map_paper_completion_v1/spatial_runtime_v2.py'),'predict','--dataset','vidstg'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);atomic(D/'Vid_collector_child.json',identity(gpu.pid));vid_started=True
  while gpu.poll()is None:maintain();atomic(D/'latest_monitor.json',dict(time=time.time(),Vid_collector=identity(gpu.pid),CPU_jobs={str(k):w for k,(c,w)in live.items()},available_RAM=mem(),disk_free=shutil.disk_usage(R).free,failed=failed));time.sleep(10)
  log.close();atomic(D/'Vid_collector_exit.json',dict(code=gpu.returncode));assert gpu.returncode==0 and read(PREV/'spatial/vidstg_sealed.json')['queries']==10303
  while live or any(not(D/f'{t}_{d}_domain_child.json').exists()for t,d in JOBS):maintain();time.sleep(5)
  assert not failed;atomic(D/'state.json',dict(status='all_recalculations_scored_audited_exported_pending_root_figures_publication',controller=who,remaining=['rootallpositive_negativeanalysis','rootbudgetPNGview','newqualitativekappa0renderandview','ALLremote_publication_verify']))
 except BaseException:atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':queue()
