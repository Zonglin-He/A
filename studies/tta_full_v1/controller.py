"""One GPU owner and two shared CPU slots for existing kappa0 work plus full TTA."""
from full_common import *
import subprocess,traceback,signal,shutil
KJOBS=[(t,d)for t in ['components','CG','budget','corruption','spatial']for d in COUNTS]
JOBS=[('TTA',d)for d in COUNTS]+KJOBS

def mem():return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
def finish_path(job):
 task,ds=job;return D/f'{ds}_reporting_receipt.json'if task=='TTA'else K0/f'{task}_{ds}_reporting_receipt.json'
def child_path(job):
 task,ds=job;return D/f'{ds}_CPU_child.json'if task=='TTA'else K0/f'{task}_{ds}_domain_child.json'
def cpu_job(ds):
 validate();who=identity();atomic(D/f'{ds}_CPU_worker.json',who)
 try:
  for command in ['score','audit']:
   atomic(D/f'{ds}_CPU_state.json',dict(status=command,worker=who));log=(D/f'{ds}_{command}.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(S/'run.py'),command,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);atomic(D/f'{ds}_{command}_child.json',identity(c.pid));rc=c.wait();log.close();atomic(D/f'{ds}_{command}_exit.json',dict(code=rc));assert rc==0,'Preserve full TTA score/audit failure'
  atomic(D/f'{ds}_CPU_state.json',dict(status='complete_scored_audited_pending_root_export_publication',worker=who))
 except BaseException:atomic(D/f'{ds}_CPU_state.json',dict(status='failed_preserved',error=traceback.format_exc(),worker=who));raise

def queue():
 p=validate();assert read(D/'preflight_receipt.json')['status']=='passed';assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who);atomic(D/'launch.json',dict(controller=who,fingerprint=p['fingerprint'],adopted_HC=read(PREV/'spatial/hc2_worker.json'),time=time.time()));live={};completed=set();failed=[];vid_started=False
 for job in KJOBS:
  cp=child_path(job)
  if cp.exists():
   w=read(cp)
   if not exited(w):live[job]=(None,w)
   else:assert finish_path(job).exists(),f'Existing failed/incomplete CPU domain {job}';completed.add(job)
 assert len(live)<=2;atomic(D/'adopted_CPU_jobs.json',{str(j):w for j,(c,w)in live.items()})
 def maintain():
  for job,(c,w)in list(live.items()):
   done=c.poll()is not None if c is not None else exited(w)
   if done:
    if(c is not None and c.returncode!=0)or not finish_path(job).exists():failed.append(job)
    else:completed.add(job)
    live.pop(job)
  for job in JOBS:
   if job in live or job in completed or job in failed:continue
   if finish_path(job).exists():completed.add(job);continue
   cp=child_path(job)
   if cp.exists():raise RuntimeError(f'Previously started incomplete domain {job}; preserve, no blind retry')
   if len(live)>=2 or mem()<8*2**30:break
   task,ds=job
   if task=='TTA'and not(D/f'{ds}_sealed.json').exists():continue
   if task=='spatial'and ds=='vidstg'and not vid_started:continue
   args=[sys.executable,'-B',__file__,'cpu-job','--dataset',ds]if task=='TTA'else[sys.executable,'-B',str(R/'studies/no_shrink_main_v1/run.py'),'domain','--task',task,'--dataset',ds];log=(D/f'{task}_{ds}_CPU.log').open('ab');c=subprocess.Popen(['taskset','-c','14,15','nice','-n','19','ionice','-c','3',*args],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},start_new_session=True);log.close();w=identity(c.pid);live[job]=(c,w);atomic(cp,w)
 def monitor(phase,gpuworker):
  atomic(D/'latest_monitor.json',dict(status=phase,time=time.time(),controller=who,active_GPU=gpuworker,CPU_jobs={str(k):w for k,(c,w)in live.items()},failed=failed,available_RAM_bytes=mem(),disk_free_bytes=shutil.disk_usage(R).free,HC_collector_progress=read(PREV/'spatial/hc2_progress.json')))
 def gpu(name,script,args):
  assert not(D/f'{name}_child.json').exists()
  while mem()<8*2**30:maintain();time.sleep(5)
  assert not failed,'CPU failure blocks further expensive work';guard_resources();log=(D/f'{name}.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(script),*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);w=identity(c.pid);atomic(D/f'{name}_child.json',w)
  while c.poll()is None:maintain();monitor(name,w);time.sleep(5)
  log.close();atomic(D/f'{name}_exit.json',dict(code=c.returncode));assert c.returncode==0,f'Preserve {name} outputs/paid attempts; no automatic retry'
 def pause():
  groups=[w for c,w in live.values()if not exited(w)]
  for w in groups:assert os.getpgid(w['pid'])==w['pid'];os.killpg(w['pid'],signal.SIGSTOP)
  atomic(K0/'deployment_CPU_pause.json',dict(groups=groups,time=time.time(),owner=who));return groups
 def resume(groups):
  for w in groups:
   if not exited(w):os.killpg(w['pid'],signal.SIGCONT)
  atomic(K0/'deployment_CPU_resume.json',dict(groups=groups,time=time.time(),owner=who))
 try:
  oldowner=read(D/'previous_no_shrink_dispatcher/controller.json');assert exited(oldowner),'Previous own dispatcher must exit before takeover';hc=read(PREV/'spatial/hc2_worker.json')
  while not exited(hc):maintain();monitor('waiting_current_HC_gridcollector_without_interruption',hc);time.sleep(5)
  assert read(PREV/'spatial/hc2_sealed.json')['queries']==1902,'Preserve incomplete paid collector'
  gpu('TTA_hc2_GPU',S/'run.py',['predict','--dataset','hc2']);assert read(D/'hc2_sealed.json')['queries']==1902;maintain()
  groups=pause()
  try:
   assert not(K0/'deployment/worker.json').exists();gpu('current_kappa0_timing',R/'studies/no_shrink_main_v1/timing.py',['run'])
  finally:resume(groups)
  # Timing must be exported before later Source work; never relabel kappa64 timings.
  log=(D/'current_timing_export.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(R/'studies/no_shrink_main_v1/timing.py'),'export'],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});assert c.wait()==0;log.close()
  vid_started=True;gpu('Vid_grid_and_TTA_GPU',S/'spatial_tta.py',[]);assert read(D/'vidstg_sealed.json')['queries']==10303 and read(PREV/'spatial/vidstg_sealed.json')['queries']==10303
  while live or len(completed)<len(JOBS):maintain();monitor('finishing_whole_domain_CPU_scores_audits',None);assert not failed;time.sleep(5)
  atomic(D/'state.json',dict(status='all_full_TTA_and_current_recalculations_scored_audited_pending_root_export_publication',controller=who,remaining=['root fullTTAanalysis','remainingkappa0rootexports','budgetPNGrootview','currenttiminganalysis','ALLremote_allpositive_negativepublication']))
 except BaseException:atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('command',choices=['queue','cpu-job']);p.add_argument('--dataset',choices=COUNTS);z=p.parse_args();queue()if z.command=='queue'else cpu_job(z.dataset)
