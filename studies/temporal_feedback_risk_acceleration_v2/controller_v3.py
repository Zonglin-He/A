"""Keep the two actual score workers; exact merge binding, then two cold audits."""
import runtime_v2 as v
from pathlib import Path
import os,sys,time,json,signal,subprocess,traceback,fcntl
D=v.D
def status(w):
 try:
  raw=(Path('/proc')/str(w['pid'])/'stat').read_text();s=raw.split(') ',1)[1].split()
  if int(s[19])!=w['startticks']:return None
  return dict(state=s[0],raw_stat=raw,raw_exit_status_field52=int(s[49]))
 except FileNotFoundError:return None
def kill_owned(w):
 if status(w):os.kill(w['pid'],signal.SIGTERM);os.kill(w['pid'],signal.SIGCONT)
def run():
 p=v.c.read(D/'execution_plan_v3.json');assert p['controller_sha256']==v.c.sha(Path(__file__));assert p['adapter_sha256']==v.c.sha(Path(__file__).with_name('merge_adapter_v3.py'));v.validate();assert v.c.read(D/'merge_preflight_v3.json')['status'].startswith('both_unchanged')
 lock=(D/'controller_v3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller_v3.json').exists();who=v.c.identity();v.put(D/'controller_v3.json',who);prior=v.c.read(D/'preserved_original_serial_score/preservation.json');owner=prior['owner'];oldfast=v.c.read(D/'controller_v2.json');inherited=[v.c.read(D/f'score_{i}_child.json')for i in [0,1]];children=[];complete=False
 try:
  v.put(D/'state_v3.json',dict(status='two_original_score_shards_continuing_no_recalculation',controller=who))
  exits=[]
  for part,w in enumerate(inherited):
   while True:
    s=status(w)
    assert s is not None,'Parent was STOP so child zombie exit status must remain readable'
    if s['state']=='Z':break
    time.sleep(2)
   code=os.waitstatus_to_exitcode(s['raw_exit_status_field52']);rec=dict(code=code,worker=w,source='actual Linux proc_pid_stat field52 encoded exit status of unreaped owned worker, not a fabricated external Python wait return',raw_stat=s['raw_stat']);v.put(D/f'score_{part}_exit_v3.json',rec);assert code==0;exits.append(rec)
  kill_owned(oldfast)
  for _ in range(100):
   if not v.live(oldfast):break
   time.sleep(.05)
  assert not v.live(oldfast);v.put(D/'previous_controller_v2_intentional_exit.json',dict(worker=oldfast,actual_exit_verified=True,reason='Exact suffix folder binding adapter; both actual score shards retained, no new score/GT loop.',external_OS_wait_return_unavailable=True))
  for stage in ['score','audit']:
   if stage=='audit':
    v.put(D/'state_v3.json',dict(status='audit_two_independent_disjoint_CPU_processes',controller=who));v.put(D/'audit_start.json',dict(monotonic_start=time.perf_counter(),wall_time=time.time()));children=[]
    for part,cpu in [(0,14),(1,15)]:
     log=(D/f'audit_{part}.log').open('xb');ch=subprocess.Popen(['taskset','-c',str(cpu),'nice','-n','19','ionice','-c','3',sys.executable,'-B',str(Path(v.__file__)),'shard','audit',str(part)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});v.put(D/f'audit_{part}_child.json',v.c.identity(ch.pid));children.append((part,ch,log))
    for part,ch,log in children:
     code=ch.wait();log.close();v.put(D/f'audit_{part}_exit.json',dict(code=code));assert code==0
   v.put(D/'state_v3.json',dict(status=stage+'_original_whole_domain_report_merge',controller=who))
   with(D/f'{stage}_merge_v3.log').open('xb')as log:
    ch=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(Path(__file__).with_name('merge_adapter_v3.py')),stage],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});v.put(D/f'{stage}_merge_child_v3.json',v.c.identity(ch.pid));code=ch.wait()
   v.put(D/f'{stage}_merge_exit_v3.json',dict(code=code));assert code==0
  assert v.c.read(v.c.D/'vidstg/reporting_receipt.json')['all_arrays_cold_rederived_exact'];originals=[prior['owner'],prior['child']];assert all(v.live(x)for x in originals);os.killpg(owner['pid'],signal.SIGTERM);os.killpg(owner['pid'],signal.SIGCONT)
  for _ in range(200):
   if not any(v.live(x)for x in originals):break
   time.sleep(.05)
  assert not any(v.live(x)for x in originals);complete=True
  v.put(D/'original_owned_exit.json',dict(status='intentional_exit_after_parallel_full_score_and_independent_cold_audit',original_processes=originals,actual_owned_exit_verified=True,OS_wait_return_code_unavailable=True,old_partial_uncheckpointed_scalar_work_not_claimed_reused=True,old_completed_query_lower_bound=prior['completed_lower_bound'],original_CPU_cost_preserved=True))
  v.put(D/'state_v3.json',dict(status='Vid_whole_official_scored_and_allarray_audited_pending_root_semantic_cost_image_publication',controller=who))
 except BaseException:v.put(D/'failure_v3.json',dict(error=traceback.format_exc(),worker=who));raise
 finally:
  for _,ch,log in children:
   if ch.poll()is None:
    ch.terminate()
    try:ch.wait(timeout=10)
    except subprocess.TimeoutExpired:ch.kill();ch.wait()
   if not log.closed:log.close()
  if not complete:
   for w in inherited:
    if v.live(w):kill_owned(w)
   if v.live(oldfast):kill_owned(oldfast)
   if v.live(owner):os.killpg(owner['pid'],signal.SIGCONT);v.put(D/'original_serial_CPU_resume_after_failure_v3.json',dict(**owner,SIGCONT_sent=True,original_RAM_preserved=True))
if __name__=='__main__':run()
