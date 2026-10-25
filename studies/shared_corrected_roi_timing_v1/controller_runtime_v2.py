"""Wait audited clean references, pause only owned CPU sync, actual timing, finally resume."""
from timing_runtime_v2 import *
import subprocess,os,signal,traceback
execution_validate();assert not(D/'controller_v2.json').exists();who=identity();atomic(D/'controller_v2.json',who);resumed=False;groups=[]
try:
 upstream=read(selected.D/'controller.json');atomic(D/'state.json',dict(status='waiting_selected_clean_whole_audits_and_owned_exit',controller=who,corruption_release_allowed=False))
 while not exited(upstream):time.sleep(10)
 gate={}
 for ds in COUNTS:
  assert read(selected.D/f'components_{ds}_domain_exit.json')['code']==0;ap=selected.D/f'components_{ds}_reporting_receipt.json';audit=read(ap);assert audit['status']=='verified_whole_domain'and audit['all_arrays_cold_rederived_exact']and audit['all_means_recomputed'];gate[ds]=dict(path=str(ap.relative_to(R)),sha256=sha(ap));assert read(selected.D/f'components_{ds}_score_receipt.json')['seal_sha256']==sha(selected.D/f'components_{ds}_sealed.json')
 atomic(D/'clean_dependency_gate.json',gate)
 w=read(R/'outputs/shared_corrected_roi_sync_v1/controller.json')
 if not exited(w):
  assert os.getpgid(w['pid'])==w['pid'];members=[]
  for path in Path('/proc').iterdir():
   if not path.name.isdecimal():continue
   try:
    z=(path/'stat').read_text().rsplit(')',1)[1].split()
    if int(z[2])!=w['pid']:continue
    cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'studies/shared_corrected_roi_sync_v1/'in cmd,cmd;members.append(dict(pid=int(path.name),startticks=int(z[19]),command=cmd))
   except(FileNotFoundError,ProcessLookupError):continue
  groups=[dict(controller=w,members=members)];atomic(D/'deployment_CPU_pause.json',dict(groups=groups,reason='Matched actual uncached timing; only owned CPU sync group'));os.killpg(w['pid'],signal.SIGSTOP)
 else:atomic(D/'deployment_CPU_pause.json',dict(groups=[]))
 atomic(D/'state.json',dict(status='actual_uncached_timing_running',controller=who,corruption_release_allowed=False));log=(D/'worker.log').open('ab');env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'};child=subprocess.Popen([sys.executable,'-B',str(S/'timing_runtime_v2.py'),'run'],stdout=log,stderr=subprocess.STDOUT,env=env);atomic(D/'timing_child.json',identity(child.pid));rc=child.wait();log.close();atomic(D/'timing_exit.json',dict(code=rc));assert rc==0,'Preserve actual measurement failure, no retry'
 atomic(D/'state.json',dict(status='measured_all192_verified_pending_root_export',controller=who,corruption_release_allowed=False))
except BaseException:
 atomic(D/'state.json',dict(status='failed_preserved',error=traceback.format_exc(),controller=who,corruption_release_allowed=False));raise
finally:
 for g in groups:
  w=g['controller']
  if not exited(w):assert os.getpgid(w['pid'])==w['pid'];os.killpg(w['pid'],signal.SIGCONT)
 atomic(D/'deployment_CPU_resume.json',dict(groups=groups,finally_resumed=True,time=time.time()))
