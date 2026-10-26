"""One low-priority CPU consumer; existing external GPU priority untouched."""
from common import *
import subprocess,traceback,fcntl
def run():
 validate();D.mkdir(parents=True,exist_ok=True);lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller_v1.json').exists();who=identity();atomic(D/'controller_v1.json',who)
 try:
  for ds in COUNTS:
   for stage in ['predict','score','audit']:
    atomic(D/'state_v1.json',dict(status=stage,dataset=ds,controller=who));log=(D/(ds+'_'+stage+'.log')).open('xb');c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'run_v1.py'),stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});atomic(D/(ds+'_'+stage+'_child.json'),identity(c.pid));code=c.wait();log.close();atomic(D/(ds+'_'+stage+'_exit.json'),dict(code=code));assert code==0,'Preserve failure; no automatic retry'
  atomic(D/'state_v1.json',dict(status='both_whole_domains_scored_audited_pending_root_analysis_publication',controller=who))
 except BaseException:atomic(D/'state_v1.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':run()
