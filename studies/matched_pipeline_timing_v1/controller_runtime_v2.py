"""One finite owner; no old queue is resumed."""
from runtime_v2_common import *
import subprocess,sys,traceback
def run():
 validate_runtime();assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who);atomic(D/'launch.json',dict(controller=who,entry=str(Path(__file__)),plan_sha256=sha(D/'plan.json')))
 try:
  for stage,file in [('GPU','worker_runtime_v2.py'),('CPU_report','report_runtime_v2.py')]:
   env={**os.environ,'PYTHONPATH':str(R),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4'if stage=='GPU'else'1','MKL_NUM_THREADS':'4'if stage=='GPU'else'1','OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
   if stage!='GPU':env['CUDA_VISIBLE_DEVICES']=''
   log=(D/(stage+'.log')).open('wb');args=['taskset','-c','14,15','nice','-n','19','ionice','-c','3',str(R/'outputs/edgetam_comparison_v1/runtime/bin/python'),'-B',str(S/file)];p=subprocess.Popen(args,cwd=R,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);time.sleep(.1);child=identity(p.pid);atomic(D/(stage+'_child.json'),child);atomic(D/'controller_state.json',dict(status='running',stage=stage,child=child,controller=who));code=p.wait();log.close();atomic(D/(stage+'_exit.json'),dict(child=child,code=code));assert code==0,f'{stage} failed; preserve no automatic retry'
  atomic(D/'controller_state.json',dict(status='all1536_complete_independent_report_verified_pending_root_analysis_publication',controller=who))
 except BaseException:atomic(D/'controller_state.json',dict(status='failed_preserved',error=traceback.format_exc(),controller=who));raise
if __name__=='__main__':run()
