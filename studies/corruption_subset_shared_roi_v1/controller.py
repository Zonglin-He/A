"""Single new owner; never resumes any historical full-corruption dispatcher."""
from subset_common import *
import subprocess,traceback
def run():
 validate();assert read(D/'preflight_receipt.json')['status']=='passed_real_pixels_cached_selected_math_no_model_GT';assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who);atomic(D/'launch.json',dict(controller=who,entry=str(Path(__file__)),plan_sha256=sha(D/'plan.json')))
 units=[(ds,c['id'])for c in conditions()for ds in COUNTS];done=[]
 try:
  for ds,name in units:
   root=unit(ds,name)
   for stage in ['predict','score','audit']:
    resources(time.perf_counter());args=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/'worker.py'),'--dataset',ds,'--condition',name]if stage=='predict'else[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/'report.py'),stage,'--dataset',ds,'--condition',name]
    root.mkdir(parents=True,exist_ok=True);env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    if stage!='predict':env['CUDA_VISIBLE_DEVICES']=''
    log=(root/(stage+'.log')).open('wb');child=subprocess.Popen(['taskset','-c','14,15','nice','-n','19','ionice','-c','3',*args],cwd=R,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);time.sleep(.08);ci=identity(child.pid);atomic(root/(stage+'_child.json'),ci);atomic(D/'state.json',dict(status='running',stage=stage,dataset=ds,condition=name,child=ci,completed_units=done,controller=who));code=child.wait();log.close();atomic(root/(stage+'_exit.json'),dict(child=ci,code=code));assert code==0,f'{ds}/{name}/{stage} failed preserved'
   done.append(dict(dataset=ds,condition=name,reporting_receipt_sha256=sha(root/'reporting_receipt.json')));atomic(D/'completed_units.json',dict(units=done,total=36,root_analysis_publication_still_required=True))
  atomic(D/'state.json',dict(status='all36_whole128_units_scored_audited_pending_root_analysis_figures_publication',completed_units=done,controller=who))
 except BaseException:atomic(D/'state.json',dict(status='failed_preserved',error=traceback.format_exc(),completed_units=done,controller=who));raise
if __name__=='__main__':run()
