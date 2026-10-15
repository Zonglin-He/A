"""Finite per-domain coordinate search; score only after each complete stage seal."""
import sys,os,time,subprocess,threading,math,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts import launch_stvg_instance_identity_v1 as resource
BASE=ROOT/'artifacts/stvg_weightfree_coordinate_v1'
UNIT='stvg-weightfree-coordinate-001.service';GiB=1024**3
PY=ROOT/'.conda/tubedetr/bin/python'

def main():
 plan=read(BASE/'DESIGN_LOCK.json');rt=read(BASE/'RUNTIME.json')
 for p,h in rt['pins'].items():assert sha(ROOT/p)==h,p
 sample=resource.sample()
 assert sample['available']>=8*GiB and sample['gpu_free']>=26*GiB and sample['disk_free']>=30*GiB
 props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MemoryHigh,MemoryMax,MemorySwapMax'],text=True).splitlines())
 assert [int(props[k])for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*GiB,20*GiB,GiB]
 write(BASE/'ACTUAL_LIMITS.json',dict(status='pass',properties=props,admission=sample))
 resource.BASE=BASE;resource.UNIT=UNIT;resource.STAGE='coordinate';stop=threading.Event()
 threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
 current=plan['initial_configs'];prior=['artifacts/stvg_weightfree_tuning_v1/confirmation003'];actual=0
 original={'hc2':dict(lr=.01,sigma=.025,tau=.05),'vidstg':dict(lr=.03,sigma=.1,tau=.25)}
 for step in plan['stages']:
  stage=BASE/step['name'];stage.mkdir(exist_ok=False);key=step['parameter'];recipes={};incumbent={}
  for ds,cfg in current.items():
   vals=step.get('values')
   if vals is None:
    center=cfg[key] if step.get('relative_to_incumbent') else original[ds][key]
    vals=[center*x for x in step['multipliers']]
   vals=sorted(set(vals+[cfg[key]]));recipes[ds]={}
   for i,v in enumerate(vals):
    name=f'v{i:02}';recipes[ds][name]=dict(cfg,**{key:v})
    if v==cfg[key]:incumbent[ds]=name
  write(stage/'STAGE_LOCK.json',dict(status='locked',parameter=key,recipes=recipes,incumbent=incumbent,reuse_directories=prior,previous_selection_sha256=sha(BASE/plan['stages'][plan['stages'].index(step)-1]['name']/'SELECTION_LOCK.json') if len(prior)>1 else sha(BASE/'DESIGN_LOCK.json'),GT_read=False,development_selection_uses_labels=True))
  write(stage/'STUDENT_RUNTIME.json',rt)
  for label,script in [('student','run_stvg_weightfree_coordinate_student_v1.py'),('root','review_stvg_weightfree_coordinate_v1.py'),('score','score_stvg_weightfree_coordinate_v1.py')]:
   with (stage/(label+'.log')).open('xb')as log:
    child=subprocess.Popen([str(PY),'-B',str(ROOT/'scripts'/script),step['name']],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
    write(stage/(label.upper()+'_LAUNCH.json'),dict(status='running',pid=child.pid,time=time.time(),GT_read=label=='score'))
    code=child.wait();write(stage/(label.upper()+'_WORKER_EXIT.json'),dict(status='exited',exit_code=code,time=time.time()))
    if code:raise RuntimeError(f'{step["name"]}/{label} exited {code}')
  summaries=read(stage/'POSTSEAL_SUMMARY.json')['summary'];chosen={};audit=[]
  for ds,rs in recipes.items():
   vals={name:next(x['mean']for x in summaries if x['dataset']==ds and x['arm']=='T_'+name and x['metric']=='m_vIoU') for name in rs}
   best=max(vals.values());ties=[n for n in rs if abs(vals[n]-best)<=1e-12]
   win=incumbent[ds] if incumbent[ds]in ties else min(ties,key=lambda n:rs[n][key]);chosen[ds]=win;current[ds]=rs[win]
   audit.append(dict(dataset=ds,parameter=key,scores=vals,selected=win,config=current[ds],boundary_winner=rs[win][key]in [min(c[key]for c in rs.values()),max(c[key]for c in rs.values())]))
  write(stage/'SELECTION_AUDIT.json',dict(status='complete',records=audit,GT_read=True,development_only=True))
  write(stage/'SELECTION_LOCK.json',dict(status='locked',configs=current,selected=chosen,GT_read=False,selected_using_development_GT=True,audit_sha256=sha(stage/'SELECTION_AUDIT.json')))
  actual+=sum(x['actual_fit_count']for x in read(stage/'STUDENT_GLOBAL_BARRIER.json')['files']);assert actual<=plan['spatial_new_fit_cap']
  prior.append(str(stage.relative_to(ROOT)))
  write(BASE/'STATUS.json',dict(status='running',stage=step['name'],done=len(prior)-1,total=len(plan['stages']),actual_fit_count=actual),mutable=True)
  print(step['name'],audit,'actual new fits',actual,flush=True)
 write(BASE/'SPATIAL_SELECTION_LOCK.json',dict(status='locked',configs=current,GT_read=False,selected_using_development_GT=True,last_selection_sha256=sha(stage/'SELECTION_LOCK.json'),confirmation_used_for_selection=False))
 write(BASE/'SPATIAL_CPU_COMPLETION.json',dict(status='complete',actual_fit_count=actual,stages=len(plan['stages']),scope='32 exposed development sources; no confirmation result yet'))
 stop.set()

if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));raise
