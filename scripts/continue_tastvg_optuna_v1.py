"""Finite serial two-dataset search; bounded trials, resumable storage, no pruning."""
import sys,os,time,subprocess,fcntl,traceback,pickle,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_optuna_common_v1 import *

def stage(dataset,name,script,args,receipt,python='.conda/tubedetr/bin/python',numerical=False):
 if receipt and Path(receipt).exists():return 0
 verify(dataset);budget();out=BASE/dataset;status(BASE/'STATUS.json',dict(status='running',dataset=dataset,stage=name,controller_pid=os.getpid(),time=time.time()))
 with (out/(name+'.log')).open('a') as log:
  env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
  if name=='spatial':env['PYTHONPATH']=str(ROOT/'.runtime/sa2va_deps')+os.pathsep+env.get('PYTHONPATH','')
  proc=subprocess.Popen(['bash','scripts/with_local_cuda.sh',str(ROOT/python),'-B',script,*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status(BASE/'STATUS.json',dict(status='running',dataset=dataset,stage=name,controller_pid=os.getpid(),worker_pid=proc.pid,time=time.time()));code=proc.wait()
 if numerical and code==42:return code
 if code or (receipt and not Path(receipt).exists()):raise RuntimeError(f'{dataset}/{name} exit={code}, missing receipt={receipt}')
 return code

def sampler_save(path,sampler):
 tmp=path.with_suffix('.tmp');tmp.write_bytes(pickle.dumps(sampler));os.replace(tmp,path)

def search(dataset):
 import optuna
 assert optuna.__version__=='4.5.0';out=BASE/dataset;p=verify(dataset);samplerfile=out/'SAMPLER.pkl'
 sampler=pickle.loads(samplerfile.read_bytes()) if samplerfile.exists() else optuna.samplers.TPESampler(seed=20260930,n_startup_trials=16,multivariate=False)
 study=optuna.create_study(study_name='tastvg_'+dataset+'_optuna_v1',storage='sqlite:///'+str(out/'study.sqlite3'),load_if_exists=True,direction='maximize',sampler=sampler,pruner=optuna.pruners.NopPruner())
 if not study.trials:
  anchors=[DEFAULT]
  for k,values in SPACE.items():
   for v in values:anchors.append({**DEFAULT,k:v})
  for values in itertools.product(*SPACE.values()):anchors.append(dict(zip(SPACE,values)))
  for config in anchors:study.enqueue_trial(config)
  sampler_save(samplerfile,study.sampler)
 State=optuna.trial.TrialState
 while True:
  alltrials=study.get_trials(deepcopy=False);attempted=[x for x in alltrials if x.state!=State.WAITING]
  running=[x for x in attempted if x.state==State.RUNNING];assert len(running)<=1
  if running:
   trial=optuna.trial.Trial(study,running[0]._trial_id)
  elif len(attempted)>=48:break
  else:trial=study.ask()
  cfg={k:trial.suggest_float(k,*v,log=True) for k,v in SPACE.items()};sampler_save(samplerfile,study.sampler);td=out/'trials'/f'{trial.number:05}';request=td/'REQUEST.json'
  r=dict(tag=f'trial{trial.number:05}',trial_number=trial.number,split='search',params=cfg)
  if request.exists():assert read(request)==r
  else:write(request,r)
  code=stage(dataset,r['tag'],'scripts/run_tastvg_optuna_trial_v1.py',[dataset,request],td/'PREDICTION_BARRIER.json',numerical=True)
  if code==42:
   trial.set_user_attr('failure','numerical_failure_preserved');study.tell(trial,state=State.FAIL)
  else:
   stage(dataset,r['tag']+'_score','scripts/score_tastvg_optuna_v1.py',[dataset,request],td/'SCORE.json');score=read(td/'SCORE.json');trial.set_user_attr('score_sha256',sha(td/'SCORE.json'));study.tell(trial,score['objective'])
  sampler_save(samplerfile,study.sampler)
  trials=study.get_trials(deepcopy=False);completed=[x for x in trials if x.state==State.COMPLETE];failed=[x for x in trials if x.state==State.FAIL]
  status(out/'SEARCH_STATUS.json',dict(status='running',attempted=len(completed)+len(failed),completed=len(completed),failed=len(failed),total=48,best_trial=study.best_trial.number if completed else None,best_value=study.best_value if completed else None,time=time.time()))
  status(out/'TRIALS.json',[dict(number=x.number,state=x.state.name,params=x.params,value=x.value,user_attrs=x.user_attrs) for x in trials]);print('SEARCH',dataset,len(completed)+len(failed),48,flush=True)
 completed=[x for x in study.trials if x.state==State.COMPLETE];assert completed and study.trials[0].state==State.COMPLETE,'Default control must complete'
 best=min(completed,key=lambda x:(-x.value,x.number));selection=out/'SELECTION.json'
 if not selection.exists():write(selection,dict(status='selected_before_confirmation',trial=best.number,params=best.params,objective=best.value,default_objective=study.trials[0].value,selection_rule='max search source-macro corrupt nonexpert delta vIoU; earliest exact tie',confirmation_used=False,time=time.time()))
 else:assert read(selection)['trial']==best.number
 status(out/'SEARCH_STATUS.json',dict(status='completed',attempted=48,completed=len(completed),failed=sum(x.state==State.FAIL for x in study.trials),best_trial=best.number,best_value=best.value,time=time.time()))
 archive(dataset+'48次搜索尝试结束，配置已封存；16源复验尚待执行')
 for name,config in [('default',DEFAULT),('selected',best.params)]:
  target=out/'confirmation'/name;request=target/'REQUEST.json'
  if name=='selected' and config==DEFAULT:
   if not (target/'REUSE.json').exists():write(target/'REUSE.json',dict(identical_to='../default',score_sha256=sha(out/'confirmation/default/SCORE.json')))
   continue
  r=dict(tag='confirmation_'+name,split='confirm',params=config)
  if not request.exists():write(request,r)
  code=stage(dataset,r['tag'],'scripts/run_tastvg_optuna_trial_v1.py',[dataset,request],target/'PREDICTION_BARRIER.json',numerical=True)
  if code==42:
   write(target/'NUMERICAL_FAILURE.json',dict(status='numerical_failure',no_reselection=True,params=config,time=time.time()));continue
  stage(dataset,r['tag']+'_score','scripts/score_tastvg_optuna_v1.py',[dataset,request],target/'SCORE.json')
 stage(dataset,'report','scripts/report_tastvg_optuna_v1.py',[dataset],out/'COMPLETION.json')
 archive(dataset+'搜索及固定配置复验完成，根核验与公开同步待执行')

def run():
 lock=(BASE/'PROCESS.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 for dataset in DATASETS:
  out=BASE/dataset
  if (out/'COMPLETION.json').exists():continue
  stage(dataset,'spatial','scripts/run_tastvg_optuna_experts_v1.py',[dataset,'spatial'],out/'experts/SPATIAL_BARRIER.json')
  stage(dataset,'temporal','scripts/run_tastvg_optuna_experts_v1.py',[dataset,'temporal'],out/'experts/TEMPORAL_BARRIER.json',python='.venv-exost/bin/python')
  stage(dataset,'capture','scripts/capture_tastvg_optuna_v1.py',[dataset],out/'CAPTURE_BARRIER.json')
  search(dataset)
 status(BASE/'STATUS.json',dict(status='completed_pending_root_review_and_publication',remaining=['root audit','public GitHub synchronization'],time=time.time()));archive('两数据集搜索与复验执行完成，根复核/公开同步待执行')
if __name__=='__main__':
 try:run()
 except BaseException as e:
  status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));archive('工程中断已保留：'+repr(e));raise
