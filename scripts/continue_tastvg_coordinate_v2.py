"""Finite ordered coarse/refined search, with sealed stage decisions and confirmations."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_coordinate_common_v2 import *
from scripts.prepare_tastvg_coordinate_tuning_v2 import refine

def stage(dataset,name,script,args,receipt,python='.conda/tubedetr/bin/python',numerical=False):
 if receipt and Path(receipt).exists():return 0
 verify(dataset);budget();out=BASE/dataset
 with (out/(name+'.log')).open('a') as log:
  env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
  if name=='spatial':env['PYTHONPATH']=str(ROOT/'.runtime/sa2va_deps')+os.pathsep+env.get('PYTHONPATH','')
  proc=subprocess.Popen(['bash','scripts/with_local_cuda.sh',str(ROOT/python),'-B',script,*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status(BASE/'STATUS.json',dict(status='running',dataset=dataset,stage=name,controller_pid=os.getpid(),worker_pid=proc.pid,time=time.time()));code=proc.wait()
 if numerical and code==42:return 42
 if code or (receipt and not Path(receipt).exists()):raise RuntimeError(f'{dataset}/{name}: exit{code}; missing{receipt}')
 return code

def choose(trials):
 valid=[r for r in trials if r['state']=='COMPLETE'];assert valid,'No valid configuration'
 return min(valid,key=lambda r:(-r['objective'],r['number']))

def search(dataset):
 out=BASE/dataset;trials=read(out/'TRIALS.json') if (out/'TRIALS.json').exists() else [];grids=read(BASE/'GRIDS.json')['coarse']
 def evaluate(tag,stage_name,params):
  existing=[r for r in trials if r['tag']==tag]
  if existing:assert existing[0]['params']==params;return existing[0]
  td=out/'search'/tag;request=td/'REQUEST.json';req=dict(tag=tag,stage=stage_name,split='search',params=params)
  if request.exists():assert read(request)==req
  else:write(request,req)
  prior=[r for r in trials if r['state']=='COMPLETE' and r['params']==params]
  if prior:
   other=prior[0];reuse=dict(source=other['result_dir'],score_sha256=sha(out/other['result_dir']/'SCORE.json'),same_config=True)
   if (td/'REUSE.json').exists():assert read(td/'REUSE.json')==reuse
   else:write(td/'REUSE.json',reuse)
   record=dict(number=len(trials),tag=tag,stage=stage_name,params=params,state='COMPLETE',objective=other['objective'],result_dir=other['result_dir'],reused_from=other['tag'])
  else:
   numerical=(td/'STATUS.json').exists() and read(td/'STATUS.json')['status']=='numerical_failure'
   code=42 if numerical else stage(dataset,tag,'scripts/run_tastvg_coordinate_trial_v2.py',[dataset,request],td/'PREDICTION_BARRIER.json',numerical=True)
   if code==42:record=dict(number=len(trials),tag=tag,stage=stage_name,params=params,state='FAIL',objective=None,result_dir=str(td.relative_to(out)))
   else:
    stage(dataset,tag+'_score','scripts/score_tastvg_coordinate_v2.py',[dataset,request],td/'SCORE.json');score=read(td/'SCORE.json');record=dict(number=len(trials),tag=tag,stage=stage_name,params=params,state='COMPLETE',objective=score['objective'],result_dir=str(td.relative_to(out)))
  trials.append(record);status(out/'TRIALS.json',trials);assert len(trials)<=72
  status(out/'SEARCH_STATUS.json',dict(status='running',stage=stage_name,scheduled=len(trials),completed=sum(r['state']=='COMPLETE' for r in trials),failed=sum(r['state']=='FAIL' for r in trials),last_tag=tag,max_scheduled=72,time=time.time()))
  print('COORDINATE',dataset,tag,record['state'],record['objective'],flush=True);return record
 selected={}
 for letter,key in [('A','lr'),('B','teacher_temperature')]:
  fixed=DEFAULT if letter=='A' else {**DEFAULT,'lr':selected['A']['params']['lr']}
  coarse=[evaluate(f'{letter}_coarse_{i:03}',letter,{**fixed,key:value}) for i,value in enumerate(grids[key])]
  if letter=='A':assert coarse[0]['state']=='COMPLETE','Default control must complete'
  winner=choose(coarse);rf=out/f'{letter}_REFINEMENT.json';r=dict(parameter=key,fixed=fixed,best_coarse=winner['tag'],best_value=winner['params'][key],values=refine(grids[key],winner['params'][key]))
  if rf.exists():assert read(rf)==r
  else:write(rf,r)
  for i,value in enumerate(r['values']):evaluate(f'{letter}_fine_{i:03}',letter,{**fixed,key:value})
  best=choose([x for x in trials if x['stage']==letter]);seal=out/f'{letter}_SELECTION.json'
  decision=dict(stage=letter,parameter=key,params=best['params'],tag=best['tag'],objective=best['objective'],result_dir=best['result_dir'],range_boundary=best['params'][key] in [min(grids[key]),max(grids[key])],confirmation_used=False,time=time.time())
  if seal.exists():assert {k:v for k,v in read(seal).items() if k!='time'}=={k:v for k,v in decision.items() if k!='time'}
  else:write(seal,decision)
  selected[letter]=read(seal);archive(dataset+' '+letter+'阶段完成并封存，参数'+str(best['params']))
 final=out/'SELECTION.json'
 if not final.exists():write(final,dict(status='selected_before_confirmation',params=selected['B']['params'],objective=selected['B']['objective'],stages=selected,objective_metric='corrupt_all_source_macro_dense_delta_vIoU',confirmation_used=False,time=time.time()))
 status(out/'SEARCH_STATUS.json',dict(status='completed',scheduled=len(trials),completed=sum(x['state']=='COMPLETE' for x in trials),failed=sum(x['state']=='FAIL' for x in trials),params=selected['B']['params'],time=time.time()))
 configs=[('default',DEFAULT),('lr_only',selected['A']['params']),('selected',selected['B']['params'])];finished=[]
 for name,params in configs:
  td=out/'confirmation'/name;req=td/'REQUEST.json';r=dict(tag='confirmation_'+name,split='confirm',params=params)
  if req.exists():assert read(req)==r
  else:write(req,r)
  previous=[n for n,c in finished if c==params]
  if previous:
   src=out/'confirmation'/previous[0];receipt='SCORE.json' if (src/'SCORE.json').exists() else 'NUMERICAL_FAILURE.json'
   if not (td/'REUSE.json').exists():write(td/'REUSE.json',dict(source=str(src.relative_to(out)),receipt=receipt,sha256=sha(src/receipt)))
   continue
  if not (td/'NUMERICAL_FAILURE.json').exists():
   numerical=(td/'STATUS.json').exists() and read(td/'STATUS.json')['status']=='numerical_failure'
   code=42 if numerical else stage(dataset,r['tag'],'scripts/run_tastvg_coordinate_trial_v2.py',[dataset,req],td/'PREDICTION_BARRIER.json',numerical=True)
   if code==42:write(td/'NUMERICAL_FAILURE.json',dict(status='numerical_failure',params=params,no_reselection=True,time=time.time()))
   else:stage(dataset,r['tag']+'_score','scripts/score_tastvg_coordinate_v2.py',[dataset,req],td/'SCORE.json')
  finished.append((name,params))
 stage(dataset,'report','scripts/report_tastvg_coordinate_v2.py',[dataset],out/'COMPLETION.json');archive(dataset+'顺序搜索和固定复验已执行完成，待根审计与公开')

def run():
 handle=(BASE/'PROCESS.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 for dataset in DATASETS:
  out=BASE/dataset
  if (out/'COMPLETION.json').exists():continue
  stage(dataset,'spatial','scripts/run_tastvg_coordinate_experts_v2.py',[dataset,'spatial'],out/'experts/SPATIAL_BARRIER.json')
  stage(dataset,'temporal','scripts/run_tastvg_coordinate_experts_v2.py',[dataset,'temporal'],out/'experts/TEMPORAL_BARRIER.json',python='.venv-exost/bin/python')
  stage(dataset,'capture','scripts/capture_tastvg_coordinate_v2.py',[dataset],out/'CAPTURE_BARRIER.json')
  search(dataset)
 status(BASE/'STATUS.json',dict(status='completed_pending_root_audit_publication',time=time.time()));archive('两数据集顺序调参和复验执行完成，待根审计与公开核验')
if __name__=='__main__':
 try:run()
 except BaseException as e:
  status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));archive('工程中断已保存 '+repr(e));raise
