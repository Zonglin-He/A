"""Finite ordered coarse/refined search, with sealed stage decisions and confirmations."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_extended_common_v3 import *
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
 out=BASE/dataset;p=verify(dataset);trials=read(out/'TRIALS.json') if (out/'TRIALS.json').exists() else [];initial=anchor(p)
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
   code=42 if numerical else stage(dataset,tag,'scripts/run_tastvg_extended_trial_v3.py',[dataset,request],td/'PREDICTION_BARRIER.json',numerical=True)
   if code==42:record=dict(number=len(trials),tag=tag,stage=stage_name,params=params,state='FAIL',objective=None,result_dir=str(td.relative_to(out)))
   else:
    stage(dataset,tag+'_score','scripts/score_tastvg_extended_v3.py',[dataset,request],td/'SCORE.json');score=read(td/'SCORE.json');record=dict(number=len(trials),tag=tag,stage=stage_name,params=params,state='COMPLETE',objective=score['objective'],result_dir=str(td.relative_to(out)))
  trials.append(record);status(out/'TRIALS.json',trials);assert len(trials)<=100
  status(out/'SEARCH_STATUS.json',dict(status='running',stage=stage_name,scheduled=len(trials),completed=sum(r['state']=='COMPLETE' for r in trials),failed=sum(r['state']=='FAIL' for r in trials),last_tag=tag,max_scheduled=100,time=time.time()))
  print('COORDINATE',dataset,tag,record['state'],record['objective'],flush=True);return record
 screen=read(BASE/'PLAN.json')['screen']
 origin=evaluate('screen_anchor','screen',initial);assert origin['state']=='COMPLETE'
 coordinates=['rho','steps','student_temperature','direction_count']
 for key in coordinates:
  for i,value in enumerate(screen[key]):
   if value!=initial[key]:evaluate(f'screen_{key}_{i:03}','screen',{**initial,key:value})
 assert len([x for x in trials if x['stage']=='screen'])==24
 winners=[]
 for index,key in enumerate(coordinates):
  candidates=[x for x in trials if x['stage']=='screen' and (x['tag'].startswith('screen_'+key+'_') or x['tag']=='screen_anchor')]
  best=choose(candidates)
  if best['objective']>origin['objective']:winners.append(dict(key=key,improvement=best['objective']-origin['objective'],winner=best,index=index))
 winners=sorted(winners,key=lambda x:(-x['improvement'],x['index']))[:2]
 allocation=dict(coordinates=winners,anchor_tag=origin['tag'],anchor_objective=origin['objective'],confirmation_used=False)
 af=out/'FOCUS_ALLOCATION.json'
 if af.exists():assert read(af)==allocation
 else:write(af,allocation)
 current=initial;decisions=[]
 import math
 for j,item in enumerate(winners):
  key=item['key'];letter=f'focus{j+1}';fixed=dict(current)
  if key in ['rho','student_temperature']:
   lo,hi=(1e-5,2.) if key=='rho' else (.01,100.)
   values=sorted(set([float(f'{10**(math.log10(lo)+(math.log10(hi)-math.log10(lo))*i/20):.14g}') for i in range(21)]+[current[key],item['winner']['params'][key]]))
  else:values=sorted(set(([1,2,3,4,5,6,8,10] if key=='steps' else [1,2,3,4,6,8,12,16])+[current[key]]))
  grid=dict(parameter=key,fixed=fixed,values=values)
  gf=out/f'{letter}_GRID.json'
  if gf.exists():assert read(gf)==grid
  else:write(gf,grid)
  coarse=[evaluate(f'{letter}_coarse_{i:03}',letter,{**fixed,key:v}) for i,v in enumerate(values)]
  best=choose(coarse)
  fine=refine(values,best['params'][key]) if key in ['rho','student_temperature'] else []
  rf=out/f'{letter}_REFINEMENT.json';r=dict(parameter=key,fixed=fixed,best_coarse=best['tag'],values=fine)
  if rf.exists():assert read(rf)==r
  else:write(rf,r)
  for i,v in enumerate(fine):evaluate(f'{letter}_fine_{i:03}',letter,{**fixed,key:v})
  best=choose([x for x in trials if x['stage']==letter]);seal=out/f'{letter}_SELECTION.json'
  decision=dict(parameter=key,params=best['params'],tag=best['tag'],objective=best['objective'],result_dir=best['result_dir'],confirmation_used=False)
  if seal.exists():assert {k:v for k,v in read(seal).items() if k!='time'}==decision
  else:write(seal,{**decision,'time':time.time()})
  decisions.append(read(seal));current=best['params'];archive(dataset+' '+letter+'已完成封存')
 final=out/'SELECTION.json'
 if not final.exists():write(final,dict(status='selected_before_confirmation',params=current,decisions=decisions,confirmation_used=False,time=time.time()))
 status(out/'SEARCH_STATUS.json',dict(status='completed',scheduled=len(trials),completed=sum(x['state']=='COMPLETE' for x in trials),failed=sum(x['state']=='FAIL' for x in trials),time=time.time()))
 done=[]
 for name,params in [('anchor',initial),('selected',current)]:
  td=out/'confirmation'/name;req=td/'REQUEST.json';r=dict(tag='confirmation_'+name,split='confirm',params=params)
  if req.exists():assert read(req)==r
  else:write(req,r)
  same=[n for n,c in done if c==params]
  if same:
   src=out/'confirmation'/same[0];receipt='SCORE.json' if (src/'SCORE.json').exists() else 'NUMERICAL_FAILURE.json'
   if not (td/'REUSE.json').exists():write(td/'REUSE.json',dict(source=str(src.relative_to(out)),receipt=receipt,sha256=sha(src/receipt)))
   continue
  if not (td/'NUMERICAL_FAILURE.json').exists():
   numerical=(td/'STATUS.json').exists() and read(td/'STATUS.json')['status']=='numerical_failure'
   code=42 if numerical else stage(dataset,r['tag'],'scripts/run_tastvg_extended_trial_v3.py',[dataset,req],td/'PREDICTION_BARRIER.json',numerical=True)
   if code==42:write(td/'NUMERICAL_FAILURE.json',dict(status='numerical_failure',params=params,no_reselection=True,time=time.time()))
   else:stage(dataset,r['tag']+'_score','scripts/score_tastvg_extended_v3.py',[dataset,req],td/'SCORE.json')
  done.append((name,params))
 stage(dataset,'report','scripts/report_tastvg_extended_v3.py',[dataset],out/'COMPLETION.json')
 archive(dataset+'扩展敏感性搜索与固定复验已执行完成，待根审计公开')

def run():
 handle=(BASE/'PROCESS.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 for dataset in DATASETS:
  out=BASE/dataset
  if (out/'COMPLETION.json').exists():continue
  stage(dataset,'spatial','scripts/run_tastvg_extended_experts_v3.py',[dataset,'spatial'],out/'experts/SPATIAL_BARRIER.json')
  stage(dataset,'temporal','scripts/run_tastvg_extended_experts_v3.py',[dataset,'temporal'],out/'experts/TEMPORAL_BARRIER.json',python='.venv-exost/bin/python')
  stage(dataset,'capture','scripts/capture_tastvg_extended_v3.py',[dataset],out/'CAPTURE_BARRIER.json')
  # Qualify exact live anchor and a refreshed multi-step trajectory without GT first.
  stage(dataset,'smoke','scripts/smoke_tastvg_extended_v3.py',[dataset],out/'SMOKE.json')
  search(dataset)
 status(BASE/'STATUS.json',dict(status='completed_pending_root_audit_publication',time=time.time()))
 archive('两数据集v3搜索复验执行结束，根审计曲线公开待执行')

if __name__=='__main__':
 try:run()
 except BaseException as e:
  status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));archive('v3工程中断已保存 '+repr(e));raise
