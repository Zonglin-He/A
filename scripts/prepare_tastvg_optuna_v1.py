import sys,hashlib,shutil,time,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_optuna_common_v1 import *
def run():
 assert read(ROOT/'artifacts/tastvg_paper48_v1/FINAL_COMPLETION.json')['status']=='completed_verified_and_published'
 for dataset,panel in [('vidstg','P1'),('hc2','P5')]:
  out=BASE/dataset
  if (out/'PLAN.json').exists():continue
  old=ROOT/'artifacts/tastvg_paper48_v1'/panel;p=read(old/'PLAN.json');indices=sorted(range(len(p['rows'])),key=lambda i:hashlib.sha256(('Optuna-cohort-v1|'+dataset+'|'+p['rows'][i]['source']).encode()).hexdigest())[:48]
  rows=[{**p['rows'][i],'ordinal':j,'original_parent':i} for j,i in enumerate(indices)];assert len({r['source'] for r in rows})==48
  splits={}
  for name,parents in [('search',list(range(32))),('confirm',list(range(32,48)))]:
   orders={f'order{k}':sorted(parents,key=lambda i:hashlib.sha256((f'Optuna-order-v1|{dataset}|{k}|'+rows[i]['source']).encode()).hexdigest()) for k in [1,2]}
   splits[name]=dict(sources=len(parents),orders=orders,total=len(parents)*12)
  conds=p['conditions'];experts=sorted({i for s in splits.values() for order in s['orders'].values() for at,i in enumerate(order) if at%4==0})
  plan=dict(dataset=dataset,paper48_panel=panel,rows=rows,splits=splits,conditions=conds,availability=25,space=SPACE,default=DEFAULT,trials=48,selection='source-hash first32 search next16 confirmation; no GT/outcome selection',historical_exposure=True)
  write(out/'PLAN.json',plan);write(out/'EXPERT_PLAN.json',dict(rows=rows,expert_needed=experts,conditions_by_parent={str(i):conds for i in experts},conditions=conds,total=len(experts)*6))
  source_bar=read(old/'SUBJECT_BARRIER.json');files={}
  for j,i in enumerate(indices):
   src=old/'subjects'/f'{i:05}.json';assert sha(src)==source_bar['files'][src.name];dst=out/'subjects'/f'{j:05}.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);files[dst.name]=sha(dst)
  write(out/'SUBJECT_BARRIER.json',dict(count=48,files=files));write(out/'COHORT.json',dict(dataset=dataset,search_sources=32,confirmation_sources=16,query_per_source=1,orders=2,conditions=conds,source_overlap=0,historical_project_exposure=True,expert_inputs=len(experts)*6,plan_sha256=sha(out/'PLAN.json'),selection_metric='corrupt future nonexpert source macro dense delta_vIoU',trials=48,space=SPACE,default=DEFAULT))
 status(BASE/'STATUS.json',dict(status='prepared',current_stage='implementation validation',time=time.time()))
 print('Prepared both 48-source pools')
if __name__=='__main__':run()
