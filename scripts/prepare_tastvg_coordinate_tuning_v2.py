"""Metadata-only registration; no model, annotation or score reads."""
import hashlib,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/tastvg_coordinate_tuning_v2'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 assert not p.exists(),f'Preserve existing registration: {p}'
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def coarse(lo,hi,anchors,first):
 values=sorted(set([float(f'{10**(math.log10(lo)+(math.log10(hi)-math.log10(lo))*i/20):.14g}') for i in range(21)]+anchors))
 return [first]+[x for x in values if x!=first]
def refine(coarse_values,best):
 values=sorted(set(coarse_values));i=values.index(best);out=[]
 for j in (i-1,i+1):
  if 0<=j<len(values):
   lo,hi=sorted((best,values[j]))
   out += [math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*k/7) for k in range(1,7)]
 return sorted(out)
def run():
 protocol=ROOT/'protocols/tastvg_coordinate_tuning_v2.md'
 grid=dict(lr=coarse(1e-6,1,[.005,.05,.5],.005),teacher_temperature=coarse(.01,100,[.05,1.,20.],1.))
 assert len(grid['lr'])+12<=36 and len(grid['teacher_temperature'])+12<=36
 for values in grid.values():
  for x in values:
   r=refine(values,x);assert len(r)<=12 and not set(r)&set(values)
 write(OUT/'GRIDS.json',dict(coarse=grid,refinement='six log-interior points per available adjacent coarse interval; k/7 k=1..6',max_scheduled_per_dataset=72))
 plans={}
 for dataset,panel in [('vidstg','P1'),('hc2','P5')]:
  prior_path=ROOT/'artifacts/tastvg_optuna_v1'/dataset/'PLAN.json';prior=read(prior_path)
  paper_path=ROOT/'artifacts/tastvg_paper48_v1'/panel/'PLAN.json';paper=read(paper_path)
  excluded={r['source'] for r in prior['rows']}
  candidates=[r for r in paper['rows'] if r['source'] not in excluded]
  candidates.sort(key=lambda r:(hashlib.sha256(('Coordinate-confirm-v2|'+dataset+'|'+r['source']).encode()).hexdigest(),r['source']))
  confirm=[{**r,'original_parent':r['ordinal'],'ordinal':32+j} for j,r in enumerate(candidates[:16])]
  rows=prior['rows'][:32]+confirm
  assert len(rows)==48 and len({r['source'] for r in rows})==48
  orders={f'order{k}':sorted(range(32,48),key=lambda i:hashlib.sha256((f'Coordinate-order-v2|{dataset}|{k}|'+rows[i]['source']).encode()).hexdigest()) for k in [1,2]}
  plan=dict(dataset=dataset,paper48_panel=panel,rows=rows,conditions=prior['conditions'],splits=dict(search=prior['splits']['search'],confirm=dict(sources=16,orders=orders,total=192)),availability=25,historical_exposure=True,selection_metric='corrupt all-arrival source-macro dense delta_vIoU',fixed_rho=.05,prior_plan_sha256=digest(prior_path),paper48_plan_sha256=digest(paper_path),old48_excluded_from_confirmation=True)
  p=OUT/dataset/'PLAN.json';write(p,plan);plans[str(p.relative_to(OUT))]=digest(p)
 write(OUT/'DESIGN_LOCK.json',dict(protocol_sha256=digest(protocol),grids_sha256=digest(OUT/'GRIDS.json'),planner_sha256=digest(Path(__file__)),plans=plans,GT_read=False,time=time.time()))
 write(OUT/'STATUS.json',dict(status='queued_after_fig1_publication_pending_runner_implementation',model_predictions=0,GT_read=False,predecessor='stvg_native_support_fig1_v1/uniform64_v2',remaining=['separate runner adaptation and qualification','VidSTG ordered search and confirmation','HC2 ordered search and confirmation','root audits and verified GitHub publication'],time=time.time()))
 print(json.dumps(dict(status='CPU_registered',coarse_counts={k:len(v) for k,v in grid.items()},datasets=2,search_sources=32,confirmation_sources=16,GT_read=False)))
if __name__=='__main__':run()
