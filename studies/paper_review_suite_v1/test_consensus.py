import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
import numpy as np
from stvg.consensus import estimate as reference,encode
from consensus_exact import estimate
from full import read,atomic,RUN
rng=np.random.default_rng(20261024);started=time.perf_counter();checks=0;maxerr=0.;nodes=[]
for n in range(2,13):
 for case in range(30):
  x=rng.normal(size=(n,4))*(.1 if case%2 else 3);x[case%n]+=case%4
  ref=reference([v[None]for v in x],[np.ones(1)]*n,np.ones(4),4,1/16,False);new=estimate(x,np.ones(4),4)
  assert new['selected']==ref['selected'];err=float(np.max(abs(new['delta']-ref['delta'])));assert err==0 and new['objective']==ref['objective'];checks+=1;maxerr=max(maxerr,err);nodes.append(new['nodes'])
for ds in ['hc2','vidstg']:
 cfg=read(R/'configs/default.json')['datasets'][ds]['E_calibration'];files=read(RUN/ds/'sealed.json')['files']
 for key,e in list(files.items())[:64]:
  with np.load(RUN/ds/e['path'])as z:x=encode(z['native_boxes'])-encode(z['Source'])[z['positions8']]
  ref=reference([v[None]for v in x],[np.ones(1)]*8,cfg['scale'],cfg['truncation'],1/16,False);new=estimate(x,cfg['scale'],cfg['truncation']);assert new['selected']==ref['selected']and np.array_equal(new['delta'],ref['delta'])and new['objective']==ref['objective'];checks+=1;nodes.append(new['nodes'])
for n in [16,32,65,130,200]:
 x=rng.normal(size=(n,4))*.01;x[::10]+=10;new=estimate(x,np.ones(4),4);assert new['exact_search_completed'];checks+=1;nodes.append(new['nodes'])
atomic(R/'outputs/paper_review_suite_v1/consensus_test.json',dict(status='passed',checks=checks,bitwise_delta_reference_checks=checks-5,maxerr=maxerr,nodes_max=max(nodes),seconds=time.perf_counter()-started,production_untouched=True,worst_case_exponential=True));print(json.dumps(dict(checks=checks,max_nodes=max(nodes),seconds=time.perf_counter()-started)))
