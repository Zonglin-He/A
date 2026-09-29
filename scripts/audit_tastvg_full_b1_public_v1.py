"""Independent primary source aggregation and source bootstrap from public source rows."""
import sys,json
from pathlib import Path
import numpy as np

def run(root):
 root=Path(root);rows=json.loads((root/'SOURCE_ROWS.json').read_text());summary=json.loads((root/'SUMMARY.json').read_text());checks=0
 for group in ['corruption','clean']:
  for sub in ['all','nonexpert','first_source_nonexpert']:
   rr=[r for r in rows if r['group']==group and r['subset']==sub];assert len(rr)==670
   for key,st in summary[group][sub].items():
    if key in ['across_order','cell_harm','query_arrivals']:continue
    a=np.array([r['metrics'][key] for r in rr if r['metrics'][key] is not None],float);assert len(a)==st['sources'];assert abs(a.mean()-st['mean'])<1e-13
    rng=np.random.default_rng(20260929);boot=[]
    for i in range(10):boot.extend(a[rng.integers(0,len(a),(1000,len(a)))].mean(1).tolist())
    np.testing.assert_allclose(np.quantile(boot,[.025,.975]),st['ci95'],rtol=0,atol=1e-13)
    assert int((a<-.05).sum())==st['source_harm_gt5pp'];checks+=4
 return dict(status='pass',checks=checks,scope='Source-level primary clean/corruption macro/CI/harm; model/state audits separate')
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
