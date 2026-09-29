"""Independent scalar source/order/harm aggregation from public rows; no models/GT."""
import json,sys
from pathlib import Path
import numpy as np

def run(root):
 root=Path(root);read=lambda n:json.loads((root/n).read_text());rows=read('ROWS.json');summary=read('SUMMARY.json');across=read('ACROSS_ORDERS.json');checks=0
 assert len(rows)==3360
 for arm,orders in summary.items():
  for order,groups in orders.items():
   for group,subsets in groups.items():
    for subset,values in subsets.items():
     rr=[r for r in rows if r['arm']==arm and r['order']==order and (r['condition']!='clean' if group=='corruption' else r['condition']==group) and (subset=='all' or not r['expert_scheduled'])];ids=sorted({r['parent'] for r in rr});assert len(rr)==values['cells'] and len(ids)==values['sources'];checks+=2
     src={key:[np.mean([r[key] for r in rr if r['parent']==i]) for i in ids] for key in ['s','t','v','delta_s','delta_t','delta_v','minus_final_s','minus_final_v']}
     for key,x in src.items():assert abs(np.mean(x)-values[key])<1e-13;checks+=1
     for m in ['s','v']:
      assert sum(x<-.05 for x in src['delta_'+m])==values['source_harm_'+m];assert sum(r['delta_'+m]<-.05 for r in rr)==values['cell_harm_'+m];checks+=2
  for group,subsets in across[arm].items():
   for subset,stats in subsets.items():
    for key,v in stats.items():
     x=np.array([summary[arm][o][group][subset][key] for o in orders]);np.testing.assert_allclose(x,v['values'],atol=1e-13,rtol=0)
     for k,z in [('mean',x.mean()),('sample_std',x.std(ddof=1)),('min',x.min()),('max',x.max()),('positive',int((x>0).sum())),('negative',int((x<0).sum())),('zero',int((x==0).sum()))]:assert abs(z-v[k])<1e-13;checks+=1
 for d in read('DIAGNOSTICS.json'):
  if 'q' in d:
   q=np.exp(-np.array(d['ranks']));q/=q.sum();np.testing.assert_allclose(q,d['q'],atol=1e-13,rtol=0);assert sorted(d['ranks'])==sorted(d['correct_ranks']);p=np.array(d['p']);assert abs(np.sum(p*np.log(p/q))-d['loss_before'])<2e-6;checks+=3
 return dict(status='pass',rows=len(rows),new_rows=1440,reused_rows=1920,checks=checks)
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
