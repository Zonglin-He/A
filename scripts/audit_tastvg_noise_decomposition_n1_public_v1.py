"""Independent scalar source/order/harm aggregation from public rows; no models/GT."""
import json,sys
from pathlib import Path
import numpy as np

def run(root):
 root=Path(root);read=lambda n:json.loads((root/n).read_text());rows=read('ROWS.json');summary=read('SUMMARY.json');across=read('ACROSS_ORDERS.json');checks=0
 assert len(rows)==5280
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
 diagnostics=read('DIAGNOSTICS.json');paired={}
 for d in diagnostics:
  if not d['scheduled']:continue
  re=d['rewards'];g=np.array(d['gt_sIoU']);ye=np.zeros(8,int) if re is None else np.array([int(v>1e-12)-int(v< -1e-12) for v in np.array(re)[1:]-re[0]])
  yg=np.array([int(v>1e-12)-int(v< -1e-12) for v in g[1:]-g[0]])
  np.testing.assert_array_equal(ye,d['teacher_labels']);np.testing.assert_array_equal(yg,d['gt_labels']);checks+=2
  arm=d['arm'];pool=[]
  for k,(e,g) in enumerate(zip(ye,yg),1):
   use=e!=0 if arm=='all' else e*g==1 if arm in ['useful','useful_matched'] else e*g==-1 if arm in ['noisy','noisy_matched'] else e==g==1 if arm=='useful_positive' else e==g==-1
   if use:pool.append(k)
  assert pool==d['eligible_indices'];checks+=1
  if arm.endswith('_matched'):
   import hashlib
   n=d['match_count'];chosen=sorted(sorted(pool,key=lambda k:hashlib.sha256(f"N1-count-v1|{d['hash_key']}|{k}".encode()).hexdigest())[:n]);paired.setdefault((d['order'],d['condition'],d['parent']),{})[arm]=(len(pool),n)
  else:chosen=pool
  assert chosen==d['selected_indices'] and len(chosen)==d['selected_count'];checks+=1
  if 'distances' in d:
   dist=np.array(d['distances']);ids=np.array(chosen);terms=np.logaddexp(0,ye[ids-1]*(dist[ids]-dist[0]));np.testing.assert_allclose(terms,d['terms'],atol=2e-6,rtol=0);assert abs(terms.mean()-d['loss_before'])<2e-6;checks+=2
 for pair in paired.values():
  u,n=pair['useful_matched'],pair['noisy_matched'];assert u[1]==n[1]==min(u[0],n[0]);checks+=1
 assert len(paired)==120
 for name,groups in read('PAIRED_CONTRASTS.json').items():
  a,b=name.split(' minus ')
  for group,subsets in groups.items():
   for subset,metrics in subsets.items():
    for metric,values in metrics.items():
     expected=np.array(across[a][group][subset][metric]['values'])-np.array(across[b][group][subset][metric]['values']);np.testing.assert_allclose(expected,values['values'],atol=1e-13,rtol=0);assert abs(expected.mean()-values['mean'])<1e-13;checks+=2
 signals=read('SIGNAL_SUMMARY.json')
 for arm,groups in signals.items():
  for group,v in groups.items():
   dd=[d for d in diagnostics if d['arm']==arm and d['scheduled'] and ((d['condition']!='clean') if group=='corruption' else d['condition']=='clean')]
   assert len(dd)==v['scheduled'];assert sum(d['valid_expert_frames']>0 for d in dd)==v['valid_specialist_cells'];assert sum(d['selected_count'] for d in dd)==v['selected_signals'];assert sum(d['updated'] for d in dd)==v['updates'];assert sum(d['selected_count']==0 for d in dd)==v['empty_selection'];checks+=5
   counts={k:0 for k in ['correct_positive','correct_negative','noisy_positive','noisy_negative','tie']}
   for d in dd:
    if not d['valid_expert_frames']:continue
    for e,g in zip(d['teacher_labels'],d['gt_labels']):counts[{(1,1):'correct_positive',(-1,-1):'correct_negative',(1,-1):'noisy_positive',(-1,1):'noisy_negative'}.get((e,g),'tie')]+=1
   for k,val in counts.items():assert val==v[k];checks+=1
   for k,source in [('mean_gradient_norm','gradient_norm'),('mean_step_norm','step_norm')]:
    vals=[d[source] for d in dd if d['updated']]
    if vals:assert abs(np.mean(vals)-v[k])<1e-13
    else:assert v[k] is None
    checks+=1
 return dict(status='pass',rows=len(rows),new_rows=3360,reused_rows=1920,checks=checks)
if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
