"""Independently reaggregate anonymous scalar CSV; no media, labels or models."""
import csv,json,sys,collections
from pathlib import Path
import numpy as np

def read_rows(path):
    rows=[]
    integers={'parent','arrival','quartile'};booleans={'expert_scheduled','updated'}
    strings={'order','condition','query_type'}
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            rows.append({k:(v if k in strings else int(v) if k in integers else v=='True' if k in booleans else float(v)) for k,v in r.items()})
    return rows

def check_summary(rows,expected):
    fields=list(expected['metrics']);buckets=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:buckets[r['parent']][r['order']].append([r[k] for k in fields])
    sources=sorted(buckets);orders=sorted({r['order'] for r in rows})
    assert len(sources)==expected['sources'] and len(rows)==expected['cells']
    a=np.array([np.mean([np.mean(buckets[s][o],axis=0) for o in sorted(buckets[s])],axis=0) for s in sources])
    rng=np.random.default_rng(20260930);boot=[]
    for _ in range(20):
        ids=rng.integers(0,len(a),(500,len(a)));boot.append(a[ids].mean(axis=1))
    ci=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    ov=np.array([[np.mean([r[k] for r in rows if r['order']==o]) for k in fields] for o in orders]);checks=2
    for j,k in enumerate(fields):
        z=expected['metrics'][k]
        np.testing.assert_allclose(a[:,j].mean(),z['mean'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(ci[:,j],z['ci95'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(ov[:,j],z['order_values'],atol=1e-12,rtol=0);checks+=3
        if len(orders)>1:np.testing.assert_allclose(ov[:,j].std(ddof=1),z['order_sample_SD'],atol=1e-12,rtol=0)
        else:assert z['order_sample_SD'] is None
        checks+=1
        for threshold,label in [(.05,'5'),(.2,'20')]:
            if k.startswith('delta_'):
                assert sum(a[:,j]<-threshold)==z[f'harm_gt{label}pp_sources']
                assert sum(r[k]<-threshold for r in rows)==z[f'harm_gt{label}pp_cells']
            else:assert z[f'harm_gt{label}pp_sources'] is None and z[f'harm_gt{label}pp_cells'] is None
            checks+=2
    return checks

def run(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text())
    rows=read_rows(root/'SCALARS.csv');summary=read('SUMMARY.json');audit=read('AUDIT.json');checks=0
    assert len(rows)==audit['state_links']==8040
    assert len({(r['parent'],r['order'],r['condition']) for r in rows})==8040
    assert len({r['parent'] for r in rows})==670
    assert sum(r['updated'] for r in rows)==audit['SGD_updates']
    assert sum(r['expert_scheduled'] for r in rows)==audit['teacher_checks'];checks+=5
    metric_names=['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5','sIoU_dense_GT','sIoU_sampled','vIoU_sampled']
    for r in rows:
        assert r['expert_scheduled']==(r['arrival']%4==0)
        assert r['quartile']==min(3,4*r['arrival']//670)
        assert r['expert_scheduled'] or (not r['updated'] and r['delta_m_tIoU']==0);checks+=3
        for k in metric_names:
            assert 0<=r['Frozen_'+k]<=1+1e-12 and 0<=r['Ours_'+k]<=1+1e-12
            assert abs(r['Ours_'+k]-r['Frozen_'+k]-r['delta_'+k])<1e-12;checks+=2
        for arm in ['Frozen','Ours']:
            for t in [.3,.5]:assert r[f'{arm}_vIoU@{t}']==float(r[arm+'_m_vIoU']>t);checks+=1
    for group,subsets in summary.items():
        selected=[r for r in rows if (r['condition']!='clean' if group=='corruption' else r['condition']==group)]
        for subset,expected in subsets.items():checks+=check_summary([r for r in selected if subset=='all' or not r['expert_scheduled']],expected)
    buckets=collections.defaultdict(list)
    for r in rows:buckets[r['parent'],r['order']].append(r)
    paired=[]
    for (parent,order),rr in sorted(buckets.items()):
        clean=next(r for r in rr if r['condition']=='clean');bad=[r['delta_m_vIoU'] for r in rr if r['condition']!='clean']
        paired.append(dict(parent=parent,order=order,delta_excess=float(np.mean(bad)-clean['delta_m_vIoU'])))
    checks+=check_summary(paired,read('CORRUPTION_EXCESS.json'))
    for q,expected in read('QUARTILES.json').items():checks+=check_summary([r for r in rows if r['condition']!='clean' and r['quartile']==int(q)],expected)
    return dict(status='pass',rows=len(rows),sources=670,checks=checks,bootstrap_replicates=10000,scope='Independent source/order/bootstrap/harm aggregation from anonymous public scalars; does not recompute metrics from private GT or rerun models')

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
