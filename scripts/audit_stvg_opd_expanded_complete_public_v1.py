"""Independent scalar-only audit of all anonymous Expanded-State outcomes."""
import collections, csv, gzip, hashlib, json, math, sys
from pathlib import Path
import numpy as np

REL = 'results/stvg_opd_claim_strengthening/2026-10-13/expanded_state_complete'
FIELDS = ['After_v','Before_v','Frozen_v','Full_minus_alpha0_v','Full_minus_alpha1_v']
ARM_FIELDS = ['After_v','Before_v','Frozen_v','After_s','After_t','After_R30','After_R50',
              'delta_total_v','delta_current_v','delta_inherited_v']
def read(path):
    if path.suffix == '.gz':
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(path.read_text())
def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def independent(part,fields):
    groups=collections.defaultdict(list)
    for r in part:groups[r['parent_id']].append(r)
    ids=sorted(groups)
    assert len({len(x) for x in groups.values()})==1
    matrix=np.array([[math.fsum(r[f] for r in groups[i])/len(groups[i]) for f in fields] for i in ids])
    rng=np.random.default_rng(20261011)
    # Same declared parent draws, separately accumulated from anonymous rows.
    boot=np.concatenate([matrix[rng.integers(0,len(ids),(100,len(ids)))].mean(1) for _ in range(100)])
    ci=np.quantile(boot,[.025,.975],axis=0)
    return dict(sources=len(ids),cells=len(part),metrics={f:dict(mean=float(matrix[:,j].mean()),
        ci95=ci[:,j].tolist(),query_macro=math.fsum(r[f] for r in part)/len(part),
        parent_values=[dict(parent_id=i,value=float(matrix[k,j])) for k,i in enumerate(ids)],
        negative_parent_count=int((matrix[:,j]<0).sum()),positive_parent_count=int((matrix[:,j]>0).sum()),
        harm_gt5pp_sources=int((matrix[:,j]<-.05).sum()),harm_gt20pp_sources=int((matrix[:,j]<-.2).sum()),
        worst_parent=float(matrix[:,j].min()),best_parent=float(matrix[:,j].max())) for j,f in enumerate(fields)})
def compare(actual,expected):
    assert actual['sources']==expected['sources'] and actual['cells']==expected['cells'];n=2
    for f,x in actual['metrics'].items():
        y=expected['metrics'][f]
        for k in ['mean','query_macro','worst_parent','best_parent']:
            assert abs(x[k]-y[k])<4e-12,(f,k);n+=1
        assert max(abs(a-b) for a,b in zip(x['ci95'],y['ci95']))<4e-12;n+=2
        for k in ['negative_parent_count','positive_parent_count','harm_gt5pp_sources','harm_gt20pp_sources']:
            assert x[k]==y[k],(f,k);n+=1
        assert len(x['parent_values'])==len(y['parent_values'])
        for a,b in zip(x['parent_values'],y['parent_values']):
            assert a['parent_id']==b['parent_id'] and abs(a['value']-b['value'])<4e-12;n+=1
    return n
def run(root):
    out=root/REL;rr=rows(out/'ROWS.jsonl.gz');assert len(rr)==5430
    assert len({r['parent_id'] for r in rr})==905
    assert all('source_id' not in r and 'query_ordinal' not in r for r in rr)
    source=read(out/'DIRECT_CONTRASTS.json.gz');arms=read(out/'ALL_ARM_STATISTICS.json.gz')
    checks=0;all_pairs=[]
    for ds,name,size in [('hc2','E_hc2_cross_clean',205),('vidstg','E_vidstg_cross_clean',700)]:
        part=[r for r in rr if r['dataset']==ds];assert len(part)==size*6
        cells={(r['arm'],r['order'],r['parent_id']):r for r in part};assert len(cells)==len(part)
        paired=[]
        for r in part:
            assert r['query_reset'] and r['Adam_reset'] and r['source_reset_per_arm_order']
            for m in ['v','s','t']:
                assert abs(r['delta_total_'+m]-r['delta_current_'+m]-r['delta_inherited_'+m])<2e-10
                assert r['After_'+m]==r['values']['After'][m]
            if r['arm']!='full':continue
            p=dict(r)
            for a in ['alpha0','alpha1']:
                c=cells[a,r['order'],r['parent_id']]
                assert c['arrival']==r['arrival'] and c['Frozen_v']==r['Frozen_v'] and c['After_t']==r['After_t']
                for m in ['v','s','t']:p['Full_minus_'+a+'_'+m]=r['After_'+m]-c['After_'+m]
            paired.append(p)
        saved=source['statistics'][name];assert saved['parent_count']==size
        for order in ['pooled','order1','order2']:
            chosen=paired if order=='pooled' else [r for r in paired if r['order']==order]
            reference=saved['pooled'] if order=='pooled' else saved['per_order'][order]
            checks+=compare(reference,independent(chosen,FIELDS))
            for a in ['alpha0','alpha1']:
                metric=reference['metrics']['Full_minus_'+a+'_v']
                entry=next(x for x in source['table'] if x['dataset']==ds and x['order']==order and x['contrast']=='Full - '+a)
                assert abs(entry['mean_pp']-metric['mean']*100)<4e-12
                assert entry['ci95_pp']==[v*100 for v in metric['ci95']]
            for a in ['full','alpha0','alpha1']:
                chosen=[r for r in part if r['arm']==a and (order=='pooled' or r['order']==order)]
                checks+=compare(arms['statistics'][name][a][order],independent(chosen,ARM_FIELDS))
        for a in ['full','alpha0','alpha1']:
            chosen=[r for r in part if r['arm']==a]
            checks+=compare(saved['arm_absolute'][a],independent(chosen,['After_v','Before_v','Frozen_v']))
        for pid in sorted({r['parent_id'] for r in paired}):
            group=sorted([r for r in paired if r['parent_id']==pid],key=lambda r:r['order'])
            x=dict(dataset=ds,parent_id=pid)
            for order,r in zip(['order1','order2'],group):
                assert r['order']==order
                for f in FIELDS:x[order+'_'+f]=r[f]
            for f in FIELDS:x['pooled_'+f]=math.fsum(r[f] for r in group)/2
            all_pairs.append(x)
    with gzip.open(out/'ALL_PARENT_CONTRASTS.csv.gz','rt') as f:table=list(csv.DictReader(f))
    assert len(table)==len(all_pairs)==905
    for x,y in zip(table,all_pairs):
        assert x['dataset']==y['dataset'] and int(x['parent_id'])==y['parent_id']
        assert all(abs(float(x[k])-v)<4e-12 for k,v in y.items() if isinstance(v,float))
    manifest=read(out/'DESIGN_PUBLIC.json')
    assert manifest['new_formal_fits']==5430 and manifest['new_DINO_calls']==0
    assert manifest['GT_used_to_select_cohort'] is False and manifest['production_promoted'] is False
    return dict(status='pass',scope='all anonymous 5430 formal scalars/905 parent pairs/three arms/two orders/full absolute and direct means/10000 paired parent bootstrap/full negative tails',
        complete_formal_rows=5430,parents=905,parent_pairs=905,scalar_comparisons=checks,
        bootstrap=10000,seed=20261011,raw_GT_read=False,new_model_calls=0,new_DINO_calls=0,
        CPU_only=True,GT_read=False,rows_sha256=digest(out/'ROWS.jsonl.gz'),
        direct_contrasts_sha256=digest(out/'DIRECT_CONTRASTS.json.gz'),all_arm_statistics_sha256=digest(out/'ALL_ARM_STATISTICS.json.gz'))
if __name__=='__main__':print(json.dumps(run(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])))
