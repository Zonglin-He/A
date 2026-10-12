"""Portable independent scalar audit; numpy and anonymous saved rows only."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PUB=ROOT/'results/stvg_opd_supplement/2026-10-11'
OLDPUB=ROOT/'results/stvg_opd_paper_hc2_revision/2026-10-08'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]

def stat(rr,field):
    q=collections.defaultdict(list)
    for r in rr:q[r['query_ordinal']].append(r)
    parent=collections.defaultdict(list);query=[]
    for key,items in sorted(q.items()):
        assert len({r['source_id'] for r in items})==1
        v=math.fsum(r[field] for r in items)/len(items);query.append(v);parent[items[0]['source_id']].append(v)
    ids=sorted(parent);values=np.array([math.fsum(parent[i])/len(parent[i]) for i in ids]);rng=np.random.default_rng(20261011)
    boots=np.concatenate([values[rng.integers(0,len(values),(100,len(values)))].mean(1) for _ in range(100)])
    return float(values.mean()),float(np.mean(query)),np.quantile(boots,[.025,.975]).tolist()

def compare(rr,expected):
    n=0
    for field,m in expected['metrics'].items():
        mean,query,ci=stat(rr,field)
        assert abs(mean-m['mean'])<4e-12 and abs(query-m['query_macro'])<4e-12 and np.max(np.abs(np.array(ci)-m['ci95']))<4e-12;n+=4
    return n

def run():
    checks=0;newarrivals=0
    orders=read(PUB/'orders/ROOT_ORDER_STATISTICS.json')['statistics']
    for name,d in orders.items():
        rr=rows(PUB/'orders'/name/'ROWS.jsonl.gz');assert len(rr)==1152;newarrivals+=len(rr);cells={(r['arm'],r['order'],r['query_ordinal']):r for r in rr};assert len(cells)==len(rr)
        paired=[]
        for r in rr:
            if r['arm']!='full':continue
            z=dict(r)
            for a in ['alpha0','alpha1']:
                c=cells[a,r['order'],r['query_ordinal']];assert r['source_id']==c['source_id'] and r['arrival']==c['arrival'] and r['Frozen_v']==c['Frozen_v'];z['Full_minus_'+a+'_v']=r['After_v']-c['After_v']
            assert abs(r['delta_current_v']+r['delta_inherited_v']-r['delta_total_v'])<1e-12;paired.append(z)
        checks+=compare(paired,d['pooled'])
        for o,s in d['per_order'].items():checks+=compare([r for r in paired if r['order']==o],s)
        for a,s in d['arm_absolute'].items():checks+=compare([r for r in rr if r['arm']==a],s)
    assert newarrivals==2304
    baseline=read(PUB/'baselines/ROOT_MATCHED_MAIN_COMPARISON.json')['datasets'];logical=0
    protocol=read(PUB/'receipts/baselines/DESIGN_LOCK.json')['projection']['datasets']
    for ds,methods in baseline.items():
        excluded=set(protocol[ds]['excluded_primary_anonymous_source_ids']);assert len(excluded)==32
        old={(r['query_ordinal'],r['order']):r for r in rows(OLDPUB/f'P1_{ds}/ROWS.jsonl.gz')}
        for method,d in methods.items():
            rr=rows(PUB/'baselines/formal'/ds/d['candidate']/'ROWS.jsonl.gz');assert len(rr)==len(old);logical+=len(rr);joined=[]
            for r in rr:
                s=old[r['query_ordinal'],r['order']];assert s['arrival']==r['arrival'] and s['source_id']==r['source_id']
                assert r['is_development_parent']==(r['source_id'] in excluded);checks+=1
                z=dict(r,OPD_v=s['After_v'],Source_v=s['Frozen_v'],selected_minus_Source_v=r['After_v']-s['Frozen_v'],OPD_minus_selected_v=s['After_v']-r['After_v']);joined.append(z)
            for scope in ['all_population','primary_excluding_32_dev_parents']:
                part=joined if scope=='all_population' else [r for r in joined if not r['is_development_parent']];checks+=compare(part,d[scope]);assert sum(r['updates']>0 for r in part)==d[scope]['actual_update_arrivals'];checks+=1
    assert logical==82710
    cost=read(PUB/'cost/END_TO_END_SUMMARY.json')['datasets'];qcount=0
    for ds,d in cost.items():
        rr=read(PUB/'cost'/ds/'QUERY_TIMINGS.json')['rows'];assert len(rr)==16;qcount+=len(rr)
        for k,v in d['timings'].items():
            assert abs(np.median([r[k] for r in rr])-v['median'])<1e-12 and abs(np.quantile([r[k] for r in rr],.95)-v['p95'])<1e-12;checks+=2
        for r in rr:
            assert abs(math.fsum(r[k] for k in ['decode_preprocess_seconds','native_student_prefix_seconds','fresh_expert_seconds','adaptation_seconds','final_readout_seconds'])+r['dispatcher_seconds']-r['total_seconds'])<1e-7;checks+=1
    assert qcount==32
    print(json.dumps(dict(status='pass',scope='independent anonymous new-order/baseline/cost arithmetic and paired bootstrap',order_arrivals=newarrivals,selected_baseline_logical_arrivals=logical,cost_query_fits=qcount,scalar_checks=checks,bootstrap=10000,all_negative_rows_retained=True)))

if __name__=='__main__':run()
