"""Model-free recomputation of every anonymous paired statistic and Q0 summary."""
import collections,gzip,json,math,sys
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]

def statistic(x):
    x=np.asarray(x,float)
    if not len(x):return dict(count=0,mean=None,ci95=None,median=None,min=None,max=None)
    gen=np.random.default_rng(20261012);inds=gen.integers(len(x),size=(10000,len(x)));b=np.sum(x[inds],axis=1)/len(x)
    return dict(count=len(x),mean=math.fsum(x)/len(x),ci95=np.percentile(b,[2.5,97.5]).tolist(),median=float(np.median(x)),
        min=float(min(x)),max=float(max(x)),harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)),positive=int(sum(x>1e-12)),negative=int(sum(x< -1e-12)))

def compare(a,b):
    if isinstance(a,dict):assert set(a)==set(b);return sum(compare(v,b[k]) for k,v in a.items())
    if isinstance(a,list):assert len(a)==len(b);return sum(compare(x,y) for x,y in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b
    else:assert abs(a-b)<2e-10*(1+abs(a))
    return 1

def run(root):
    p=Path(root)/'results/stvg_opd_temporal_spatial_complementarity/2026-10-12';rr=rows(p/'ROWS.jsonl.gz');s=read(p/'SUMMARY.json');checks=0
    assert len(rr)==192 and len({r['parent_id'] for r in rr})==64
    for r in rr:
        for arm in ['Native','OPD','Direct','Capped_Direct']:
            for metric in ['v','t','s','R30','R50']:
                checks+=compare(r[f'{arm}_T1_{metric}']-r[f'{arm}_T0_{metric}'],r[f'delta_T_{arm}_{metric}'])
            assert r[f'{arm}_T0_s']==r[f'{arm}_T1_s']
        for t in ['T0','T1']:
            assert len({r[f'{a}_{t}_t'] for a in ['Native','OPD','Direct','Capped_Direct']})==1
            for c in ['Direct','Capped_Direct','Native']:
                for m in ['v','t','s','R30','R50']:checks+=compare(r[f'OPD_{t}_{m}']-r[f'{c}_{t}_{m}'],r[f'OPD_minus_{c}_{t}_{m}'])
        for m in ['v','t','s','R30','R50']:checks+=compare(r[f'delta_T_OPD_{m}']-r[f'delta_T_Native_{m}'],r[f'interaction_OPD_Native_{m}'])
        assert r['gained_outside_Native_frames']==r['gained_supported_frames']+r['gained_unsupported_frames']
        assert r['T1_event_frames']-r['T0_event_frames']==r['gained_outside_Native_frames']-r['lost_event_frames']
    for panel,data in s.items():
        for ds,b in data.items():
            parents=collections.defaultdict(list)
            for r in rr:
                if r['panel']==panel and r['dataset']==ds:parents[r['parent_id']].append(r)
            assert len(parents)==b['parents']==32
            for f,v in b['metrics'].items():checks+=compare(statistic([math.fsum(r[f] for r in rs)/len(rs) for rs in parents.values()]),v)
    qr=rows(p/'Q0_GROUP_ROWS.jsonl.gz');qp=rows(p/'Q0_PAIR_ROWS.jsonl.gz');qs=read(p/'Q0_DIAGNOSTIC_SUMMARY.json');assert len(qr)==192 and len(qp)==1308
    for ds,data in qs.items():
        for g,b in data.items():
            r=[x for x in qr if x['dataset']==ds and x['group']==g];pr=[x for x in qp if x['dataset']==ds and x['group']==g]
            assert len(r)==b['groups']==32 and len(pr)==b['pairs'];assert sum(not v['equal_amplitudes'] for v in pr)==b['unequal_amplitude_pairs']
            for f,st in b['statistics'].items():checks+=compare(dict(**statistic([v[f] for v in r if v[f] is not None]),undefined=sum(v[f] is None for v in r)),st)
            for f,st in b['pair_statistics'].items():
                v=np.array([x[f] for x in pr]);checks+=compare(dict(count=len(v),mean=float(v.mean()),median=float(np.median(v)),p05_p95=np.quantile(v,[.05,.95]).tolist()),st)
            for v in r:
                if v['weighted_absolute_norm']>1e-15:checks+=compare(v['mean_norm']/v['weighted_absolute_norm'],v['cancellation_ratio'])
    return dict(status='pass',scope='all anonymous paired contrasts/coverage/parent bootstrap/Q0 group and pair summaries',
        scalar_checks=checks,factorial_cells=192,logical_outputs=1536,distinct_parents=64,Q0_groups=192,Q0_pairs=1308,CPU_only=True,GT_read=False,new_model_calls=0)

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),allow_nan=False))
