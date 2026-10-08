"""Reproduce every P0 parent statistic from anonymous rows using NumPy only."""
import collections,gzip,json,sys
from pathlib import Path
import numpy as np

def read(p):
    with Path(p).open() as f:return json.load(f)

def run(base):
    base=Path(base);root=read(base/'P0_ROOT_STATISTICS.json');checks=0
    for ds in ['hc2','vidstg']:
        summary=read(base/('P0_'+ds)/'SUMMARY.json')
        with gzip.open(base/('P0_'+ds)/'ROWS.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
        assert len(rows)==768 and len({r['source_id'] for r in rows})==128
        for arm in ['on_policy','frozen_rollout','shuffled_feedback']:
            rr=[r for r in rows if r['arm']==arm];groups=collections.defaultdict(list)
            fields=list(summary['arms'][arm]['metrics'])
            for r in rr:groups[r['source_id']].append(r)
            assert all(len(v)==2 and {r['order'] for r in v}=={'order1','order2'} for v in groups.values())
            matrix=np.array([[sum(r[f] for r in groups[s])/2 for f in fields] for s in sorted(groups)])
            assert np.isfinite(matrix).all()
            rng=np.random.default_rng(20261006)
            bootstrap=np.concatenate([np.average(matrix[rng.integers(0,128,(100,128))],axis=1) for _ in range(100)])
            lo,hi=np.quantile(bootstrap,[.025,.975],axis=0)
            for j,f in enumerate(fields):
                expected=summary['arms'][arm]['metrics'][f];independent=root['datasets'][ds][arm][f]
                for target in [expected,independent]:
                    assert abs(matrix[:,j].mean()-target['mean'])<2e-12
                    assert np.max(abs(np.array([lo[j],hi[j]])-target['ci95']))<2e-12
                    assert abs(np.mean([r[f] for r in rr])-target['query_macro'])<2e-12
                    if f.startswith('delta_'):
                        assert int((matrix[:,j]<-.05).sum())==target['harm_gt5pp_sources']
                        assert int((matrix[:,j]<-.20).sum())==target['harm_gt20pp_sources']
                    checks+=5
            for r in rr:
                assert abs(r['delta_total_v']-r['delta_current_v']-r['delta_inherited_v'])<2e-12
                assert r['Frozen_t']==r['Before_t']==r['After_t']
                for t in [.3,.5]:
                    assert r['correct_to_wrong_'+str(t)]==float(r['Frozen_v']>t and r['After_v']<=t)
                    assert r['wrong_to_correct_'+str(t)]==float(r['Frozen_v']<=t and r['After_v']>t)
                checks+=6
        grouped=collections.defaultdict(dict)
        for r in rows:grouped[(r['source_id'],r['order'])][r['arm']]=r
        assert len(grouped)==256
        for cell in grouped.values():
            full=cell['on_policy']
            assert abs(full['on_policy_minus_shuffled_v']-full['After_v']+cell['shuffled_feedback']['After_v'])<2e-12
            assert abs(full['on_policy_minus_frozen_rollout_v']-full['After_v']+cell['frozen_rollout']['After_v'])<2e-12
            assert all(r['Frozen_v']==full['Frozen_v'] and r['Frozen_t']==full['Frozen_t'] and r['Frozen_s']==full['Frozen_s'] for r in cell.values())
            checks+=5
    result=dict(status='pass',scalar_checks=checks,requires_private_media_GT_or_weights=False,
        reconstructed_model_gradients_or_Jacobian=False,bootstrap_resamples=10000)
    print(json.dumps(result));return result

if __name__=='__main__':run(sys.argv[1])
