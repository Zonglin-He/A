"""Reclassify all fixed v1 decisions using the measured branch flag and every qualifying slot."""
from collections import Counter, defaultdict
import numpy as np

from registry_swap import HERE, OUT, digest, read
from calibrate import csv_file

PARENT=HERE.parent/'3eed_v1_subject_margin_attribution_20261120'


def run():
    raw=PARENT/'run_PRIVATE'
    seal=read(raw/'SEALED.json')
    assert not seal['FINAL_access']
    total=defaultdict(Counter)
    err=0.
    for direction,pin in seal['directions'].items():
        folder=raw/direction
        assert digest(folder/'SEALED.json')==pin
        path=raw/('scored_'+direction+'.npz')
        audit=read(path.with_suffix('.json'))
        assert digest(path)==audit['file_sha256'] and audit['prediction_seal']==digest(raw/'SEALED.json')
        with np.load(path) as f:
            gt=f['pool_iou'].copy()
        for name,h in read(folder/'SEALED.json')['files'].items():
            assert digest(folder/name)==h
            with np.load(folder/name) as z:
                pool=gt[z['ordinals']]
                slate=np.take_along_axis(pool,z['slots'],axis=1)
                prior,score=z['prior'].astype(float),z['final']
                pick=z['choices'][:,0]
                np.testing.assert_array_equal(score.argmax(1),pick)
                err=max(err,float(abs(prior+z['unary']+z['rel']-score).max()))
                bound=1.5+z['active'].astype(int)
                for t in (.25,.50):
                    good=slate>t
                    won=slate[np.arange(len(pick)),pick]>t
                    gap=prior.max(1)-np.where(good,prior,-np.inf).max(1)
                    kinds=np.select([won,pool.max(1)<=t,~good.any(1),gap>=bound],
                                     ['hit','no_pool','miss_slate','blocked'],default='reachable_miss')
                    for scope in (direction,'all_records'):
                        total[scope,t].update(kinds.tolist())
    results=[]
    for (scope,t),counts in sorted(total.items()):
        n=sum(counts.values())
        values={k:counts[k] for k in ('hit','no_pool','miss_slate','blocked','reachable_miss')}
        errors=n-counts['hit']
        results.append(dict(scope=scope,threshold=t,n=n,**values,errors=errors,
                    blocked_pct_all=100*counts['blocked']/n,blocked_pct_errors=100*counts['blocked']/errors))
    assert sum(total['all_records',.5].values())==18620
    assert total['all_records',.25]['blocked']==83 and total['all_records',.5]['blocked']==36
    assert err<1.e-6
    OUT.mkdir(exist_ok=True)
    csv_file(OUT/'bound_recheck.csv',results)


if __name__=='__main__':
    run()
