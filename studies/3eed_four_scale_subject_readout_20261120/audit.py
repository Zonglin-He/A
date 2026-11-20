"""Recount the public row table and check the exact readout on small edge cases."""
from collections import defaultdict
import csv
import gzip

import numpy as np

from replay import OUT, digest, regions, save, selection


def check():
    rows = defaultdict(list)
    seen = set()
    with gzip.open(OUT/'ordinary_rows.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            key = r['source'],r['case']
            assert key not in seen and float(r['previous_shared'])==float(r['subject_only_shared'])
            seen.add(key)
            rows[r['source']+'_to_'+r['target']].append(r)
    with (OUT/'metrics.csv').open() as f:
        metrics = {(r['direction'],r['arm']):r for r in csv.DictReader(f)}
    assert len(rows)==6 and len(seen)==18620
    arms = ('previous_shared','subject_only_shared','independent')
    for arm in arms:
        by_direction = []
        for direction,group in rows.items():
            old = np.array([float(r['previous_shared']) for r in group])
            val = np.array([float(r[arm]) for r in group])
            ht = val[:,None]>[.25,.5]
            h0 = old[:,None]>[.25,.5]
            numbers = [*np.mean(ht,axis=0)*100,100*val.mean()]
            result = metrics[direction,arm]
            np.testing.assert_allclose(numbers,[float(result[k]) for k in ('acc25','acc50','miou')],rtol=0,atol=1e-10)
            np.testing.assert_array_equal(np.count_nonzero(ht & ~h0,axis=0),[int(result[k]) for k in ('rescue25','rescue50')])
            np.testing.assert_array_equal(np.count_nonzero(~ht & h0,axis=0),[int(result[k]) for k in ('harm25','harm50')])
            by_direction.append(numbers)
        np.testing.assert_allclose(np.mean(by_direction,axis=0),[float(metrics['macro',arm][k]) for k in
                                  ('acc25','acc50','miou')],rtol=0,atol=1.e-10)

    # Nonempty raw frame, foreground rejected at every scale.
    word = dict(xyz=np.zeros((1,3)),extent=np.ones((1,len(regions.CLASSES)-1,3)))
    seg = dict(xyz=word['xyz'],labels=np.full((4,1),-1,dtype=int),plane=np.zeros(3))
    assert regions.assemble(seg,word)['completed'].shape==(0,len(regions.CLASSES)-1,7)
    rng = np.random.default_rng(42)
    checks = 0
    for n in (0,1,4,17):
        for rep in range(16):
            valid = rng.random((n,n))>.5
            np.fill_diagonal(valid,False)
            j = valid*(rng.random((n,n)) if rep%3 else float(rep%2))
            K = np.eye(n)
            us = rng.random(n) if rep%4 else np.ones(n)
            ur = rng.random(n) if rep%4 else np.ones(n)
            old,ref,score = selection.medoid(j,K,us,ur,valid)
            actual,ignored,score2 = selection.medoid(j,K,us,ur,valid,read_reference=False)
            assert old==actual and ignored==-1
            np.testing.assert_array_equal(score2,score)
            if old>=0:
                assert valid[old,ref] and j[old,ref]>0
            checks+=1
    save(OUT/'audit.json',dict(ordinary_rows=len(seen),directions=len(rows),metric_rows=len(metrics),
         medoid_checks=checks,empty_inventory_legacy_shape=True,table_sha256=digest(OUT/'ordinary_rows.csv.gz')))


if __name__=='__main__':
    check()
