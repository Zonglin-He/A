"""Verify the direct reduced geometry against paired v2 tensor projections."""
from concurrent.futures import ProcessPoolExecutor
import sys
from time import perf_counter

import cv2
import numpy as np
import torch

from cache_mono import guard
from lean_geometry import describe,reduced_pool
from models import EDGE_INDEX,NODE_INDEX
from provenance import HERE,RUN,digest,runtime,write
from eval_minimal_verifier import dev_jobs,mono_jobs
from prepare_blind import prepare
from provenance import retain
from source_features import pool_features


def query(job):
    case,row,previous = job
    if case is None:
        import minimal

        base = minimal.predict(row['frame'],row['raw'])
        case = prepare((row['frame']['uid'],row,dict(outputs=dict(axis_2d=base))))
    args = case,row['frame'],row['raw']
    full,(x,e,_) = pool_features(*args)
    assert full['pool']==previous['pool']
    keep = retain(full['pool'])
    result = []
    for col,cap in enumerate((17,13)):
        pool,evidence = reduced_pool(*args,cap)
        selected = np.flatnonzero(keep[:,col])
        expected = [full['pool'][i]['box'] for i in selected]
        assert pool==expected,(case['uid'],cap,pool,expected)
        a,b = describe(evidence,pool)
        selected_nodes = x[np.ix_(selected,NODE_INDEX)]
        selected_edges = e[np.ix_(selected,selected,EDGE_INDEX)]
        dn = float(np.max(abs(a-selected_nodes)))
        de = float(np.max(abs(b-selected_edges)))
        np.testing.assert_allclose(a,selected_nodes,rtol=1.e-6,atol=1.e-7)
        np.testing.assert_allclose(b,selected_edges,rtol=1.e-6,atol=1.e-7)
        result.append((dn,de))
    return result


def main():
    sys.addaudithook(guard)
    runtime()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    begin = perf_counter()
    checks = {}
    with ProcessPoolExecutor(8) as workers:
        for scope,(jobs,_) in (('development',dev_jobs()),('mono',mono_jobs())):
            seen = list(workers.map(query,jobs,chunksize=4))
            new = np.asarray(seen).max(0).tolist()
            checks[scope] = dict(n=len(seen),compare_all_boxes=True,
                max_error=dict(zip(('cap17','cap13'),new,strict=True)))
            print(scope,checks[scope],flush=True)
    write(RUN/'lean-geometry-audit.json',dict(scopes=checks,labels_read=False,
          seconds=perf_counter()-begin,code={k:digest(HERE/k) for k in ('audit_lean.py','lean_geometry.py')}))


if __name__=='__main__':
    main()
