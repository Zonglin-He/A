"""Describe paired uncertainty over Waymo recording groups for sealed readouts."""
from collections import defaultdict

import numpy as np

from catalog import RESULTS, atomic_json, csv_rows, sha


CONTRASTS = (
    ('B_verified','A_known_only'),
    ('B_verified','raw_D'),
    ('C_old_head','B_verified'),
    ('D_new_evidence','C_old_head'),
    ('D_new_evidence','raw_D'),
    ('D_new_evidence','D_rgb_off'),
    ('D_new_evidence','D_rgb_shuffled'),
)


def measure(rows, pair):
    by=defaultdict(dict)
    for x in rows:
        if x['status']=='covered' and x['platform']=='waymo' and x['arm'] in pair:
            by[(x['recording'],x['case'])][x['arm']]=x
    records=defaultdict(list)
    for (r,_),tables in by.items():
        assert tables.keys()==set(pair)
        a,b=(tables[k] for k in pair)
        hits=int(a['winner_status']=='correct')-int(b['winner_status']=='correct')
        mrr=1/int(a['instance_rank'])-1/int(b['instance_rank'])
        records[r].append((hits,mrr,1))
    rank=np.array([np.add.reduce(records[r]) for r in sorted(records)])
    rng=np.random.default_rng(251119)
    draw=rng.integers(len(rank),size=(10000,len(rank)))
    total=rank[draw].sum(1)
    draws=total[:,:2]/total[:,2,None]
    effect=rank[:,:2].sum(0)/rank[:,2].sum()
    return dict(recordings=len(rank),covered=int(rank[:,2].sum()),
                paired_top1_diff=float(effect[0]),top1_95_percentiles=np.quantile(draws[:,0],[.025,.975]).tolist(),
                paired_mrr_diff=float(effect[1]),mrr_95_percentiles=np.quantile(draws[:,1],[.025,.975]).tolist())


if __name__=='__main__':
    f=('all_rows.csv','seed_repeat_rows.csv')
    atomic_json(RESULTS/'RECORDING_INTERVALS.json',dict(
        design='descriptive recording-paired bootstrap of fixed predictions, no training refits or gating',
        n_draws=10000,scope='Waymo covered',inputs={k:sha(RESULTS/k) for k in f},
        paired={k:{a+'__vs__'+b:measure(csv_rows(RESULTS/k),(a,b)) for a,b in CONTRASTS} for k in f},
        protected_data_access=False))
