"""Equal-parent/equal-query expected within-query std from sealed moments."""
from collections import Counter
import torch


def aggregate_query_std(rows, variance_floor=1e-12):
    if not rows or variance_floor<=0:
        raise ValueError('nonempty moments and positive variance floor required')
    keys=[r['key'] for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('duplicate query identities')
    counts=Counter(r['source'] for r in rows)
    weights=torch.tensor([1/(len(counts)*counts[r['source']]) for r in rows],dtype=torch.float64)
    result={}
    for src,dst in [('referent','spatial'),('event','event')]:
        mean=torch.stack([torch.as_tensor(r['moments'][src]['mean'],dtype=torch.float64) for r in rows])
        second=torch.stack([torch.as_tensor(r['moments'][src]['second_moment'],dtype=torch.float64) for r in rows])
        if mean.ndim!=2 or second.shape!=mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(second).all():
            raise ValueError('finite channel-vector first/second moments required')
        variance=second-mean.square()
        if variance.min() < -1e-10:
            raise ValueError('inconsistent negative within-query variance')
        std=variance.clamp_min(variance_floor).sqrt()
        global_mean=weights@mean
        result[dst]={'mean':global_mean,'expected_query_std':weights@std,
            'within_query_rms_std':(weights@variance.clamp_min(variance_floor)).sqrt(),
            'population_std':((weights@second)-global_mean.square()).clamp_min(variance_floor).sqrt()}
    return {'branches':result,'queries':len(rows),'parents':len(counts),'weights':weights}
