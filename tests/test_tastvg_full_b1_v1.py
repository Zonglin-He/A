"""Contracts for all-query source aggregation and source-group stream ordering."""
import numpy as np
from scripts.score_tastvg_full_b1_v1 import stats

def test_source_mean_not_query_weighted():
    # Source0 has two queries, source1 one: macro must give equal source weight.
    queries={0:[0.,1.],1:[1.]};per_source=np.array([np.mean(v) for v in queries.values()]);assert per_source.mean()==.75
    assert per_source.mean()!=np.mean([x for v in queries.values() for x in v])

def test_condition_order_axes_preserved():
    x=np.arange(2*4*3*3*4*3,dtype=float).reshape(2,4,3,3,4,3)
    idx=[1,2,3];actual=np.mean(np.mean(x[:,:,:,0][:,idx],axis=1),axis=1)
    reference=np.stack([np.mean([x[s,c,o,0] for c in idx for o in range(3)],axis=0) for s in range(2)])
    np.testing.assert_array_equal(actual,reference)

def test_missing_subset_not_zero_filled():
    x=np.array([[np.nan,.2,.4],[.1,np.nan,np.nan]])
    values=np.nanmean(x,axis=1);np.testing.assert_allclose(values,[.3,.1]);assert np.isclose(values.mean(),.2)
    s=stats(values);assert s['sources']==2 and s['source_harm_gt5pp']==0 and np.isclose(s['mean'],.2)
