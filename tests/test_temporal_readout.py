import numpy as np
from stvg.temporal_readout import BASE,tables,combine,posterior,decode
from stvg.global6 import posterior as original
from stvg.temporal import final_mbr
from test_temporal_profile import sample

def test_default_is_exact_original():
    ids,r,z=sample();q=posterior(r,z,ids,BASE);assert np.array_equal(q,original(r,z,ids));assert decode(ids,q,BASE)['interval']==final_mbr(ids,q)['interval']
def test_mixture_uses_exact_same_physical_support():
    ids,r,z=sample();p={**BASE,'hull_weight':0.};t=tables(r,z,ids,p);q=combine(t,ids,p)
    assert abs(q.sum()-1)<1e-12 and not q[np.tril_indices(len(ids))].any()
    # Each native offset contributes exactly half the mass; no coordinate interpolation.
    a=np.arange(0,len(ids),2);assert abs(q[np.ix_(a,a)].sum()-.5)<1e-12
    assert np.allclose(posterior(r,z,ids,{**BASE,'hull_weight':.5}),.5*(q+original(r,z,ids)))
def test_start_and_end_calibration_are_distinct():
    ids,r,z=sample();a=posterior(r,z,ids,{**BASE,'start_temperature':.75});b=posterior(r,z,ids,{**BASE,'end_temperature':.75});assert not np.allclose(a,b)
def test_last_layer_count_is_direct_mean():
    ids,r,z=sample();t=tables(r,z,ids,BASE);assert np.array_equal(combine(t,ids,{**BASE,'last_layers':3}),t[-3:].mean(0))
def test_blocked_generalized_risk_matches_dense_sum():
    ids,r,z=sample();q=posterior(r,z,ids,BASE)
    for power in [.5,2.]:
        v=decode(ids,q,{**BASE,'risk_power':power});truth=np.zeros_like(q)
        for a in range(len(ids)-1):
            for b in range(a+1,len(ids)):
                total=0.
                for i in range(len(ids)-1):
                    for j in range(i+1,len(ids)):
                        inter=max(0,min(ids[b]+1,ids[j]+1)-max(ids[a],ids[i]));union=max(ids[b]+1,ids[j]+1)-min(ids[a],ids[i]);total+=q[i,j]*(inter/union)**power
                truth[a,b]=total
        legal=np.triu_indices(len(ids),1);assert np.max(abs(v['utility_grid'][legal]-truth[legal]))<1e-12
