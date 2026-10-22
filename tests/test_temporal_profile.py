import numpy as np
from stvg.temporal_profile import BASE,combine,tables
from stvg.global6 import posterior

def sample():
 ids=[0,2,4,6,8,10];records=[dict(frame_ids=ids[i::2])for i in [0,1]];rng=np.random.default_rng(41);logits=[rng.normal(size=(6,1,3,2)).astype(np.float32)for _ in [0,1]];return ids,records,logits

def test_default_profile_is_bitwise_original():
 ids,r,z=sample();q=combine(tables(r,z,ids),ids,BASE);assert np.array_equal(q,posterior(r,z,ids))
def test_positive_slope_weights_late_layers():
 ids,r,z=sample();t=tables(r,z,ids);w=np.exp(2*np.arange(6)/5);w/=w.sum();assert np.array_equal(combine(t,ids,{**BASE,'layer_slope':2}),np.sum(t*w[:,None,None],axis=0))
def test_length_tilt_preserves_order_and_shifts_mass_shorter():
 ids,r,z=sample();t=tables(r,z,ids);a=combine(t,ids,BASE);b=combine(t,ids,{**BASE,'length_power':.5});L=np.array(ids)[None]+1-np.array(ids)[:,None];assert (b*L).sum()<(a*L).sum();assert abs(b.sum()-1)<1e-12;assert not b[np.tril_indices(len(ids))].any()
