import numpy as np
from test_temporal_profile import sample
from stvg.temporal_boundary_pairing import BASE,tables,combine,native_grids
from stvg.global6 import posterior

def test_zero_pairing_is_exact_original():
 ids,r,z=sample();assert np.array_equal(combine(tables(r,z,ids,BASE),ids,BASE),posterior(r,z,ids))
def test_two_event_bridge_loses_relative_to_single_pair():
 # Two plausible events at[0,2] and[5,7]; a cross-event[0,7] pairs unrelated endpoints.
 rec=[dict(frame_ids=list(range(9)))]*2;s=np.full(9,-5.);e=s.copy();s[[0,5]]=3.;e[[2,7]]=3.;z=np.stack([s,e],axis=1)[None].astype(np.float32);g0=native_grids(rec,[z,z],BASE)[0];g1=native_grids(rec,[z,z],{**BASE,'pair_strength':2.})[0];pairs=list(map(tuple,g0['ij'].T));bridge=pairs.index((0,7));single=pairs.index((0,2));assert g1['p'][bridge]/g1['p'][single]<g0['p'][bridge]/g0['p'][single]
def test_pairing_and_agreement_preserve_probability_and_legal_grid():
 ids,r,z=sample()
 for strength in [-1.,.5,2.,4.]:
  p={**BASE,'pair_strength':strength};q=combine(tables(r,z,ids,p),ids,p);assert abs(q.sum()-1)<1e-12 and not q[np.tril_indices(len(ids))].any()
