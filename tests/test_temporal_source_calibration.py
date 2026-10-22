import numpy as np
from test_temporal_profile import sample
from stvg.temporal_source_calibration import BASE,tables,combine,decode,pair_geometry
from stvg.global6 import posterior
from stvg.temporal import final_mbr

def test_default_source_profile_is_exact_global6():
 ids,r,z=sample();q=combine(tables(r,z,ids,BASE),ids,BASE,.01);assert np.array_equal(q,posterior(r,z,ids))
def test_source_only_is_exact_last_layer_time():
 ids,r,z=sample();p={**BASE,'source_mix':1.};a=tables(r,z,ids,p);q=combine(a,ids,p,.01);assert np.array_equal(q,a[1,-1]);assert decode(ids,q,p)['interval']==final_mbr(ids,a[1,-1])['interval']
def test_agreement_and_tail_preserve_support_and_mass():
 ids,r,z=sample()
 for change in [{'offset_agreement':2.},{'nucleus_mass':.9},{'adaptive_start':1.},{'pool_power':0.},{'source_trust':1.}]:
  p={**BASE,**change};q=combine(tables(r,z,ids,p),ids,p,.01);assert np.isfinite(q).all()and abs(q.sum()-1)<1e-12 and not q[np.tril_indices(len(ids))].any()
def test_agreement_pairwise_sum_matches_enumeration():
 from stvg.temporal import probabilities
 ids,r,z=sample();p={**BASE,'offset_agreement':2.};a=tables(r,z,ids,p)[0,0];g=probabilities(r,[v[0]for v in z]);m=np.zeros_like(a);lookup={f:i for i,f in enumerate(ids)}
 for(a0,b0),pa in zip(g[0]['ij'].T,g[0]['p']):
  for(a1,b1),pb in zip(g[1]['ij'].T,g[1]['p']):
   s0,e0=r[0]['frame_ids'][a0],r[0]['frame_ids'][b0]+1;s1,e1=r[1]['frame_ids'][a1],r[1]['frame_ids'][b1]+1;iou=max(0,min(e0,e1)-max(s0,s1))/(max(e0,e1)-min(s0,s1));m[lookup[min(s0,s1)],lookup[max(e0,e1)-1]]+=pa*pb*np.exp(2*iou)
 m/=m.sum();assert np.allclose(a,m,rtol=0,atol=1e-14)
