"""Exact merged MAP versus independent enumeration/CDF and deterministic tie contracts."""
import unittest,numpy as np
from scripts import stvg_temporal_marginal_decode_util256_v1 as f
class Contracts(unittest.TestCase):
 def test_two_offset_mass_and_cdf(self):
  for overlap in [False,True]:
   rng=np.random.default_rng(19);records=[dict(frame_ids=list(range(k if not overlap else 0,12+(k if not overlap else 0),2)))for k in [0,1]];zs=[rng.normal(size=(1,6,2)).astype(np.float32)for _ in records];iv,v,ids,M,gs=f.decode(records,zs);D=np.zeros_like(M);idx={v:i for i,v in enumerate(ids)}
   for (a,b),p in zip(gs[0]['ij'].T,gs[0]['p']):
    for (c,d),q in zip(gs[1]['ij'].T,gs[1]['p']):D[idx[min(gs[0]['ids'][a],gs[1]['ids'][c])],idx[max(gs[0]['ids'][b],gs[1]['ids'][d])]]+=p*q
   np.testing.assert_array_equal(M,D);np.testing.assert_allclose(M,f.independent_cdf(gs,ids),atol=1e-14,rtol=0);self.assertEqual(iv,f.mode(ids,D)[0])
 def test_zero_change_exact(self):
  r=[dict(frame_ids=[0,2,4]),dict(frame_ids=[1,3,5])];z=[np.arange(6,dtype=np.float32).reshape(1,3,2)*.03]*2;a=f.decode(r,z);b=f.decode(r,[v.copy()for v in z]);self.assertEqual(a[:2],b[:2]);np.testing.assert_array_equal(a[3],b[3])
 def test_tie_and_no_clamp(self):
  M=np.zeros((4,4));M[0,2]=M[1,3]=.5;self.assertEqual(f.mode([0,1,2,3],M)[0],[0,3]);r=[dict(frame_ids=[0,2,4])]*2
  with self.assertRaises(AssertionError):f.decode(r,[np.full((1,3,2),np.nan,np.float32)]*2)
if __name__=='__main__':unittest.main()
