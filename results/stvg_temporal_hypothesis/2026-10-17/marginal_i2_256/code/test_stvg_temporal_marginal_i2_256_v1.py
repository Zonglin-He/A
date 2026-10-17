"""CPU contracts for the only changed ranking, not a model-effect test."""
import unittest,numpy as np
from scripts import stvg_temporal_hypothesis_fit32_v1 as f
from scripts.stvg_temporal_marginal_i2_select256_v1 import select
class Contracts(unittest.TestCase):
 def off(self):
  rng=np.random.default_rng(42);out=[]
  for k in range(2):
   n=7;ij=np.asarray(np.triu_indices(n,1));z=rng.normal(size=(n,2));out.append(dict(hidden=rng.normal(size=(n,256))*.03,ij=ij,base=z[ij[0],0]+z[ij[1],1],native_logits=z,ids=list(range(k,2*n+k,2))))
  return out
 def test_I1_exact_and_ranking_independent(self):
  off=self.off();new=select(off,list(range(15)));old=f.select(off,list(range(15)));self.assertEqual(new['hypotheses'][0],old['hypotheses'][0]);self.assertEqual(len(new['hypotheses']),2)
  from collections import defaultdict
  m=defaultdict(float)
  for i,pa in enumerate(f.original._prob(off[0]['base'])[0]):
   for j,pb in enumerate(f.original._prob(off[1]['base'])[0]):m[tuple(f.merge(off,[off[0]['ij'][:,i],off[1]['ij'][:,j]]))]+=pa*pb
  ordered=sorted(m,key=lambda iv:(-m[iv],iv));alt=next(iv for iv in ordered if f.sampled(iv,list(range(15)))is not None and f.sampled(iv,list(range(15)))!=new['hypotheses'][0]['sampled_physical_frame_ids']);self.assertEqual(list(alt),new['hypotheses'][1]['interval_frames']);self.assertEqual(list(ordered[0]),new['marginal_no_update_interval'])
 def test_original_fitter_and_no_information(self):
  off=self.off();s=select(off,list(range(15)));no=f.fit(off,s['hypotheses'],[.2,.2]);self.assertEqual(no['optimizer_steps'],0);self.assertEqual(np.count_nonzero(no['theta']),0);f.audit(off,no)
  yes=f.fit(off,s['hypotheses'],[-.7,.4]);self.assertGreater(yes['optimizer_steps'],0);f.audit(off,yes)
if __name__=='__main__':unittest.main()
