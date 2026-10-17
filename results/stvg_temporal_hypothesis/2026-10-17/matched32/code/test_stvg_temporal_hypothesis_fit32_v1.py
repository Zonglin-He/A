import unittest,numpy as np
from scripts import stvg_temporal_hypothesis_fit32_v1 as f
class Contracts(unittest.TestCase):
 def off(self):
  rng=np.random.default_rng(42);out=[]
  for k in range(2):
   n=7;ij=np.asarray(np.triu_indices(n,1));z=rng.normal(size=(n,2));out.append(dict(hidden=rng.normal(size=(n,256))*.03,ij=ij,base=z[ij[0],0]+z[ij[1],1],native_logits=z,ids=list(range(k,2*n+k,2))))
  return out
 def test_actual_two_observations_and_native(self):
  off=self.off();s=f.select(off,list(range(15)))
  self.assertEqual(s['Native_interval'],f.native.native_decode(off,np.zeros(512))[0]);self.assertEqual(len(s['hypotheses']),2);self.assertNotEqual(*[h['sampled_physical_frame_ids'] for h in s['hypotheses']])
 def test_equal_rewards_exact_noop(self):
  off=self.off();s=f.select(off,list(range(15)));z=f.fit(off,s['hypotheses'],[.2,.2]);self.assertEqual(z['optimizer_steps'],0);self.assertEqual(np.count_nonzero(z['theta']),0);f.audit(off,z)
 def test_gradient_independent_and_direction(self):
  off=self.off();s=f.select(off,list(range(15)));pre=f.prepare(off,s['hypotheses'],[-.7,.4]);self.assertGreater(pre['q'][1],f.original._prob(pre['candidate_base'])[0][1]);z=f.fit(off,s['hypotheses'],[-.7,.4]);self.assertGreater(z['optimizer_steps'],0);f.audit(off,z)
  theta=np.full(512,.003);_,g,_=f.loss_gradient(pre,theta,'x',f.CFG)
  for i in [0,57,233,511]:
   a=theta.copy();b=theta.copy();a[i]+=1e-5;b[i]-=1e-5;fd=(f.loss_gradient(pre,a,'x',f.CFG)[0]-f.loss_gradient(pre,b,'x',f.CFG)[0])/2e-5;self.assertLess(abs(fd-g[i]),1e-8)
if __name__=='__main__':unittest.main()
