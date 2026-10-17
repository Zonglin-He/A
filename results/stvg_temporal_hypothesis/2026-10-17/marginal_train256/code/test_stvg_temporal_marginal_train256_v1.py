"""Independent mass, gradient, target, no-information, and original-config tests."""
import unittest,numpy as np
from scripts import stvg_temporal_marginal_train_fit256_v1 as f,stvg_temporal_hypothesis_fit32_v1 as old
class Contracts(unittest.TestCase):
 def setup_case(self,overlap=False):
  rng=np.random.default_rng(42);out=[]
  for k in range(2):
   n=7;ij=np.asarray(np.triu_indices(n,1));z=rng.normal(size=(n,2));out.append(dict(hidden=rng.normal(size=(n,256))*.03,ij=ij,base=z[ij[0],0]+z[ij[1],1],native_logits=z,ids=list(range(0 if overlap else k,2*n+(0 if overlap else k),2))))
  ids=list(range(15));selection=old.select(out,ids);return out,selection['hypotheses']
 def test_mass_and_gradient_disjoint_and_shared_endpoints(self):
  for overlap in [False,True]:
   off,hs=self.setup_case(overlap);pre=f.prepare(off,hs,[-.3,.6]);pairs=f.brute_pairs(off,hs);theta=np.full((256,2),.003);L,g,lw=f.independent_torch(off,theta,pre['q'],pairs);a,b,_=f.loss_gradient(pre,theta.ravel(),'x',f.CFG);self.assertLess(abs(a-L),1e-11);self.assertLess(np.max(abs(b-g.ravel())),1e-10)
   for i in [0,57,233,511]:
    plus=theta.ravel().copy();minus=plus.copy();plus[i]+=1e-5;minus[i]-=1e-5;fd=(f.loss_gradient(pre,plus,'x',f.CFG)[0]-f.loss_gradient(pre,minus,'x',f.CFG)[0])/2e-5;self.assertLess(abs(fd-b[i]),1e-8)
   base=np.array([f.event_logweight(off,[o['base']for o in off],e,False)[0]for e in pre['events']]);zero=f.independent_torch(off,np.zeros((256,2)),pre['q'],pairs)[2];self.assertLess(np.max(abs(base-zero)),1e-12)
 def test_targets_and_equal_reward_exact_noop(self):
  off,hs=self.setup_case();pre=f.prepare(off,hs,[-.3,.6]);self.assertGreater(pre['q'][1],pre['Native_conditional'][1]);z=f.fit(off,hs,[.2,.2]);self.assertEqual(z['optimizer_steps'],0);self.assertEqual(np.count_nonzero(z['theta']),0);self.assertEqual(z['final_interval'],f.native.native_decode(off,np.zeros(512))[0]);f.audit(off,z)
 def test_original_optimizer_and_config(self):
  self.assertEqual(f.CFG,old.CFG);off,hs=self.setup_case();z=f.fit(off,hs,[-.7,.4]);self.assertGreater(z['optimizer_steps'],0);f.audit(off,z)
if __name__=='__main__':unittest.main()
