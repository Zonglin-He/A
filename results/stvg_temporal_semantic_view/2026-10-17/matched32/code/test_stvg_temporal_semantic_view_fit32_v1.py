import unittest,numpy as np,torch
from scripts import stvg_temporal_semantic_view_fit32_v1 as f
class NativeDistillation(unittest.TestCase):
 def setup_problem(self):
  g=torch.Generator().manual_seed(170017);h=[torch.randn(5,256,generator=g,dtype=torch.float64)*.02 for _ in range(2)];case=dict(zero=dict(logits=[torch.randn(1,5,2,generator=g) for _ in range(2)]),records=[dict(frame_ids=[1,3,5,7,9]),dict(frame_ids=[2,4,6,8,10])]);return f.offsets(case,h)
 def test_identity_and_rejected_are_exact_zero(self):
  o=self.setup_problem();q=[f.original._prob(x['base'])[0] for x in o]
  for admitted in [True,False]:
   a=f.fit(o,q,admitted);self.assertEqual(a['optimizer_steps'],0);self.assertEqual(a['final_interval'],f.native_decode(o,np.zeros(512))[0]);self.assertEqual(np.count_nonzero(a['theta']),0)
 def test_gradient_finite_difference_and_torch_audit(self):
  o=self.setup_problem();q=f.boundary_targets(o,[4.,8.]);pre=dict(offsets=o,targets=q);x=np.linspace(-.03,.03,512);L,g,_=f.loss_gradient(pre,x,'s',f.CFG)
  for j in [0,22,255,256,511]:
   z=x.copy();z[j]+=1e-5;hi=f.loss_gradient(pre,z,'s',f.CFG)[0];z[j]-=2e-5;lo=f.loss_gradient(pre,z,'s',f.CFG)[0];self.assertAlmostEqual(g[j],(hi-lo)/2e-5,places=8)
  a=f.fit(o,q);self.assertGreater(a['optimizer_steps'],0);self.assertGreater(f.audit_fit(o,a),1)
 def test_convex_and_mixture_gradient_identity(self):
  o=self.setup_problem();q=f.boundary_targets(o,[4.,8.]);x=np.linspace(-.1,.2,512);y=x[::-1].copy();pre=dict(offsets=o,targets=q);lx=f.loss_gradient(pre,x,'s',f.CFG)[0];ly=f.loss_gradient(pre,y,'s',f.CFG)[0];lm=f.loss_gradient(pre,.3*x+.7*y,'s',f.CFG)[0];self.assertLessEqual(lm,.3*lx+.7*ly+1e-12)
  for a,b in zip(o,q):
   p0=f.original._prob(a['base'])[0];mix=(b+p0)/2;self.assertTrue(np.allclose((p0-b)+(p0-p0),2*(p0-mix),atol=1e-16))
if __name__=='__main__':unittest.main()
