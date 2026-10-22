import unittest,numpy as np,torch
from research.native_event_text_v1.core import posteriors,masked_inputs
class Controls(unittest.TestCase):
 def test_equal_interventions_no_effect(self):
  q=np.triu(np.ones((5,5)),1);q=q/q.sum();e=q.copy();e[0,2]*=2;e/=e.sum();r,_=posteriors(q,e,e,q)
  np.testing.assert_allclose(r['Selective'],q,atol=1e-15)
 def test_mask_preserves_nonselected(self):
  a=torch.arange(10)[None];b=masked_inputs(a,[3,5],99);self.assertEqual(b.shape,a.shape);self.assertTrue(torch.equal(a[:,[0,1,2,4,6,7,8,9]],b[:,[0,1,2,4,6,7,8,9]]));self.assertEqual(a[0,3],3)
 def test_uniform_identity_is_sharpen(self):
  q=np.triu(np.arange(25).reshape(5,5),1).astype(float);q/=q.sum();u=(q>0)/(q>0).sum();r,_=posteriors(q,u,u,u);np.testing.assert_allclose(r['IdentityContrast'],r['Sharpen'],atol=1e-15)
if __name__=='__main__':unittest.main()
