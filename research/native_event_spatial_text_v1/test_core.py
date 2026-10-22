import unittest,torch
from research.native_event_spatial_text_v1.core import footprint,reflected,replace
class Tests(unittest.TestCase):
 def test_coverage_reflection(self):
  valid=torch.ones((7,10),dtype=torch.bool);box=torch.tensor([.3,.4,.2,.3]);m=footprint(box,7,10,valid);self.assertAlmostEqual(float(m.sum()),.2*.3*70,places=5);r=reflected(m,valid);self.assertAlmostEqual(float(r.sum()),float(m.sum()),places=10)
 def test_padding_and_zero(self):
  valid=torch.ones((1,7,10),dtype=torch.bool);valid[:,6,:]=False;valid[:,:,9]=False;x=torch.arange(3*7*10).reshape(1,3,7,10).half();mask=torch.zeros_like(valid,dtype=torch.float64);z,n=replace(x,mask,valid);self.assertTrue(torch.equal(z,x));mask[0]=footprint(torch.tensor([.2,.3,.3,.2]),7,10,valid[0]);z,n=replace(x,mask,valid);self.assertTrue(torch.equal(z.permute(0,2,3,1)[~valid],x.permute(0,2,3,1)[~valid]));self.assertEqual(n,0)
 def test_full_coverage_fallback(self):
  x=torch.randn(1,3,2,2).half();valid=torch.ones(1,2,2,dtype=torch.bool);out,n=replace(x,valid.double(),valid);self.assertEqual(n,1);self.assertTrue(torch.equal(out,x.float().mean((2,3),keepdim=True).expand_as(x).half()))
if __name__=='__main__':unittest.main()
