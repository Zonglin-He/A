import unittest,torch
from research.native_event_support_view_v1.core import keep_region,inside_fill,reflected
class Tests(unittest.TestCase):
 def test_inside_padding_and_corner(self):
  valid=torch.ones(1,7,10,dtype=torch.bool);valid[:,6,:]=False;x=torch.randn(1,4,7,10).half();keep=torch.zeros_like(valid);keep[:,2:5,4:7]=True;y=inside_fill(x,keep,valid);self.assertTrue(torch.equal(y.permute(0,2,3,1)[keep|~valid],x.permute(0,2,3,1)[keep|~valid]));mean=x.float()[:,:,2:5,4:7].mean((2,3)).half();self.assertTrue(torch.equal(y[:,:,0,0],mean));self.assertFalse(bool(keep[0,0,0]))
 def test_allkeep_and_fallback(self):
  valid=torch.ones((7,10),dtype=torch.bool);keep,fallback=keep_region(torch.tensor([3.,3.,.1,.1]),valid);self.assertTrue(fallback);self.assertTrue(torch.equal(keep,valid));x=torch.randn(1,4,7,10).half();self.assertTrue(torch.equal(inside_fill(x,keep[None],valid[None]),x))
 def test_reflected_initial_counts(self):
  valid=torch.ones((7,10),dtype=torch.bool);valid[:,9]=False;k,f=keep_region(torch.tensor([.3,.4,.2,.3]),valid);c=reflected(k,valid);self.assertEqual(int(k.sum()),int(c.sum()));a=k.clone();b=c.clone();a[0,0]=b[0,0]=True;self.assertLessEqual(abs(int(a.sum())-int(b.sum())),1)
if __name__=='__main__':unittest.main()
