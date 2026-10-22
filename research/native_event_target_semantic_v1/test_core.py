import unittest,torch
from types import SimpleNamespace
from research.native_event_target_semantic_v1.core import *
class Tests(unittest.TestCase):
 def test_masked_softmax(self):
  z=torch.arange(24,dtype=torch.float64).reshape(2,2,1,6)/10;p=z.softmax(-1);keep=torch.tensor([[1,0,1,0,1,0],[0,1,0,1,0,1]],dtype=torch.bool);out=restrict_prob(p,keep);want=z.masked_fill(~keep[:,None,None,:],-torch.inf).softmax(-1);torch.testing.assert_close(out,want,atol=1e-15,rtol=1e-15)
 def test_allkeys_exact_identity(self):
  p=torch.randn(2,3,1,7).softmax(-1);self.assertIs(restrict_prob(p,torch.ones(2,7,dtype=torch.bool)),p)
 def test_nonempty_geometry_and_reflection(self):
  valid=torch.ones(7,10,dtype=torch.bool);m,fb=keep_region(torch.tensor([.2,.3,.1,.1]),valid);c=reflected(m,valid);self.assertEqual(int(m.sum()),int(c.sum()));self.assertFalse(fb)
  m,fb=keep_region(torch.tensor([2.,2.,.01,.01]),valid);self.assertTrue(fb);self.assertTrue(torch.equal(m,valid))
 def test_time_index(self):
  aux=torch.tensor([[.1+.01*i,.3,.05,.05]for i in range(9)]);valid=torch.ones(5,7,10,dtype=torch.bool);_,s=masks(aux,0,5,7,10,valid,'WrongTimeContext');self.assertEqual([x['source_index']for x in s],[8,6,4,2,0])
 def test_hook_cleanup_on_exception(self):
  drop=torch.nn.Dropout(0).eval();layer=SimpleNamespace(attention=SimpleNamespace(self=SimpleNamespace(dropout=drop)));head=SimpleNamespace(layer_ca=[layer]);keep=torch.ones(2,7,dtype=torch.bool)
  with self.assertRaises(RuntimeError):
   with support_keys(head,keep):
    p=torch.randn(2,3,1,7).softmax(-1);self.assertIs(drop(p),p);raise RuntimeError('intentional cleanup test')
  self.assertFalse(drop._forward_pre_hooks)
if __name__=='__main__':unittest.main()
