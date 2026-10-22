import unittest,torch
from research.native_event_caption_restore_v1.core import weights,restore,inside_fill,observe,caption_positions
class Tests(unittest.TestCase):
 def test_equal_mass_multiset_and_padding(self):
  torch.manual_seed(42);v=torch.ones(5,3,4,dtype=torch.bool);v[:,:,3]=False;k=torch.zeros_like(v);k[:,1,1]=True
  a=torch.rand(5,3,4);ws,stats=weights(a,k,v);x=torch.randn(5,2,3,4).half();d=inside_fill(x,k,v)
  for w in ws.values():
   z=restore(x,d,w,k,v);self.assertTrue(torch.equal(z.permute(0,2,3,1)[k|~v],x.permute(0,2,3,1)[k|~v]))
   self.assertTrue(torch.all(w[k]==1));self.assertTrue(torch.all(w[~v]==0))
  self.assertTrue(torch.equal(restore(x,d,torch.ones_like(ws['Caption']),k,v),x))
 def test_zero_attention(self):
  v=torch.ones(2,3,4,dtype=torch.bool);k=v.clone();k[:,0,:]=False;w,s=weights(torch.zeros(2,3,4),k,v)
  self.assertTrue(all(x['zero_fallback']for x in s));self.assertTrue(torch.all(w['Caption']==1))
 def test_observer_noop_and_cleanup(self):
  m=torch.nn.MultiheadAttention(8,2).eval();x=torch.randn(7,2,8);base=m(x,x,x);out=[]
  with observe(m,[0,1],2,3,out):z=m(x,x,x)
  self.assertTrue(torch.equal(z[0],base[0]));self.assertEqual(out[0].shape,(2,2));self.assertFalse(m._forward_hooks)
  with self.assertRaises(RuntimeError):
   with observe(m,[0],2,3,[]):raise RuntimeError('intentional')
  self.assertFalse(m._forward_hooks)
 def test_caption_offset_and_original_ids(self):
  class Tok:
   def __call__(self,*a,**kw):return dict(input_ids=torch.tensor([[0,5,6,7,2]]),offset_mapping=torch.tensor([[[0,0],[0,3],[4,7],[8,12],[0,0]]]))
  p,n=caption_positions(Tok(),dict(subject='man',caption='man runs',full_input_ids=torch.tensor([[0,5,6,7,2]])))
  self.assertEqual(p,[2,3]);self.assertEqual(n,5)
if __name__=='__main__':unittest.main()
