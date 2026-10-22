import unittest,torch
from research.native_event_modality_context_v1.core import focus,keep_region,reflected
class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(1);self.a=torch.randn(5,4,7,10).half();self.m=torch.randn_like(self.a);self.v=torch.ones(5,7,10,dtype=torch.bool);self.v[:,:,-1]=False;self.k=torch.zeros_like(self.v);self.k[:,2:5,2:6]=True
 def test_modality_preservation(self):
  for kind in ['Full','MotionTarget','MotionReflected','AppearanceTarget','AppearanceReflected','WrongTimeMotion']:
   a,m=focus(self.a,self.m,self.k,self.v,kind)
   if not kind.startswith('Appearance'):self.assertTrue(torch.equal(a,self.a))
   if kind.startswith('Appearance')or kind=='Full':self.assertTrue(torch.equal(m,self.m))
 def test_inside_and_padding(self):
  for kind in ['MotionTarget','AppearanceTarget','WrongTimeMotion']:
   a,m=focus(self.a,self.m,self.k,self.v,kind)
   for x,y in [(a,self.a),(m,self.m)]:self.assertTrue(torch.equal(x.permute(0,2,3,1)[~self.v],y.permute(0,2,3,1)[~self.v]))
   original=self.m.flip(0)if kind=='WrongTimeMotion'else self.m
   self.assertTrue(torch.equal(m.permute(0,2,3,1)[self.k],original.permute(0,2,3,1)[self.k]))
 def test_reflection_and_fallback(self):
  for v in self.v:
   k,f=keep_region(torch.tensor([.3,.5,.2,.4]),v);c=reflected(k,v);self.assertEqual(int(k.sum()),int(c.sum()))
   k,f=keep_region(torch.tensor([3.,3.,.1,.1]),v);self.assertTrue(f);self.assertTrue(torch.equal(k,v))
 def test_temporal_index_roundtrip(self):
  ids=torch.arange(65);o=[ids[0::2],ids[1::2]];back=[x.flip(0)for x in o];out=torch.empty_like(ids);out[0::2]=back[0];out[1::2]=back[1];self.assertTrue(torch.equal(out,ids.flip(0)))
  self.assertTrue(torch.equal(self.m.flip(0).flip(0),self.m))
if __name__=='__main__':unittest.main()
