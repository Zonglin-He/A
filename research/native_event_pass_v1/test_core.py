import unittest,torch
from research.native_event_pass_v1.core import collect_passes,split_passes
class Decoder(torch.nn.Module):
 def forward(self,x):return x,x
class Tests(unittest.TestCase):
 def test_observer_and_call_order(self):
  m=Decoder();xs=[torch.full((6,1,n,256),float(k))for k,n in enumerate([33,33,32,32])]
  with collect_passes(m)as values:
   for x in xs:
    a,b=m(x);self.assertIs(a,x);self.assertIs(b,x)
  self.assertFalse(m._forward_hooks);first,second=split_passes(values)
  self.assertTrue(torch.equal(first[0],xs[0])and torch.equal(first[1],xs[2])and torch.equal(second[0],xs[1])and torch.equal(second[1],xs[3]));values[0].add_(1);self.assertEqual(float(xs[0].sum()),0.)
 def test_cleanup_on_failure(self):
  m=Decoder()
  with self.assertRaisesRegex(RuntimeError,'deliberate'):
   with collect_passes(m):raise RuntimeError('deliberate')
  self.assertFalse(m._forward_hooks)
if __name__=='__main__':unittest.main()
