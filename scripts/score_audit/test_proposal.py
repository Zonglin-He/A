import unittest
import torch
from scripts.score_audit.proposal import retained_teacher,build
class ProposalTests(unittest.TestCase):
 def test_seven_selection(self):
  l=torch.zeros(7,2,1);l[:,1]=2;b=torch.ones(7,2,4)*.5;m=torch.zeros(7,2,2,2)
  q,_,s=retained_teacher(l,b,m,(2,2),(2,2),torch.ones(7,4,dtype=torch.bool))
  self.assertEqual(q,1);self.assertTrue(torch.equal(s,torch.full_like(s,.5)))
 def test_source_fallback_and_no_hidden(self):
  torch.manual_seed(7);f=torch.randn(9,4,256);v=torch.ones(9,4,dtype=torch.bool)
  S=torch.tensor([[.5,.5,.6,.6]],dtype=torch.float64).repeat(9,1);ids=torch.arange(9)
  b=build(S,ids,f,v,list(range(7)),torch.zeros(7,2,2),S[:7],torch.ones(9,dtype=torch.bool),(2,2))
  self.assertTrue(torch.equal(b['boxes'][:,0],S));self.assertFalse(b['active'].any())
  self.assertTrue(torch.equal(b['valid'].sum(-1),torch.ones(9,dtype=torch.long)))
 def test_requires_seven(self):
  with self.assertRaises(AssertionError):retained_teacher(torch.zeros(8,2,1),torch.ones(8,2,4),torch.zeros(8,2,2,2),(2,2),(2,2),torch.ones(8,4,dtype=torch.bool))
if __name__=='__main__':unittest.main()
