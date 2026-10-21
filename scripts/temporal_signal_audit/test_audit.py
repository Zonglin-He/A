"""Small semantic tests for tied ranking, physical-time masks and attention ROI."""
import unittest,numpy as np,torch
from scripts.temporal_signal_audit.evaluate import auc,aligned,bh
from scripts.temporal_signal_audit.extract import cosine_change
from stvg.tube_replay import occupancy_prior
class AuditTests(unittest.TestCase):
 def test_auc(self):
  y=[0,0,1,1];self.assertEqual(auc(y,[1,1,1,1]),.5);self.assertEqual(auc(y,[0,1,2,3]),1);self.assertEqual(auc(y,[3,2,1,0]),0);self.assertIsNone(auc([0,1],[0,1]))
 def test_alignment(self):
  y=np.r_[np.zeros(10),np.ones(10)];r=aligned(y,y);self.assertEqual(r['auc'],1);self.assertGreater(r['alignment_delta'],.4)
 def test_uniform_attention(self):
  legal=torch.ones(4,70,dtype=torch.bool);b=torch.tensor([[.5,.5,.2,.3]]*4);m,_=occupancy_prior(b,(7,10),legal);p,_=occupancy_prior(b,(7,10),legal,permuted=True)
  self.assertTrue(torch.allclose(m.sum(-1),p.sum(-1)));self.assertTrue(torch.allclose((m/70).sum(-1),(p/70).sum(-1)));self.assertTrue(torch.allclose((m/70).sum(-1),m.mean(-1)))
 def test_no_cross_offset_artifact(self):
  v=np.zeros((9,2));v[::2,0]=1;v[1::2,1]=1;self.assertTrue(np.allclose(cosine_change(v),0))
 def test_halfopen_and_mask(self):
  t=np.array([0.,1.,2.,3.]);inside=(t>=1)&(t<3);self.assertEqual(inside.tolist(),[False,True,True,False]);obs=np.array([0,1,0,1],bool);self.assertEqual((inside&~obs).sum(),1)
 def test_bh(self):self.assertEqual(bh({'a':.01,'b':.04}),{'b':.04,'a':.02})
if __name__=='__main__':unittest.main()
