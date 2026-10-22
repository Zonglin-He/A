import unittest,torch
from research.native_event_role_context_v1.core import quota_mask,keymask,contexts
class Checks(unittest.TestCase):
 def test_quota(self):
  v=torch.ones(3,4,dtype=torch.bool);t=v.clone();t[:,2:]=False;p=torch.arange(12).reshape(3,4)
  x=quota_mask(t,p,3,v);self.assertEqual(int((x&~t).sum()),3);self.assertTrue(bool(x[t].all()))
 def test_text_mask(self):
  a=torch.zeros(3,16,dtype=torch.bool);k=torch.ones(3,12,dtype=torch.bool);self.assertIs(keymask(a,k),a);k[:,0]=False;z=keymask(a,k);self.assertTrue(torch.equal(z[:,:4],a[:,:4]));self.assertTrue(bool(z[:,4].all()));self.assertFalse(bool(a.any()))
 def test_extra_and_parity(self):
  b=torch.tensor([[.3,.5,.2,.4]]*7);c=b.clone();c[:,0]=.8;v=torch.ones(3,5,7,dtype=torch.bool);m,s=contexts(b,c,b,1,v)
  self.assertEqual([x['frame_index']for x in s],[1,3,5]);self.assertTrue(all(torch.equal(m[k].sum(1),m['Pair'].sum(1))for k in m if k!='TargetOnly'))
if __name__=='__main__':unittest.main()
