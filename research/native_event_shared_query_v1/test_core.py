import unittest
import torch
from research.native_event_shared_query_v1.core import expanded_like,shared_second_pass

class Decoder(torch.nn.Module):
    def forward(self,**kw):return kw

class SharedTests(unittest.TestCase):
    def test_same_values_and_expanded_stride(self):
        q=torch.arange(8).float()[None,None].expand(5,1,8);stored=q.clone();v=expanded_like(stored,q)
        self.assertTrue(torch.equal(q,v));self.assertEqual(q.stride(),v.stride())
    def test_only_second_pass_and_hook_removed(self):
        d=Decoder();full=[torch.full((5,1,8),2.),torch.full((4,1,8),3.)]
        with shared_second_pass(d,full)as calls:
            for i,n in enumerate([5,5,4,4]):
                q=torch.ones(8)[None,None].expand(n,1,8);x=d(query_tgt=q,other=q)
                self.assertTrue(torch.equal(x['other'],q));self.assertEqual(float(x['query_tgt'][0,0,0]),[1,2,1,3][i])
        self.assertEqual(len(calls),4);self.assertFalse(d._forward_pre_hooks)
    def test_cleanup_on_failure(self):
        d=Decoder()
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with shared_second_pass(d,[None,None]):raise RuntimeError('deliberate')
        self.assertFalse(d._forward_pre_hooks)

if __name__=='__main__':unittest.main()
