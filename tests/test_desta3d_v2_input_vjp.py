import unittest
import torch
from vg_tta.desta3d_v2_input_vjp import postcast_leaf, transport_summary
from vg_tta.desta3d_v2_cast_probe import cast_difference


class InputVJPTests(unittest.TestCase):
    def test_post_injection_leaf_preserves_forward_and_scope(self):
        torch.manual_seed(13)
        merger = torch.nn.Linear(3,4).to(torch.bfloat16).requires_grad_(False)
        head = torch.nn.Linear(4,5).to(torch.bfloat16).requires_grad_(False)
        x = torch.randn(2,3).to(torch.bfloat16); labels = torch.tensor([1,2])
        h = merger.register_forward_hook(lambda m,a,y: (y.float()+.0001).to(y.dtype))
        original = merger(x); expected = torch.nn.functional.cross_entropy(head(original).float(),labels)
        with postcast_leaf(merger) as capture:
            observed = merger(x)
            loss = torch.nn.functional.cross_entropy(head(observed).float(),labels)
            g = torch.autograd.grad(loss,capture['leaf'])[0]
        h.remove()
        self.assertTrue(torch.equal(original,observed))
        self.assertTrue(torch.equal(expected,loss))
        reference = original.detach().requires_grad_(True)
        control = torch.autograd.grad(torch.nn.functional.cross_entropy(head(reference).float(),labels),reference)[0]
        self.assertTrue(torch.equal(g,control))
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in [merger,head] for p in m.parameters()))
        self.assertEqual(len(merger._forward_hooks),0)

    def test_sparse_dot_matches_dense_control_and_support_guard(self):
        a = torch.tensor([1.,1.0038,-1.,0.]); b = a+torch.tensor([.0001,.0002,-.0001,0.])
        difference = cast_difference(a,a.bfloat16(),b,b.bfloat16())
        g = torch.tensor([1.,2.,3.,4.]).bfloat16()
        result = transport_summary(g,difference)
        expected = g.double().dot(b.bfloat16().double()-a.bfloat16().double())
        self.assertEqual(result['dot_postcast_delta'],float(expected))
        self.assertEqual(result['dot_precast_delta'],float(g.double().dot((b-a).double())))
        with self.assertRaises(AssertionError): transport_summary(g[:2],difference)


if __name__ == '__main__': unittest.main()
