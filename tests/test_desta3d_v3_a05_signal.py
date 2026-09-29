import unittest
import torch
from vg_tta.desta3d_v3_a05_signal import signal_loss,balanced_direction,select_dev16,passes
class SignalContract(unittest.TestCase):
    def test_full_vocab_and_detached_teacher(self):
        torch.manual_seed(1);t=torch.randn(2,4,152775,requires_grad=True);x=t.detach().clone().requires_grad_()
        k=signal_loss(x,t,'U-Consistency');self.assertEqual(float(k),0.)
        k.backward();self.assertIsNone(t.grad);self.assertLess(float(x.grad.abs().max()),8*torch.finfo(torch.float32).eps*float(x.detach().softmax(-1).max()))
        x=torch.randn(2,4,152775,requires_grad=True);h=signal_loss(x,t,'U-Entropy')
        log=x.double().log_softmax(-1);expected=-(log.exp()*log).sum(-1).mean()
        self.assertLess(abs(float(h)-float(expected)),5e-6)
        h.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.norm()),0)
        with self.assertRaises(ValueError):signal_loss(x,t[:,:,:1001],'U-Consistency')
    def test_zero_leaf_chain_scope(self):
        torch.manual_seed(2);q=torch.linalg.qr(torch.randn(24,4).double()).Q.float()
        c=torch.zeros(1,2,2,2,4,requires_grad=True);f=torch.randn(1,2,2,2,24)
        w=torch.randn(24,13);new=f+c@q.T;self.assertTrue(torch.equal(new,f))
        logits=(new@w).flatten(0,-2);teacher=logits.detach().clone()
        loss=signal_loss(logits,teacher,'U-Entropy');gc,gf=torch.autograd.grad(loss,(c,new))
        self.assertTrue(torch.allclose(gc,gf@q,atol=1e-8,rtol=1e-6));self.assertIsNone(w.grad)
    def test_balance_selection_and_gates(self):
        g=torch.tensor([3.,4.]);z=torch.zeros(2)
        self.assertTrue(torch.allclose(balanced_direction(g,z),-g.double()/5))
        self.assertTrue(torch.equal(balanced_direction(z,z),z.double()))
        self.assertTrue(torch.equal(balanced_direction(g,-g),z.double()))
        rows=[dict(source=p,key=f'{p}-{i}') for p in range(16) for i in range(4)]
        a=select_dev16(rows);self.assertEqual(len(a),16);self.assertEqual(len({r['source'] for r in a}),16)
        self.assertEqual([r['key'] for r in a],[r['key'] for r in select_dev16(list(reversed(rows)))])
        self.assertTrue(passes(.1,11,16,11,16));self.assertFalse(passes(.099,16,16,16,16));self.assertFalse(passes(.2,10,16,16,16));self.assertFalse(passes(.2,0,0,16,16))
if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
