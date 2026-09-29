import unittest
import torch
from vg_tta.desta3d_v3_a04_factorized import FactorizedDirectionMixer,route
from vg_tta.desta3d_v3_a01_triage import FitabilityDirectionMixer
from vg_tta.desta3d_v3_a0_screen import coefficients
from vg_tta.desta3d_v3_gap_candidates import coefficient_direction_loss
from vg_tta.optimizer_checkpoint import cpu_clone,restore_optimizer

class A04Contract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.u=torch.linalg.qr(torch.randn(2560,256)).Q
        self.b=torch.linalg.qr(torch.randn(256,16).double()).Q
    def test_scope_and_local_initialization(self):
        torch.manual_seed(20260928);old=FitabilityDirectionMixer(self.u)
        torch.manual_seed(20260928);m=FactorizedDirectionMixer(self.u,self.b)
        for k,v in m.state_dict().items():
            if k.startswith(('input.','local.','mix.')):self.assertTrue(torch.equal(v,old.state_dict()[k]))
        self.assertEqual(sum(p.numel() for p in m.parameters()),76688)
        c={k:torch.randn(1,3,2,2,n,requires_grad=True) for k,n in [('z',128),('evidence8',8)]}
        c.update(qT=torch.randn(1,128,requires_grad=True),qS=torch.randn(1,128,requires_grad=True),state33=torch.randn(1,3,33,requires_grad=True))
        p=m(c);self.assertEqual(p.shape,(1,3,2,2,16))
        h=torch.nn.functional.silu(m.input(torch.cat((c['z'],c['qT'][:,None,None,None].expand(1,3,2,2,128),c['qS'][:,None,None,None].expand(1,3,2,2,128),c['evidence8'],c['state33'][:,:,None,None].expand(1,3,2,2,33)),-1)))
        expected=m.output(torch.nn.functional.silu(m.mix(h+torch.nn.functional.silu(m.local(h.movedim(-1,1)).movedim(1,-1)))))
        self.assertTrue(torch.equal(p,expected))
        coefficient_direction_loss(p,torch.randn_like(p))['cosine'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertTrue(all(v.grad is None for v in c.values()))
        self.assertTrue(all(not x.requires_grad for x in m.buffers()))
    def test_projection_cosine_and_budget(self):
        a=torch.randn(27,256).double();p=torch.randn(27,16).double();t=a@self.b
        cos=lambda x,y:(x*y).sum()/x.norm()/y.norm()
        full=p@self.b.T
        self.assertAlmostEqual(float(cos(full,a)),float(cos(p,t)*t.norm()/a.norm()),places=13)
        d=full@self.u.double().T;cap=.13545580427763146*5;d=d*cap/d.norm()
        self.assertAlmostEqual(float(d.norm()),cap,places=13)
    def test_continuation_and_routes(self):
        torch.manual_seed(5);m=torch.nn.Linear(5,2);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
        x=torch.randn(3,5);y=torch.randn(3,2)
        for _ in range(2):o.zero_grad();(m(x)-y).square().sum().backward();o.step()
        state=cpu_clone(o.state_dict());other=torch.nn.Linear(5,2);other.load_state_dict(m.state_dict());oo=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=0);restore_optimizer(oo,state)
        for mm,opt in [(m,o),(other,oo)]:opt.zero_grad();(mm(x)-y).square().sum().backward();opt.step()
        self.assertTrue(all(torch.equal(v,other.state_dict()[k]) for k,v in m.state_dict().items()))
        self.assertTrue(all(int(v['step'])==3 for v in oo.state.values()))
        self.assertEqual(route(.3,.1,200),'dev64_native')
        self.assertEqual(route(.3,.099,200),'conditioning_generalization_stop')
        self.assertEqual(route(.299,.2,200),'continue_same_trajectory_to_2000')
        self.assertEqual(route(.299,.2,2000),'stop_offline_predictor_tuning_next_R16_optimization_feasibility')
if __name__=='__main__':unittest.main()
