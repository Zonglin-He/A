import unittest,torch
from scripts.stvg_semantic_preference_student32_v1 import policy
from scripts.stvg_semantic_preference_contract_v1 import weighted_fixed_support_nll
class SupportGradient(unittest.TestCase):
    def test_same_mass_and_exact_mean_gradient_both_feedbacks(self):
        support=dict(boxes_cxcywh=[[.2,.3,.2,.4],[.7,.6,.3,.3],[.4,.8,.1,.2],[.5,.4,.4,.5]],posterior=[.1,.4,.2,.2,.1])
        for arm in ['GEOM','PREF']:
            mean=torch.tensor([[.2,-.3,.1,-.4]],dtype=torch.float64,requires_grad=True);ev=torch.tensor([support['boxes_cxcywh'][0]],dtype=torch.float64)
            p=policy(mean,ev,'fixed',0,config=dict(sigma=.25,tau=.1),support=support,arm=arm)
            self.assertAlmostEqual(float(p['weights'].sum()),.9,places=12);self.assertFalse(p['samples'].requires_grad);self.assertFalse(p['weights'].requires_grad)
            f=weighted_fixed_support_nll(mean,p['samples'],p['weights'],.25);g=torch.autograd.grad(f,mean)[0];expected=(p['weights'][:,:,None]*(mean[:,None,:]-p['samples'])).sum(1)/.25**2
            self.assertTrue(torch.allclose(g,expected,atol=1e-12,rtol=0))
            for i in range(4):
                q=mean.detach().clone();q[0,i]+=1e-6;v1=weighted_fixed_support_nll(q,p['samples'],p['weights'],.25);q[0,i]-=2e-6;v0=weighted_fixed_support_nll(q,p['samples'],p['weights'],.25)
                self.assertAlmostEqual(float((v1-v0)/2e-6),float(g[0,i]),places=6)
    def test_fixed_action_chart_not_silently_clamped(self):
        support=dict(boxes_cxcywh=[[.5,.5,1.,.4]]*4,posterior=[.1,.4,.2,.2,.1])
        with self.assertRaises(AssertionError):policy(torch.zeros(1,4),torch.ones(1,4)*.5,'fixed',0,config=dict(sigma=.25,tau=.1),support=support,arm='PREF')
if __name__=='__main__':unittest.main()
