import copy,unittest,torch
from scripts.stvg_semantic_preference_4anchor32_v1 import policy,rollout_audit
from scripts.stvg_semantic_preference_student32_v1 import policy as single
from scripts.stvg_semantic_preference_contract_v1 import weighted_fixed_support_nll
class MultiAnchor(unittest.TestCase):
    def setUp(self):
        self.active=[dict(position=i,boxes_cxcywh=[[.2,.3,.2,.4],[.7,.6,.3,.3],[.4,.8,.1,.2],[.5,.4,.4,.5]],posterior=[.1,.4,.2,.2,.1]) for i in range(4)]
        self.active[1]['posterior']=[.2,.1,.5,.1,.1]
    def test_multi_anchor_gradient_and_independent_audit(self):
        for arm in ['GEOM','PREF']:
            mean=torch.zeros(4,4,dtype=torch.float64,requires_grad=True);e=mean.new_tensor([a['boxes_cxcywh'][0] for a in self.active]);cfg=dict(sigma=.25,tau=.1)
            p=policy(mean,e,'fixed',0,config=cfg,support=dict(active=self.active),arm=arm);loss=weighted_fixed_support_nll(mean,p['samples'],p['weights'],cfg['sigma']);g=torch.autograd.grad(loss,mean)[0]
            expected=(p['weights'][:,:,None]*(mean[:,None]-p['samples'])).sum(1)/(cfg['sigma']**2*4);self.assertTrue(torch.allclose(g,expected,atol=1e-12,rtol=0));self.assertTrue(torch.allclose(p['weights'].sum(-1),mean.new_tensor([.9,.8,.9,.9])))
            for i in range(4):
                m=mean.detach().clone();m[i,0]+=1e-6;hi=weighted_fixed_support_nll(m,p['samples'],p['weights'],cfg['sigma']);m[i,0]-=2e-6;lo=weighted_fixed_support_nll(m,p['samples'],p['weights'],cfg['sigma']);self.assertAlmostEqual(float((hi-lo)/2e-6),float(g[i,0]),places=6)
            row=dict(rollout=p,loss=float(loss),mean_gradient=g);r=rollout_audit(row,e.numpy(),cfg,list(range(4)),'fixed',0,dict(active=self.active),arm);self.assertLess(r['mean_gradient_error'],1e-12)
            with self.assertRaises(AssertionError):rollout_audit(row,e.numpy(),cfg,[1,0,2,3],'fixed',0,dict(active=self.active),arm)
    def test_one_anchor_is_original_exact(self):
        for arm in ['GEOM','PREF']:
            m=torch.zeros(1,4);e=torch.ones(1,4)*.3;cfg=dict(sigma=.25,tau=.1);a=self.active[0]
            x=policy(m,e,'fixed',0,config=cfg,support=dict(active=[a]),arm=arm);y=single(m,e,'fixed',0,config=cfg,support=a,arm=arm)
            for k in ['samples','weights','rewards']:self.assertTrue(torch.equal(x[k],y[k]))
    def test_no_clamp(self):
        a=copy.deepcopy(self.active);a[2]['boxes_cxcywh'][0][2]=1.
        with self.assertRaises(AssertionError):policy(torch.zeros(4,4),torch.ones(4,4)*.3,'fixed',0,config=dict(sigma=.25,tau=.1),support=dict(active=a),arm='PREF')
if __name__=='__main__':unittest.main()
