"""Meaningful CPU ordering/ranking/statistics/state and Gaussian contracts; no GT."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.stvg_opd_hc2_coordinate_common_v1 import *
from scripts.finalize_stvg_opd_hc2_coordinate_v1 import independent
from vg_tta.decota_spatial_opd_tunable_v1 import rollout,likelihood_loss,optimizer_step

class Contracts(unittest.TestCase):
    def test_one_coordinate_per_candidate_and_all_current_values_present(self):
        cfg=START.copy()
        self.assertEqual(sum(map(len,GRIDS.values())),26)
        for k in COORDINATES:
            candidates=candidate_configs(cfg,k)
            self.assertIn(cfg,candidates)
            for c in candidates:self.assertEqual({n for n in c if c[n]!=cfg[n]}-{k},set())
            cfg=candidates[-1]
    def test_incumbent_then_step_then_ordinal_ties(self):
        def s(cfg,mean=.1,harm=0):return dict(config=cfg,statistics=dict(metrics=dict(delta_total_v=dict(mean=mean,harm_gt20pp_sources=harm))))
        other={**START,'sigma':.25}
        self.assertGreater(rank(s(START),START,3),rank(s(other),START,0))
        self.assertGreater(rank(s(other,mean=.2),START,0),rank(s(START),START,3))
        self.assertGreater(rank(s(other,harm=0),START,0),rank(s(START,harm=1),START,3))
        self.assertGreater(rank(s({**other,'steps':3}),START,4),rank(s(other),START,0))
        self.assertGreater(rank(s(other),START,0),rank(s(other),START,1))
    def test_state_commit_resets_query_and_scales_only_delta(self):
        ini={'spatial.query_residual':torch.ones(256),'LN':torch.tensor([2.,-2.])};fin={n:v+4 for n,v in ini.items()}
        for alpha in GRIDS['writeback']:
            x=commit(ini,fin,alpha);self.assertEqual(torch.count_nonzero(x['spatial.query_residual']),0)
            torch.testing.assert_close(x['LN'],ini['LN']+4*alpha,rtol=0,atol=0)
    def test_every_sigma_and_tau_applies_to_sampling_and_weights(self):
        mean=torch.tensor([[.2,-.3,.4,-.5]]);evidence=torch.tensor([[.5,.5,.3,.3]])
        reference=rollout(mean,evidence,'contract',0,config=START)
        for sigma in GRIDS['sigma']:
            r=rollout(mean,evidence,'contract',0,config={**START,'sigma':sigma})
            torch.testing.assert_close((r['samples']-mean[:,None])/sigma,(reference['samples']-mean[:,None])/.1,atol=2e-6,rtol=1e-6)
        for tau in GRIDS['tau']:
            r=rollout(mean,evidence,'contract',0,config={**START,'tau':tau})
            self.assertTrue(torch.equal(r['samples'],reference['samples']))
            x=r['rewards'].numpy().astype(float)/tau;x-=x.max(-1,keepdims=True);expected=np.exp(x);expected/=expected.sum(-1,keepdims=True)
            np.testing.assert_allclose(r['weights'].numpy(),expected,atol=2e-7,rtol=2e-6)
    def test_true_likelihood_derivative_all_exploration_scales(self):
        for sigma in GRIDS['sigma']:
            m=torch.tensor([[.3,-.1,.4,.2]],requires_grad=True);old=m.detach()+.02
            r=rollout(old,torch.tensor([[.5,.5,.2,.3]]),'derivative',0,config={**START,'sigma':sigma})
            g=torch.autograd.grad(likelihood_loss(m,old,r['samples'],r['weights'],sigma=sigma),m)[0].numpy()
            z=r['samples'].numpy().astype(float);w=r['weights'].numpy().astype(float);mu=m.detach().numpy().astype(float)
            expected=(mu-old.numpy())/sigma**2-((w-1/32)[...,None]*(z-mu[:,None])).sum(1)/sigma**2
            np.testing.assert_allclose(g,expected,atol=2e-5,rtol=2e-5)
    def test_no_feedback_preserves_existing_optimizer_moments(self):
        p=torch.tensor([.2],requires_grad=True);opt=torch.optim.Adam([p],lr=.03)
        optimizer_step(opt,(p-1).square().sum(),[p],True);before=p.detach().clone();step=opt.state[p]['step'].clone()
        self.assertIsNone(optimizer_step(opt,p.square().sum(),[p],False));self.assertTrue(torch.equal(before,p));self.assertTrue(torch.equal(step,opt.state[p]['step']))
    def test_independent_all_sources_and_order_resampling(self):
        rows=[]
        for s in range(32):
            for j in range(2):rows.append(dict(source_id=s,order=str(j),**{f:.12 if f=='delta_total_v' else 0. for f in FIELDS}))
        a,m=independent(rows);self.assertEqual(a.shape,(32,len(FIELDS)));np.testing.assert_allclose(m['delta_total_v']['ci95'],[.12,.12],atol=1e-15)
        with self.assertRaises(AssertionError):independent(rows[:-1])
        with self.assertRaises(AssertionError):independent(rows[:-1]+[rows[0]])

if __name__=='__main__':unittest.main()
