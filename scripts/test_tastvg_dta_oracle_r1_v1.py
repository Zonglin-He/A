"""Meaningful support, gradient, native-decoding and reset checks for R1."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from vg_tta.tastvg_dta_oracle_v1 import *
from methods.decota_final_simplified_v1.objectives import native_view_indices

class Tests(unittest.TestCase):
    def test_full_legal_distribution(self):
        z=torch.tensor([[1.,4.],[2.,3.],[3.,2.],[4.,1.]])
        lp,ij=joint(z)
        self.assertEqual(lp.numel(),6);self.assertTrue((ij[0]<ij[1]).all())
        self.assertAlmostEqual(float(lp.exp().sum()),1.)
        self.assertTrue(torch.allclose(joint(z+torch.tensor([4.,-8.]))[0],lp))

    def test_gaussian_original_coordinates_no_clipping(self):
        ids=[10,20,30,40];lp,sigma=gaussian(ids,[22,36]);ij=torch.triu_indices(4,4,1)
        expected=-((torch.tensor(ids,dtype=torch.float64)[ij[0]]-22)**2+
                     (torch.tensor(ids,dtype=torch.float64)[ij[1]]+1-36)**2)/200.
        self.assertTrue(torch.equal(lp,expected-expected.logsumexp(0)))
        outside,_=gaussian(ids,[-40,90]);self.assertTrue(torch.isfinite(outside).all())
        self.assertEqual(sigma,10.)

    def test_official_offset_map_and_tie(self):
        ids=[3,7,14,20,27,36];parts=offsets(ids)
        for z in [torch.zeros(6,2),torch.tensor([[1.,2.],[2.,5.],[3.,1.],[3.,2.],[2.,5.],[1.,2.]])]:
            a=native_decode(z,ids,parts)
            raw=[native_view_indices(z[p][None]) for p in parts]
            expected=[[parts[i][s],parts[i][e]] for i,(s,e) in enumerate(raw)]
            self.assertEqual(a['offset_indices'],expected)

    def test_analytic_joint_gradient(self):
        torch.manual_seed(8);x=torch.randn(6,4);w=torch.randn(2,4,requires_grad=True);b=torch.randn(2,requires_grad=True)
        z=x@w.T+b;lp,ij=joint(z);q,_=gaussian([0,4,8,12,16,20],[5,17]);p0=(lp.detach()*0).log_softmax(0)
        loss=kl(q,lp)+kl(p0,lp);gw,gb=torch.autograd.grad(loss,(w,b))
        diff=2*lp.exp()-q.exp()-p0.exp();gz=torch.zeros(6,2,dtype=torch.float64)
        gz[:,0].index_add_(0,ij[0],diff);gz[:,1].index_add_(0,ij[1],diff)
        self.assertTrue(torch.allclose(gw.double(),gz.T@x.double(),atol=2e-6))
        self.assertLess(float(gb.norm()),2e-6)

    def test_sgd_reset_and_no_mutation(self):
        torch.manual_seed(2);h=torch.randn(8,256)
        head={'0.weight':torch.eye(256),'0.bias':torch.zeros(256),
              '1.weight':torch.randn(2,256)*.02,'1.bias':torch.zeros(2)}
        orig={k:v.clone() for k,v in head.items()};ids=list(range(0,16,2))
        a=fit_query(h,ids,head,[4,11],.001);b=fit_query(h,ids,head,[4,11],.001)
        self.assertEqual(a['head_final_sha256'],b['head_final_sha256'])
        self.assertEqual(len(a['states']),4);self.assertEqual(len(a['trace']),3)
        for k in head:self.assertTrue(torch.equal(head[k],orig[k]))
        for before,after in zip(a['states'][:-1],a['states'][1:]):
            for p in ['weight','bias']:
                expected=before[p].clone().add_(after[p+'_gradient'],alpha=-.001)
                self.assertTrue(torch.equal(after[p],expected))

    def test_teacher_map_two_offset_envelope(self):
        ids=list(range(10));head={'0.weight':torch.eye(256),'0.bias':torch.zeros(256),
          '1.weight':torch.zeros(2,256),'1.bias':torch.zeros(2)}
        a=fit_query(torch.zeros(10,256),ids,head,[2,8],.001)
        # Equal-distance offset endpoint ties retain the earlier index.
        self.assertEqual(a['teacher_MAP']['physical_interval'],[1,8])
        self.assertEqual(a['before'],a['after'])

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
