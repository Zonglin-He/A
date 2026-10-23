"""Reference math, positive control and meaningful representation invariants."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from stvg.prototype import evidence,KAPPA
from stvg.representation import weights,pool
from stvg.prototype import membership,inclusive_scores,update

class TestContentFeedback(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(91);self.z=rng.normal(size=(18,7))
        q=np.triu(rng.uniform(size=(18,18)));self.q=q/q.sum()
    def test_scalar_reference(self):
        L,_=evidence(self.z,self.q);w=membership(self.q)
        for t in range(18):
            fit=np.arange(18)%2!=t%2;v=self.z[fit];center=v.mean(0)
            u=(v-center)/np.linalg.norm(v-center,axis=1)[:,None]
            a=self.z[t]-center;a/=np.linalg.norm(a)
            k=np.array([(1+np.dot(a,b))/2 for b in u]);m=k.mean()
            p=(sum(k*w[fit])+KAPPA*m)/(sum(w[fit])+KAPPA)
            n=(sum(k*(1-w[fit]))+KAPPA*m)/(sum(1-w[fit])+KAPPA)
            self.assertAlmostEqual(L[t],np.log(p/n),places=13)
    def test_rotation_translation_scale(self):
        L,_=evidence(self.z,self.q);Q=np.linalg.qr(np.random.default_rng(9).normal(size=(7,7)))[0]
        other,_=evidence((self.z@Q)*3+5,self.q)
        np.testing.assert_allclose(L,other,atol=2e-14,rtol=1e-12)
    def test_constant_exact_native(self):
        L,_=evidence(np.full((18,11),.27),self.q)
        self.assertTrue(np.array_equal(L,np.zeros(18)))
        self.assertTrue(np.array_equal(update(self.q,inclusive_scores(L)),self.q))
    def test_synthetic_event_separation(self):
        z=np.zeros((20,2));z[:,0]=1;z[6:14]=[0,1]
        q=np.ones((20,20))*0;q[np.triu_indices(20)]=.001;q[5,14]=1;q/=q.sum()
        L,_=evidence(z,q)
        self.assertGreater(L[6:14].min(),0);self.assertLess(L[:4].max(),0)
    def test_region_pool_and_reflection(self):
        valid=np.ones((5,7),bool);a,_=weights([.2,.3,.2,.2],5,7,valid)
        b=a[::-1,::-1];self.assertAlmostEqual(a.sum(),b.sum())
        x=np.random.default_rng(8).normal(size=(5,7,3));y=x.copy();y[a==0]+=500
        self.assertTrue(np.array_equal(pool(x,a),pool(y,a)))
        valid[:]=False;valid[1,1]=True;c,fallback=weights([.9,.9,.01,.01],5,7,valid)
        self.assertTrue(fallback);self.assertEqual(c.sum(),1)
    def test_parity_train_isolation(self):
        L,stats=evidence(self.z,self.q)
        self.assertEqual(stats[0]['positive_weight'],membership(self.q)[1::2].sum())
        self.assertEqual(stats[1]['positive_weight'],membership(self.q)[::2].sum())

if __name__=='__main__':unittest.main()
