import unittest,numpy as np
from research.native_event_information_audit_v1.core import moments,combine,fit_predict,holdout_stats,auc
class Tests(unittest.TestCase):
 def test_weighted_reference(self):
  rng=np.random.default_rng(4);xs=[rng.normal(size=(n,6))for n in [7,13,31]];ys=[rng.normal(size=len(x))for x in xs];stat=combine([moments(x,y)for x,y in zip(xs,ys)]);z=rng.normal(size=(10,6));p=fit_predict(stat,z);w=np.concatenate([np.full(len(x),1/len(x))for x in xs]);X=np.concatenate(xs);Y=np.concatenate(ys);xm=np.average(X,axis=0,weights=w);ym=np.average(Y,weights=w);scale=np.sqrt(np.average(((X-xm)**2).sum(1),weights=w));C=(X-xm)/scale;W=np.linalg.solve(C.T@(w[:,None]*C)+np.eye(6),C.T@(w*(Y-ym)));self.assertLess(abs(p-((z-xm)/scale@W+ym)).max(),1e-14)
 def test_all_parent_members_excluded(self):
  stats=[moments(np.ones((3,2))*i,np.arange(3))for i in range(4)];a,excluded=holdout_stats(stats,['a','b','a','c'],2);self.assertEqual(excluded,{0,2});self.assertEqual(a['total'],2.)
 def test_heldout_truth_not_used(self):
  rng=np.random.default_rng(4);xs=[rng.normal(size=(10,3))for _ in range(3)];ys=[rng.normal(size=10)for _ in xs];stats=[moments(x,y)for x,y in zip(xs,ys)];a,_=holdout_stats(stats,['a','b','c'],0);p=fit_predict(a,xs[0]);stats[0]=moments(xs[0],ys[0]+300);b,_=holdout_stats(stats,['a','b','c'],0);self.assertTrue(np.array_equal(p,fit_predict(b,xs[0])))
 def test_synthetic_crossquery_signal(self):
  rng=np.random.default_rng(4);xs=[rng.normal(size=(30,4))for _ in range(12)];ys=[(x[:,0]>0).astype(float)for x in xs];stats=[moments(x,y)for x,y in zip(xs,ys)];a,_=holdout_stats(stats,list(range(12)),0);self.assertGreater(auc(fit_predict(a,xs[0]),ys[0]),.95)
 def test_rotation_scale(self):
  rng=np.random.default_rng(4);x=rng.normal(size=(40,6));y=rng.normal(size=40);u,_=np.linalg.qr(rng.normal(size=(6,6)));a=fit_predict(moments(x,y),x);b=fit_predict(moments(13*x@u,y),13*x@u);self.assertLess(abs(a-b).max(),1e-14)
if __name__=='__main__':unittest.main()
