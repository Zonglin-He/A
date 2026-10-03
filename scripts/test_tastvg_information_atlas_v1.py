"""Meaningful controls for weights, undefined metrics, physical labels and fits."""
import unittest
import numpy as np
from scripts.tastvg_information_atlas_math_v1 import *

class AtlasTests(unittest.TestCase):
    def test_label_conventions(self):
        f=frame_labels([10,20,30,40],[20,40]);np.testing.assert_equal(f['event'],[0,1,1,0])
        np.testing.assert_allclose(f['start_distance']+f['end_distance'],20/31)
        c=candidate_labels([[10,50],[20,25],[20,40]],[20,40],0)
        np.testing.assert_allclose(c['precision'],[.5,1,1]);np.testing.assert_allclose(c['recall'],[1,.25,1])
        np.testing.assert_allclose(c['delta'],[0,-.25,.5])
    def test_source_weights(self):
        w=equal_source_weights([0,0,0,1]);np.testing.assert_allclose(w,[1/6,1/6,1/6,.5])
    def test_shuffle(self):
        y=np.arange(32);a,ids,p=shuffle_labels(y,'s','candidate',anchor=3)
        self.assertEqual(a[3],3);np.testing.assert_equal(np.sort(a),y)
        np.testing.assert_equal(a,shuffle_labels(y,'s','candidate',anchor=3)[0])
    def test_ridge_analytic(self):
        x=np.arange(20,dtype=float)[:,None];y=2*x[:,0]+3;g=np.repeat(np.arange(4),5)
        path,stats,selected=ridge_path(x,y,g,x,y,g)
        self.assertEqual(selected,[0]);self.assertLess(stats[0]['equation_error'][0],1e-10)
        mean,std=normalization(x,equal_source_weights(g));z=(x-mean)/std
        w=np.linalg.solve(z.T@z/20+ALPHAS[0]*np.eye(1),z.T@(y-y.mean())/20)
        np.testing.assert_allclose(path[0]['weight'][:,0],w)
    def test_zero_delta(self):
        x=np.array([[0.],[1.],[-1.]]);y=np.array([0.,.4,-.4]);g=np.array([0,0,0])
        p,_,_=ridge_path(x,y,g,x,y,g,intercept=False)
        self.assertEqual(float(predict(split(p[0]),[[0.]])[0]),0)
    def test_undefined_auc_r2(self):
        m=moments([1,1],[.8,.9],True);self.assertIsNone(m['auc']);self.assertIsNone(m['within_r2'])
    def test_paired_source_bootstrap(self):
        rows=[]
        for s in range(3):
            a=moments([0,1],[0,1]);b=moments([0,1],[.5,.5])
            rows.append(dict(source_index=s,order='o',condition='c',metrics={'a':a,'b':b}))
        d=paired_difference(rows,'a','b','r2',draws=100);self.assertEqual(d['mean'],1);self.assertEqual(d['ci95'],[1,1])
    def test_logistic_KKT(self):
        x=np.arange(-4,4,dtype=float)[:,None];y=(x[:,0]>0).astype(float);g=np.repeat([0,1],4)
        p,stat,at=logistic_path(x,y,g,x,y,g)
        self.assertLess(max(z['KKT_error'] for z in stat),1e-5);self.assertGreater(predict(p[at],[[3.]])[0],predict(p[at],[[-3.]])[0])

def split(m):
    m=dict(m);m['weight']=m['weight'][:,0];m['bias']=float(m['bias'][0]);return m
if __name__=='__main__':unittest.main()
