import unittest
import numpy as np
from scripts.tastvg_latent_quality_math_v1 import interval_features,geometry_features,temporal_iou,fit_path,predict,pick

class TestLatent(unittest.TestCase):
    def test_context_and_offsets(self):
        h=np.arange(8)[:,None]*np.ones((8,256));ids=[0,2,4,6,8,10,12,14]
        x,c=interval_features(h,ids,[[2,4]],4)
        self.assertEqual(x.shape,(1,1792))
        np.testing.assert_array_equal(x[0].reshape(7,256)[:,0],[2,4,3,.5,5.5,1.5,-1.5])
        self.assertEqual(c,[dict(left_frames=2,right_frames=2,inside_frames=3)])
    def test_edge_context_is_zero(self):
        x,c=interval_features(np.ones((9,256)),range(9),[[0,8]],2)
        np.testing.assert_array_equal(x[0].reshape(7,256)[3:5],0)
        self.assertEqual(c[0]['left_frames'],0)
    def test_labels_and_geometry(self):
        np.testing.assert_allclose(temporal_iou([[0,4],[2,6],[7,8]],[0,4]),[1,1/3,0])
        np.testing.assert_allclose(geometry_features([10,12,14,16],[[0,2]]),[[0,5/7,5/7]])
    def test_ridge_and_training_scaler(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(96,7));w=np.arange(7)/20;y=x@w+.4
        model,path,at=fit_path(x,y,x[:32]+.2,y[:32]+.2*w.sum(),np.repeat(range(4),8))
        np.testing.assert_allclose(model['mean'],x.mean(0))
        self.assertLess(max(p['normal_equation_max_error'] for p in path),1e-10)
        self.assertLess(np.mean((predict(model,x)-y)**2),.02)
    def test_tie_and_conservative_baseline(self):
        self.assertEqual(pick([.2,.3,.3],0,3),0)
        self.assertEqual(pick([.2,.1,.4],0,3),2)
        self.assertEqual(pick([.2,.2,.1],1,3),1)

if __name__=='__main__':unittest.main()
