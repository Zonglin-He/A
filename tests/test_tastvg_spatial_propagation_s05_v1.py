import unittest
import numpy as np
import torch
from vg_tta.tastvg_spatial_propagation_s05_v1 import affinity,propagate

class Propagation(unittest.TestCase):
    def test_affinity_matches_independent_numpy(self):
        rng=np.random.default_rng(18);h=rng.normal(size=(3,4,5));r=rng.normal(size=(8,5));fg=np.array([1,0,1,0,0,1,0,0],bool)
        def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
        expected=.5*(unit(h)@unit(r[fg].mean(0))-unit(h)@unit(r[~fg].mean(0)))+.5*np.max(unit(h)@unit(r[fg]).T,axis=-1)
        got=affinity(torch.tensor(h),torch.tensor(r),torch.tensor(fg)).numpy()
        np.testing.assert_allclose(got,expected,atol=1e-12)
    def fixture(self,empty=False):
        tok=torch.zeros(7,4,2);tok[:,:,1]=1;tok[:,0]=torch.tensor([1.,0.]);tok[2,0]=torch.tensor([0.,1.]);tok[2,3]=torch.tensor([1.,0.])
        data=dict(frame_ids=list(range(7)),views=[dict(H=tok[o::2].transpose(0,1),info=dict(fea_map_size=(2,2),encoded_mask=torch.zeros(len(tok[o::2]),4,dtype=torch.bool))) for o in (0,1)])
        masks=np.zeros((5,4,4),bool)
        if not empty:masks[:,:2,:2]=True
        expert=dict(masks=masks,positions=[0,1,3,5,6],parent=0,condition='clean',pixel_sha256='toy')
        return data,expert
    def test_interleaving_geometry_and_reference_preservation(self):
        d,e=self.fixture();z=propagate(d,e)
        self.assertEqual(z['diagnostics']['propagated_frames'],2)
        np.testing.assert_allclose(z['boxes'][2],[.75,.75,.5,.5]);np.testing.assert_allclose(z['boxes'][4],[.25,.25,.5,.5])
        np.testing.assert_allclose(z['boxes'][0],[.25,.25,.5,.5]);self.assertAlmostEqual(z['diagnostics']['threshold'],.25)
    def test_empty_no_invented_foreground(self):
        d,e=self.fixture(True);z=propagate(d,e)
        self.assertFalse(z['valid'].any());self.assertNotIn('evidence',z)

if __name__=='__main__':unittest.main()
