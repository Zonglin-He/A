import unittest
import numpy as np
import torch
from vg_tta.tastvg_spatial_expansion_s0_v1 import mask_boxes,alignment

class SpatialExpansion(unittest.TestCase):
    def test_sparse_mask_boxes_and_empty(self):
        masks=np.zeros((3,10,20),bool);masks[0,2:6,4:12]=True;masks[2,9,19]=True
        valid,boxes=mask_boxes(masks,[0,3,7],8)
        self.assertEqual(np.flatnonzero(valid).tolist(),[0,7])
        np.testing.assert_allclose(boxes[0],[.4,.4,.4,.4]);np.testing.assert_allclose(boxes[7],[.975,.95,.05,.1])
        self.assertFalse(mask_boxes(None,[0,3,7],8)[0].any())
    def test_loss_matches_scalar_geometry_and_gradient(self):
        p=torch.tensor([[.4,.5,.2,.3],[.6,.3,.1,.15],[.1,.2,.12,.11]],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([[.5,.4,.3,.2],[.2,.6,.1,.1],[.8,.8,.15,.2]],dtype=torch.float64)
        keep=torch.tensor([True,False,True]);a=alignment(p,q,keep,5,3)
        vals=[]
        for i in [0,2]:
            x,y=p.detach().numpy()[i],q.numpy()[i];ax,ay=x[:2]-x[2:]/2,x[:2]+x[2:]/2;bx,by=y[:2]-y[2:]/2,y[:2]+y[2:]/2
            inter=max(0,min(ay[0],by[0])-max(ax[0],bx[0]))*max(0,min(ay[1],by[1])-max(ax[1],bx[1]))
            union=x[2]*x[3]+y[2]*y[3]-inter;enclosing=(max(ay[0],by[0])-min(ax[0],bx[0]))*(max(ay[1],by[1])-min(ax[1],bx[1]))
            vals.append(5*abs(x-y).sum()+3*(1-inter/union+(enclosing-union)/enclosing))
        self.assertAlmostEqual(float(a),np.mean(vals),places=12)
        self.assertTrue(torch.autograd.gradcheck(lambda x:alignment(x,q,keep,5,3),(p,)))
        grad=torch.autograd.grad(a,p)[0];self.assertTrue(torch.equal(grad[1],torch.zeros(4,dtype=torch.float64)))
    def test_empty_loss_zero_gradient(self):
        p=torch.rand(3,4,requires_grad=True);loss=alignment(p,p.detach(),torch.zeros(3,dtype=torch.bool),5,3)
        self.assertEqual(float(loss),0);self.assertEqual(float(torch.autograd.grad(loss,p)[0].abs().sum()),0)

if __name__=='__main__':unittest.main()
