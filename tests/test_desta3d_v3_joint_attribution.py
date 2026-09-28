import unittest
import numpy as np
from vg_tta.desta3d_v3_joint_attribution import bases, split, attribution, proj


class Contracts(unittest.TestCase):
    def test_nonorthogonal_exact_sum_not_double_projection(self):
        t=np.array([[1.],[0.],[0.]]);s=np.array([[1.],[1.],[0.]])/2**.5
        b=bases(t,s);j=np.array([[2.,3.,1.]])@b['U'];p=split(j,b)
        for d in p.values():np.testing.assert_allclose(sum(d.values()),j,atol=1e-14)
        for k in ['T_first','S_first']:
            a,c=p[k].values();self.assertAlmostEqual(float((a*c).sum()),0.)
        a,c=p['direct_sum'].values();self.assertNotAlmostEqual(float((a*c).sum()),0.)
        self.assertGreater(attribution(j,{'T':j},b)['naive_sum_relative_error'],.1)

    def test_exact_intersection_is_not_unique_ownership(self):
        e=np.eye(4);b=bases(e[:,:2],e[:,1:3]);self.assertEqual(b['report']['intersection_dimension'],1)
        j=np.array([[1.,2.,3.,0.]])@b['U'];p=split(j,b)
        for d in p.values():np.testing.assert_allclose(sum(d.values()),j,atol=1e-14)

    def test_zero_and_perpendicular(self):
        e=np.eye(4);b=bases(e[:,:1],e[:,1:2]);x=np.array([[1.,2.,3.,4.]]);j=x@b['U'];r=x-j@b['U'].T
        self.assertAlmostEqual(float((x*x).sum()),float((j*j).sum()+(r*r).sum()))
        z=attribution(j*0,{'T':j},b);self.assertEqual(z['T_first']['components']['T']['energy'],0.)

    def test_signed_cross_objective_and_rotation_invariance(self):
        rng=np.random.default_rng(8);q=np.linalg.qr(rng.normal(size=(8,4)))[0];qt=q[:,:2];qs=q[:,2:]
        g=rng.normal(size=(3,8));x=rng.normal(size=(3,8));out=[]
        for sign in [1,-1]:
            b=bases(sign*qt,qs[:,::-1]);out.append(attribution(x@b['U'],{'T':g@b['U'],'S':-g@b['U']},b))
        for key in ['T','S_given_T']:
            a=out[0]['T_first']['components'][key]['local_descent'];self.assertAlmostEqual(a['T'],-a['S'])
            self.assertAlmostEqual(a['T'],out[1]['T_first']['components'][key]['local_descent']['T'])

if __name__=='__main__':unittest.main()
