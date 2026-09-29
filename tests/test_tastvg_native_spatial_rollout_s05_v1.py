import unittest
import torch
from vg_tta.tastvg_native_spatial_rollout_s05_v1 import directions,rollout_states

class NativeRollouts(unittest.TestCase):
    def test_directions_are_fixed_orthogonal(self):
        q=directions();self.assertTrue(torch.equal(q,directions()));torch.testing.assert_close(q@q.T,torch.eye(4,dtype=torch.double),atol=1e-12,rtol=0)
    def test_antithetic_bound_and_center(self):
        center={'query':torch.zeros(256),'norm':torch.ones(1536)};states,d,b=rollout_states(center)
        v=lambda s:torch.cat([x.flatten().double() for x in s.values()]);base=v(center)
        self.assertTrue(torch.equal(v(states[0]),base))
        for i in range(1,9,2):
            a,c=v(states[i])-base,v(states[i+1])-base
            torch.testing.assert_close(a,-c,atol=1e-7,rtol=0);self.assertAlmostEqual(float(a.norm()/base.norm()),.05,places=7)
    def test_state_storage_is_independent(self):
        x={'a':torch.zeros(256),'b':torch.ones(1536)};states,_,_=rollout_states(x);states[1]['a'].add_(1)
        self.assertEqual(float(x['a'].sum()),0);self.assertEqual(float(states[0]['a'].sum()),0)

if __name__=='__main__':unittest.main()
