import unittest
import numpy as np
from vg_tta.tastvg_spatial_critic_s06_v1 import rewards,pair_record,ALL_PAIRS,ANTITHETIC

class Critic(unittest.TestCase):
    def test_reward_ignores_empty_and_matches_geometry(self):
        boxes=np.array([[[.5,.5,.4,.4],[.1,.1,.1,.1]],[[.6,.5,.4,.4],[.9,.9,.2,.2]]]);e=np.array([[.5,.5,.4,.4],[0,0,0,0]])
        np.testing.assert_allclose(rewards(boxes,e,[True,False]),[1,.12/.20],atol=1e-12)
        self.assertIsNone(rewards(boxes,e,[False,False]))
    def test_ties_and_direction_symmetry(self):
        self.assertEqual(pair_record(0,.1)['accuracy'],.5);self.assertIsNone(pair_record(.1,0)['accuracy']);self.assertEqual(pair_record(-.1,-.2)['accuracy'],1)
        for a,b in [(.1,.2),(-.2,.1),(0,.2),(.1,0)]:self.assertEqual(pair_record(a,b)['accuracy'],pair_record(-a,-b)['accuracy'])
    def test_pair_structure(self):
        self.assertEqual(len(ALL_PAIRS),36);self.assertEqual(ANTITHETIC,[(1,2),(3,4),(5,6),(7,8)]);self.assertTrue(all(p in ALL_PAIRS for p in ANTITHETIC))

if __name__=='__main__':unittest.main()
