import unittest
import numpy as np
from scripts.local224.core import crop_windows,single_posterior

class GeometryTest(unittest.TestCase):
    def test_full(self):
        r,s=crop_windows([[.5,.5,1,1]],640,360);self.assertEqual(r,(0,0,640,360));self.assertEqual(r,s)
    def test_margin_same_size(self):
        r,s=crop_windows([[.02,.02,.1,.1]],640,360)
        self.assertEqual(r[:2],(0,0));self.assertEqual((r[2]-r[0],r[3]-r[1]),(s[2]-s[0],s[3]-s[1]))
        self.assertTrue(0<=s[0]<s[2]<=640 and 0<=s[1]<s[3]<=360)
    def test_fixed_union(self):
        r,s=crop_windows([[.2,.5,.1,.2],[.8,.5,.1,.2]],100,100)
        self.assertLessEqual(r[0],15);self.assertGreaterEqual(r[2],85)
    def test_posterior_physical(self):
        q=single_posterior([2,8,20],np.zeros((1,3,2),np.float32),[2,4,8,14,20]);self.assertAlmostEqual(q.sum(),1);self.assertEqual(np.count_nonzero(q),3);self.assertTrue((np.tril(q)==0).all())
    def test_invalid(self):
        with self.assertRaises(AssertionError):crop_windows([[.5,.5,-1,1]],100,100)
        with self.assertRaises(AssertionError):single_posterior([0,1],np.zeros((1,2,2)),[0,1])
if __name__=='__main__':unittest.main()
