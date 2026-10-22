import unittest,numpy as np
from research.native_event_episode_context_v1.core import window,slices,embed
class Tests(unittest.TestCase):
 def test_boundaries(self):
  for i in range(65):
   for j in range(i+1,65):
    lo,hi,s=window(i,j);self.assertLessEqual(lo,i);self.assertGreaterEqual(hi,j);self.assertEqual(hi-lo+1,max(49,j-i+1));self.assertTrue(0<=s<=65-(hi-lo+1))
 def test_parities(self):
  for lo in range(17):
   rows=slices(lo,lo+48);self.assertEqual([rows[i%2][1][i//2]for i in range(49)],list(range(lo,lo+49)))
   for off,pos,sel in rows:self.assertEqual([off+2*x for x in sel],pos)
 def test_embed(self):
  q=np.triu(np.ones((49,49)),1);q/=q.sum();a=embed(q,3,51);self.assertTrue(np.array_equal(a[3:52,3:52],q));self.assertAlmostEqual(a.sum(),1)
if __name__=='__main__':unittest.main()
