"""Small structural checks of censoring and actual physical-grid handling."""
import unittest
import numpy as np
from research.native_event_context_v1.core import censored_evidence,log_ratio

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.ids=[3,9,20,36,80,105,140];self.w=self.ids[:5]
        self.u=np.triu(np.ones((5,5)),1);self.u/=self.u.sum()
    def test_null_exact_zero(self):
        s,c=censored_evidence(self.ids,self.w,self.u,self.u)
        self.assertTrue(np.array_equal(s,np.zeros_like(s)));self.assertEqual(sum(c.values()),21)
    def test_unseen_is_unknown_and_edges_censored(self):
        q=self.u.copy();q[1,3]*=3;q/=q.sum();s,_=censored_evidence(self.ids,self.w,q,self.u)
        self.assertEqual(s[4,6],0);self.assertEqual(s[0,4],0)
        self.assertAlmostEqual(s[1,6],log_ratio(q.sum(1),self.u.sum(1))[1])
        self.assertAlmostEqual(s[0,3],log_ratio(q.sum(0),self.u.sum(0))[3])
    def test_joint_not_two_marginals(self):
        q=self.u.copy();q[1,3]*=4;q/=q.sum();s,_=censored_evidence(self.ids,self.w,q,self.u)
        self.assertAlmostEqual(s[1,3],np.log(q[1,3]/self.u[1,3]))

if __name__=='__main__':unittest.main()
