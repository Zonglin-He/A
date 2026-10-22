import unittest
from scripts.branch_decomposition.supplement import boundary_ready
class BarrierTests(unittest.TestCase):
 def test_running_never_ready(self):self.assertFalse(boundary_ready(dict(status='running',jobs=dict(hc2='scored',vidstg='running'))))
 def test_scored_not_published_never_ready(self):self.assertFalse(boundary_ready(dict(status='awaiting_publication',jobs=dict(hc2='scored',vidstg='scored'))))
 def test_published_complete(self):self.assertTrue(boundary_ready(dict(status='complete_published',jobs=dict(hc2='scored',vidstg='scored'))))
 def test_failed_never_ready(self):self.assertFalse(boundary_ready(dict(status='failed',jobs=dict(hc2='scored',vidstg='scored'))))
if __name__=='__main__':unittest.main()
