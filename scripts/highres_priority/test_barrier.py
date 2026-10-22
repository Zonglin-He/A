import os,unittest
from unittest.mock import patch
from scripts.highres_priority import run

class BarrierTest(unittest.TestCase):
    def test_running_full_rejected(self):
        with patch.object(run,'read',return_value={'status':'running','paused_completed':6088}):
            with self.assertRaises(AssertionError):run.assert_pause_boundary()
    def test_wrong_pause_count_rejected(self):
        with patch.object(run,'read',return_value={'status':'paused_by_user','paused_completed':6089}):
            with self.assertRaises(AssertionError):run.assert_pause_boundary()
    def test_prefix_timer_preserves_output_and_restores(self):
        import stvg.temporal_capture as tc
        marker=object()
        with patch.object(tc,'_prefix_view',return_value=marker)as original,patch.object(run.torch.cuda,'synchronize'):
            with run.prefix_costs()as costs:self.assertIs(tc._prefix_view(None,None,None),marker)
            self.assertIs(tc._prefix_view,original);self.assertEqual(len(costs),1)
    def test_prefix_timer_exception_restores(self):
        import stvg.temporal_capture as tc
        with patch.object(tc,'_prefix_view',side_effect=ValueError('preserved'))as original,patch.object(run.torch.cuda,'synchronize'):
            with self.assertRaises(ValueError):
                with run.prefix_costs():tc._prefix_view(None,None,None)
            self.assertIs(tc._prefix_view,original)
if __name__=='__main__':unittest.main()
