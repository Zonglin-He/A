import unittest,numpy as np
from vg_tta.desta3d_v3_policy_gate_views import build_views,wrong_temporal_keep,translated_rectangle
from vg_tta.external_privileged_views import parse_teacher_text,spatial_view,temporal_view
class Controls(unittest.TestCase):
 def test_matched_wrong_and_original_constructor(self):
  x=np.random.default_rng(10).integers(0,256,(7,12,19,3),dtype=np.uint8);ids=[0,1,3,5,8,9,20]
  e=parse_teacher_text('{.1,.6} .15:[.2,.1,.5,.8] .5:[.3,.2,.6,.9]',ids)
  v,m=build_views(x,ids,e);self.assertEqual(sum(m['wrong_keep']),sum(m['correct_keep']))
  np.testing.assert_array_equal(v['S'],spatial_view(x,ids,e)[0]);np.testing.assert_array_equal(v['T'],temporal_view(x,ids,e['interval_physical'])[0])
  for out in v.values():self.assertEqual(out.shape,x.shape);self.assertEqual(out.dtype,x.dtype)
  self.assertTrue(m['time_distinguishable']);self.assertGreater(m['spatial_distinguishable_frames'],0)
 def test_missing_full_image_and_tie(self):
  x=np.arange(3*12*19*3,dtype=np.uint8).reshape(3,12,19,3);ids=[0,4,8]
  e=parse_teacher_text('malformed',ids);v,m=build_views(x,ids,e)
  for a in v.values():np.testing.assert_array_equal(a,x)
  self.assertFalse(m['time_distinguishable']);self.assertEqual(translated_rectangle([0,0,1,1],19,12),([0,0,19,12],[0,0,19,12]))
  wrong,shift,valid=wrong_temporal_keep([1,0,1,0]);self.assertTrue(valid);self.assertEqual(shift,1)
if __name__=='__main__':unittest.main()
