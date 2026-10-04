import sys,unittest,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.spatial_guided_temporal_import_v1 import core as c

class Geometry(unittest.TestCase):
    def test_box_conversion(self):
        np.testing.assert_allclose(c.a_xyxy([[.5,.5,.2,.4]],100,50),[[40,15,60,35]])
    def test_gt_extension_never_drops_frames(self):
        out,r=c.extend_gt({'2':[2,2,4,4],'4':[4,4,8,8]},[0,2,3,4,6])
        np.testing.assert_equal(out,[[2,2,4,4],[2,2,4,4],[3,3,6,6],[4,4,8,8],[4,4,8,8]])
        self.assertEqual(r['extrapolated_frames'],2);self.assertFalse(r['temporal_span_used'])
    def test_border_square_preserves_centre(self):
        f=np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3);w=c.window([0,0,2,2],5,4)
        out=c.crop(f,w);self.assertEqual(out.shape,(3,3,3));np.testing.assert_equal(out[0,0],f[0,0])
    def test_invalid_box_full_fallback(self):
        f=np.zeros((4,5,3),np.uint8);s=c.window([2,2,1,1],5,4)
        self.assertTrue(s['fallback_full']);self.assertIs(c.crop(f,s),f)
    def test_time_grid_unchanged(self):
        ids=[10,20,30,40];p,d=c.sample_indices(ids,10)
        np.testing.assert_equal(p,[0,0,1,1,2,2]);self.assertEqual(d,3.1)
    def test_argmax_duplicates_ties(self):
        self.assertEqual(c.top([[0,2],[0,2],[1,3]],[.5,.5,.4])['index'],0)
    def test_half_open_teacher_metric(self):
        self.assertEqual(c.iou([0,2],[2,3]),0);self.assertEqual(c.iou([0,2],[1,3]),1/3)
    def test_deployable_label_guard(self):
        for p in ['/tmp/GT_LABELS_search.json','/tmp/GT_BOXES_INPUT.json','/tmp/ROWS.json']:
            with self.assertRaises(PermissionError):c.data_guard('open',(p,'r',0))
        c.data_guard('open',('/tmp/COHORT.json','r',0))

if __name__=='__main__':unittest.main()
