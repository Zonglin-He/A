import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from copy import copy
import numpy as np
from scripts.tastvg_paper48_p5_common_v1 import frame_ids
from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics

class P5Contracts(unittest.TestCase):
    def test_sampler_matches_official_test_function(self):
        p=Path('external/TA-STVG/datasets/data_utils.py')
        n=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='make_hcstvg_input_clip')
        scope=dict(copy=copy,np=np)
        exec(compile(ast.Module(body=[n],type_ignores=[]),str(p),'exec'),scope)
        cfg=SimpleNamespace(INPUT=SimpleNamespace(SAMPLE_FPS=1.6))
        for count in [200,499,500,501,600]:
            row={k:None for k in ['item_id','vid','width','height','description','object','bboxs','gt_temp_bound']}
            row.update(frame_count=count,frame_ids=list(range(count-1)),actioness=np.zeros(count-1),start_heatmap=np.zeros(count-1),end_heatmap=np.zeros(count-1))
            actual=scope['make_hcstvg_input_clip'](cfg,'test',row)['frame_ids']
            self.assertEqual(frame_ids(count),actual)

    def test_hc_official_geometry_inclusive_truth_end_is_preserved(self):
        metric=HC2DenseMetric();ids=[0,3,7,11]
        boxes=np.array([[0,0,10,10],[1,0,11,10],[1,1,11,11],[0,1,10,11]],float)
        truth={i:[0,0,10,10] for i in range(2,10)}
        for interval in [[0,12],[3,8],[10,12]]:
            a=metric(boxes,ids,interval,truth,[2,9]);b=dense_official_metrics(boxes,ids,interval,truth,[2,9])
            for k in b:self.assertAlmostEqual(a[k],b[k],places=12)
        exact=metric(np.tile([0,0,10,10],(4,1)),ids,[2,9],truth,[2,9])
        self.assertEqual(exact['m_vIoU'],1.)
        self.assertEqual(exact['sIoU_dense_GT'],1.)

if __name__=='__main__':unittest.main()
