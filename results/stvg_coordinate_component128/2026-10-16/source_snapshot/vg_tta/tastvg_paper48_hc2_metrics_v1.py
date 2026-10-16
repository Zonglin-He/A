"""Official HC-STVG evaluator, preserving its literal endpoint conventions."""
import ast
import numpy as np
from vg_tta.tastvg_paper48_metrics_v1 import official_functions,ROOT

class HC2DenseMetric:
    def __init__(self):
        self.functions=official_functions()
        p=ROOT/'external/TA-STVG/datasets/evaluation/hcstvg_eval.py'
        nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HCSTVGiouEvaluator']
        assert len(nodes)==1
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),self.functions)

    def __call__(self,boxes,ids,interval,truth,gt_interval):
        tube=self.functions['linear_interp']({int(fid):[np.asarray(box,float).tolist()] for fid,box in zip(ids,boxes)})
        cls=self.functions['HCSTVGiouEvaluator'];e=cls.__new__(cls)
        e.vid2steds={0:list(map(int,gt_interval))};e.vid2box={0:{int(k):[list(v)] for k,v in truth.items()}}
        e.vid2names={0:'anonymous'};e.vid2sents={0:''};e.iou_thresholds=[.3,.5]
        z=e.evaluate({0:tube},{0:dict(sted=list(map(int,interval)))},{})[0][0]
        return dict(m_tIoU=z['tiou'],m_vIoU=z['viou'],sIoU_dense_GT=z['gt_viou'],**{'vIoU@0.3':z['viou@0.3'],'vIoU@0.5':z['viou@0.5']})
