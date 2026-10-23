"""Independent CPU CG evaluator, loaded from the pinned official source AST."""
import ast
from functools import lru_cache
import typing
import numpy as np
from cgstvg_runtime import VENDOR


@lru_cache(maxsize=1)
def official_functions():
    import torch
    scope=dict(np=np,torch=torch,Dict=typing.Dict,List=typing.List,Tuple=typing.Tuple)
    files=[('utils/box_utils.py',{'np_box_iou','_box_inter_union','np_box_area'}),
           ('engine/evaluate.py',{'linear_interp'}),
           ('datasets/evaluation/hcstvg_eval.py',{'HCSTVGiouEvaluator'}),
           ('datasets/evaluation/vidstg_eval.py',{'VidSTGiouEvaluator'})]
    for name,names in files:
        path=VENDOR/name;tree=ast.parse(path.read_text())
        nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name in names]
        assert len(nodes)==len(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope


def native_official_metric(boxes,ids,metadata,truth,span,interval,dataset):
    """CG's own interpolation/evaluator; no constructors opening datasets."""
    from stvg.metrics_base import xyxy
    scope=official_functions();pixels=xyxy(boxes,metadata['width'],metadata['height'])
    # Both CG datasets use the same native postprocessor clamp. Predictions are
    # already converted with that rule before serialization; assert it here.
    assert np.min(pixels)> -1e-10
    tube=scope['linear_interp']({int(f):[b.tolist()]for f,b in zip(ids,pixels)})
    cls=scope['HCSTVGiouEvaluator'if dataset=='hc2'else'VidSTGiouEvaluator'];e=cls.__new__(cls)
    e.vid2steds={0:list(map(int,span))};e.vid2box={0:{int(k):[list(v)]for k,v in truth.items()}}
    e.vid2names={0:'anonymous'};e.vid2sents={0:''};e.iou_thresholds=[.3,.5]
    arguments=[{0:tube},{0:dict(sted=list(map(int,interval)),qtype='declarative')}]
    if dataset=='vidstg':arguments.append({})  # native required, unused pred_conf
    v=e.evaluate(*arguments)[0][0]
    return dict(v=float(v['viou']),t=float(v['tiou']),s=float(v['gt_viou']))
