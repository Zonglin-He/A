import ast,typing
from functools import lru_cache

from pathlib import Path

import numpy as np

import torch
ROOT=Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def official_functions():
 scope=dict(np=np,torch=torch,Dict=typing.Dict,List=typing.List,Tuple=typing.Tuple)
 for relative,names in [('external/TA-STVG/utils/box_utils.py',{'np_box_iou','_box_inter_union','np_box_area'}),('external/TA-STVG/engine/evaluate.py',{'linear_interp'}),('external/TA-STVG/datasets/evaluation/vidstg_eval.py',{'VidSTGiouEvaluator'})]:
  p=ROOT/relative;tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
 return scope

class DenseMetric:
 def __init__(self):
  self.functions=official_functions()
 def __call__(self,boxes,ids,interval,truth,gt_interval):
  boxes=np.asarray(boxes,float);truth={int(k):np.asarray(v,float).reshape(4).tolist() for k,v in truth.items()}
  tube=self.functions['linear_interp']({int(fid):[box.tolist()] for fid,box in zip(ids,boxes)})
  cls=self.functions['VidSTGiouEvaluator'];e=cls.__new__(cls);e.vid2steds={0:list(map(int,gt_interval))};e.vid2box={0:{k:[v] for k,v in truth.items()}};e.vid2names={0:'fixed'};e.vid2sents={0:''};e.iou_thresholds=[.3,.5]
  out=e.evaluate({0:tube},{0:dict(sted=list(map(int,interval)),qtype='declarative')},{},{})[0][0]
  return {'m_tIoU':out['tiou'],'m_vIoU':out['viou'],'vIoU@0.3':out['viou@0.3'],'vIoU@0.5':out['viou@0.5'],'sIoU_dense_GT':out['gt_viou']}

def xyxy(boxes,width,height):
 b=np.asarray(boxes,float);return np.concatenate([b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2],-1)*np.array([width,height,width,height])
