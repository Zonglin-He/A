"""Use the official TA-STVG interpolation/evaluator functions without data-loading imports."""
import ast,typing
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

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

def source_summary(rows,fields):
 """Average each source over eligible cells; sources are bootstrap units."""
 result={};sources=sorted({r['parent'] for r in rows});orders=sorted({r['order'] for r in rows})
 if not rows:return dict(sources=0,cells=0,metrics={})
 # Equal conditions then equal orders within a source; no query replication.
 for field in fields:
  per_source=[]
  for source in sources:
   vals=[]
   for order in orders:
    v=[r[field] for r in rows if r['parent']==source and r['order']==order]
    if v:vals.append(np.mean(v))
   per_source.append(np.mean(vals))
  a=np.array(per_source);rng=np.random.default_rng(20260930);boot=np.concatenate([a[rng.integers(0,len(a),(min(500,10000-i),len(a)))].mean(1) for i in range(0,10000,500)])
  ov=[np.mean([r[field] for r in rows if r['order']==o]) for o in orders]
  result[field]=dict(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),order_values=list(map(float,ov)),order_sample_SD=float(np.std(ov,ddof=1)) if len(ov)>1 else None,harm_gt5pp_sources=int((a<-.05).sum()) if field.startswith('delta_') else None,harm_gt20pp_sources=int((a<-.20).sum()) if field.startswith('delta_') else None,harm_gt5pp_cells=sum(r[field]<-.05 for r in rows) if field.startswith('delta_') else None,harm_gt20pp_cells=sum(r[field]<-.20 for r in rows) if field.startswith('delta_') else None)
 return dict(sources=len(sources),cells=len(rows),metrics=result)
