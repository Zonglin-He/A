"""Small trajectory-conditioned temporal reranker; pure arrays, no GT/model IO.

The original physical expected-tIoU utility remains the dominant score. An
optional evidence bonus ranks the same twelve view decisions plus Global6.
"""
import numpy as np
from scipy.stats import rankdata
from stvg.temporal import final_mbr
from scripts.interval_pool.core import choose,map_view

def iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);inter=np.maximum(0,np.minimum(a[...,1],b[...,1])-np.maximum(a[...,0],b[...,0]));return inter/(a[...,1]-a[...,0]+b[...,1]-b[...,0]-inter)
def bounded_curve(x):
 x=np.asarray(x,float);scale=max(float(np.quantile(x,.75)-np.quantile(x,.25)),1e-6);return np.tanh((x-np.median(x))/scale)
def mean_curve(t,z,a,b):
 """Physical-time mean of the piecewise-linear curve, clipped to saved support."""
 a=max(float(a),float(t[0]));b=min(float(b),float(t[-1]))
 if b<=a+1e-12:return None
 nodes=np.r_[a,t[(t>a)&(t<b)],b];v=np.interp(nodes,t,z);return float(np.trapz(v,nodes)/(b-a))
def boundary_features(intervals,t,hidden,spatial):
 h=(rankdata(hidden,method='average')-.5)/len(hidden);z=bounded_curve(spatial);out=[]
 for start,end in intervals:
  # The interval column is physical last-frame+1, already expressed in seconds.
  w=min((end-start)/4.,1.)
  hi=np.interp([start,end],t,h);before=mean_curve(t,z,start-w,start);first=mean_curve(t,z,start,start+w);last=mean_curve(t,z,end-w,end);after=mean_curve(t,z,end,end+w)
  enter=max(0.,(first-before)/2)if first is not None and before is not None else 0.
  leave=max(0.,(last-after)/2)if last is not None and after is not None else 0.
  # Missing flanking evidence contributes no bonus, not a forced event boundary.
  out.append([float(hi.mean()),float((hi[0]*enter+hi[1]*leave)/2),enter,leave])
 a=np.asarray(out);assert np.isfinite(a).all()and(a>=-1e-12).all()and(a<=1+1e-12).all();return a

def candidate_pool(pack,ids):
 reference=final_mbr(ids,pack['hull']);base=reference['interval'];all_views=[]
 for views in pack['views']:
  for grid in views:all_views.append(choose(grid['ids'],map_view(grid,grid['ids'])))
 # Incumbent comes first for deterministic tie protection; duplicates keep first.
 intervals=np.asarray(list(dict.fromkeys(tuple(x)for x in [base]+all_views)),int)
 index={int(f):i for i,f in enumerate(ids)};ij=np.asarray([[index[int(a)],index[int(b)-1]]for a,b in intervals]);utilities=reference['utility_grid'][ij[:,0],ij[:,1]]
 agreement=float(np.mean(iou(np.asarray(all_views),base)))
 gate=bool(reference['utility']<.5 or agreement<.5)
 return dict(intervals=intervals,utilities=utilities,base_interval=base,base_expected_iou=reference['utility'],view_agreement=agreement,ambiguous=gate,original_13_intervals=np.asarray(all_views+[base]))

def select(pool,feature,weight,kind='joint',gated=True):
 assert 0<=weight<=.1 and kind in ('hidden','joint')
 if gated and not pool['ambiguous']:return 0
 f=feature[:,0 if kind=='hidden'else 1];score=pool['utilities']+weight*f;index=int(np.argmax(score))
 assert pool['utilities'][0]-pool['utilities'][index]<=weight+1e-12
 return index
