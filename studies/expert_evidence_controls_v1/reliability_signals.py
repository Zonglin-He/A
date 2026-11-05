"""No-label deployment signals, fixed before any target reliability evaluation."""
import numpy as np
from stvg.consensus import encode,estimate
from stvg.readout import prediction
import torch
def signals(source,logits,records,ids,pos,teacher,calibration):
 b=np.asarray(source,np.float64);pos=list(map(int,pos));ref=np.asarray(teacher['boxes'],np.float64);assert ref.shape==(8,4)and len(set(pos))==8
 scores=np.asarray(teacher['selected_frame_scores'],np.float64).reshape(-1);assert scores.shape==(8,)and np.isfinite(scores).all()and(scores>=0).all()and(scores<=1).all()
 z=encode(b);r=encode(ref);scale=np.asarray(calibration['scale'],float);cut=float(calibration['truncation']);pools=[r[i:i+1]-z[j]for i,j in enumerate(pos)];loo=[]
 for i in range(8):
  rest=pools[:i]+pools[i+1:];est=estimate(rest,[np.ones(1)for _ in rest],scale,cut,1/16,diagnostics=False)
  # Prediction of the held-out raw expert residual from the other seven.
  loo.append(float(np.square((pools[i][0]-est['delta'])/scale).sum()/cut))
 entropy=[]
 for value in logits:
  v=np.asarray(value.detach().cpu()if hasattr(value,'detach')else value,np.float64);assert v.ndim==3 and v.shape[0]==1 and v.shape[2]==2
  for k in range(2):
   q=v[0,:,k]-v[0,:,k].max();q=np.exp(q);q/=q.sum();entropy.append(float(-(q*np.log(np.maximum(q,1e-300))).sum()/np.log(len(q))))
 iv=prediction(logits,torch.as_tensor(b),records,list(map(int,ids)))['raw_physical_intervals'];assert len(iv)==2
 a,c=iv[0];d,e=iv[1];inter=max(0,min(c,e)-max(a,d));iou=inter/(c-a+e-d-inter)
 result=dict(expert_weakness=1-float(scores.mean()),LOO_geometric_instability=float(np.median(loo)),Source_temporal_confidence=float((1-np.mean(entropy))*iou),expert_mean_score=float(scores.mean()),expert_min_score=float(scores.min()),LOO_residuals=loo,Source_normalized_endpoint_entropies=entropy,Source_twooffset_temporal_IoU=float(iou),GT_read=False)
 assert all(np.isfinite(result[k])for k in ['expert_weakness','LOO_geometric_instability','Source_temporal_confidence']);return result
def empirical_rank(reference,x):
 a=np.sort(np.asarray(reference,np.float64));assert a.ndim==1 and len(a)>0
 # Midrank, including a matched half-weight for source ties; no target fitting.
 return float((np.searchsorted(a,x,side='left')+np.searchsorted(a,x,side='right'))/(2*len(a)))
def risks(train_cases,case):
 weak=empirical_rank([v['expert_weakness']for v in train_cases],case['expert_weakness']);geom=empirical_rank([v['LOO_geometric_instability']for v in train_cases],case['LOO_geometric_instability']);expert=(weak+geom)/2;confidence=case['Source_temporal_confidence']
 return dict(expert_risk=expert,Source_confidence=confidence,combined=expert*confidence)
