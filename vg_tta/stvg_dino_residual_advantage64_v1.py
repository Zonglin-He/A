"""Offline geometry and instance-evidence diagnostics; not a teacher/model."""
import numpy as np

def xyxy(b):
 b=np.asarray(b,float);return np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)

def ious(boxes,truth,row,ds):
 b=xyxy(np.asarray(boxes,float).reshape(-1,4))*np.array([row['input']['width'],row['input']['height']]*2)
 if ds=='hc2':b=np.maximum(b,0)
 t=np.asarray(truth,float);inter=np.maximum(0,np.minimum(b[:,2:],t[2:])-np.maximum(b[:,:2],t[:2])).prod(-1);union=np.maximum(0,b[:,2:]-b[:,:2]).prod(-1)+np.maximum(0,t[2:]-t[:2]).prod()-inter
 return np.divide(inter,union,out=np.zeros(len(b)),where=union>0)

def normalized_truth(truth,row):
 t=np.asarray(truth,float)/np.array([row['input']['width'],row['input']['height']]*2);return np.r_[(t[:2]+t[2:])/2,t[2:]-t[:2]]

def geometry(E,truth,row,ds):
 t=normalized_truth(truth,row);e=np.asarray(E,float);center=np.max(abs(e[:2]-t[:2])/t[2:]);scale=np.max(np.r_[e[2:]/t[2:],t[2:]/e[2:]]);iou=float(ious(e[None],truth,row,ds)[0]);bad=iou<=.3
 return dict(E_IoU=iou,normalized_center_error=float(center),scale_factor=float(scale),strata=dict(all=True,bad=bool(bad),far=bool(bad and center>=1),scale=bool(bad and scale>=2),scale_near_center=bool(bad and scale>=2 and center<=.25)))

def legal(b):
 a=np.asarray(b,float);return np.isfinite(a).all(-1)&(a>0).all(-1)&(a<1).all(-1)

def select(boxes,scores,threshold=.35):
 b=np.asarray(boxes,float).reshape(-1,4);s=np.asarray(scores,float);valid=legal(b)&np.isfinite(s)&(s>=threshold)
 if not valid.any():return None
 ids=np.flatnonzero(valid);return int(ids[np.argmax(s[ids])])

def mutual_iou(a,b):
 aa,bb=xyxy(np.asarray(a)),xyxy(np.asarray(b));inter=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])).prod();union=np.maximum(aa[2:]-aa[:2],0).prod()+np.maximum(bb[2:]-bb[:2],0).prod()-inter;return float(inter/union)if union>0 else 0.

def recovered(iou,E_iou):return bool(iou>=.5 and iou-E_iou>=.1)

def properties():
 row=dict(input=dict(width=100,height=80));t=[20,16,40,48];b=normalized_truth(t,row);assert abs(ious(b[None],t,row,'vidstg')[0]-1)<1e-12
 e=b.copy();e[0]+=.6;g=geometry(e,t,row,'vidstg');assert g['strata']['far'];e=b.copy();e[2:]*=3;g=geometry(e,t,row,'vidstg');assert g['strata']['scale_near_center'];assert abs(mutual_iou(b,b)-1)<1e-12;assert select(np.array([b,b]),[.2,.4])==1;assert select(np.array([b,b]),[.1,.2])is None
 return dict(status='pass',pixel_normalized_roundtrip=True,known_far_scale_positive_controls=True,existing_035_legality=True,IoU_identity=True,no_model_or_adaptation=True)
