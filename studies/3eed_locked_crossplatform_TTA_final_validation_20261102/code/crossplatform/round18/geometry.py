"""Label-free candidate grouping and cross-view geometry correspondence."""
import numpy as np
from scipy.optimize import linear_sum_assignment
def iou_matrix(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);sa=np.maximum(a[:,3:],1e-6);sb=np.maximum(b[:,3:],1e-6)
 lo=np.maximum(a[:,None,:3]-sa[:,None]/2,b[None,:,:3]-sb[None]/2);hi=np.minimum(a[:,None,:3]+sa[:,None]/2,b[None,:,:3]+sb[None]/2);inter=np.maximum(hi-lo,0).prod(-1);return inter/np.maximum(sa.prod(-1)[:,None]+sb.prod(-1)[None]-inter,1e-12)
def groups(boxes,scores):
 boxes=np.asarray(boxes);scores=np.asarray(scores);mat=iou_matrix(boxes,boxes);order=np.argsort(-scores,kind='stable');g=[];reps=[]
 for q in order:
  j=next((i for i,r in enumerate(reps) if mat[q,r]>=.5),None)
  if j is None:reps.append(int(q));g.append([]);j=len(g)-1
  g[j].append(int(q))
 return g,np.array(reps,int)
def match_boxes(anchor,other):
 iou=iou_matrix(anchor,other);scale=np.maximum((np.maximum(anchor[:,3:],.1)[:,None]+np.maximum(other[:,3:],.1)[None])/2,.1)
 centre=np.mean(np.abs(anchor[:,None,:3]-other[None,:,:3])/scale,-1);size=np.mean(np.abs(np.log(np.maximum(anchor[:,None,3:],.05)/np.maximum(other[None,:,3:],.05))),-1)
 cost=1-iou+centre+.25*size;r,c=linear_sum_assignment(cost);ok=(iou[r,c]>=.1)|((centre[r,c]<.5)&(size[r,c]<.7))
 return r[ok],c[ok],{'assigned':len(r),'accepted':int(ok.sum()),'mean_accepted_IoU':float(iou[r[ok],c[ok]].mean()) if ok.any() else None}
