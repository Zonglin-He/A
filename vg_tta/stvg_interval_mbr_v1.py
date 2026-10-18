"""Exact finite legal-interval expected-tIoU decisions, no GT or actionness."""
import numpy as np

def legal_distribution(logits, frame_ids):
 z=np.asarray(logits,dtype=np.float64).reshape(-1,2);f=np.asarray(frame_ids,dtype=np.int64)
 assert len(z)==len(f) and np.isfinite(z).all() and (np.diff(f)>0).all()
 s,e=np.triu_indices(len(z),1);a=z[s,0]+z[e,1];a-=a.max();q=np.exp(a);q/=q.sum()
 return np.stack((f[s],f[e]+1),1),q,np.stack((s,e),1)

def expected_iou(intervals,q,block=128):
 iv=np.asarray(intervals,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
 assert (iv[:,1]>iv[:,0]).all() and (q>=0).all() and abs(q.sum()-1)<1e-10
 out=np.empty(len(iv));length=iv[:,1]-iv[:,0]
 for i in range(0,len(iv),block):
  a=iv[i:i+block];inter=np.maximum(0,np.minimum(a[:,None,1],iv[None,:,1])-np.maximum(a[:,None,0],iv[None,:,0]))
  out[i:i+block]=(inter/(length[i:i+block,None]+length[None,:]-inter))@q
 return out

def mbr(logits,frame_ids):
 iv,q,ij=legal_distribution(logits,frame_ids);utility=expected_iou(iv,q);k=int(utility.argmax())
 return dict(interval=iv[k].tolist(),indices=ij[k].tolist(),expected_tIoU=float(utility[k]),map_expected_tIoU=float(utility[int(q.argmax())]),posterior_sum=float(q.sum()))

def event_membership(logits,records,frame_ids):
 ids=np.asarray(frame_ids);p=np.zeros(len(ids))
 for z,r in zip(logits,records):
  iv,q,_=legal_distribution(z,r['frame_ids']);p+=(((ids[:,None]>=iv[None,:,0])&(ids[:,None]<iv[None,:,1]))@q)/len(records)
 return p
