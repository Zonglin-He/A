"""Stable crop geometry and one-offset physical posterior. No model or labels."""
import numpy as np
from scipy.special import logsumexp

def crop_windows(track,width,height,context=1.5):
    b=np.asarray(track,dtype=float)
    assert b.ndim==2 and b.shape[1]==4 and np.isfinite(b).all() and (b[:,2:]>0).all()
    lo=(b[:,:2]-b[:,2:]/2).min(0)*[width,height]
    hi=(b[:,:2]+b[:,2:]/2).max(0)*[width,height]
    center=(lo+hi)/2;extent=(hi-lo)*context
    start=np.floor(center-extent/2).astype(int);end=np.ceil(center+extent/2).astype(int)
    size=np.minimum(end-start,[width,height]);assert (size>0).all()
    start=np.maximum(0,np.minimum(start,np.array([width,height])-size));end=start+size
    real=tuple(map(int,[*start,*end]));w,h=map(int,size)
    corners=[(x,y,x+w,y+h)for x,y in [(0,0),(0,height-h),(width-w,0),(width-w,height-h)]]
    c=(start+end)/2
    shifted=min(corners,key=lambda z:(-(((np.array(z[:2])+z[2:])/2-c)**2).sum(),z))
    return real,shifted

def crop_stats(track,real,shifted,width,height):
    def iou(a,b):
        a=np.asarray(a);b=np.asarray(b);s=np.maximum(a[:2],b[:2]);e=np.minimum(a[2:],b[2:]);v=np.maximum(0,e-s).prod();return float(v/(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-v))
    b=np.asarray(track,float);targets=np.concatenate([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],1)*[width,height,width,height]
    area=(real[2]-real[0])*(real[3]-real[1])/(width*height)
    return dict(area_fraction=float(area),vertical_magnification=height/(real[3]-real[1]),horizontal_magnification=width/(real[2]-real[0]),windows_equal=real==shifted,window_iou=iou(real,shifted),real_predicted_box_mean_iou=float(np.mean([iou(real,b)for b in targets])),shifted_predicted_box_mean_iou=float(np.mean([iou(shifted,b)for b in targets])))

def single_posterior(ids,logits,full_ids):
    z=np.asarray(logits);assert z.dtype==np.float32 and z.shape==(1,len(ids),2) and np.isfinite(z).all()
    assert len(ids)>=2 and np.all(np.diff(ids)>0) and np.all(np.diff(full_ids)>0)
    a,b=np.triu_indices(len(ids),1);scores=z[0,a,0].astype(float)+z[0,b,1].astype(float);p=np.exp(scores-logsumexp(scores))
    loc={int(v):i for i,v in enumerate(full_ids)};q=np.zeros((len(full_ids),len(full_ids)),float)
    np.add.at(q,([loc[int(ids[i])]for i in a],[loc[int(ids[j])]for j in b]),p)
    assert abs(q.sum()-1)<1e-12 and np.isfinite(q).all();return q
