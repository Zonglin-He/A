"""Exact soft feature-footprint interventions, leaving source padding unchanged."""
import torch,numpy as np

def footprint(box,h,w,valid):
 box=box.double();yy=torch.arange(h,dtype=torch.float64,device=box.device)/h;xx=torch.arange(w,dtype=torch.float64,device=box.device)/w
 x0,y0=box[:2]-box[2:]/2;x1,y1=box[:2]+box[2:]/2
 dx=(torch.minimum(xx+1/w,x1)-torch.maximum(xx,x0)).clamp_min(0)*w;dy=(torch.minimum(yy+1/h,y1)-torch.maximum(yy,y0)).clamp_min(0)*h
 return (dy[:,None]*dx[None,:]).clamp(0,1)*valid

def reflected(mask,valid):
 ij=torch.nonzero(valid);assert len(ij)>0;y0,x0=ij.min(0).values.tolist();y1,x1=ij.max(0).values.tolist();assert bool(valid[y0:y1+1,x0:x1+1].all())and int(valid.sum())==(y1-y0+1)*(x1-x0+1)
 out=torch.zeros_like(mask);out[y0:y1+1,x0:x1+1]=mask[y0:y1+1,x0:x1+1].flip((0,1));assert abs(float(out.sum()-mask.sum()))<1e-10;return out

def replace(x,masks,valid):
 assert x.ndim==4 and masks.shape==valid.shape==(x.shape[0],x.shape[2],x.shape[3]);m=masks.float()*valid;bg=(1-m)*valid;den=bg.sum((1,2));fallback=den==0;weights=torch.where(fallback[:,None,None],valid.float(),bg);mean=(x.float()*weights[:,None]).sum((2,3))/weights.sum((1,2))[:,None];out=(x.float()*(1-m[:,None])+mean[:,:,None,None]*m[:,None]).to(x.dtype)
 assert torch.equal(out.permute(0,2,3,1)[~valid],x.permute(0,2,3,1)[~valid]);return out,int(fallback.sum())

def posteriors(qs):
 q0=qs['Full0'];take=q0>0
 def log(q):
  assert np.array_equal(q>0,take);out=np.zeros_like(q);out[take]=np.log(q[take]);return out
 from research.native_event_text_v1.core import normalize
 z={k:log(v)for k,v in qs.items()};target=z['Full0']-z['FullTarget']-z['Id0']+z['IdTarget'];control=z['Full0']-z['FullControl']-z['Id0']+z['IdControl'];score=target-control;spatial=z['FullControl']-z['FullTarget'];identity=z['IdControl']-z['IdTarget'];direct=np.zeros_like(q0);v=score[take];v=np.exp(v-v.max());direct[take]=v/v.sum()
 return dict(NativeG6=q0,InteractionResidual=normalize(q0,score),DirectInteraction=direct,TargetInteraction=normalize(q0,target),SpatialOnly=normalize(q0,spatial),IdentityOnly=normalize(q0,identity),TextOnly=normalize(q0,z['Full0']-z['Id0']),Sharpen=normalize(q0,z['Full0']),ReverseInteraction=normalize(q0,score[::-1,::-1].T.copy()),NoInteraction=q0.copy()),score
