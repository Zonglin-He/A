"""Float64 batched yaw OBB IoU via 2D rectangle edge intersections and z overlap.

Independent from the 3D convex-hull auditor. Centers are shifted to GT origin;
full side lengths and yaw radians. No data or labels are read by this module.
"""
import numpy as np
S=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
def corners(boxes):
 b=np.asarray(boxes,np.float64);p=b[:,None,3:5]*S[None]/2;c=np.cos(b[:,6])[:,None];s=np.sin(b[:,6])[:,None]
 return np.stack((c*p[...,0]-s*p[...,1],s*p[...,0]+c*p[...,1]),-1)+b[:,None,:2]
def many_iou(pred,gt):
 p=np.asarray(pred,np.float64).copy();g=np.asarray(gt,np.float64).copy()
 if p.ndim!=2 or p.shape[1]!=7 or g.shape!=(7,):raise ValueError('pred[N,7], gt[7] required')
 if not np.isfinite(p).all() or not np.isfinite(g).all() or np.any(g[3:6]<=0):raise ValueError('finite positive GT required')
 out=np.zeros(len(p));valid=np.all(p[:,3:6]>0,axis=1)
 # Shift to avoid global-coordinate cancellation.
 p[:,:3]-=g[:3];g[:3]=0
 a=corners(p);b=corners(g[None])[0]
 z=np.maximum(0.,np.minimum(p[:,2]+p[:,5]/2,g[5]/2)-np.maximum(p[:,2]-p[:,5]/2,-g[5]/2))
 live=valid&(z>0)&(a.max(1)>=b.min(0)-1e-12).all(1)&(a.min(1)<=b.max(0)+1e-12).all(1)
 ix=np.flatnonzero(live)
 if not len(ix):return out
 q=p[ix];a=a[ix];n=len(ix);b=np.broadcast_to(b,(n,4,2));u=np.roll(a,-1,axis=1)-a;v=np.roll(b,-1,axis=1)-b
 cg,sg=np.cos(g[6]),np.sin(g[6]);ag=np.stack((cg*a[...,0]+sg*a[...,1],-sg*a[...,0]+cg*a[...,1]),-1)
 amask=(np.abs(ag)<=g[None,None,3:5]/2+1e-10).all(-1)
 delta=b-q[:,None,:2];cp=np.cos(q[:,6])[:,None];sp=np.sin(q[:,6])[:,None]
 bp=np.stack((cp*delta[...,0]+sp*delta[...,1],-sp*delta[...,0]+cp*delta[...,1]),-1)
 bmask=(np.abs(bp)<=q[:,None,3:5]/2+1e-10).all(-1)
 d=b[:,None,:,:]-a[:,:,None,:];den=cross(u[:,:,None,:],v[:,None,:,:]);nonparallel=np.abs(den)>1e-15
 t=np.zeros_like(den);s=np.zeros_like(den)
 np.divide(cross(d,v[:,None,:,:]),den,out=t,where=nonparallel)
 np.divide(cross(d,u[:,:,None,:]),den,out=s,where=nonparallel)
 imask=nonparallel&(t>=-1e-10)&(t<=1+1e-10)&(s>=-1e-10)&(s<=1+1e-10)
 inter=a[:,:,None,:]+t[:,:,:,None]*u[:,:,None,:]
 points=np.concatenate((a,b,inter.reshape(n,16,2)),1);mask=np.concatenate((amask,bmask,imask.reshape(n,16)),1)
 count=mask.sum(1);mean=(points*mask[...,None]).sum(1)/np.maximum(count,1)[:,None]
 angle=np.arctan2(points[...,1]-mean[:,None,1],points[...,0]-mean[:,None,0]);angle[~mask]=np.inf
 order=np.argsort(angle,axis=1,kind='stable');poly=np.take_along_axis(points,order[...,None],axis=1)
 edges=cross(poly[:,:-1],poly[:,1:]);edgevalid=np.arange(23)[None,:]<np.maximum(count-1,0)[:,None]
 last=poly[np.arange(n),np.maximum(count-1,0)];area=np.abs((edges*edgevalid).sum(1)+cross(last,poly[:,0]))/2;area[count<3]=0
 volumes=np.prod(q[:,3:6],axis=1);gv=np.prod(g[3:6]);intersection=np.minimum(np.maximum(area*z[ix],0),np.minimum(volumes,gv))
 out[ix]=intersection/(volumes+gv-intersection)
 return out
