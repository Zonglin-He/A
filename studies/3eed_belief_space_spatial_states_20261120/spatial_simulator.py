"""Binary metric questions on a pair of fixed identities in a level camera frame."""
import numpy as np
NAMES=('right','forward','right_after_yaw45','separation_grows_after_shift2')
def leveled(boxes,camera):
    right=camera[0,:2].copy()
    n=np.linalg.norm(right)
    if n<1.e-8:
        raise ValueError("Camera right has no defined horizontal direction")
    right/=n
    forward=np.array([-right[1],right[0]])
    origin=np.linalg.solve(camera[:3,:3],-camera[:3,3])[:2]
    return (boxes[:,:2]-origin)@np.stack([right,forward]).T

def outcomes(subject_boxes,reference_boxes,camera):
    s=leveled(subject_boxes,camera);r=leveled(reference_boxes,camera)
    d=s[:,None,:]-r[None,:,:]
    x,f=d[...,0],d[...,1]
    initial=np.sum(d*d,axis=2)
    shifted=np.sum((d+[2.,0.])**2,axis=2)
    return (x>0,f>0,(x-f)/np.sqrt(2.)>0,shifted>initial)

def pushforward(p,events):
    if p.sum()==0:return np.full(len(events),.5)
    return np.array([np.clip(p[e].sum(),0.,1.) for e in events])
