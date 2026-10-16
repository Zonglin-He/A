"""Isolated, fixed spatial self-supervision diagnostics, not deployment code."""
import hashlib
import numpy as np
import torch
from methods.decota_final_simplified_v1.objectives import generalized_iou


def image_hash(frames):
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def inverse_boxes(boxes, transforms=None, flip=False):
    out = boxes.clone()
    if flip:
        out[:, 0] = 1-out[:, 0]
    if transforms is not None:
        tr = torch.as_tensor(transforms, dtype=out.dtype, device=out.device)
        out = torch.cat((out[:, :2]*tr[:, 2:]+tr[:, :2], out[:, 2:]*tr[:, 2:]), -1)
    return out


def views(frames, native):
    """Pixel views and exact normalized inverse maps, with no evaluation labels."""
    import cv2
    n,h,w,_ = frames.shape
    l,t = int(np.floor(.05*w)), int(np.floor(.05*h))
    r,b = int(np.ceil(.95*w)), int(np.ceil(.95*h))
    tr = np.tile([l/w,t/h,(r-l)/w,(b-t)/h], (n,1))
    crop = frames[:, t:b, l:r].copy()
    roi, trs, pixels = [], [], []
    for frame, box in zip(frames, np.asarray(native)):
        cx,cy,bw,bh = box*np.array([w,h,w,h])
        rw,rh = max(16,2*bw),max(16,2*bh)
        x0,y0 = int(np.floor(cx-rw/2)),int(np.floor(cy-rh/2))
        x1,y1 = int(np.ceil(cx+rw/2)),int(np.ceil(cy+rh/2))
        x0,y0 = max(0,min(w-1,x0)),max(0,min(h-1,y0))
        x1,y1 = max(x0+1,min(w,x1)),max(y0+1,min(h,y1))
        # In-border minimum, deterministic and independent of GT.
        if x1-x0<min(16,w):
            x0=max(0,min(x0,w-16));x1=min(w,x0+16)
        if y1-y0<min(16,h):
            y0=max(0,min(y0,h-16));y1=min(h,y0+16)
        roi.append(cv2.resize(frame[y0:y1,x0:x1],(224,224),interpolation=cv2.INTER_LINEAR))
        trs.append([x0/w,y0/h,(x1-x0)/w,(y1-y0)/h]);pixels.append([x0,y0,x1,y1])
    return {
        'flip':dict(frames=frames[:,:,::-1].copy(), transforms=None, flip=True),
        'crop':dict(frames=crop, transforms=tr, flip=False, pixel_crop=[l,t,r,b]),
        'cycle':dict(frames=np.stack(roi), transforms=np.array(trs), flip=False, pixel_crop=pixels),
        'roundtrip_noop':dict(frames=frames[:,:,::-1][:,:,::-1].copy(),transforms=None,flip=False),
    }


class BoxLoss:
    def __init__(self, target, positions, denominator):
        self.target = target.detach().clone()
        self.positions = torch.as_tensor(positions, device=target.device, dtype=torch.long)
        self.denominator = max(float(denominator),1.)
        if len(self.target) != len(self.positions): raise ValueError('support mismatch')

    def __call__(self, boxes):
        if not len(self.positions): return boxes.sum()*0
        p = boxes[self.positions]
        return (5*(p-self.target).abs().sum(-1)+2*(1-generalized_iou(p,self.target))).sum()/self.denominator


def velocity_energy(boxes, times):
    dt = times[1:]-times[:-1]
    return (((boxes[1:]-boxes[:-1])/dt[:,None]).square().sum(-1)*dt).sum()/(times[-1]-times[0])


def gradient_vector(replay, loss):
    pp = [p for _,p in replay.named]
    grads = torch.autograd.grad(loss,pp,allow_unused=False)
    return torch.cat([g.detach().reshape(-1).double() for g in grads])


def cosine(a,b):
    den = a.norm()*b.norm()
    return float(torch.dot(a,b)/den) if den > 1e-20 else None
