"""Native local features with explicit physical time and spatial correspondence.

No motion branch, text fusion, class vocabulary or fixed feature width required.
Coordinates are normalized ORIGINAL-image xyxy, not crop/patch/query indices.
Compressed tokens without spatial support cannot be silently treated as a grid.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class LocalFrame:
    time: float
    values: np.ndarray        # [local tokens, D], D arbitrary within a video
    regions: np.ndarray       # [local tokens, 4], original-image xyxy
    valid: np.ndarray
    grid_shape: tuple | None = None

    def __post_init__(self):
        v=np.asarray(self.values);b=np.asarray(self.regions);m=np.asarray(self.valid)
        if v.ndim!=2 or b.shape!=(len(v),4)or m.shape!=(len(v),):raise ValueError('Explicit token/region/valid correspondence required')
        if not np.isfinite(v).all()or not np.isfinite(b).all()or not np.isfinite(self.time):raise ValueError('Nonfinite local representation')
        if not np.all(b[:,2:]>b[:,:2])or not m.any():raise ValueError('Positive spatial support and a valid token required')
        if self.grid_shape is not None:
            h,w=self.grid_shape
            if h*w!=len(v):raise ValueError('Grid layout differs from token count')
            np.testing.assert_allclose(b,grid_regions(h,w),rtol=0,atol=1e-14)

def grid_regions(h,w):
    x,y=np.arange(w)/w,np.arange(h)/h
    xx,yy=np.meshgrid(x,y)
    return np.stack([xx,yy,xx+1/w,yy+1/h],axis=-1).reshape(-1,4)

def aggregate_frame(frame,box,context_scale=1.5):
    """Area coverage/cell area weights; empty ROI falls back to valid native support.

    box is original-image normalized cxcywh. Regular-grid branch deliberately
    preserves the selected TA reduction order; irregular support has its explicit
    coverage rule and is not claimed numerically equivalent to another adapter.
    """
    box=np.asarray(box,dtype=np.float64).copy()
    if box.shape!=(4,)or not np.isfinite(box).all()or not np.all(box[2:]>0):raise ValueError('Positive normalized cxcywh trajectory required')
    box[2:]*=context_scale;lo,hi=box[:2]-box[2:]/2,box[:2]+box[2:]/2
    values=np.asarray(frame.values,dtype=np.float64);valid=np.asarray(frame.valid,dtype=bool)
    if frame.grid_shape is not None:
        h,w=frame.grid_shape;x,y=np.arange(w)/w,np.arange(h)/h
        dx=np.maximum(0,np.minimum(x+1/w,hi[0])-np.maximum(x,lo[0]))*w
        dy=np.maximum(0,np.minimum(y+1/h,hi[1])-np.maximum(y,lo[1]))*h
        weight=dy[:,None]*dx[None,:]*valid.reshape(h,w)
        fallback=not bool(weight.sum()>0)
        if fallback:weight=valid.reshape(h,w).astype(np.float64).copy()
        pooled=np.einsum('hwc,hw->c',values.reshape(h,w,-1),weight)/weight.sum()
    else:
        region=np.asarray(frame.regions,dtype=np.float64)
        overlap=np.maximum(0,np.minimum(region[:,2:],hi)-np.maximum(region[:,:2],lo))
        weight=overlap.prod(1)/(region[:,2:]-region[:,:2]).prod(1)*valid
        fallback=not bool(weight.sum()>0)
        if fallback:weight=valid.astype(np.float64).copy()
        pooled=np.einsum('jc,j->c',values,weight)/weight.sum()
    return pooled,dict(fallback=bool(fallback),area=float(weight.sum()),tokens=int(valid.sum()))

def aggregate_sequence(frames,boxes,context_scale=1.5):
    if len(frames)!=len(boxes)or not frames:raise ValueError('Align feature frames and full-time trajectory explicitly')
    times=np.asarray([f.time for f in frames]);width=frames[0].values.shape[1]
    if not np.all(np.diff(times)>0)or any(f.values.shape[1]!=width for f in frames):raise ValueError('Increasing physical time and a consistent within-video feature space required')
    rows=[aggregate_frame(f,b,context_scale)for f,b in zip(frames,boxes)]
    return np.stack([x for x,_ in rows]),[s for _,s in rows]
