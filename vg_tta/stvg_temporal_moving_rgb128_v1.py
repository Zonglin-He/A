"""Exact legacy per-frame cycle crop, used only for frozen temporal observations."""
import math,hashlib
import numpy as np


def rectangles(boxes,width,height):
    b=np.asarray(boxes,dtype=np.float64);assert b.ndim==2 and b.shape[1]==4 and np.isfinite(b).all()and width>=16 and height>=16
    out=[]
    for box in b:
        cx,cy,bw,bh=box*np.array([width,height,width,height]);rw,rh=max(16,2*bw),max(16,2*bh)
        x0,y0=int(np.floor(cx-rw/2)),int(np.floor(cy-rh/2));x1,y1=int(np.ceil(cx+rw/2)),int(np.ceil(cy+rh/2))
        x0,y0=max(0,min(width-1,x0)),max(0,min(height-1,y0));x1,y1=max(x0+1,min(width,x1)),max(y0+1,min(height,y1))
        if x1-x0<min(16,width):x0=max(0,min(x0,width-16));x1=min(width,x0+16)
        if y1-y0<min(16,height):y0=max(0,min(y0,height-16));y1=min(height,y0+16)
        out.append([x0,y0,x1,y1])
    return out


def independent_rectangles(boxes,width,height):
    ans=[]
    for row in boxes:
        cx=float(row[0])*width;cy=float(row[1])*height;rw=max(16.,2*(float(row[2])*width));rh=max(16.,2*(float(row[3])*height))
        l=max(0,min(width-1,math.floor(cx-rw/2)));t=max(0,min(height-1,math.floor(cy-rh/2)));r=max(l+1,min(width,math.ceil(cx+rw/2)));b=max(t+1,min(height,math.ceil(cy+rh/2)))
        if r-l<16:l=max(0,min(l,width-16));r=min(width,l+16)
        if b-t<16:t=max(0,min(t,height-16));b=min(height,t+16)
        ans.append([l,t,r,b])
    return ans


def crop(frames,rects):
    import cv2
    assert len(frames)==len(rects)
    return np.stack([cv2.resize(frame[t:b,l:r],(224,224),interpolation=cv2.INTER_LINEAR)for frame,(l,t,r,b)in zip(frames,rects)])


def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def properties():
    from vg_tta.spatial_ssl_diagnostic_v1 import views
    rng=np.random.default_rng(4128);frames=rng.integers(0,256,(7,71,103,3),dtype=np.uint8);boxes=np.array([[.5,.5,.1,.2],[0.,.5,.01,.01],[1.,1.,.4,.2],[1.5,-.2,.01,.4],[.2,.3,1.,1.],[.4,.4,.14,.07],[.2,.6,.3,.5]])
    rect=rectangles(boxes,103,71);legacy=views(frames,boxes)['cycle'];assert rect==independent_rectangles(boxes,103,71)==legacy['pixel_crop'];assert np.array_equal(crop(frames,rect),legacy['frames'])
    return dict(status='pass',legacy_pixel_rectangles_exact=True,legacy_real_RGB_resize_exact=True,scalar_rectangles_exact=True,outside_and_minimum_cases_included=True,no_model_or_GT=True)
