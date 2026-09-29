"""Fixed observation privileges and no-GT matched-wrong location controls."""
import numpy as np
from PIL import Image,ImageFilter
from vg_tta.external_privileged_views import temporal_view,spatial_view,dense_spatial_support

def wrong_temporal_keep(keep):
    keep=np.asarray(keep,dtype=bool);n=len(keep);order=[n//2]+[j for j in range(1,n) if j!=n//2]
    for j in order:
        wrong=np.roll(keep,j)
        if not np.array_equal(wrong,keep):return wrong,j,True
    return keep.copy(),0,False

def translated_rectangle(box,width,height):
    a=np.floor(np.asarray(box[:2])*[width,height]).astype(int);b=np.ceil(np.asarray(box[2:])*[width,height]).astype(int)
    w,h=b-a;corners=[(0,0),(0,height-h),(width-w,0),(width-w,height-h)]
    center=(a+b)/2
    chosen=max(corners,key=lambda p:float(np.sum((np.asarray(p)+[w/2,h/2]-center)**2)))
    x,y=chosen;return [int(x),int(y),int(x+w),int(y+h)],[int(a[0]),int(a[1]),int(b[0]),int(b[1])]

def build_views(frames,ids,ev,dim=.25,radius=8):
    tv,td=temporal_view(frames,ids,ev['interval_physical'],dim);sv,sd=spatial_view(frames,ids,ev,radius);ts,_=spatial_view(tv,ids,ev,radius)
    if ev['interval_physical'] is None:keep=np.ones(len(ids),dtype=bool)
    else:keep=(np.asarray(ids)>=ev['interval_physical'][0])&(np.asarray(ids)<=ev['interval_physical'][1])
    wk,shift,changed=wrong_temporal_keep(keep);wt=frames.copy();wt[~wk]=np.rint(wt[~wk].astype(np.float32)*dim).clip(0,255).astype(np.uint8)
    boxes,diag=dense_spatial_support(ids,ev);rects=[]
    for box in boxes:
        rects.append(None if box is None else translated_rectangle(box,frames.shape[2],frames.shape[1]))
    def wrong_space(src):
        out=src.copy()
        for i,rect in enumerate(rects):
            if rect is None:continue
            x1,y1,x2,y2=rect[0];im=np.asarray(Image.fromarray(src[i]).filter(ImageFilter.GaussianBlur(radius))).copy();im[y1:y2,x1:x2]=src[i,y1:y2,x1:x2];out[i]=im
        return out
    views=dict(B1=frames.copy(),T=tv,S=sv,TS=ts,T_wrong=wt,S_wrong=wrong_space(frames),TS_wrong=wrong_space(wt))
    meta=dict(temporal=td,spatial=sd,correct_keep=keep.tolist(),wrong_keep=wk.tolist(),wrong_time_shift=shift,time_distinguishable=changed,
      wrong_rectangles=rects,spatial_distinguishable_frames=sum(x is not None and x[0]!=x[1] for x in rects),same_spatial_availability=True)
    assert int(keep.sum())==int(wk.sum())
    for r in rects:
        if r is not None:assert (r[0][2]-r[0][0])*(r[0][3]-r[0][1])==(r[1][2]-r[1][0])*(r[1][3]-r[1][1])
    return views,meta
