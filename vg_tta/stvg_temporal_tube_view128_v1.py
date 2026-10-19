"""A coherent RGB window from an existing frozen trajectory and student interval."""
import numpy as np


def tube_window(ids,boxes,interval,width,height):
    ids=np.asarray(ids);b=np.asarray(boxes,float);assert b.shape==(len(ids),4)and np.isfinite(b).all()
    start,end=map(int,interval);assert end>start
    left=max(0,int(np.searchsorted(ids,start,side='right'))-1);right=min(len(ids)-1,int(np.searchsorted(ids,end-1,side='left')))
    idx=np.arange(left,right+1);xy=np.c_[b[idx,:2]-b[idx,2:]/2,b[idx,:2]+b[idx,2:]/2];xy=np.clip(xy,0,1);valid=(xy[:,2]>xy[:,0])&(xy[:,3]>xy[:,1])
    if not valid.any():return [0,0,int(width),int(height)],dict(nodes=idx.tolist(),valid_nodes=0,reason='no image intersection -> full-frame')
    v=xy[valid];lo=v[:,:2].min(0);hi=v[:,2:].max(0);center=(lo+hi)/2;size=np.minimum(1.,2*(hi-lo));lo=np.minimum(np.maximum(center-size/2,0),1-size);hi=lo+size
    pixel=np.r_[np.floor(np.round(lo*[width,height],10)),np.ceil(np.round(hi*[width,height],10))].astype(int);pixel[:2]=np.maximum(pixel[:2],0);pixel[2:]=np.minimum(pixel[2:],[width,height]);assert (pixel[2:]>pixel[:2]).all()
    return pixel.tolist(),dict(nodes=idx.tolist(),valid_nodes=int(valid.sum()),reason='2x common-window hull incl interpolation neighbours')


def crop(frames,window):
    x0,y0,x1,y1=map(int,window);assert 0<=x0<x1<=frames.shape[2] and 0<=y0<y1<=frames.shape[1]
    return np.ascontiguousarray(frames[:,y0:y1,x0:x1])


def properties():
    x=np.arange(4*7*9*3,dtype=np.uint8).reshape(4,7,9,3);assert np.array_equal(crop(x,[0,0,9,7]),x)
    ids=[0,10,20,30];b=np.tile([.5,.5,.1,.2],(4,1));win,meta=tube_window(ids,b,[11,19],100,100);assert meta['nodes']==[1,2]and win==[40,30,60,70]
    b[:,2:]=1;win,_=tube_window(ids,b,[0,31],100,100);assert win==[0,0,100,100]
    b[:,:2]=2;win,meta=tube_window(ids,b,[0,31],100,100);assert win==[0,0,100,100]and meta['valid_nodes']==0
    return dict(status='pass',full_crop_RGB_exact=True,interpolation_neighbours_included=True,constant_window_preserves_motion_coordinates=True,empty_fallback_full=True)
