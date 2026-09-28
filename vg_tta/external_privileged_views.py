"""Geometry-preserving evidence interfaces. No model, labels, or optimizer."""
from __future__ import annotations
import re
import numpy as np
from PIL import Image, ImageFilter

NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

def repeated_frame_indices(frame_ids, count=100):
    ids=np.asarray(frame_ids,dtype=np.int64)
    if ids.ndim!=1 or len(ids)<2 or np.any(np.diff(ids)<=0):
        raise ValueError('At least two distinct ordered physical frames required')
    positions=np.rint(np.linspace(0,len(ids)-1,count)).astype(np.int64)
    return positions,ids[positions]

def time_to_physical(value,teacher_frame_ids):
    ids=np.asarray(teacher_frame_ids,dtype=np.float64)
    if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid normalized time')
    if len(ids)<2 or np.any(np.diff(ids)<0):raise ValueError('Invalid teacher time support')
    return float(np.interp(value,np.linspace(0,1,len(ids)),ids))

def parse_teacher_text(text,teacher_frame_ids):
    # Same token bin normalization as official replace_and_normalize, without
    # silently merging duplicate time-box pairs in a dictionary.
    def token(m):return f'{int(m.group(2))/99:.8f},'
    normalized=re.sub(r'<(TEMP|WIDTH|HEIGHT)-(\d+)>',token,text)
    normalized=normalized.replace(',]',']').replace(',}','}')
    spans=re.findall(r'\{\s*('+NUMBER+r')\s*,\s*('+NUMBER+r')\s*\}',normalized)
    pairs=re.findall(r'('+NUMBER+r')\s*,?\s*:\s*\[\s*('+NUMBER+r')\s*,\s*('+NUMBER+r')\s*,\s*('+NUMBER+r')\s*,\s*('+NUMBER+r')\s*\]',normalized)
    interval=None; errors=[];boxes=[]
    if len(spans)!=1:errors.append('missing_or_ambiguous_interval')
    else:
        a,b=map(float,spans[0])
        if 0<=a<=b<=1:interval=[time_to_physical(a,teacher_frame_ids),time_to_physical(b,teacher_frame_ids)]
        else:errors.append('invalid_interval')
    seen=set()
    for entry in pairs:
        t,*box=map(float,entry)
        valid=all(np.isfinite(box)) and all(0<=v<=1 for v in box) and box[0]<box[2] and box[1]<box[3]
        if not 0<=t<=1:errors.append('invalid_box_time');continue
        physical=time_to_physical(t,teacher_frame_ids)
        if physical in seen:errors.append('duplicate_physical_box_time')
        seen.add(physical)
        boxes.append(dict(normalized_time=t,physical_frame=physical,box=box,valid_geometry=valid))
        if not valid:errors.append('invalid_box_geometry')
    if not boxes:errors.append('missing_boxes')
    return dict(normalized_text=normalized,interval_physical=interval,boxes=boxes,errors=errors,
        temporal_usable=interval is not None,spatial_usable=bool(boxes) and not any(e in errors for e in ['invalid_box_time','invalid_box_geometry','duplicate_physical_box_time']))

def temporal_view(frames,frame_ids,interval,dim=.25):
    frames=np.asarray(frames);ids=np.asarray(frame_ids)
    if frames.dtype!=np.uint8 or frames.shape[0]!=len(ids):raise ValueError('RGB uint8 frame identity mismatch')
    if interval is None:return frames.copy(),dict(fallback='no_temporal_support',changed_frames=0)
    if len(interval)!=2 or not np.isfinite(interval).all() or interval[0]>interval[1]:raise ValueError('Invalid interval')
    keep=(ids>=interval[0])&(ids<=interval[1]);out=frames.copy()
    out[~keep]=np.rint(out[~keep].astype(np.float32)*dim).clip(0,255).astype(np.uint8)
    return out,dict(fallback=None,changed_frames=int((~keep).sum()),keep=keep.tolist())

def spatial_view(frames,frame_ids,evidence,radius=8):
    out=np.asarray(frames).copy();ids=np.asarray(frame_ids,dtype=np.float64)
    if out.dtype!=np.uint8 or out.shape[0]!=len(ids):raise ValueError('RGB uint8 frame identity mismatch')
    if not evidence['spatial_usable']:return out,dict(fallback='no_valid_spatial_support',changed_frames=0)
    pairs=sorted(evidence['boxes'],key=lambda r:r['physical_frame']);ts=np.array([p['physical_frame'] for p in pairs]);bs=np.array([p['box'] for p in pairs])
    changed=[];h,w=out.shape[1:3]
    for i,t in enumerate(ids):
        if t<ts[0] or t>ts[-1]:continue  # No extrapolation outside explicit support.
        box=np.array([np.interp(t,ts,bs[:,j]) for j in range(4)])
        x1,y1=np.floor(box[:2]*[w,h]).astype(int);x2,y2=np.ceil(box[2:]*[w,h]).astype(int)
        blurred=np.asarray(Image.fromarray(out[i]).filter(ImageFilter.GaussianBlur(radius))).copy()
        blurred[y1:y2,x1:x2]=out[i,y1:y2,x1:x2];out[i]=blurred;changed.append(i)
    return out,dict(fallback=None,changed_frames=len(changed),positions=changed,interpolation='physical-time linear; inside explicit support only')
