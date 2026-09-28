"""Physical-time evidence; raw errors retained, usability local to observations."""
from __future__ import annotations
import re
from itertools import combinations
import numpy as np
from PIL import Image, ImageFilter
NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def time_support(frame_ids,clip_bounds=None):
    ids=np.asarray(frame_ids,dtype=np.float64)
    if ids.ndim!=1 or len(ids)<2 or not np.isfinite(ids).all() or np.any(np.diff(ids)<=0):
        raise ValueError('Distinct ordered physical observations required')
    lo,hi=map(float,clip_bounds if clip_bounds is not None else (ids[0],ids[-1]))
    if not np.isfinite([lo,hi]).all() or lo>=hi or ids[0]<lo or ids[-1]>hi:raise ValueError('Invalid physical clip bounds')
    return ids,lo,hi


def nearest_observations(ids,times):
    """Earlier observation on a tie, including roundoff within 8*float64 epsilon at the support scale."""
    times=np.atleast_1d(np.asarray(times,dtype=np.float64))
    distances=np.abs(ids[:,None]-times)
    tolerance=8*np.finfo(np.float64).eps*np.maximum(1,np.maximum(np.abs(ids).max(),np.abs(times)))
    return (distances<=distances.min(axis=0)+tolerance).argmax(axis=0)


def repeated_frame_indices(frame_ids,count=100,*,clip_bounds=None):
    ids,lo,hi=time_support(frame_ids,clip_bounds)
    if count<2:raise ValueError('At least two physical slots required')
    slots=np.linspace(lo,hi,count)
    positions=nearest_observations(ids,slots)
    return positions,np.asarray(frame_ids)[positions]


def time_to_physical(value,frame_ids,*,clip_bounds=None):
    _,lo,hi=time_support(frame_ids,clip_bounds)
    if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid normalized time')
    return float(lo+value*(hi-lo))


def box_iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    inter=float(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod())
    union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)
    return inter/union if union>0 else 0.


def parse_teacher_text(text,frame_ids,*,clip_bounds=None):
    ids,lo,hi=time_support(frame_ids,clip_bounds)
    normalized=re.sub(r'<(TEMP|WIDTH|HEIGHT)-(\d+)>',lambda m:repr(int(m.group(2))/99)+',',text)
    normalized=normalized.replace(',]',']').replace(',}','}')
    spans=re.findall(r'\{\s*('+NUMBER+r')\s*,\s*('+NUMBER+r')\s*\}',normalized)
    pairs=list(re.finditer(r'('+NUMBER+r')\s*,?\s*:\s*\[([^\]]*)\]',normalized))
    interval=None;errors=[];boxes=[]
    if len(spans)!=1:errors.append('missing_or_ambiguous_interval')
    else:
        a,b=map(float,spans[0])
        if 0<=a<=b<=1:interval=[lo+a*(hi-lo),lo+b*(hi-lo)]
        else:errors.append('invalid_interval')
    for index,m in enumerate(pairs):
        t=float(m.group(1));parts=[v.strip() for v in m.group(2).split(',') if v.strip()]
        syntax=len(parts)==4 and all(re.fullmatch(NUMBER,v) for v in parts)
        box=list(map(float,parts)) if syntax else None
        valid_time=bool(np.isfinite(t) and 0<=t<=1)
        physical=lo+t*(hi-lo) if valid_time else None
        pos=int(nearest_observations(ids,[physical])[0]) if valid_time else None
        valid=bool(syntax and np.isfinite(box).all() and all(0<=v<=1 for v in box) and box[0]<box[2] and box[1]<box[3])
        saved_box=None if box is None else [v if np.isfinite(v) else None for v in box]
        boxes.append(dict(raw_index=index,raw_match=m.group(0),normalized_time=t if np.isfinite(t) else None,physical_frame=physical,
            observation_position=pos,observation_frame=None if pos is None else float(ids[pos]),box=saved_box,
            valid_time=valid_time,valid_geometry=valid,valid_format=bool(syntax)))
        if not valid_time:errors.append('invalid_box_time')
        if not syntax:errors.append('invalid_box_format')
        elif not valid:errors.append('invalid_box_geometry')
    if not boxes:errors.append('missing_boxes')
    if normalized.count(':')!=len(pairs):errors.append('unparsed_box_entries')
    groups=[]
    for pos in sorted({r['observation_position'] for r in boxes if r['valid_time']}):
        rows=[r for r in boxes if r['observation_position']==pos];valid=[r['box'] for r in rows if r['valid_geometry']]
        values=np.asarray(valid,dtype=float);pairwise=[box_iou(a,b) for a,b in combinations(valid,2)]
        groups.append(dict(observation_position=pos,physical_frame=float(ids[pos]),raw_indices=[r['raw_index'] for r in rows],
            raw_count=len(rows),duplicate_count=max(0,len(rows)-1),valid_count=len(valid),invalid_count=len(rows)-len(valid),
            usable=bool(valid),box=np.median(values,axis=0).tolist() if valid else None,
            pairwise_iou=pairwise,pairwise_iou_min=min(pairwise) if pairwise else None,
            coordinate_range=np.ptp(values,axis=0).tolist() if valid else None,coordinate_std=np.std(values,axis=0).tolist() if valid else None))
    return dict(schema='physical_clip_time_v2',normalized_text=normalized,clip_bounds=[lo,hi],observation_frame_ids=ids.tolist(),
        interval_physical=interval,boxes=boxes,frame_groups=groups,errors=errors,format_valid=not errors,
        temporal_usable=interval is not None,spatial_usable=any(g['usable'] for g in groups),
        aggregation='nearest existing physical observation (ties earlier); coordinate-wise median of valid local boxes',
        duplicate_count=sum(g['duplicate_count'] for g in groups),disagreement_threshold=None)


def dense_spatial_support(frame_ids,evidence):
    ids=np.asarray(frame_ids,dtype=np.float64)
    if not np.array_equal(ids,evidence['observation_frame_ids']):raise ValueError('Evidence observation support mismatch')
    groups=evidence['frame_groups'];direct={g['observation_position']:g for g in groups}
    valid=sorted([g for g in groups if g['usable']],key=lambda g:g['physical_frame'])
    dense=[None]*len(ids);kind=['uncovered']*len(ids);gaps=[]
    for i,t in enumerate(ids):
        if i in direct:
            if direct[i]['usable']:dense[i]=direct[i]['box'];kind[i]='direct_aggregate'
            else:kind[i]='invalid_local_fallback'
            continue
        left=[g for g in valid if g['physical_frame']<t];right=[g for g in valid if g['physical_frame']>t]
        if not left or not right:continue
        a,b=left[-1],right[0]
        if any(not g['usable'] and a['physical_frame']<g['physical_frame']<b['physical_frame'] for g in groups):
            kind[i]='invalid_anchor_barrier';continue
        alpha=(t-a['physical_frame'])/(b['physical_frame']-a['physical_frame'])
        dense[i]=((1-alpha)*np.asarray(a['box'])+alpha*np.asarray(b['box'])).tolist();kind[i]='interpolated'
        gaps.append(float(b['physical_frame']-a['physical_frame']))
    ts=[g['physical_frame'] for g in valid]
    rawts=sorted(r['physical_frame'] for r in evidence['boxes'] if r['valid_time'] and r['valid_geometry'])
    diag=dict(raw_box_count=len(evidence['boxes']),valid_anchor_count=len(valid),duplicate_count=evidence['duplicate_count'],
        covered_observations=sum(b is not None for b in dense),coverage_fraction=sum(b is not None for b in dense)/len(ids),
        max_anchor_gap_physical=float(max(np.diff(ts),default=0.)),max_raw_box_gap_physical=float(max(np.diff(rawts),default=0.)),
        max_used_interpolation_gap_physical=max(gaps,default=0.),gap_limit=None,support_kind=kind,
        interpolation='linear physical time within valid anchors; no extrapolation; invalid anchor barrier')
    return dense,diag


def temporal_view(frames,frame_ids,interval,dim=.25):
    frames=np.asarray(frames);ids=np.asarray(frame_ids)
    if frames.dtype!=np.uint8 or frames.shape[0]!=len(ids):raise ValueError('RGB uint8 frame identity mismatch')
    if interval is None:return frames.copy(),dict(fallback='no_temporal_support',changed_frames=0)
    if len(interval)!=2 or not np.isfinite(interval).all() or interval[0]>interval[1]:raise ValueError('Invalid interval')
    keep=(ids>=interval[0])&(ids<=interval[1]);out=frames.copy()
    out[~keep]=np.rint(out[~keep].astype(np.float32)*dim).clip(0,255).astype(np.uint8)
    return out,dict(fallback=None,changed_frames=int((~keep).sum()),keep=keep.tolist())


def spatial_view(frames,frame_ids,evidence,radius=8):
    out=np.asarray(frames).copy()
    if out.dtype!=np.uint8 or len(out)!=len(frame_ids):raise ValueError('RGB uint8 frame identity mismatch')
    dense,diagnostic=dense_spatial_support(frame_ids,evidence);changed=[];h,w=out.shape[1:3]
    for i,box in enumerate(dense):
        if box is None:continue
        box=np.asarray(box);x1,y1=np.floor(box[:2]*[w,h]).astype(int);x2,y2=np.ceil(box[2:]*[w,h]).astype(int)
        blurred=np.asarray(Image.fromarray(out[i]).filter(ImageFilter.GaussianBlur(radius))).copy()
        blurred[y1:y2,x1:x2]=out[i,y1:y2,x1:x2]
        if not np.array_equal(blurred,out[i]):changed.append(i)
        out[i]=blurred
    return out,dict(diagnostic,fallback=None if evidence['spatial_usable'] else 'no_valid_spatial_support',changed_frames=len(changed),positions=changed)
