"""CPU-only paper readouts and prelocked schedules; no model or GT loading."""
import numpy as np

MIXED_REGIMES=['clean','frame_drop_5','motion_blur_10','clean','occlusion_5','exposure_10','frame_freeze_5']

def standard_columns(values):
    """s,t,v -> t,v,recall. v must already be computed by the declared evaluator."""
    a=np.asarray(values,dtype=float)
    if a.shape[-1]!=3 or not np.isfinite(a).all():raise ValueError('Expected finite s,t,v')
    return np.stack([a[...,1],a[...,2],a[...,2]>.3,a[...,2]>.5],axis=-1)

def arrival_kind(first,expert):
    return 'expert' if expert else ('first_source_nonexpert' if first else 'later_source_nonexpert')

def quartile(arrival,total):
    if not 0<=arrival<total:raise ValueError('Invalid arrival')
    return min(3,4*arrival//total)

def mixed_schedule(rows,sequence):
    sources=[];seen=set();last=None
    for i in sequence:
        s=rows[i]['source']
        if s!=last:
            if s in seen:raise ValueError('Expected contiguous source blocks')
            sources.append(s);seen.add(s);last=s
    assignment={s:MIXED_REGIMES[min(6,j*7//len(sources))] for j,s in enumerate(sources)}
    return [dict(parent=i,arrival=j,condition=assignment[rows[i]['source']],reset=(j==0),expert_scheduled=j%4==0) for j,i in enumerate(sequence)]

def dense_official_metrics(sampled_xyxy,frame_ids,pred_interval,gt_xyxy,gt_interval):
    """Official-style dense linear interpolation + physical-frame union.

    Input boxes share coordinates. No extrapolation beyond sampled support.
    Caller must supply actual dense annotation; never fabricate it from sparse GT.
    This is a metric kernel, not a completed official-data equivalence audit.
    """
    b=np.asarray(sampled_xyxy,dtype=float);ids=np.asarray(frame_ids,dtype=int)
    if b.shape!=(len(ids),4) or not np.isfinite(b).all() or len(ids)==0 or np.any(np.diff(ids)<=0):raise ValueError('Invalid sampled tube')
    a,z=map(int,pred_interval);g,h=map(int,gt_interval)
    if z<=a or h<=g:raise ValueError('Expected positive half-open intervals')
    inter=max(0,min(z,h)-max(a,g));tiou=inter/(z-a+h-g-inter)
    total=0.
    for fid,truth in gt_xyxy.items():
        fid=int(fid)
        if not max(a,g)<=fid<min(z,h) or not ids[0]<=fid<=ids[-1]:continue
        pred=np.array([np.interp(fid,ids,b[:,k]) for k in range(4)])
        truth=np.asarray(truth,dtype=float).reshape(4)
        area=np.maximum(np.minimum(pred[2:],truth[2:])-np.maximum(pred[:2],truth[:2]),0).prod()
        union=np.maximum(pred[2:]-pred[:2],0).prod()+np.maximum(truth[2:]-truth[:2],0).prod()-area
        total+=area/union if union>0 else 0.
    v=total/max(max(z,h)-min(a,g),1)
    return dict(m_tIoU=tiou,m_vIoU=v,**{'vIoU@0.3':float(v>.3),'vIoU@0.5':float(v>.5)})
