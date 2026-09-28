"""Independent scalar evidence scoring on the locked observed physical grid."""
import numpy as np
from vg_tta.external_privileged_views import dense_spatial_support

def external_prediction(evidence,frame_ids):
    boxes,diagnostic=dense_spatial_support(frame_ids,evidence)
    return {'frame_ids':list(frame_ids),'dense_xyxy':[[0.,0.,0.,0.] if b is None else b for b in boxes],
        'interval_physical':evidence['interval_physical'],'format_valid':evidence['format_valid'],'errors':evidence['errors'],
        'spatial_support':diagnostic}

def scalar_external(pred,label):
    assert pred['frame_ids']==label['frame_ids']
    ids=pred['frame_ids'];support=[a and b for a,b in zip(label['box_valid'],label['event_active'])]
    ious=[]
    for a,b in zip(pred['dense_xyxy'],label['boxes_xyxy']):
        inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
        union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter
        ious.append(inter/max(union,1e-7))
    n=sum(support);s=sum(v for v,yes in zip(ious,support) if yes)/max(1,n)
    interval=pred['interval_physical'];v=t=0.
    if interval is not None:
        p0,p1=interval[0],interval[1]+1
        g0,g1=label['event_interval']['begin_fid'],label['event_interval']['end_fid']
        lo,hi=max(p0,g0),min(p1,g1);u0,u1=min(p0,g0),max(p1,g1)
        t=max(0.,hi-lo)/max(1.,u1-u0)
        v=sum(iou for fid,iou,yes in zip(ids,ious,support) if yes and lo<=fid<hi)/max(1,sum(u0<=fid<u1 for fid in ids))
    return {'vIoU':v,'sIoU':s,'tIoU':t,'format_ok':pred['format_valid'],'spatial_support_frames':n}

def tensor_metrics(pred,label,*,external=False):
    """Separate vectorized geometry/reductions, used only to crosscheck readback."""
    import torch
    dtype=torch.float64;ids=torch.tensor(label['frame_ids'],dtype=dtype)
    support=torch.tensor(label['box_valid'],dtype=torch.bool)&torch.tensor(label['event_active'],dtype=torch.bool)
    if external:
        p=torch.tensor(pred['dense_xyxy'],dtype=dtype);interval=pred['interval_physical']
        interval=None if interval is None else [interval[0],interval[1]+1]
    else:
        p=torch.zeros(len(ids),4,dtype=dtype)
        valid=pred['format_ok'] and pred['interval'] is not None
        if not valid:return {m:0. for m in ('vIoU','sIoU','tIoU')}
        for i,pos in enumerate(pred['positions']):
            if pred['geometry_valid'][i]:
                c=torch.as_tensor(pred['boxes_cxcywh'][i],dtype=dtype);p[pos]=torch.cat([c[:2]-c[2:]/2,c[:2]+c[2:]/2])
        interval=[float(ids[pred['interval'][0]]),float(ids[pred['interval'][1]])+1]
    g=torch.tensor(label['boxes_xyxy'],dtype=dtype)
    inter=(torch.minimum(p[:,2:],g[:,2:])-torch.maximum(p[:,:2],g[:,:2])).clamp(min=0).prod(-1)
    union=(p[:,2:]-p[:,:2]).clamp(min=0).prod(-1)+(g[:,2:]-g[:,:2]).clamp(min=0).prod(-1)-inter
    iou=inter/union.clamp(min=1e-7);s=float(iou[support].sum()/support.sum().clamp(min=1));v=t=0.
    if interval is not None:
        a=torch.tensor(interval,dtype=dtype);b=torch.tensor([label['event_interval']['begin_fid'],label['event_interval']['end_fid']],dtype=dtype)
        lo=torch.maximum(a[0],b[0]);hi=torch.minimum(a[1],b[1]);u0=torch.minimum(a[0],b[0]);u1=torch.maximum(a[1],b[1])
        t=float((hi-lo).clamp(min=0)/(u1-u0).clamp(min=1))
        v=float(iou[support&(ids>=lo)&(ids<hi)].sum()/((ids>=u0)&(ids<u1)).sum().clamp(min=1))
    return {'vIoU':v,'sIoU':s,'tIoU':t}
