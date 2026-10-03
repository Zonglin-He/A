"""Metric/readout-only GT interventions; no label-conditioned decoder interface."""
import numpy as np

def event_positions(frame_ids,span):
    eligible=np.flatnonzero((np.asarray(frame_ids)>=span[0])&(np.asarray(frame_ids)<span[1])).tolist()
    if len(eligible)<5:return dict(eligible=False,available=len(eligible),positions=[],reason='sampling_support_insufficient')
    j=np.rint(np.linspace(0,len(eligible)-1,5)).astype(int)
    positions=[eligible[i] for i in j];assert len(set(positions))==5
    return dict(eligible=True,available=len(eligible),positions=positions,reason='GT_event_quantiles')

class DenseTube:
    """Vectorized official dense IoU, preserving no-extrapolation and HC endpoints."""
    def __init__(self,normalized,row,truth,span,clip=False):
        from vg_tta.tastvg_paper48_metrics_v1 import xyxy
        ids=np.asarray(row['frame_ids']);boxes=xyxy(normalized,row['input']['width'],row['input']['height'])
        if clip:boxes=np.maximum(boxes,0)
        self.fids=np.array(sorted(truth));self.span=span
        t=np.array([truth[int(f)] for f in self.fids]);p=np.column_stack([np.interp(self.fids,ids,boxes[:,k]) for k in range(4)])
        inter=np.maximum(np.minimum(p[:,2:],t[:,2:])-np.maximum(p[:,:2],t[:,:2]),0).prod(1)
        union=np.maximum(p[:,2:]-p[:,:2],0).prod(1)+np.maximum(t[:,2:]-t[:,:2],0).prod(1)-inter
        self.iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        self.iou[(self.fids<ids[0])|(self.fids>ids[-1])]=0
        self.s=float(self.iou.mean());self.sample_support=[int(ids[0]),int(ids[-1])]
    def score(self,interval):
        a,z=map(int,interval);g,h=self.span;assert z>a and h>g
        overlap=max(0,min(z,h)-max(a,g));union=z-a+h-g-overlap
        mask=(self.fids>=max(a,g))&(self.fids<min(z,h))
        return dict(v=float(self.iou[mask].sum()/max(max(z,h)-min(a,g),1)),t=float(overlap/union),s=self.s)

def physical(indices,ids):return [ids[indices[0]],ids[indices[1]]+1]

def official(normalized,row,truth,span,interval,dataset,gt_space=False):
    from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
    from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
    if gt_space:
        ids=sorted(truth);boxes=np.array([truth[f] for f in ids])
    else:
        ids=row['frame_ids'];boxes=xyxy(normalized,row['input']['width'],row['input']['height'])
        if dataset=='hc2':boxes=np.maximum(boxes,0)
    m=(DenseMetric() if dataset=='vidstg' else HC2DenseMetric())(boxes,ids,interval,truth,span)
    return dict(v=float(m['m_vIoU']),t=float(m['m_tIoU']),s=float(m['sIoU_dense_GT']))

def box_iou(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    inter=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0).prod(-1)
    union=np.maximum(a[...,2:]-a[...,:2],0).prod(-1)+np.maximum(b[...,2:]-b[...,:2],0).prod(-1)-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
