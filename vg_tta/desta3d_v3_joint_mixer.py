"""Learned shared correction in a frozen output union. No labels or decoder here."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class JointCorrectionMixer(nn.Module):
    def __init__(self, basis, hidden=128, radius=math.sqrt((.087687**2+.170316**2)/2)):
        super().__init__()
        if basis.ndim != 2 or not torch.allclose(basis.double().T@basis.double(), torch.eye(basis.shape[1], device=basis.device, dtype=torch.float64), atol=2e-6, rtol=0):
            raise ValueError('Expected audited orthonormal union')
        self.register_buffer('basis', basis.detach().float().clone())
        self.radius = float(radius)
        # Frozen B1 common stem and its two query pools supply 3*hidden channels.
        # Distinct evidence channels are inputs, never scalar gates on F.
        self.input = nn.Linear(3*hidden+8, hidden)
        self.local = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, basis.shape[1])
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z, qT, qS, evidence, stock):
        shape = z.shape[:-1]
        if evidence.shape != (*shape, 8) or stock.shape[:-1] != shape:
            raise ValueError('THW evidence/visual support mismatch')
        if not all(torch.isfinite(x).all() for x in (z,qT,qS,evidence,stock)):
            raise ValueError('Nonfinite mixer input')
        qT=qT[:,None,None,None].expand(*shape,-1)
        qS=qS[:,None,None,None].expand(*shape,-1)
        h=F.silu(self.input(torch.cat((z,qT,qS,evidence),-1)))
        h=h+F.silu(self.local(h.movedim(-1,1)).movedim(1,-1))
        a=torch.tanh(self.output(F.silu(self.mix(h))))
        # Smooth zero initialization, global norm bound, no normalize-zero or
        # output-dependent state selection. Each coefficient stays in [-1,1].
        scale=stock.detach().float().square().mean().sqrt()*math.sqrt(stock.shape[-1]/a.shape[-1])*self.radius
        return F.linear(a*scale,self.basis)


def frozen_context(adapter, fields):
    with torch.no_grad():
        z=adapter.input_proj(fields['visual_grid'])
        z=adapter.norm_stem(adapter.shared_stem(z.movedim(-1,1))).movedim(1,-1)
        t,_=adapter.query_pool_event(fields['query_tokens'],fields['query_mask'])
        s,_=adapter.query_pool_spatial(fields['query_tokens'],fields['query_mask'])
    return z.detach(),t.detach(),s.detach()


def source_evidence(label, frame_ids, height, width):
    """GT privilege: event activity/boundaries, box occupancy/knownness, THW.

    Missing boxes are UNKNOWN (occupancy0 + known0), independent of event time.
    Temporal endpoints use the first/last observed event frame; no GT prefix.
    """
    from vg_tta.desta3d_v3_latent_oracle import masks_from_source_record
    m=masks_from_source_record(label,frame_ids,height,width)
    event=m['event'];spatial=m['spatial'];n=len(frame_ids)
    known=torch.tensor(label['box_valid'],dtype=torch.float32)[None,:,None,None].expand_as(event)
    spatial=spatial*known
    pos=event[0,:,0,0].nonzero().flatten()
    boundary0=torch.zeros_like(event);boundary1=torch.zeros_like(event)
    if len(pos):boundary0[:,pos[0]]=1;boundary1[:,pos[-1]]=1
    times=torch.tensor(frame_ids,dtype=torch.float32)
    times=(times-times[0])/(times[-1]-times[0]).clamp_min(1)
    tt=times[None,:,None,None].expand_as(event)
    yy=((torch.arange(height)+.5)/height)[None,None,:,None].expand_as(event)
    xx=((torch.arange(width)+.5)/width)[None,None,None,:].expand_as(event)
    return torch.stack((event,boundary0,boundary1,spatial,known,tt,yy,xx),-1),pos


def native_supervision(label, trace, branch):
    """Native action support only. Missing support is explicit, never GT-filled."""
    from vg_tta.desta3d_v3_actuation_support import spatial_positions
    index=0 if branch=='event' else 1
    kind='time' if branch=='event' else 'coordinate'
    if len(trace['branches'])<=index or kind not in trace['branches'][index]['logits']:
        return None,'missing_native_'+branch+'_support'
    if branch=='event':
        p=torch.tensor(label['event_active']).nonzero().flatten()
        if not len(p):return None,'no_observed_event_frame'
        return (torch.stack((p[0],p[-1])),torch.ones(2,dtype=torch.bool)),None
    positions=spatial_positions(trace)
    valid=torch.tensor([label['box_valid'][p] for p in positions],dtype=torch.bool)[:,None].expand(-1,4).clone()
    if not valid.any():return None,'no_annotated_box_on_native_anchors'
    boxes=torch.tensor([label['boxes_xyxy'][p] for p in positions],dtype=torch.float64)
    indices=(boxes*1000).round().long()
    if ((indices[valid]<0)|(indices[valid]>1000)).any():raise ValueError('Invalid source coordinates')
    ids=torch.tensor(trace['coordinate_ids'])
    return (ids[indices],valid),None
