"""Diagnostic query-only optimization. Negative objective never receives GT boxes."""
import torch
from methods.decota_final_simplified_v1.replay import SpatialReplay
from methods.decota_final_simplified_v1.objectives import xyxy


def iou(a, b):
    a, b = xyxy(a.double()), xyxy(b.double())
    lo = torch.maximum(a[..., :2], b[..., :2])
    hi = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (hi-lo).clamp_min(0).prod(-1)
    area_a = (a[..., 2:]-a[..., :2]).clamp_min(0).prod(-1)
    area_b = (b[..., 2:]-b[..., :2]).clamp_min(0).prod(-1)
    return inter/(area_a+area_b-inter).clamp_min(1e-12)


def proposals(teacher):
    """Deterministic GT-free union of two existing proposal pools."""
    collected = {}
    for arm in ('B', 'A'):
        for obs in teacher['arms'][arm]['observations']:
            pos = int(obs['position'])
            pool = collected.setdefault(pos, dict(position=pos, frame_id=int(obs['frame_id']),
                                                  boxes=[], provenance=[]))
            for j, box in enumerate(obs['probe']['boxes']):
                box = box.detach().cpu().float()
                if not torch.isfinite(box).all() or (box[2:] <= 0).any():
                    raise ValueError('Invalid proposal, not silently ignored')
                if any(float(iou(box, old)) >= .8 for old in pool['boxes']):
                    continue
                pool['boxes'].append(box)
                pool['provenance'].append(dict(arm=arm, rank=j,
                                              score=float(obs['probe']['target_scores'][j])))
    return [{**v, 'boxes': torch.stack(v['boxes'])} for _,v in sorted(collected.items()) if v['boxes']]


def roi_weights(boxes, h, w):
    """Exact fractional-cell overlap, normalized per rectangle, no hard pixel center."""
    bounds = xyxy(boxes.double()).clamp(0, 1)
    xx = torch.arange(w, device=boxes.device, dtype=torch.float64)/w
    yy = torch.arange(h, device=boxes.device, dtype=torch.float64)/h
    dx = (torch.minimum(bounds[:,2,None],xx[None]+1/w)-
          torch.maximum(bounds[:,0,None],xx[None])).clamp_min(0)
    dy = (torch.minimum(bounds[:,3,None],yy[None]+1/h)-
          torch.maximum(bounds[:,1,None],yy[None])).clamp_min(0)
    area = (dy[:,:,None]*dx[:,None,:]).flatten(1)
    if (area.sum(-1) <= 0).any():
        raise ValueError('Proposal outside physical image')
    return area/area.sum(-1,keepdim=True)


def negative_masks(pool, boxes, valid):
    """Oracle preparation only. The fitting loss consumes flags, NOT boxes/IoUs."""
    flags, diagnostics = {}, []
    for p in pool:
        pos=p['position'];m=len(p['boxes']);v=bool(valid[pos])
        overlaps=iou(p['boxes'],boxes[pos]) if v else None
        neg=(overlaps < .05) if v else torch.zeros(m,dtype=torch.bool)
        eligible=v and m>=2 and 0<int(neg.sum())<m
        if eligible: flags[pos]=neg.clone()
        diagnostics.append(dict(position=pos, M=m, valid=v,
            negative_count=int(neg.sum()) if v else None,
            survivors=int((~neg).sum()) if v else None, eligible=eligible,
            candidate_iou=overlaps.tolist() if v else None))
    return flags,diagnostics


def shuffled_masks(flags,seed):
    g=torch.Generator().manual_seed(seed)
    result={}
    for pos,n in sorted(flags.items()):
        mask=torch.zeros_like(n);mask[torch.randperm(len(n),generator=g)[:int(n.sum())]]=True
        result[pos]=mask
    return result


def candidate_log_probs(attentions,pool,grids):
    result={}
    for p in pool:
        pos=p['position'];offset=pos%2;index=pos//2;h,w=grids[offset]
        a=attentions[offset][index,0,:h*w].double()
        weights=roi_weights(p['boxes'].to(a.device),h,w)
        density=weights@a
        result[pos]=torch.log_softmax((density+1e-12).log(),dim=0)
    return result


def complementary(logp, masks):
    losses=[]
    for pos,mask in sorted(masks.items()):
        if not 0<int(mask.sum())<len(mask):
            raise ValueError('Closed-set complementary labels need at least one survivor')
        lp=logp[pos]
        # log(1-p_j) from other candidates: stable even for saturated probabilities.
        for j in torch.nonzero(mask,as_tuple=False).flatten().tolist():
            others=torch.arange(len(lp),device=lp.device)!=j
            losses.append(-torch.logsumexp(lp[others],dim=0))
    if not losses:
        raise ValueError('Empty supervision: explicit no-op, not fabricated gradient')
    return torch.stack(losses).mean()


class QueryReplay(SpatialReplay):
    def __init__(self,model,views,n):
        super().__init__(model,views,n,cached=True)
        self.decoder.requires_grad_(False)
        self.named=[('spatial.query_residual',self.delta)]
        self.initial=self.state()
        assert self.delta.numel()==256

    def values_with_attention(self):
        captured=[]
        block=self.decoder.decoder.layers[0]
        module=block.cross_attn if block.from_scratch_cross_attn else block.cross_attn_image
        def capture(module,args,out):
            if not isinstance(out,tuple) or out[1] is None:
                raise RuntimeError('Native trained attention weights unavailable')
            captured.append(out[1])
        handle=module.register_forward_hook(capture)
        try:
            result=self.values()
        finally:
            handle.remove()
        if len(captured)!=2:
            raise RuntimeError('Expected the final spatial pass for both offsets')
        return result,captured
