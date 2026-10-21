"""Continuous rectangle/soft-mask area rewards; same gradients are available to Direct."""
import torch
import torch.nn.functional as F


def box_overlap(a,b):
    lo=torch.maximum(a[:,:2,None]-a[:,2:,None]/2,(b[...,:2]-b[...,2:]/2).transpose(1,2))
    hi=torch.minimum(a[:,:2,None]+a[:,2:,None]/2,(b[...,:2]+b[...,2:]/2).transpose(1,2))
    inter=(hi-lo).clamp_min(0).prod(1)
    union=a[:,2:].prod(1)[:,None]+b[...,2:].prod(-1)-inter
    return inter/union.clamp_min(1e-12)


class MaskAreaReward:
    """masks[T,Q,H,W], selected track first; boxes[T,K,4] in normalized cxcywh.

    A bilinearly interpolated integral image gives the exact integral of a
    piecewise-constant pixel density. Valid ratios trim only canvas padding;
    off-image candidate area stays in the denominator. No mask threshold.
    """
    def __init__(self,masks,valid_ratio,eligible_alternatives):
        if masks.ndim!=4 or masks.requires_grad or not torch.isfinite(masks).all():
            raise ValueError('Expected frozen finite[T,Q,H,W] probabilities')
        if not((masks>=0)&(masks<=1)).all():raise ValueError('Masks must be probabilities')
        self.prefix=F.pad(masks.cumsum(-1).cumsum(-2),(1,0,1,0))
        self.ratio=torch.as_tensor(valid_ratio,device=masks.device,dtype=masks.dtype)
        self.size=torch.tensor([masks.shape[-1],masks.shape[-2]],device=masks.device,dtype=masks.dtype)
        if self.ratio.shape!=(len(masks),2) or not((self.ratio>0)&(self.ratio<=1)).all():raise ValueError('Invalid canvas')
        self.extent=self.ratio*self.size
        self.eligible=torch.as_tensor(eligible_alternatives,device=masks.device,dtype=torch.bool)
        if self.eligible.shape!=(len(masks),masks.shape[1]-1):raise ValueError('Invalid alternative eligibility')
        full=torch.tensor([.5,.5,1.,1.],device=masks.device,dtype=masks.dtype)[None,None].expand(len(masks),1,4)
        self.total=self.mass(full)[:,0]

    def mass(self,boxes):
        b=boxes.to(self.prefix)
        if b.ndim!=3 or len(b)!=len(self.prefix) or b.shape[-1]!=4 or not torch.isfinite(b).all() or not(b[...,2:]>0).all():
            raise ValueError('Invalid normalized positive boxes')
        lo=(b[...,:2]-b[...,2:]/2)*self.extent[:,None]
        hi=(b[...,:2]+b[...,2:]/2)*self.extent[:,None]
        lo=torch.minimum(lo.clamp_min(0),self.extent[:,None]);hi=torch.minimum(hi.clamp_min(0),self.extent[:,None])
        corners=torch.stack([lo,torch.stack([hi[...,0],lo[...,1]],-1),torch.stack([lo[...,0],hi[...,1]],-1),hi],2)
        grid=(2*corners/self.size-1).reshape(len(b),-1,1,2)
        v=F.grid_sample(self.prefix,grid,mode='bilinear',padding_mode='border',align_corners=True)
        v=v.reshape(len(b),self.prefix.shape[1],b.shape[1],4).permute(0,2,1,3)
        return (v[...,3]-v[...,1]-v[...,2]+v[...,0]).clamp_min(0)

    def scores(self,boxes):
        b=boxes.to(self.prefix);area=(b[...,2:]*self.extent[:,None]).prod(-1)
        dice=2*self.mass(b)/(area[:,:,None]+self.total[:,None]).clamp_min(1e-12)
        pos=dice[:,:,0]
        if dice.shape[-1]==1:neg=torch.zeros_like(pos)
        else:neg=dice[:,:,1:].masked_fill(~self.eligible[:,None],0).max(-1).values
        return dict(MaskDice=pos,MaskContrast=pos-neg)
