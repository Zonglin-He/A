"""Content-marginal boundary reward; candidate-box gradients also serve Direct."""
import torch
import torch.nn.functional as F


class CDFExtentReward:
    def __init__(self,mass,valid_ratio,tail=.025):
        if mass.ndim!=4 or mass.shape[1]!=1 or mass.requires_grad or not torch.isfinite(mass).all()or not(mass>=0).all():
            raise ValueError('Expected frozen nonnegative[T,1,H,W] content mass')
        if not 0<tail<.5:raise ValueError('Invalid fixed tail probability')
        self.tail=tail;self.prefix=F.pad(mass.cumsum(-1).cumsum(-2),(1,0,1,0))
        self.ratio=torch.as_tensor(valid_ratio,device=mass.device,dtype=mass.dtype)
        if self.ratio.shape!=(len(mass),2)or not((self.ratio>0)&(self.ratio<=1)).all():raise ValueError('Invalid valid canvas')
        self.size=mass.new_tensor([mass.shape[-1],mass.shape[-2]]);self.extent=self.ratio*self.size
        self.total=self.integral(self.extent[:,None])[:,0];self.nonempty=self.total>0

    def integral(self,points):
        grid=(2*points/self.size-1)[:,:,None]
        return F.grid_sample(self.prefix,grid,align_corners=True,padding_mode='border')[:,0,:,0]

    def cdf(self,boxes):
        b=boxes.to(self.prefix)
        if b.ndim!=3 or b.shape[0]!=len(self.prefix)or b.shape[-1]!=4 or not torch.isfinite(b).all()or not(b[...,2:]>0).all():raise ValueError('Invalid positive cxcywh boxes')
        lo=(b[...,:2]-b[...,2:]/2).clamp(0,1)*self.extent[:,None];hi=(b[...,:2]+b[...,2:]/2).clamp(0,1)*self.extent[:,None]
        end=self.extent[:,None].expand_as(lo)
        corners=torch.stack([torch.stack([lo[...,0],end[...,1]],-1),torch.stack([hi[...,0],end[...,1]],-1),torch.stack([end[...,0],lo[...,1]],-1),torch.stack([end[...,0],hi[...,1]],-1)],2)
        values=self.integral(corners.reshape(len(b),-1,2)).reshape(*b.shape[:2],4)
        return values/self.total[:,None,None].clamp_min(torch.finfo(values.dtype).tiny)

    def scores(self,boxes):
        c=self.cdf(boxes);target=c.new_tensor([self.tail,1-self.tail,self.tail,1-self.tail])
        value=-(c-target).abs().mean(-1)
        return torch.where(self.nonempty[:,None],value,torch.zeros_like(value))
