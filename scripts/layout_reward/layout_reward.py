"""Frozen same-frame ReferDINO stride16 projected features, ordered 3x3 ROI."""
import torch
import torch.nn.functional as F

class LayoutReward:
    def __init__(self, features, mask, target, device='cuda'):
        self.features=features.detach().float().to(device)
        self.mask=mask.detach().bool().to(device)
        self.target=target.detach().double().to(device)
        assert self.features.shape[0]==8 and self.features.shape[1]==256
        self.valid_hw=[]
        for m in self.mask:
            good=~m;h=int(good.any(1).sum());w=int(good.any(0).sum())
            assert h>0 and w>0 and good[:h,:w].all() and int(good.sum())==h*w
            self.valid_hw.append((h,w))
        with torch.no_grad():self.template,self.template_valid=self.sample(self.target[:,None])
        self.template=self.template.detach();self.template_valid=self.template_valid.detach()
        self.active=self.template_valid.any(-1).squeeze(1)

    def sample(self, boxes):
        assert boxes.shape[0]==8 and boxes.shape[-1]==4
        f=self.features;h,w=f.shape[-2:];n=boxes.shape[1]
        t=torch.tensor([-1/3,0,1/3],device=boxes.device,dtype=boxes.dtype)
        yy,xx=torch.meshgrid(t,t,indexing='ij');delta=torch.stack((xx.flatten(),yy.flatten()),-1)
        xy=boxes[:,:,:2,None].transpose(-1,-2)+boxes[:,:,2:,None].transpose(-1,-2)*delta[None,None]
        # Input normalized image coordinates map only to each frame's non-padding extent.
        ratio=torch.tensor([[vw/w,vh/h]for vh,vw in self.valid_hw],device=boxes.device,dtype=boxes.dtype)
        grid=2*xy*ratio[:,None,None]-1
        sampled=F.grid_sample(f,grid.float().reshape(8,n*9,1,2),mode='bilinear',padding_mode='zeros',align_corners=False)
        sampled=sampled.squeeze(-1).transpose(1,2).reshape(8,n,9,256)
        norm=sampled.norm(dim=-1);inside=((xy>=0)&(xy<=1)).all(-1)
        valid=inside & (norm>1e-8)
        return F.normalize(sampled,p=2,dim=-1,eps=1e-8),valid

    def __call__(self,boxes):
        z,valid=self.sample(boxes);cos=(z*self.template).sum(-1).double()
        # Invalid candidate cells receive -1, not an artificial zero similarity.
        value=torch.where(valid,cos,-torch.ones_like(cos))
        weights=self.template_valid.double();den=weights.sum(-1).clamp_min(1)
        return (value*weights).sum(-1)/den
