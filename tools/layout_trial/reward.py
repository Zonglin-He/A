"""Same-frame teacher ROI layout reward and exact OPD feedback substitution."""
import types
import numpy as np
import torch
import torch.nn.functional as F
from stvg import policy


def descriptors(features,boxes,bins=3,valid_ratio=None):
    """Average a shared 6x6 sampling lattice into ordered 3x3 or pooled1 bins.

    features[T,C,H,W] and normalized cxcywh boxes[T,K,4]. Pixel-stride padding
    is accounted for by valid_ratio[T,2] (width,height). No box clipping: points
    outside the image contribute zero, and zero descriptors remain zero.
    """
    if bins not in [1,3]:raise ValueError('Only ordered3 or pooled1 are defined')
    if features.ndim!=4 or boxes.ndim!=3 or boxes.shape[0]!=len(features)or boxes.shape[-1]!=4:
        raise ValueError('Expected same-frame feature maps and ROI boxes')
    if not torch.isfinite(features).all()or not torch.isfinite(boxes).all()or not(boxes[...,2:]>0).all():
        raise ValueError('Nonfinite feature/ROI or nonpositive extent')
    if features.requires_grad:raise ValueError('Teacher feature maps must be frozen')
    b=boxes.to(device=features.device,dtype=features.dtype);t,k,_=b.shape
    ratio=torch.ones(t,2,device=features.device,dtype=features.dtype)if valid_ratio is None else torch.as_tensor(valid_ratio,device=features.device,dtype=features.dtype)
    if ratio.shape!=(t,2)or not((ratio>0)&(ratio<=1)).all():raise ValueError('Invalid feature canvas ratios')
    axis=(torch.arange(6,device=b.device,dtype=b.dtype)+.5)/6-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');unit=torch.stack((xx,yy),-1)
    coords=b[:,:,:2][:,:,None,None,:]+b[:,:,2:][:,:,None,None,:]*unit
    inside=((coords>=0)&(coords<=1)).all(-1)
    grid=(2*coords*ratio[:,None,None,None,:]-1).reshape(t,k*6,6,2)
    z=F.grid_sample(features,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    z=z.reshape(t,features.shape[1],k,6,6).permute(0,2,1,3,4)
    z=z*inside[:,:,None].to(z)
    z=F.adaptive_avg_pool2d(z.reshape(t*k,features.shape[1],6,6),(bins,bins))
    z=z.flatten(2).transpose(1,2).reshape(t,k,bins*bins,features.shape[1])
    return F.normalize(z,dim=-1,eps=1e-12)


class LayoutReward:
    def __init__(self,features,teacher_boxes,bins=3,valid_ratio=None):
        self.features=features.detach();self.bins=bins;self.ratio=valid_ratio
        self.target=descriptors(self.features,teacher_boxes[:,None],bins,valid_ratio).detach()
    def __call__(self,boxes):
        return (descriptors(self.features,boxes,self.bins,self.ratio)*self.target).sum(-1).mean(-1).double()
    def mean_loss(self,boxes,target_ignored):
        return 1-self(boxes[:,None])[:,0]
    @torch.no_grad()
    def feedback(self,proposal,D,target,alias,seed,step,scales,tau):
        # Exactly the original antithetic two-scale proposals; no geometric
        # loss is computed or mixed into layout feedback.
        raw,key=policy.noise_for(alias,seed,step,len(proposal),torch.float32)
        eps=torch.cat((raw[:,:8],raw[:,16:24],raw[:,8:16],raw[:,24:]),1).to(proposal)
        sig=torch.tensor([scales[0]]*16+[scales[1]]*16,device=proposal.device,dtype=proposal.dtype)
        actions=(proposal[:,None]+D[:,None]*sig[None,:,None]*eps).detach()
        boxes=policy.unchart(actions);reward=self(boxes);weights=torch.softmax(reward/tau,-1).detach()
        xy0=boxes[...,:2]-boxes[...,2:]/2;xy1=boxes[...,:2]+boxes[...,2:]/2
        outside=((xy0<0).any(-1)|(xy1>1).any(-1))
        empty=(torch.minimum(xy1,torch.ones_like(xy1))<=torch.maximum(xy0,torch.zeros_like(xy0))).any(-1)
        if not torch.isfinite(reward).all()or not torch.isfinite(weights).all():raise ValueError('Nonfinite layout feedback')
        return dict(proposal_mean=proposal.clone(),D=D.clone(),actions=actions,boxes=boxes,cost=-reward,
            reward=reward,weights=weights,eps=eps,rng_seed=key,informative=bool((reward!=reward[:,:1]).any()),
            outside_image=int(outside.sum()),zero_image_overlap=int(empty.sum()),invalid_positive_size=0,
            feedback='same-frame teacher ordered ROI cosine')



def fit_layout(rp,teacher_boxes,key,seed,cfg,reward):
    # Same original loop, Adam, antithetic candidates, mixture loss, analytic
    # gradient check, content injection and finally reset. No frozen file edits.
    original=policy.fit.__wrapped__
    scope=dict(original.__globals__);scope.update(feedback=reward.feedback,local_cost=reward.mean_loss)
    loop=types.FunctionType(original.__code__,scope,'fit_layout',original.__defaults__,original.__closure__)
    return torch.enable_grad()(loop)(rp,teacher_boxes,key,'Live',seed,cfg)


@torch.enable_grad()
def fit_direct(rp,teacher_boxes,cfg,reward=None):
    target=torch.as_tensor(teacher_boxes,device=rp.state.device,dtype=torch.float64).detach()
    opt=torch.optim.Adam([rp.state],lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    path=[];gradients=[];before_zero=False
    with torch.no_grad():rp.state.zero_()
    try:
        for step in range(cfg['steps']+1):
            rp.check();v=rp.values();boxes=v['boxes'][rp.positions]
            loss=policy.local_cost(boxes,target).mean()if reward is None else reward.mean_loss(boxes,target).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite Direct loss')
            path.append(dict(step=step,boxes=v['boxes'].detach().cpu(),anchor_loss=float(loss.detach())))
            if step==0:before_zero=torch.equal(v['boxes'],rp.E)
            if step==cfg['steps']:break
            opt.zero_grad(set_to_none=True);gradient=torch.autograd.grad(loss,rp.state)[0]
            if not torch.isfinite(gradient).all():raise ValueError('Nonfinite Direct state gradient')
            gradients.append(float(gradient.norm()));rp.state.grad=gradient.detach();opt.step()
        if not before_zero:raise ValueError('Direct did not start from same zero content/E')
        return dict(final=path[-1]['boxes'],path=path,actual_updates=cfg['steps'],
                    gradient_norm_mean=float(np.mean(gradients)),gradient_nonzero=any(g>0 for g in gradients))
    finally:
        with torch.no_grad():rp.state.zero_()


def entropy(weights):
    return -(weights*weights.clamp_min(1e-300).log()).sum(-1)/np.log(weights.shape[-1])


def temperature_from_unlabeled(rewards,geometry,target_tau=1.):
    target=float(entropy(torch.softmax(-geometry/target_tau,-1)).mean())
    grid=[.0001,.0003,.001,.003,.01,.03,.1,.3,1.]
    values={str(t):float(entropy(torch.softmax(rewards/t,-1)).mean())for t in grid}
    tau=min(grid,key=lambda t:(abs(values[str(t)]-target),t))
    return dict(tau=tau,target_geometry_entropy=target,layout_entropy=values[str(tau)],all_entropy=values,
                target_GT_used=False,calibration='one global tau matched to mean initial geometric feedback entropy on fixed128')
