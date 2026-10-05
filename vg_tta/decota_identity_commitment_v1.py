"""Frozen E step, hard path contrastive M step; no label access.

This is a geometric preference surrogate, not an EM likelihood guarantee.
Complete Cartesian paths retain duplicates; a singleton has no contrast.
"""
import math
import torch
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.objectives import generalized_iou
from vg_tta.c1_enabling_tricks_v1 import TrickReplay
from vg_tta.decota_optimizer_posterior_r1_v1 import Energy,flat
from vg_tta.decota_track_critic_r3_v1 import TrackEnergy

ARMS=['top1','marginal','map_contrastive']

class IdentityContrastive(TrackEnergy):
    def __init__(self,expert,boxes,ids,tau=1.):
        super().__init__(expert,boxes,ids)
        assert tau==1.,'No temperature search in this experiment'
        self.tau=tau
        self.map_index=None if self.empty else int(self.logpath.argmax())
        self.no_competitor=self.empty or len(self.paths)==1
    def __call__(self,boxes):
        if self.no_competitor:return boxes.sum()*0
        compat=boxes.new_zeros(len(self.paths))
        for t,(pos,ev,_) in enumerate(self.frames):
            compat=compat+generalized_iou(boxes[pos],ev)[self.path_index[:,t]]
        compat=compat/(len(self.frames)*self.tau)
        return torch.logsumexp(compat,0)-compat[self.map_index]
    def metadata(self):
        return dict(super().metadata(),map_index=self.map_index,
            map_path=[] if self.empty else list(self.paths[self.map_index]),
            no_competitor=self.no_competitor,tau=self.tau,
            M_loss='GIoU-frame-mean contrastive cross entropy',E_step_fixed=True)

def factory(expert,boxes,ids,arm):
    if arm=='top1':return Energy(expert,boxes,'top1')
    if arm=='marginal':return TrackEnergy(expert,boxes,ids)
    if arm=='map_contrastive':return IdentityContrastive(expert,boxes,ids)
    raise ValueError(arm)

@torch.enable_grad()
def fit(base,initial,expert,ids,key,arm):
    base.restore(initial);rp=TrickReplay(base,ids,key,{})
    with torch.no_grad():before=detached(rp.values()['boxes'])
    fn=factory(expert,before,ids,arm);skip=fn.empty or getattr(fn,'no_competitor',False)
    origin=rp.state();named=rp.named;params=[p for _,p in named]
    assert sum(p.numel() for p in params)==1792
    opt=torch.optim.Adam(params,lr=.03,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    history=[];best=math.inf;selected=0
    try:
        for k in range(1 if skip else 11):
            opt.zero_grad(set_to_none=True);boxes=rp.values()['boxes'];loss=fn(boxes);value=float(loss.detach())
            assert math.isfinite(value)
            if value<best:best,selected=value,k
            h=dict(step=k,loss=value,state=detached(rp.state(),'cpu'),boxes=detached(boxes,'cpu'));history.append(h)
            if skip or k==10:break
            gs=torch.autograd.grad(loss,params);g=flat(gs).detach();assert torch.isfinite(g).all()
            for p,gg in zip(params,gs):p.grad=gg.detach()
            old=flat(params).detach().clone();opt.step();raw=flat(params).detach()-old
            assert torch.isfinite(flat(params)).all()
            h['update']=dict(gradient=g.cpu(),raw=raw.cpu(),names=[n for n,_ in named],lr=.03,optimizer='adam')
        proposal=history[selected]['state'];rp.restore(proposal)
        with torch.no_grad():final=detached(rp.values()['boxes'],'cpu')
        return dict(arm=arm,before=detached(before,'cpu'),initial=detached(origin,'cpu'),state=proposal,
            final=final,path=history,selected_step=selected,gradient_calls=len(history)-1,
            empty=fn.empty,no_competitor=getattr(fn,'no_competitor',False),active_parameters=1792,
            metadata=fn.metadata() if isinstance(fn,TrackEnergy) else dict(frame_metadata=fn.frame_metadata),GT_used=False)
    finally:rp.restore(origin)
