"""Matched actuation experiments. No label imports, no production changes."""
import math
import torch
import numpy as np
from scipy.special import logsumexp
from methods.decota_final_simplified_v1.objectives import SpatialLoss
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.tastvg_decota_critic_p0_v1 import CriticEnergy
from vg_tta.c1_enabling_tricks_v1 import TrickReplay

OBJECTIVES = ['direct', 'all', 'admit', 'top1']
ARMS = [f'{o}_{v}' for o in OBJECTIVES for v in ['adam','sgd']] + ['all_lr_authority','all_post_authority']

class Energy(CriticEnergy):
    def __init__(self, expert, boxes, support):
        super().__init__(expert, boxes)
        if support in ['admit','top1']:
            keep=[i for i,m in enumerate(self.frame_metadata) if m['old_admitted']]
            self.frames=[self.frames[i] for i in keep]
            self.frame_metadata=[self.frame_metadata[i] for i in keep]
        if support=='top1':
            frames=[]
            for (pos,box,logw),m in zip(self.frames,self.frame_metadata):
                k=int(logw.argmax());frames.append((pos,box[k:k+1],logw.new_zeros(1)))
                m.update(proposals=1,weights=[1.],scores=[m['scores'][k]],evidence_entropy=0.)
            self.frames=frames
        self.empty=not self.frames

def objective(expert, boxes, name):
    return SpatialLoss(expert['anchors']['single4'], boxes) if name=='direct' else Energy(expert,boxes,name)

def authority(lossfn):
    if lossfn.empty: return 0.
    values=[]
    for m in lossfn.frame_metadata:
        n=m['proposals'];values.append(1. if n==1 else max(0.,min(1.,1-m['evidence_entropy']/math.log(n))))
    return float(np.mean(values))

def flat(values):return torch.cat([x.reshape(-1) for x in values])

@torch.enable_grad()
def fit(base,initial,expert,ids,key,arm,sgd_lr=None,steps=10):
    name,mode=arm.split('_',1);base.restore(initial)
    replay=TrickReplay(base,ids,key,{});origin=replay.state();params=[p for _,p in replay.named]
    fn=objective(expert,base.zero['boxes'],name)
    a=authority(fn) if mode.endswith('authority') else 1.
    lr=sgd_lr if mode=='sgd' else .03*a if mode=='lr_authority' else .03
    assert lr is not None and lr>=0
    opt=torch.optim.SGD(params,lr=lr,momentum=0) if mode=='sgd' else torch.optim.Adam(params,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    history=[];best=math.inf;selected=0
    try:
        for k in range(1 if fn.empty else steps+1):
            opt.zero_grad(set_to_none=True);val=replay.values();loss=fn(val['boxes']);lv=float(loss.detach())
            assert math.isfinite(lv)
            if lv<best:best,selected=lv,k
            h=dict(step=k,loss=lv,state=detached(replay.state(),'cpu'),boxes=detached(val['boxes'],'cpu'))
            history.append(h)
            if fn.empty or k==steps:break
            gg=torch.autograd.grad(loss,params);g=flat(gg).detach();assert torch.isfinite(g).all()
            for p,t in zip(params,gg):p.grad=t.detach()
            before=flat(params).detach().clone();opt.step();after=flat(params).detach()
            h['update']=dict(gradient=g.cpu(),raw=(after-before).cpu(),optimizer=mode,lr=lr)
            assert torch.isfinite(after).all()
        proposal=history[selected]['state'];used=proposal
        if mode=='post_authority':
            # Adam evolves its unmodified proposal; interpolate only its selected final state.
            used={n:origin[n]+a*(proposal[n].to(origin[n])-origin[n]) for n in origin}
            replay.restore(used)
            with torch.no_grad():out=detached(replay.values()['boxes'],'cpu')
        else:out=history[selected]['boxes']
        return dict(arm=arm,initial=detached(origin,'cpu'),state=detached(used,'cpu'),
            proposal_state=detached(proposal,'cpu'),final=out,selected_step=selected,path=history,
            skipped=fn.empty,gradient_calls=len(history)-1,parameter_count=1792,
            frame_metadata=getattr(fn,'frame_metadata',[]),authority=a,lr=lr,
            authority_is_not_correctness_probability=True,GT_used=False)
    finally:replay.restore(origin)

def box_displacement(a,b):return float((a-b).abs().mean())

def posterior(logp0,cost,ij,ids,beta=1.):
    """Exact physical centre groups; no discretisation of centre or new endpoints."""
    lp=np.asarray(logp0,dtype=np.float64);r=-np.asarray(cost,dtype=np.float64)
    ij=np.asarray(ij,dtype=np.int64);ids=np.asarray(ids,dtype=np.int64)
    assert lp.ndim==1 and ij.shape==(2,len(lp)) and np.isfinite(lp).all() and np.isfinite(r).all()
    assert (ij[0]<ij[1]).all()
    lp=lp-logsumexp(lp);logfull=lp+beta*r;logfull-=logsumexp(logfull)
    centres=ids[ij[0]]+ids[ij[1]]+1
    unique,group=np.unique(centres,return_inverse=True)
    logextent=np.empty_like(lp);pcentre=[]
    for g in range(len(unique)):
        at=group==g;mass=logsumexp(lp[at]);pcentre.append(np.exp(mass))
        logextent[at]=mass+lp[at]+beta*r[at]-logsumexp(lp[at]+beta*r[at])
    p0,full,extent=np.exp(lp),np.exp(logfull),np.exp(logextent)
    assert abs(full.sum()-1)<1e-12 and abs(extent.sum()-1)<1e-12
    actual=np.bincount(group,weights=extent,minlength=len(unique))
    assert np.max(np.abs(actual-np.asarray(pcentre)))<2e-12
    return dict(ij=ij,logp0=lp,reward=r,logfull=logfull,logextent=logextent,
        full_map=int(logfull.argmax()),extent_map=int(logextent.argmax()),
        centre_marginal_max_error=float(np.max(np.abs(actual-pcentre))),
        centre_groups=len(unique),singleton_groups=int((np.bincount(group)==1).sum()),
        native_KL_full=float(np.sum(full*(logfull-lp))),native_KL_extent=float(np.sum(extent*(logextent-lp))),beta=beta)
