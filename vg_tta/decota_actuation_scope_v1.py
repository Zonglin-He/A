"""Conditional cached spatial experiments; current readout and write separated."""
import math
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.c1_enabling_tricks_v1 import TrickReplay,QUERY
from vg_tta.decota_optimizer_posterior_r1_v1 import Energy,authority,flat
from vg_tta.decota_track_critic_r3_v1 import TrackEnergy,FrameSum

def factory(expert,boxes,ids,kind):
    if kind=='frame':return Energy(expert,boxes,'all')
    if kind=='frame_sum':return FrameSum(expert,boxes)
    if kind in ['track','track_authority']:return TrackEnergy(expert,boxes,ids)
    if kind=='track_giou':return TrackEnergy(expert,boxes,ids,similarity='giou')
    raise ValueError(kind)

def concentration(fn):return fn.path_authority if isinstance(fn,TrackEnergy) else authority(fn)

@torch.enable_grad()
def fit_scope(base,initial,expert,ids,key,config,scope='joint'):
    base.restore(initial);rp=TrickReplay(base,ids,key,{})
    with torch.no_grad():before=detached(rp.values()['boxes'])
    fn=factory(expert,before,ids,config['evidence'])
    a=authority(Energy(expert,before,'all')) if config.get('authority_source')=='frame' else concentration(fn)
    mode=config['optimizer'];eta=config['lr'];use_a=config.get('authority',False)
    lr=eta*a if mode=='sgd' and use_a or mode=='lr_authority' else eta
    named=[(n,p) for n,p in rp.named if scope!='u_only' or n==QUERY]
    params=[p for _,p in named];origin=rp.state()
    opt=torch.optim.SGD(params,lr=lr,momentum=0) if mode=='sgd' else torch.optim.Adam(params,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    history=[];best=math.inf;selected=0;backwards=0
    try:
        for k in range(1 if fn.empty else 11):
            opt.zero_grad(set_to_none=True);v=rp.values();loss=fn(v['boxes']);value=float(loss.detach());assert math.isfinite(value)
            if value<best:best,selected=value,k
            h=dict(step=k,loss=value,state=detached(rp.state(),'cpu'),boxes=detached(v['boxes'],'cpu'));history.append(h)
            if fn.empty or k==10:break
            gs=torch.autograd.grad(loss,params);g=flat(gs).detach();assert torch.isfinite(g).all()
            for p,gg in zip(params,gs):p.grad=gg.detach()
            old=flat(params).detach().clone();opt.step();raw=flat(params).detach()-old
            h['update']=dict(gradient=g.cpu(),raw=raw.cpu(),names=[n for n,_ in named],lr=lr,optimizer='sgd' if mode=='sgd' else 'adam')
            backwards+=1
        proposal=history[selected]['state'];used={n:v.to(origin[n]) for n,v in proposal.items()}
        if mode=='post_authority' or use_a and mode!='sgd':used={n:origin[n]+a*(used[n]-origin[n]) for n in origin}
        if scope=='small_LN':
            # Existing full fast correction; a separately limits LN use/write.
            used={n:v if n==QUERY else origin[n]+a*(v-origin[n]) for n,v in used.items()}
        rp.restore(used)
        with torch.no_grad():final=detached(rp.values()['boxes'],'cpu')
        write_proposal=detached(used,'cpu');slow=None
        if scope=='u_only' and not fn.empty:
            ln=[(n,p) for n,p in rp.named if n!=QUERY];gg=torch.autograd.grad(fn(rp.values()['boxes']),[p for _,p in ln]);backwards+=1
            lo=torch.optim.SGD([p for _,p in ln],lr=lr,momentum=0) if mode=='sgd' else torch.optim.Adam([p for _,p in ln],lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            old=flat([p for _,p in ln]).detach().clone()
            for (_,p),g in zip(ln,gg):p.grad=g.detach()
            lo.step();raw=flat([p for _,p in ln]).detach()-old
            candidate=rp.state()
            if mode=='post_authority' or use_a and mode!='sgd':candidate={n:used[n]+a*(candidate[n]-used[n]) for n in candidate}
            write_proposal=detached(candidate,'cpu')
            slow=dict(gradient=flat(gg).detach().cpu(),raw=raw.cpu(),lr=lr,optimizer='sgd' if mode=='sgd' else 'adam',names=[n for n,_ in ln],steps=1)
        result=dict(scope=scope,config=config,before=detached(before,'cpu'),initial=detached(origin,'cpu'),state=detached(used,'cpu'),
            proposal_state=proposal,write_proposal=write_proposal,final=final,path=history,selected_step=selected,authority=a,
            lr=lr,gradient_calls=backwards,active_parameters=sum(p.numel() for p in params),slow_LN=slow,
            empty=fn.empty,GT_used=False,metadata=fn.metadata() if isinstance(fn,TrackEnergy) else dict(valid_frames=len(fn.frames)))
        return result
    finally:rp.restore(origin)

def commit_state(initial,proposal):
    return {n:torch.zeros_like(v) if n==QUERY else v+(proposal[n].to(v)-v)*(1/16) for n,v in initial.items()}
