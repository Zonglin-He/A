"""Matched local response factors and bounded native-gradient step control.

No label reader. Pretrained weights and original source code remain unchanged.
"""
import contextlib,copy,inspect,textwrap,types
import torch
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from vg_tta.stvg_dense_e_opd64_v1 import NativeReplay
from vg_tta.stvg_query_e_opd64_v1 import QUERY
from vg_tta.decota_spatial_opd_tunable_v1 import rollout,likelihood_loss,optimizer_step
from vg_tta.decota_spatial_opd_boundary_chart_proposal003 import proposed_coordinates


@contextlib.contextmanager
def reference_graph(decoder,full):
    if not full:
        yield;return
    assert 'forward' not in decoder.__dict__
    src=textwrap.dedent(inspect.getsource(type(decoder).forward))
    old='pred_boxes = new_pred_boxes.detach()'
    assert src.count(old)==1
    ns={};exec(compile(src.replace(old,'pred_boxes = new_pred_boxes'),'<isolated-reference-gradient>','exec'),inspect.getmodule(type(decoder)).__dict__,ns)
    decoder.forward=types.MethodType(ns['forward'],decoder)
    try:yield
    finally:del decoder.forward


def chart(value,support):
    return proposed_coordinates(value['boxes'],value['native_raw_logits'])[0][support]


def objective(mu,rr,sigma):
    # Double precision scalar for the finite-forward audit/acceptance only.
    return likelihood_loss(mu.double(),rr['mean'].double(),rr['samples'].double(),rr['weights'].double(),sigma)


@torch.enable_grad()
def local(base,ids,key,fit,k):
    """Same historical state/rollout; a 2x2x2 local factorial, not new streams."""
    before=fit['path'][k];rnd=fit['rounds'][k];support=fit['feedback_positions'];cfg=fit['config']
    rp=NativeReplay(base,ids,key,{});base.restore(before['state'])
    flags=[(p,p.requires_grad)for _,p in rp.named]
    for n,p in rp.named:p.requires_grad_(n==QUERY)
    u=base.delta;u0=u.detach().clone();out={};grads={};frozen=state_hash(base.decoder.state_dict())
    rr=None if rnd.get('rollout')is None else detached(rnd['rollout'],u.device)
    count=dict(forwards=0,backwards=0)
    def evaluate(delta,absolute=None):
        with torch.no_grad():
            u.copy_(u0+delta if absolute is None else absolute);v=rp.values();count['forwards']+=1
            b=v['boxes'].detach().cpu();mu=chart(v,support)
            loss=None if rr is None else float(objective(mu,rr,cfg['sigma']))
        return dict(boxes=b,loss=loss)
    try:
        zero=evaluate(torch.zeros_like(u));assert torch.equal(zero['boxes'],before['boxes'])
        if rnd['updated']:
            for full in [False,True]:
                with torch.no_grad():u.copy_(u0)
                with reference_graph(base.decoder.decoder,full):
                    v=rp.values();count['forwards']+=1
                    assert torch.equal(v['boxes'].cpu().detach(),before['boxes'])
                    loss=likelihood_loss(chart(v,support),rr['mean'],rr['samples'],rr['weights'],cfg['sigma'])
                    grads[full]=torch.autograd.grad(loss,u)[0].detach();count['backwards']+=1
                del v,loss
            assert torch.equal(grads[False].cpu(),before['update']['gradient'])
            # Reproduce the original Adam step exactly with its saved prior state.
            previous=next((fit['path'][j]['Adam_after']for j in range(k-1,-1,-1)if 'Adam_after'in fit['path'][j]),None)
            opt=torch.optim.Adam([u],lr=cfg['lr'],betas=(.9,.999),eps=1e-8)
            if previous is not None:opt.load_state_dict(copy.deepcopy(previous))
            with torch.no_grad():u.copy_(u0)
            u.grad=grads[False].clone();opt.step()
            actual=u.detach()-u0;assert torch.equal(actual.cpu(),before['update']['raw'])
            assert torch.equal(u.detach().cpu(),fit['path'][k+1]['state'][QUERY])
            after=before['Adam_after']['state'][0];n=int(after['step'])
            vhat=after['exp_avg_sq'].to(u)/(1-.999**n)
            mprev=torch.zeros_like(u)if previous is None else previous['state'][0]['exp_avg'].to(u)
            directions={}
            for full in [False,True]:
                g=grads[full]
                for hist in [True,False]:
                    m=(.9*mprev+.1*g)/(1-.9**n)if hist else g
                    d=-m/(vhat.sqrt()+1e-8)
                    d=d*(actual.norm()/d.norm())if d.norm()>0 else torch.zeros_like(d)
                    if not full and hist:d=actual.clone()
                    directions[full,hist]=d
                    for amplitude in [1.,.5]:
                        name=f'{"chain"if full else"detach"}_{"history"if hist else"current"}_{"full"if amplitude==1 else"half"}'
                        exact=fit['path'][k+1]['state'][QUERY].to(u)if not full and hist and amplitude==1 else None
                        z=evaluate(d*amplitude,exact)
                        z.update(delta=(d*amplitude).cpu(),gradient_dot_step=float(g@(d*amplitude)),
                            same_native_gradient_dot_step=float(grads[False]@(d*amplitude)),full_chain_gradient_dot_step=float(grads[True]@(d*amplitude)))
                        out[name]=z
            assert torch.equal(out['detach_history_full']['boxes'],fit['path'][k+1]['boxes'])
            fd=[]
            for full,hist in [(False,True),(True,False)]:
                d=directions[full,hist];norm=d.norm()
                if norm==0:continue
                unit=d/norm
                for h in [.001,.0005]:
                    a=evaluate(h*unit);b=evaluate(-h*unit)
                    fd.append(dict(direction='native_Adam'if not full else'chain_current',h=h,
                        finite_difference=(a['loss']-b['loss'])/(2*h),native_autograd=float(grads[False]@unit),chain_autograd=float(grads[True]@unit)))
        else:
            fd=[]
            for full in [False,True]:
                for hist in [True,False]:
                    for amp in ['full','half']:
                        name=f'{"chain"if full else"detach"}_{"history"if hist else"current"}_{amp}'
                        out[name]=dict(**zero,delta=torch.zeros_like(u).cpu(),gradient_dot_step=0.,same_native_gradient_dot_step=0.,full_chain_gradient_dot_step=0.)
        assert state_hash(base.decoder.state_dict())==frozen
        return dict(round=k,updated=rnd['updated'],before=zero,variants=out,gradients=detached(grads,'cpu'),FD=fd,compute=count)
    finally:
        base.restore(base.initial)
        for p,f in flags:p.requires_grad_(f)


@torch.enable_grad()
def controlled_fit(base,ids,key,teacher,positions,cfg,active,sparse):
    base.restore(base.initial);rp=NativeReplay(base,ids,key,{})
    flags=[(p,p.requires_grad)for _,p in rp.named]
    for n,p in rp.named:p.requires_grad_(n==QUERY)
    u=base.delta;initial=detached(rp.state(),'cpu');target=teacher.detach().to(u)
    support=positions if sparse else list(range(len(ids)))
    opt=torch.optim.Adam([u],lr=cfg['lr'],betas=(.9,.999),eps=1e-8)
    path=[];rounds=[];forwards=0;backwards=0
    frozen=state_hash(base.decoder.state_dict())
    try:
        for k in range(cfg['steps']+1):
            v=rp.values();forwards+=1;b=v['boxes'];bc=b.detach().cpu()
            path.append(dict(step=k,boxes=bc,state=detached(rp.state(),'cpu')))
            if k==cfg['steps']:break
            if not active:
                rounds.append(dict(round=k,updated=False,no_expert_support=True,trials=[]));continue
            mu=chart(v,support);fullmu=proposed_coordinates(b,v['native_raw_logits'])[0]
            fullrr=rollout(fullmu.detach(),target,key,k,config=cfg)
            rr={n:(x[support]if torch.is_tensor(x)and x.ndim and len(x)==len(ids)else x)for n,x in fullrr.items()}
            informative=bool((rr['used_rewards']!=rr['used_rewards'][:,:1]).any())
            loss=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights'],cfg['sigma'])
            l0=float(objective(mu.detach(),rr,cfg['sigma']))
            oldopt=copy.deepcopy(opt.state_dict());u0=u.detach().clone()
            update=optimizer_step(opt,loss,[u],informative);trials=[];accepted=None
            if update is not None:
                backwards+=1;step=update['raw'].to(u);proposal_u=u.detach().clone()
                for scale in [1.,.5,.25,.125]:
                    with torch.no_grad():
                        u.copy_(proposal_u if scale==1 else u0+scale*step);vv=rp.values();forwards+=1
                        lv=float(objective(chart(vv,support),rr,cfg['sigma']))
                        move=vv['boxes']-b.detach()
                        # Fixed numerical tolerance; no task labels or output-quality gate.
                        tolerance=1e-7*(1+abs(l0))
                        ok=lv<=l0+tolerance
                        trials.append(dict(scale=scale,loss=lv,loss_change=lv-l0,tolerance=tolerance,accepted=ok,
                            output_L2=float(move.norm()),output_abs_mean=float(move.abs().mean())))
                    if ok:accepted=scale;break
                if accepted is None:
                    with torch.no_grad():u.copy_(u0)
                    opt.load_state_dict(oldopt)
                path[-1]['proposal']=update
            rounds.append(dict(round=k,updated=accepted is not None,informative=informative,accepted_scale=accepted,
                loss_before=l0,trials=trials,rollout=detached(rr,'cpu'),optimizer_before=detached(oldopt,'cpu'),optimizer_after=detached(opt.state_dict(),'cpu'),
                rejected_state_restored=update is not None and accepted is None))
            del v,b,loss,mu
        assert all(torch.equal(x['state'][n],initial[n])for x in path for n in initial if n!=QUERY)
        assert state_hash(base.decoder.state_dict())==frozen
        return dict(path=path,rounds=rounds,final=path[-1]['boxes'],state=path[-1]['state'],compute=dict(forwards=forwards,backwards=backwards),
            selected_step=cfg['steps'],all_pretrained_LN_frozen=True,support=support,GT_used=False)
    finally:
        base.restore(base.initial)
        for p,f in flags:p.requires_grad_(f)
