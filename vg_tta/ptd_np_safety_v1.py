"""Label-free monitor and genuinely matched single adapter update."""
import torch
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter

def js(lp,lq):
    lp=lp.double().log_softmax(-1);lq=lq.double().log_softmax(-1)
    mix=torch.logaddexp(lp,lq)-torch.log(torch.tensor(2.,dtype=torch.float64))
    return .5*((lp.exp()*(lp-mix)).sum(-1)+(lq.exp()*(lq-mix)).sum(-1))

def matched_step(z,teacher_logits,j,epsilon=.001,seed=20260924):
    h=z['h'].float();base=z['logits'].float();model=Adapter(h.shape[-1],seed=seed)
    lp=(base+model(h)).double().log_softmax(-1)
    q=(.5*(base[j].double().log_softmax(-1)+teacher_logits[j].double().log_softmax(-1))).log_softmax(-1).detach()
    loss=(lp[j].exp()*(lp[j]-q)).sum(-1).mean();loss.backward()
    assert torch.count_nonzero(model.down.weight.grad)==0
    g=model.up.weight.grad.detach();norm=float(g.norm())
    result=dict(block=j,loss=float(loss.detach()),gradient_norm=norm,epsilon=epsilon,steps=1,GT_used=False)
    if not torch.isfinite(g).all() or norm<1e-12:
        return dict(**result,matched=False,reason='zero_or_nonfinite_gradient',up=model.up.weight.detach(),tokens=base.argmax(-1),actual_KL=0.,displacement_norm=0.)
    direction=-g/norm
    with torch.no_grad():
        model.up.weight.copy_(direction);d=model(h).double();origin=base.double().log_softmax(-1);p=origin.exp()
        def displacement(alpha):
            return float((torch.logsumexp(origin+alpha*d,-1)-alpha*(p*d).sum(-1)).mean())
        lo,hi=0.,1.
        for _ in range(30):
            if displacement(hi)>=epsilon:break
            hi*=2
        else:raise RuntimeError('NO_KL_BRACKET')
        for _ in range(30):
            mid=(lo+hi)/2
            if displacement(mid)>epsilon:hi=mid
            else:lo=mid
        alpha=(lo+hi)/2;model.up.weight.copy_(direction*alpha)
        updated=base+model(h);newlp=updated.double().log_softmax(-1)
        kl=float((p*(origin-newlp)).sum(-1).mean())
        assert abs(kl-epsilon)<=max(1e-6,.01*epsilon),(kl,epsilon)
        assert torch.isfinite(updated).all()
        return dict(**result,matched=True,reason='matched',up=model.up.weight.detach().clone(),tokens=updated.argmax(-1),
            actual_KL=kl,displacement_norm=float(model.up.weight.norm()),effective_LR=alpha/norm,
            changed_coordinates=int((updated.argmax(-1)!=base.argmax(-1)).sum()),
            after_loss=float((newlp[j].exp()*(newlp[j]-q.double())).sum(-1).mean()))

def synthetic_checks():
    torch.manual_seed(73);z=dict(h=torch.randn(5,4,16),logits=torch.randn(5,4,1001))
    same=matched_step(z,z['logits'],2);assert not same['matched'] and torch.equal(same['tokens'],z['logits'].argmax(-1))
    other=matched_step(z,z['logits']+.7*torch.randn(5,4,1001),2)
    assert other['matched'] and abs(other['actual_KL']-.001)<1e-6
    a=Adapter(16);a.up.weight.data.copy_(other['up']);lp=(z['logits']+a(z['h'])).double().log_softmax(-1);op=z['logits'].double().log_softmax(-1)
    assert abs(float((op.exp()*(op-lp)).sum(-1).mean())-other['actual_KL'])<1e-12
    assert float(js(z['logits'],z['logits']).abs().max())<1e-12
    return dict(zero_consensus_no_update=True,nonzero_matched_update=True,independent_reinsertion=True,JS_identical_zero=True)
