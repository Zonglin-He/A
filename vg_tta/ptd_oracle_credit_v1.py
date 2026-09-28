"""Exact finite-candidate oracle distillation. No label I/O."""
import copy
import torch
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter

SEEDS=(20260924,20260925,20260926)
BETA=.01

def policy(logits,block):
    coord=logits[block['j']].double().log_softmax(-1)
    b=block['boxes'].long()
    raw=coord[torch.arange(4)[None,:],b].sum(-1)
    return raw.log_softmax(0),raw.logsumexp(0)

def targets(base,blocks):
    out=[]
    for b in blocks:
        lp,mass=policy(base,b);credit=b['credit'].double()
        lq=(lp+credit/BETA).log_softmax(0)
        out.append(dict(lp=lp.detach(),lq=lq.detach(),log_mass=float(mass),
            target_KL=float((lq.exp()*(lq-lp)).sum()),
            reverse_KL=float((lp.exp()*(lp-lq)).sum()),
            native_expected_credit=float((lp.exp()*credit).sum()),
            target_expected_credit=float((lq.exp()*credit).sum()),
            best_prior=float(lp[b['best']].exp()),best_target=float(lq[b['best']].exp())))
    return out

def objective(logits,blocks,ts):
    if not any(bool(b['credit'].ne(0).any()) for b in blocks):return logits.sum()*0
    losses=[]
    for b,t in zip(blocks,ts):
        lp,_=policy(logits,b);losses.append((lp.exp()*(lp-t['lq'])).sum())
    return torch.stack(losses).mean() if losses else logits.sum()*0

def fit(z,blocks,seed):
    model=Adapter(z['h'].shape[-1],seed=seed);initial=copy.deepcopy(model.state_dict())
    ts=targets(z['logits'],blocks);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0)
    hist=[];updates=[]
    for step in range(4):
        logits=z['logits']+model(z['h']);loss=objective(logits,blocks,ts)
        with torch.no_grad():
            lp0=z['logits'].double().log_softmax(-1);lp=logits.double().log_softmax(-1)
            expected=[float((policy(logits,b)[0].exp()*b['credit']).sum()) for b in blocks]
            hist.append(dict(step=step,state=copy.deepcopy(model.state_dict()),tokens=logits.argmax(-1).detach(),
                loss=float(loss),whole_coordinate_KL=float((lp0.exp()*(lp0-lp)).sum(-1).mean()),
                expected_credits=expected,changed_coordinates=int((logits.argmax(-1)!=z['base_tokens']).sum())))
        if step==3:break
        before=[p.detach().clone() for p in model.parameters()]
        optimizer.zero_grad();loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if norm>0:optimizer.step()
        displacement=sum(float((p.detach()-q).double().square().sum()) for p,q in zip(model.parameters(),before))**.5
        updates.append(dict(step=step+1,gradient_norm=norm,displacement=displacement,optimizer_applied=norm>0))
    fresh=Adapter(z['h'].shape[-1],seed=seed)
    assert all(torch.equal(v,initial[k]) for k,v in fresh.state_dict().items())
    fresh.load_state_dict(hist[-1]['state'])
    assert torch.equal((z['logits']+fresh(z['h'])).argmax(-1),hist[-1]['tokens'])
    return dict(seed=seed,history=hist,updates=updates,targets=ts,GT_used=True,reset_exact=True,reload_exact=True)

def synthetic():
    # Independent closed-form derivative wrt unnormalized candidate scores.
    x=torch.tensor([.2,-.7,.4],dtype=torch.float64,requires_grad=True)
    credit=torch.tensor([0.,.015,-.02],dtype=torch.float64)
    lp=x.log_softmax(0);lq=(lp.detach()+credit/BETA).log_softmax(0)
    loss=(lp.exp()*(lp-lq)).sum();grad=torch.autograd.grad(loss,x,retain_graph=True)[0]
    analytic=lp.exp()*(lp-lq-loss);err=float((analytic-grad).abs().max())
    def f(v):
        p=v.log_softmax(0);return (p.exp()*(p-lq)).sum()
    fd=torch.stack([(f(x.detach()+torch.eye(3,dtype=x.dtype)[i]*1e-6)-f(x.detach()-torch.eye(3,dtype=x.dtype)[i]*1e-6))/2e-6 for i in range(3)])
    ferr=float((fd-grad).abs().max());assert err<1e-12 and ferr<1e-8
    zero=(lp.exp()*(lp-lp.detach())).sum();zg=torch.autograd.grad(zero,x)[0];assert zg.abs().max()<1e-14
    ratios=lq-lp.detach();assert ratios[1]>ratios[0]>ratios[2]
    return dict(passed=True,analytic_error=err,finite_difference_error=ferr,zero_credit_gradient=float(zg.abs().max()),oracle_tilt_order=True)
