"""Explicitly new PTD posterior-occupancy bridge; no GT or teacher inputs."""
import copy
import torch
from torch.nn import functional as F
from scripts.ptd_temporal_opd_probe_v1 import TimeAdapter
from methods.decota_final_simplified_v1.objectives import time_cells,temporal_loss

def posterior(logits):
    ij=torch.triu_indices(logits.shape[-1],logits.shape[-1],0)
    score=logits[0,ij[0]].double()+logits[1,ij[1]].double()
    return score.log_softmax(0),ij

@torch.no_grad()
def evidence(logits,frame_ids):
    lp,ij=posterior(logits);p=lp.exp();t=torch.arange(len(frame_ids))
    membership=(t[None]>=ij[0,:,None])&(t[None]<=ij[1,:,None])
    occupancy=(p[:,None]*membership).sum(0)
    raw=torch.logit(occupancy.clamp(1e-6,1-1e-6))
    median=torch.quantile(raw,.5)
    mad=torch.quantile((raw-median).abs(),.5)
    u=(raw-.5*median)/(mad+1e-6)
    edges,weight=time_cells(frame_ids)
    la,ln=F.logsigmoid(u),F.logsigmoid(-u)
    prefix=torch.cat([u.new_zeros(1),(weight*(la-ln)).cumsum(0)])
    cost=-(weight*ln).sum()-(prefix[ij[1]+1]-prefix[ij[0]])
    score=-cost+.1*lp;at=int(score.argmax())
    target=dict(ij=ij,target=at,target_indices=ij[:,at].tolist(),logp0=lp,occupancy=occupancy,
        raw_logits=raw,standardized_logits=u,omega=weight,cell_edges=edges,cost=cost,score=score)
    return dict(offsets=[target],teacher_frozen=True,GT_online=False,
        evidence_source='frozen_PTD_legal_joint_posterior_occupancy',new_unvalidated_interface=True)

def train(probe,frame_ids):
    h,base=probe['h'],probe['logits'];ev=evidence(base,frame_ids)
    model=TimeAdapter(h.shape[-1]);initial=copy.deepcopy(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.1,betas=(.9,.999),eps=1e-4,weight_decay=0.)
    history=[];updates=[]
    for step in range(6):
        z=base+model(h);loss=temporal_loss([z.T[None]],ev,.2)
        assert torch.isfinite(loss)
        lp,ij=posterior(z)
        history.append(dict(step=step,loss=float(loss.detach()),state=copy.deepcopy(model.state_dict()),logits=z.detach().clone(),interval=ij[:,int(lp.argmax())].tolist()))
        if step==5:break
        opt.zero_grad();loss.backward()
        grad=torch.cat([p.grad.flatten() for p in model.parameters()]);assert torch.isfinite(grad).all()
        updates.append(dict(step=step+1,gradient_norm=float(grad.norm())));opt.step()
    state={k:initial[k]+.25*(v-initial[k]) for k,v in model.state_dict().items()};model.load_state_dict(state)
    with torch.no_grad():z=base+model(h);lp,ij=posterior(z)
    return dict(state=state,initial=initial,history=history,updates=updates,evidence=ev,logits=z,interval=ij[:,int(lp.argmax())].tolist(),GT_used=False,selection='step5_parameter_shrink_.25')

def checks():
    logits=torch.tensor([[0.,1.,2.],[2.,1.,0.]])
    ev=evidence(logits,[0,10,20]);e=ev['offsets'][0];lp,ij=posterior(logits)
    brute=torch.tensor([sum(float(lp[k].exp()) for k,(s,t) in enumerate(ij.T) if int(s)<=j<=int(t)) for j in range(3)],dtype=torch.float64)
    assert torch.allclose(brute,e['occupancy'],atol=1e-15,rtol=0)
    z=logits.T[None].double().requires_grad_();loss=temporal_loss([z],ev,.2)
    at=e['target'];manual=-lp[at]+(.2-(lp[at]-torch.cat([lp[:at],lp[at+1:]]).max())).clamp_min(0)
    assert abs(float(loss-manual))<1e-12;loss.backward();assert torch.isfinite(z.grad).all()
    # Independent per-span frame-cell cost.
    for k,(s,t) in enumerate(ij.T):
        inside=(torch.arange(3)>=s)&(torch.arange(3)<=t)
        c=-(e['omega']*torch.where(inside,F.logsigmoid(e['standardized_logits']),F.logsigmoid(-e['standardized_logits']))).sum()
        assert abs(float(c-e['cost'][k]))<1e-12
    return dict(passed=True,occupancy_brute=True,projection_cost_brute=True,original_loss_exact=True,singleton_spans_preserved=True)
