"""Native-consistent temporal candidate scores and a persistent linear ranker."""
import hashlib
import torch
from torch.nn import functional as F
from vg_tta.spatial_online_state_v1 import arrival


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def native_scores(logits,records,intervals):
    """Best native legal two-offset span pair yielding each physical envelope.

    Native decoding maximizes each legal offset span then takes their envelope.
    Scoring a candidate by nearest endpoints would generally change that policy.
    Four boundary flags per offset compute the exact constrained joint maximum.
    """
    out=[]
    for start,end in intervals:
        tables=[]
        for z,r in zip(logits,records):
            ids=torch.tensor(r['frame_ids']);n=len(ids);lp=z.detach().cpu()[0].log_softmax(0)
            score=lp[:,0,None]+lp[None,:,1]
            i,j=torch.meshgrid(torch.arange(n),torch.arange(n),indexing='ij')
            valid=(i<j)&(ids[:,None]>=start)&(ids[None,:]<end)
            flags=(ids[:,None]==start).long()+2*(ids[None,:]==end-1).long()
            table=[]
            for flag in range(4):
                keep=valid&(flags==flag);table.append(float(score[keep].max()) if keep.any() else float('-inf'))
            tables.append(table)
        value=max(tables[0][a]+tables[1][b] for a in range(4) for b in range(4) if (a|b)==3)/2
        assert value!=float('-inf'),'Candidate outside native envelope support'
        out.append(value)
    return torch.tensor(out,dtype=torch.float64)


def features(hidden,indices):
    assert hidden.ndim==2 and hidden.shape[1]==256
    return torch.stack([torch.cat([hidden[s],hidden[e],hidden[s:e+1].mean(0)]) for s,e in indices]).double()


def source_state():return {'w':torch.zeros(768,dtype=torch.float64),'b':torch.zeros((),dtype=torch.float64)}


def arrive(previous):return arrival(source_state(),previous,'O-all')


def score(base,phi,state):return base+phi@state['w']+state['b']


def pairs(teacher):return [(i,j) for i in range(len(teacher)) for j in range(len(teacher)) if teacher[i]>teacher[j]+1e-12]


def update(base,phi,state,teacher,lr=.001):
    pp=pairs(teacher);w=state['w'].detach().clone().requires_grad_(True)
    # Common bias cancels analytically, so its gradient is exactly zero.
    def objective(value):
        ss=base+phi@value
        return torch.stack([F.softplus(-(ss[i]-ss[j])) for i,j in pp]).mean() if pp else value.sum()*0
    before=objective(w);g=torch.autograd.grad(before,w)[0]
    new={'w':(w-lr*g).detach(),'b':state['b'].detach().clone()}
    after=objective(new['w']);assert torch.isfinite(new['w']).all()
    return new,dict(loss_before=float(before.detach()),loss_after=float(after.detach()),gradient_norm=float(g.norm()),step_norm=float((new['w']-state['w']).norm()),pairs=len(pp),lr=lr,steps=1,momentum=0,bias_gradient=0.)
