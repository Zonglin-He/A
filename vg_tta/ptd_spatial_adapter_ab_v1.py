"""Small output-only adapter and exact coordinate product policy; no labels/I/O."""
import torch
from torch import nn
from torch.nn import functional as F

class Adapter(nn.Module):
    def __init__(self,dim,rank=16,seed=20260924):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.down=nn.Linear(dim,rank,bias=False,dtype=torch.float32)
            self.up=nn.Linear(rank,1001,bias=False,dtype=torch.float32)
            nn.init.zeros_(self.up.weight)
    def forward(self,h):
        x=F.layer_norm(h.float(),(h.shape[-1],))
        return self.up(F.gelu(self.down(x)))

def boxes_from_tokens(tokens):
    xy=tokens.float()/1000
    valid=(xy[...,2:]>xy[...,:2]).all(-1)
    boxes=torch.cat(((xy[...,:2]+xy[...,2:])/2,xy[...,2:]-xy[...,:2]),-1)
    boxes=torch.where(valid[...,None],boxes,torch.zeros_like(boxes))
    return boxes,valid

def anchor_indices(n):
    if n==0:return []
    return sorted(set(min(n-1,int((i+.5)*n/4)) for i in range(4)))

def sample(logits,anchors,seed,m=4):
    generator=torch.Generator(device=logits.device).manual_seed(seed)
    lp=logits.float().log_softmax(-1);p=lp.exp()
    # [anchor, candidate, coordinate]; repeated draws remain repeated.
    tokens=torch.multinomial(p[anchors].reshape(-1,1001),m,replacement=True,generator=generator).reshape(len(anchors),4,m).permute(0,2,1)
    logp=lp[anchors,None].expand(-1,m,-1,-1).gather(-1,tokens[...,None]).squeeze(-1).sum(-1)
    boxes,valid=boxes_from_tokens(tokens)
    return dict(tokens=tokens.cpu(),logp=logp.cpu(),boxes=boxes.cpu(),valid=valid.cpu(),seed=seed)

def kl(q,logp):return (q*(q.clamp_min(1e-30).log()-logp)).sum(-1)

def fit_A(h,base,teacher,anchors,lr,steps=3,misaligned=False,seed=20260924):
    model=Adapter(h.shape[-1],seed=seed).to(h.device);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    origin=base.softmax(-1).detach();target=teacher[anchors].detach()
    if misaligned:target=target.roll(1,0)
    result=[]
    for step in range(steps+1):
        logits=base+model(h);lp=logits.log_softmax(-1)
        loss=kl(target,lp[anchors]).sum()/16+.1*kl(origin,lp).mean()
        result.append(dict(step=step,loss=float(loss.detach()),tokens=logits.argmax(-1).detach().cpu(),
                           state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
        if step==steps:break
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    return result
