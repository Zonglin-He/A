"""Exact v0 small-head policy: prior .7, six zero weights, guarded memory."""
import torch
import torch.nn.functional as F


def initial_weights(cache):return cache['support'].new_zeros(6)


def prior(cache,t):
    valid=cache['valid'][t];n=int(valid.sum());p=cache['support'].new_zeros(3);p[0]=.7 if n>1 else 1
    if n>1:p[valid&~F.one_hot(torch.tensor(0,device=p.device),3).bool()]=.3/(n-1)
    return p.clamp_min(1e-300).log().masked_fill(~valid,-1e9)


def state(cache,t,memory,previous,use_memory=True):
    desc=cache['descriptors'][t];H=(desc@F.normalize(memory,dim=-1))if use_memory else torch.zeros(3,dtype=desc.dtype,device=desc.device)
    short=desc@cache['initial_memory']if t==0 else desc@(previous@cache['descriptors'][t-1])
    link=torch.zeros_like(H)if t==0 else previous@cache['links'][t-1]
    source=torch.zeros_like(H);source[0]=1
    x=torch.stack([H,short,link,desc@desc[0],cache['deviation'][t],source],-1)
    reward=cache['support'][t]+.25*link+.1*H-.1*cache['deviation'][t]
    q=(reward/.2).masked_fill(~cache['valid'][t],-1e9).log_softmax(-1)
    return x,q


def proxy(cache,path):
    path=path.to(cache['boxes'].device);T=len(path);index=torch.arange(T,device=path.device);u=cache['support'][index,path];g=cache['deviation'][index,path];observed=torch.zeros(T,dtype=torch.bool,device=path.device);observed[cache['positions']]=True;c=cache['confidence']*cache['active'];result=u.new_tensor(0.)
    for group in [observed,~observed]:
        w=c*group;result=result+(u*w).sum()/w.sum().clamp_min(1e-12)
    pair=(c[:-1]*c[1:]).sqrt();links=cache['links'][torch.arange(T-1,device=path.device),path[:-1],path[1:]]
    result=result+.25*(links*pair).sum()/pair.sum().clamp_min(1e-12)-.1*(g*cache['inside']).sum()/cache['inside'].sum().clamp_min(1)
    return float(result.detach())


def rollout(cache,phi,gumbel=None,kind='OPD',greedy=False,teacher_path=False,use_memory=True):
    memory=cache['initial_memory'];previous=memory.new_tensor([1.,0.,0.]);losses=[];regularizers=[];states=[];targets=[];choices=[];probabilities=[];writes=[]
    T=len(cache['boxes'])
    for t in range(T):
        x,logq=state(cache,t,memory,previous,use_memory);logpi0=prior(cache,t);logits=(logpi0+x@phi).masked_fill(~cache['valid'][t],-1e9);logp=logits.log_softmax(-1);p=logp.exp();c=cache['confidence'][t]*cache['active'][t]
        target=logq if kind=='DirectSoft'else logq.detach();losses.append(c*(target.exp()*(target-logp)).sum());regularizers.append((p*(logp-logpi0)).sum())
        selection=logq if teacher_path else logits
        if not greedy:selection=selection+gumbel[t]
        index=int(selection.argmax());hard=F.one_hot(torch.tensor(index,device=phi.device),3).to(phi)
        action=selection.softmax(-1)if kind=='DirectSoft'and not greedy and not teacher_path else hard
        # Reasonable bounded writes, identical in all methods. Invalid evidence
        # never changes actual boxes or memory, even after shared weight updates.
        amount=.2*cache['confidence'][t]*cache['active'][t]*(action*cache['support'][t]).sum()if use_memory else memory.new_tensor(0.)
        memory=F.normalize((1-amount)*memory+amount*(action@cache['descriptors'][t]),dim=-1);previous=action
        states.append(x.detach());targets.append(logq.detach());choices.append(index);probabilities.append(p.detach());writes.append(float(amount.detach()))
    den=(cache['confidence']*cache['active']).sum().clamp_min(1e-12)
    loss=torch.stack(losses).sum()/den+.05*torch.stack(regularizers).mean()
    path=torch.tensor(choices,device=phi.device)
    return dict(loss=loss,features=torch.stack(states),logq=torch.stack(targets),choices=path,probabilities=torch.stack(probabilities),writes=writes,proxy=proxy(cache,path))


def fixed_loss(cache,phi,paths):
    values=[];base=torch.stack([prior(cache,t)for t in range(len(cache['boxes']))])
    for path in paths:
        logits=base+torch.einsum('tkd,d->tk',path['features'],phi);logp=logits.log_softmax(-1);q=path['logq'].exp();w=cache['confidence']*cache['active']
        dist=((q*(path['logq']-logp)).sum(-1)*w).sum()/w.sum().clamp_min(1e-12);reg=(logp.exp()*(logp-base)).sum(-1).mean();values.append(dist+.05*reg)
    return torch.stack(values).mean()


def graph_path(cache):
    # Exact optimizer of the fixed acceptance score; no memory reward.
    T=len(cache['boxes']);c=cache['confidence']*cache['active'];observed=torch.zeros(T,dtype=torch.bool,device=c.device);observed[cache['positions']]=True;w=torch.zeros_like(c)
    for mask in [observed,~observed]:z=c*mask;w=w+z/z.sum().clamp_min(1e-12)
    node=cache['support']*w[:,None]-.1*cache['deviation']*cache['inside'][:,None]/cache['inside'].sum().clamp_min(1);node=node.masked_fill(~cache['valid'],-1e9);pair=(c[:-1]*c[1:]).sqrt();pair=pair/pair.sum().clamp_min(1e-12);v=node[0];back=[]
    for t in range(1,T):
        best,index=(v[:,None]+.25*pair[t-1]*cache['links'][t-1]).max(0);v=best+node[t];back.append(index)
    path=[int(v.argmax())]
    for b in reversed(back):path.append(int(b[path[-1]]))
    return torch.tensor(path[::-1],device=c.device)


def output(cache,path):
    index=path.to(cache['boxes'].device);return cache['boxes'][torch.arange(len(index),device=index.device),index]
