"""v0: top-K direct reference matching and <=3 connected-region candidates."""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
from tools.universal_trial.core import cell_occupancy,chart


def iou(a,b):
    lo=torch.maximum(a[:2]-a[2:]/2,b[:2]-b[2:]/2);hi=torch.minimum(a[:2]+a[2:]/2,b[:2]+b[2:]/2);inter=(hi-lo).clamp_min(0).prod()
    return inter/(a[2:].prod()+b[2:].prod()-inter).clamp_min(1e-12)


def correspondence(target,reference,k=16,tau=.07):
    sim=target@reference.T/tau;values,indices=sim.topk(min(k,sim.shape[-1]),-1);weights=values.softmax(-1)
    return torch.zeros_like(sim).scatter(-1,indices,weights)


def dice(a,b,area):return 2*(torch.minimum(a,b)*area).sum()/((a+b)*area).sum().clamp_min(1e-12)


def projection_dice(support,occupancy,cells,h,w):
    # Axis projections by max occupancy; interval-cell integration is exact
    # at the cached grid resolution. No semantic mask ground truth is used.
    s=support.reshape(h,w);o=occupancy.reshape(h,w)
    dx=(cells[:w,2]-cells[:w,0]);dy=(cells[::w,3]-cells[::w,1])
    return .5*(dice(s.amax(0),o.amax(0),dx)+dice(s.amax(1),o.amax(1),dy))


def build(features,cells,times,positions,seeds,source,teacher,window,confidence_threshold=.05):
    T,N,C=features.shape;f=F.normalize(features,dim=-1);area=(cells[:,2:]-cells[:,:2]).prod(-1);h=len(cells[:,1].unique());w=N//h
    boxes=source[:,None].repeat(1,3,1);valid=torch.zeros(T,3,device=f.device,dtype=torch.bool);valid[:,0]=True;maps=[];reliability=[]
    gap=(times[positions][1:]-times[positions][:-1]).median().clamp_min(1e-8)
    for t in range(T):
        left=max([a for a,p in enumerate(positions)if p<=t],default=0);right=min([a for a,p in enumerate(positions)if p>=t],default=7);parts=[];rs=[];regions=[]
        for a in dict.fromkeys([left,right]):
            pos=positions[a]
            if t==pos:p=seeds[a];back=p
            else:
                A=correspondence(f[t],f[pos]);p=A@seeds[a];back=correspondence(f[pos],f[t])@p
            contrast=(p.max()-p.min()).clamp(0,1);r=dice(seeds[a],back,area)*contrast*torch.exp(-(times[t]-times[pos]).abs()/gap).to(f)
            parts.append(p);rs.append(r)
            binary=(p>=.5*p.max())&(p>1e-8);components,n=label(binary.cpu().numpy().reshape(h,w))
            for k in range(1,n+1):
                mask=torch.from_numpy((components==k).reshape(-1)).to(f.device);lo=cells[mask,:2].amin(0);hi=cells[mask,2:].amax(0);b=torch.cat([.5*(lo+hi),hi-lo]);quality=float((p[mask]*area[mask]).sum()*r);regions.append((quality,b))
        if t in positions:regions.insert(0,(float('inf'),teacher[positions.index(t)].clone()))
        regions.sort(key=lambda x:x[0],reverse=True);j=1
        for quality,b in regions:
            if quality<=0:continue
            if any(float(iou(b,boxes[t,k]))>=.7 for k in range(j)):continue
            boxes[t,j]=b;valid[t,j]=True;j+=1
            if j==3:break
        rs=torch.stack(rs);maps.append((torch.stack(parts)*rs[:,None]).sum(0)/rs.sum().clamp_min(1e-12));reliability.append(rs.max())
    maps=torch.stack(maps);confidence=torch.stack(reliability);inside=(times>=window[0])&(times<window[1]);active=(confidence>=confidence_threshold)&inside&(valid.sum(-1)>1)
    valid[~active,1:]=False
    occ=torch.stack([cell_occupancy(boxes[:,j],cells)for j in range(3)],1);a=(occ*area).sum(-1)
    desc=F.normalize(torch.einsum('tkn,tnc->tkc',occ*area,f)/a[:,:,None].clamp_min(1e-12),dim=-1)
    support=torch.stack([torch.stack([projection_dice(maps[t],occ[t,j],cells,h,w)for j in range(3)])for t in range(T)])
    anchor=F.normalize(torch.einsum('an,anc->ac',seeds*area,f[positions]),dim=-1)
    D=torch.cat([source[:,2:],torch.ones_like(source[:,2:])],-1).clamp_min(1e-5);delta=((chart(boxes)-chart(source)[:,None])/D[:,None]).square().sum(-1);deviation=1-(-.5*delta).exp()
    links=[]
    for t in range(T-1):
        A=correspondence(f[t+1],f[t]);prop=occ[t]@A.T
        links.append(torch.stack([torch.stack([projection_dice(prop[i],occ[t+1,j],cells,h,w)for j in range(3)])for i in range(3)]))
    return dict(boxes=boxes,valid=valid,active=active,descriptors=desc,support=support,deviation=deviation,links=torch.stack(links),confidence=confidence,initial_memory=anchor[int(torch.nonzero(seeds.sum(-1)>1e-8)[0])]if bool((seeds.sum(-1)>1e-8).any())else desc[0,0],positions=torch.tensor(positions,device=f.device),inside=inside)
