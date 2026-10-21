"""Backbone-independent video evidence, parameter subspace and trajectory score.

No model, target-query, offset, ASA or TTS definitions appear in this module.
Transition consistency is an observable heuristic, not an identity annotation.
"""
import torch
import torch.nn.functional as F


def chart(boxes):
    return torch.cat([boxes[...,:2],boxes[...,2:].clamp_min(1e-8).log()],-1)


def cell_occupancy(boxes,cells):
    lo=boxes[...,:2]-boxes[...,2:]/2;hi=boxes[...,:2]+boxes[...,2:]/2
    overlap=(torch.minimum(hi[...,None,:],cells[None,:,2:])-torch.maximum(lo[...,None,:],cells[None,:,:2])).clamp_min(0).prod(-1)
    area=(cells[:,2:]-cells[:,:2]).prod(-1)
    return (overlap/area).clamp(0,1)


def evidence_subspace(gradients,rms=.01,max_coordinate=.05):
    # Each column keeps its evidence provenance; no feature-PCA or per-model
    # learned propagation. Rank/collinearity must be measured, not assumed.
    G=torch.stack(gradients,-1).detach();norm=G.norm(dim=0);unit=G/norm.clamp_min(1e-20)
    length=torch.minimum(G.new_full(norm.shape,rms*G.shape[0]**.5),max_coordinate/unit.abs().amax(0).clamp_min(1e-20))
    return unit*length,unit.T@unit


def initial_candidates(anchors,device,dtype):
    eye=torch.eye(anchors,device=device,dtype=dtype);mean=torch.ones(1,anchors,device=device,dtype=dtype)/anchors
    first=mean.clone();first[:,anchors//2:]=0;first/=first.sum();last=mean.clone();last[:,:anchors//2]=0;last/=last.sum()
    return torch.cat([torch.zeros_like(mean),eye,-eye,mean,-mean,first,last],0)


def component_update(candidates,rewards,temperature=.1):
    """Reward-weighted parameter-policy search, not claimed new distillation.

    Keep separate evidence directions; never average different components into
    one output tube. Only within-component[zero,+e_a,-e_a] means are formed.
    """
    anchors=(len(candidates)-5)//2;centers=[];weights=[]
    for a in range(anchors):
        index=torch.tensor([0,1+a,1+anchors+a],device=candidates.device)
        w=(rewards[index]/temperature).softmax(0);centers.append((w[:,None]*candidates[index]).sum(0));weights.append(w)
    return torch.stack(centers),torch.stack(weights)


class VideoEvidence:
    def __init__(self,features,cells,times,positions,seed_mass,source,window,temperature=.07):
        # features[T,N,C], cells[N,xyxy], timestamps in seconds. All frozen.
        if features.requires_grad or seed_mass.requires_grad or source.requires_grad:raise ValueError('Evidence must be frozen')
        T,N,C=features.shape;assert cells.shape==(N,4)and len(times)==T and len(positions)==8 and seed_mass.shape==(8,N)
        assert bool((times[1:]>times[:-1]).all())and len(set(positions))==8
        self.cells=cells.detach();self.source=source.detach();self.area=(cells[:,2:]-cells[:,:2]).prod(-1);assert torch.allclose(self.area.sum(),self.area.new_tensor(1.),atol=1e-5)
        feature=F.normalize(features.float(),dim=-1);prior=self.area/self.area.sum();logarea=prior.log()
        forward=[];reverse=[]
        for t in range(T-1):
            sim=feature[t]@feature[t+1].T/temperature
            forward.append((sim+logarea[None]).softmax(-1));reverse.append((sim.T+logarea[None]).softmax(-1))
        self.forward=torch.stack(forward);self.reverse=torch.stack(reverse)
        self.time_weight=((times>=window[0])&(times<window[1])).to(feature);assert self.time_weight.sum()>0
        anchor_times=times[torch.tensor(positions,device=times.device)];gap=(anchor_times[1:]-anchor_times[:-1]).median().clamp_min(1e-8)
        density=[];confidence=[]
        for a,pos in enumerate(positions):
            seed=seed_mass[a];seed=seed/seed.sum()if seed.sum()>0 else prior.clone();paths=[None]*T;messages=[None]*T;paths[pos]=seed;messages[pos]=seed
            for t in range(pos,T-1):paths[t+1]=paths[t]@self.forward[t];messages[t+1]=self.reverse[t]@messages[t]
            for t in range(pos-1,-1,-1):paths[t]=paths[t+1]@self.reverse[t];messages[t]=self.forward[t]@messages[t+1]
            p=torch.stack(paths);cycle=(p*torch.stack(messages)).sum(-1)/seed.square().sum().clamp_min(1e-20);cycle=cycle.clamp(0,1)
            contrast=.5*(p-prior).abs().sum(-1);decay=(-(times-times[pos]).abs()/gap).exp().to(p)
            density.append(p);confidence.append(contrast*cycle*decay)
        self.density=torch.stack(density).detach();self.confidence=torch.stack(confidence).detach();self.evidence_strength=self.confidence.amax(0)
        assert torch.isfinite(self.density).all()and torch.isfinite(self.confidence).all()

    def score(self,boxes,association_weight=.5,change_weight=.25,hypothesis_temperature=.1):
        # Same continuous reward for OPD and Direct, evaluated on actual tubes.
        occ=cell_occupancy(boxes,self.cells);area=(occ*self.area).sum(-1);inside=(self.density*occ[None]).sum(-1)
        c=self.confidence*self.time_weight[None];support=((inside-area[None])*c).sum(-1)/c.sum(-1).clamp_min(1e-12)
        conditional=self.density*occ[None]/inside[:,:,None].clamp_min(1e-12)
        f=torch.einsum('atn,tnm->atm',conditional[:,:-1],self.forward);r=torch.einsum('atn,tnm->atm',conditional[:,1:],self.reverse)
        hit=.5*((f*occ[None,1:]).sum(-1)-area[None,1:]+(r*occ[None,:-1]).sum(-1)-area[None,:-1])
        pair=(c[:,:-1]*c[:,1:]).clamp_min(0).sqrt();association=(hit*(inside[:,:-1]*inside[:,1:]).clamp_min(0).sqrt()*pair).sum(-1)/pair.sum(-1).clamp_min(1e-12)
        D=torch.cat([self.source[:,2:],torch.ones_like(self.source[:,2:])],-1).clamp_min(1e-5)
        delta=((chart(boxes)-chart(self.source))/D).square().sum(-1);lack=(1-self.evidence_strength)*self.time_weight
        change=((1-(-.5*delta).exp())*lack).sum()/self.time_weight.sum()
        per_anchor=support+association_weight*association-change_weight*change
        total=hypothesis_temperature*(torch.logsumexp(per_anchor/hypothesis_temperature,0)-torch.log(per_anchor.new_tensor(len(per_anchor))))
        return dict(total=total,support=support,association=association,change=change,anchor_rewards=per_anchor,anchor_posterior=(per_anchor/hypothesis_temperature).softmax(0))
