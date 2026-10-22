"""Finite numeric profiles; preserve upstream operations and exact DP tie order."""
import copy,hashlib,json
import torch
import torch.nn.functional as F
from stvg.predictor import output_writeback
from ._spatial_math import occupancy,extent_kernel,soft_dice,EPS,scoped_single_thread
from .spatial import BoxFixedBank,evidence_dp


def profile_id(p):return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:16]


def baseline(dataset):return dict(gamma=.5 if dataset=='hc2'else 1.,admission=.1,guard=.1,link=.25)


@scoped_single_thread
def view_geometry(base,evidence,Source,raw,p):
    E=torch.from_numpy(output_writeback(Source,raw,p['gamma'])).double();g=copy.copy(base);g.metadata=dict(base.metadata)
    if not torch.equal(E,base.E):
        g.E=E;T=g.frames;cells=evidence.valid;features=evidence.features
        g.side_reference_boxes=base.side_reference_boxes.clone()
        for side in [0,1]:g.side_reference_boxes[~g.side_available[:,side],side]=E[~g.side_available[:,side]]
        g.boxes=E[:,None].expand(-1,3,-1).clone()
        for side in [0,1]:g.boxes[g.side_available[:,side],side+1]=g.side_reference_boxes[g.side_available[:,side],side]
        occ,_,_=occupancy(g.boxes,g.height,g.width);occ*=cells.reshape(T,1,g.height,g.width)
        pooled=torch.bmm(occ.flatten(-2),features.double())/occ.flatten(-2).sum(-1).clamp_min(EPS)[...,None]
        g.features=F.normalize(pooled,dim=-1,eps=1e-12);g.E_features=g.features[:,0]
        g.extent_scores=extent_kernel(g.boxes[:,:,None],g.side_reference_boxes[:,None]);g.association_scores=torch.einsum('tkd,tsd->tks',g.features,g.side_prototypes).clamp(0,1)
        side_score=.5*g.extent_scores+.5*g.association_scores;g.projection=(side_score*g.side_reliability[:,None]).sum(-1)/g.side_reliability.sum(-1).clamp_min(EPS)[:,None];g.u=g.projection
        dz=torch.cat(((g.boxes[...,:2]-E[:,None,:2])/E[:,None,2:],g.boxes[...,2:].log()-E[:,None,2:].log()),-1);g.guard=-torch.expm1(-(torch.linalg.vector_norm(dz,dim=-1)/2).square())
        masked=(occ*g.merged_targets[:,None]).flatten(-2);previous=evidence.transport_adjacent(masked[:-1].transpose(0,1)).transpose(0,1);g.link=soft_dice(previous[:,:,None],masked[1:,None])
    g.reliable=(g.confidence>=p['admission'])&g.side_available.any(-1);g.valid=torch.zeros(g.frames,3,dtype=torch.bool);g.valid[:,0]=True;g.candidate_origin=torch.full((g.frames,3),-1,dtype=torch.long);g.candidate_quality=torch.zeros(g.frames,3,dtype=torch.float64)
    for t in range(g.frames):
        for side in [0,1]:
            if g.side_available[t,side]and(g.reliable[t]or g.observed[t]):g.valid[t,side+1]=True;g.candidate_origin[t,side+1]=-2 if g.observed[t]else side;g.candidate_quality[t,side+1]=g.side_reliability[t,side]
        if g.valid[t,1]and g.valid[t,2]and torch.equal(g.boxes[t,1],g.boxes[t,2]):g.valid[t,2]=False
    return g


@scoped_single_thread
def predict(base,evidence,Source,raw,interval,p):
    g=view_geometry(base,evidence,Source,raw,p);bank=BoxFixedBank(g,interval=interval)
    if p['guard']!=.1 or p['link']!=.25:
        obs=bank.event&bank.observed;unobs=bank.event&~bank.observed
        w=.5*obs.double()/max(1,int(obs.sum()))+.5*bank.confidence*unobs/max(1,int(unobs.sum()))
        bank.score_unary=w[:,None]*bank.projection-p['guard']*bank.guard/bank.frames
        ew=p['link']*bank.event_edges*bank.confidence[:-1]*bank.confidence[1:]/max(1,int(bank.event_edges.sum()));bank.score_pair=ew[:,None,None]*bank.link
    result=evidence_dp(bank);result['candidate_admitted_frames']=int(bank.active.sum());result['changed_from_ERef_frames']=int((result['final']!=g.E).any(-1).sum());return result
