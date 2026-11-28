"""Models for the staged verifier ablation."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models import AttentionControl, RISK, TAU
from verifier import DROP, HEADS


@dataclass(frozen=True)
class Design:
    pool: str = 'base'
    channels: int = 14
    depth: int = 3
    width: int = 128
    anchor: bool = False


CONTROL = 'S5-pass'
ORIGINAL = {
    'Anchor': Design(depth=0,anchor=True),
    'Attention1-128': Design(depth=1),
    'Attention1-64': Design(depth=1,width=64),
    'Nodes8': Design(channels=8),
    'Nodes4': Design(channels=4),
    'CompactGraph': Design(channels=8,depth=0,width=64,anchor=True),
}
PROPOSAL = {
    'No90': Design(pool='no90'),
    'Extent': Design(pool='extent'),
    'Geometry': Design(pool='geometry'),
    'Adaptive': Design(pool='adaptive'),
    'Compact': Design(pool='geometry',channels=8,depth=0,width=64,anchor=True),
    'CompactAdaptive': Design(pool='adaptive_geometry',channels=8,depth=0,width=64,anchor=True),
}
DESIGNS = {CONTROL:Design(),**ORIGINAL,**PROPOSAL}


def reduce_nodes(nodes,channels):
    if channels==14:
        return nodes
    dim = torch.linalg.vector_norm(nodes[...,10:13],dim=-1,keepdim=True)
    if channels==8:
        return torch.cat((nodes[...,[0,1,4,5,6,7]],dim,nodes[...,13:14]),dim=-1)
    assert channels==4
    return torch.cat((nodes[...,[0,5,7]],dim),dim=-1)


class AnchorSet(nn.Module):
    def __init__(self,channels,width):
        super().__init__()
        features = 2*channels+8
        self.phi = nn.Sequential(nn.LayerNorm(features),nn.Linear(features,width),nn.GELU(),
                                 nn.Dropout(DROP),nn.Linear(width,width))
        self.score = nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,width),nn.GELU(),
                                   nn.Dropout(DROP),nn.Linear(width,1))

    def forward(self,nodes,edges,valid):
        anchor = nodes[:,:1].expand_as(nodes)
        z = self.phi(torch.cat((nodes,anchor,edges),dim=-1))
        pooled = (z*valid[...,None]).sum(1,keepdim=True)/valid.sum(1)[:,None,None]
        return self.score(torch.cat((z,pooled.expand_as(z)),dim=-1)).squeeze(-1)


class CompactAttention(nn.Module):
    def __init__(self,channels,width):
        super().__init__()
        self.width = width
        self.embedding = nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,width),nn.GELU(),
                                       nn.Dropout(DROP),nn.Linear(width,width))
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width,3*width)
        self.edge_bias = nn.Linear(8,HEADS)
        self.edge_value = nn.Linear(8,width)
        self.out = nn.Linear(width,width)
        self.ff = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),
                                nn.GELU(),nn.Dropout(DROP),nn.Linear(2*width,width))
        self.drop = nn.Dropout(DROP)
        self.rank = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,nodes,edge,valid):
        x = self.embedding(nodes)
        n,k,w = x.shape
        d = w//HEADS
        q,key,v = [t.reshape(n,k,HEADS,d).transpose(1,2) for t in self.qkv(self.norm(x)).chunk(3,dim=-1)]
        logits = (q@key.transpose(-1,-2))*d**-.5 + self.edge_bias(edge).permute(0,3,1,2)
        weight = self.drop(logits.masked_fill(~valid[:,None,None,:],-1.e4).softmax(-1))
        edge_v = self.edge_value(edge).reshape(n,k,k,HEADS,d)
        relation = (weight.permute(0,2,3,1)[...,None]*edge_v).sum(2).flatten(2)
        x = x+self.drop(self.out((weight@v).transpose(1,2).reshape(n,k,w)+relation))
        return self.rank(x+self.drop(self.ff(x))).squeeze(-1)


class FullAttention(AttentionControl):
    def __init__(self,channels,depth):
        super().__init__(channels,8,True)
        self.layers = self.layers[:depth]

    def forward(self,nodes,edges,valid):
        x = self.embedding(nodes)
        for layer in self.layers:
            x = layer(x,edges,valid)
        return self.rank(self.final(x)).squeeze(-1)


def model_for(design):
    if design.anchor:
        return AnchorSet(design.channels,design.width)
    if design.width==64:
        return CompactAttention(design.channels,design.width)
    return FullAttention(design.channels,design.depth)


def transform(data,design):
    nodes,edges,valid,*rest = data
    return (reduce_nodes(nodes,design.channels),edges[:,0] if design.anchor else edges,valid,*rest)


def risk_loss(scores,iou,valid):
    utility = iou-RISK*((iou[:,:1]>.50)&(iou<=.50))
    target = (utility/TAU).masked_fill(~valid,-1.e4).softmax(1)
    return F.kl_div(scores.masked_fill(~valid,-1.e4).log_softmax(1),target,reduction='batchmean')


def equivalence():
    from models import risk_loss as prior_loss

    gen = torch.Generator().manual_seed(20261128)
    scores0 = torch.randn((128,13),generator=gen,dtype=torch.float64)
    iou = torch.rand((128,13),generator=gen,dtype=torch.float64)
    valid = torch.rand((128,13),generator=gen)>.4
    valid[:,0] = True
    result = {}
    for dtype in (torch.float64,torch.float32):
        tol = 1.e-12 if dtype==torch.float64 else 2.e-6
        scores = scores0.to(dtype).requires_grad_()
        truth = iou.to(dtype)
        adjusted = truth-RISK*((truth[:,:1]>.50)&(truth<=.50))
        old = ((truth-truth[:,:1]-RISK*((truth[:,:1]>.50)&(truth<=.50)))/TAU)
        p_old = old.masked_fill(~valid,-1.e4).softmax(1)
        p_new = (adjusted/TAU).masked_fill(~valid,-1.e4).softmax(1)
        a,b = prior_loss(scores-scores[:,:1],truth,valid),risk_loss(scores,truth,valid)
        ga,gb = (torch.autograd.grad(loss,scores)[0] for loss in (a,b))
        errors = dict(probability=float((p_old-p_new).abs().max()),loss=float(abs(a-b).detach()),
                      gradient=float((ga-gb).abs().max()))
        assert max(errors.values())<tol,errors
        result[str(dtype)] = errors
    return result
