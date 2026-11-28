"""Shared pairwise geometry, attention controls, and a single utility policy."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from provenance import CAPACITY
from descriptors import EDGE_FIELDS,NODE_FIELDS
from verifier import DROP, WIDTH, RelativeVerifier, top_nonanchor

NODE = ('inside','shell','span_x','span_y','hit','ray_huber','pass','image_iou','area_log','hull_fill',
        'prior_log_l','prior_log_w','prior_log_h','prior_cos2')
EDGE = ('iou3d','shift_l','shift_w','shift_h','log_l','log_w','log_h','yaw_cos2')
NODE_INDEX = tuple(NODE_FIELDS.index(k) for k in NODE)
EDGE_INDEX = tuple(EDGE_FIELDS.index(k) for k in EDGE)
RISK = .10
TAU = .05


@dataclass(frozen=True)
class Arm:
    cap: int = 21
    mean: bool = False
    utility: bool = False
    lean: bool = False


S0 = Arm()
ARMS = {
    'S1': Arm(cap=17),
    'S1-13': Arm(cap=13),
    'S2': Arm(mean=True),
    'S3': Arm(utility=True),
    'S4': Arm(lean=True),
    'S5-full17': Arm(cap=17,mean=True,utility=True,lean=True),
    'S5-full13': Arm(cap=13,mean=True,utility=True,lean=True),
}


class Pairwise(nn.Module):
    def __init__(self,nodes,edges,utility):
        super().__init__()
        joint = 2*nodes+edges
        self.message = nn.Sequential(nn.LayerNorm(joint),nn.Linear(joint,32),nn.GELU(),
                                     nn.Dropout(DROP),nn.Linear(32,32))
        self.evaluate = nn.Sequential(nn.LayerNorm(joint+32),nn.Linear(joint+32,32),nn.GELU(),
                                      nn.Dropout(DROP),nn.Linear(32,1 if utility else 3))
        self.utility = utility

    def forward(self,nodes,edges,valid):
        k = nodes.shape[1]
        own = nodes[:,:,None].expand(-1,-1,k,-1)
        other = nodes[:,None].expand(-1,k,-1,-1)
        msg = self.message(torch.cat((own,other,edges),dim=-1))
        total = (msg*valid[:,None,:,None]).sum(2)/valid.sum(1)[:,None,None]
        anchor = nodes[:,:1].expand_as(nodes)
        outputs = self.evaluate(torch.cat((nodes,anchor,edges[:,0],total),dim=-1))
        if self.utility:
            return outputs[...,0]-outputs[:,:1,0]
        return tuple(outputs.unbind(-1))


class AttentionControl(RelativeVerifier):
    def __init__(self,nodes,edges,utility):
        super().__init__()
        if nodes != 46:
            self.embedding[0] = nn.LayerNorm(nodes)
            self.embedding[1] = nn.Linear(nodes,WIDTH)
        if edges != 14:
            for layer in self.layers:
                layer.edge_bias = nn.Linear(edges,layer.edge_bias.out_features)
                layer.edge_value = nn.Linear(edges,WIDTH)
            self.gate[0] = nn.Linear(3*WIDTH+edges,WIDTH)
        self.utility = utility
        if utility:
            del self.gate

    def forward(self,nodes,edges,valid):
        if not self.utility:
            return super().forward(nodes,edges,valid)
        x = self.embedding(nodes)
        for layer in self.layers:
            x = layer(x,edges,valid)
        scores = self.rank(self.final(x)).squeeze(-1)
        return scores-scores[:,:1]


def model_for(arm):
    if arm == S0 or not any((arm.mean,arm.utility,arm.lean)):
        return RelativeVerifier()
    cls = Pairwise if arm.mean else AttentionControl
    return cls(14 if arm.lean else 46,8 if arm.lean else 14,arm.utility)


def risk_loss(pred,truth,valid):
    anchor = truth[:,:1]
    u = truth-anchor-RISK*((anchor>.50)&(truth<=.50))
    logits = pred.masked_fill(~valid,-1.e4)
    target = (u/TAU).masked_fill(~valid,-1.e4).softmax(1)
    return F.kl_div(logits.log_softmax(1),target,reduction='batchmean')


def raw_choice(output,valid,single):
    if not single:
        return torch.stack(top_nonanchor(*output,valid),1)
    top,i = output[:,1:].masked_fill(~valid[:,1:],-torch.inf).max(1)
    return torch.stack((i+1,top),1)


def relative_nodes(nodes,valid):
    v = nodes[:,:,[20,16,2,18]]
    count = valid.sum(1)[:,None,None]
    mean = (v*valid[...,None]).sum(1,keepdim=True)/count
    sigma = (((v-mean).square()*valid[...,None]).sum(1,keepdim=True)/count).sqrt()
    peers = valid[:,None,:,None]
    left = ((v[:,:,None]>v[:,None])&peers).sum(2)
    tied = ((v[:,:,None]==v[:,None])&peers).sum(2)
    rank = (left+.5*(tied+1))/count
    nodes[:,:,32:44] = torch.cat(((v-mean)/(sigma+1.e-8),rank,v-v[:,:1]),2)
    return nodes.masked_fill(~valid[...,None],0.)


def project_tensors(data,membership,arm):
    """Compact in original order and return the original candidate indices."""
    nodes,edges,valid,labels = data
    k = nodes.shape[1]
    indices = torch.arange(k,device=nodes.device)[None].expand(len(nodes),-1)
    if arm.cap < 21:
        keep = membership[:,:,int(arm.cap==13)]&valid
        i = indices.masked_fill(~keep,k).sort(1).values[:,:arm.cap]
        valid = i<k
        i = i.clamp(max=k-1)
        row = torch.arange(len(nodes),device=nodes.device)
        if arm.lean:
            nodes = nodes[:,:,NODE_INDEX]
            edges = edges[:,:,:,EDGE_INDEX]
        nodes = nodes[row[:,None],i]
        edges = edges[row[:,None,None],i[:,:,None],i[:,None,:]]
        labels = labels.gather(1,i).masked_fill(~valid,0.)
        nodes = nodes.masked_fill(~valid[...,None],0.)
        edges = edges.masked_fill(~(valid[:,:,None]&valid[:,None,:])[...,None],0.)
        if not arm.lean:
            nodes = relative_nodes(nodes,valid)
        return (nodes,edges,valid,labels),i
    if arm.lean:
        nodes,edges = nodes[:,:,NODE_INDEX],edges[:,:,:,EDGE_INDEX]
    return (nodes,edges,valid,labels),indices
