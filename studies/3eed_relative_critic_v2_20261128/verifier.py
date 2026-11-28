"""Edge-biased set attention and the asymmetric anchor-relative objective."""
import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 128
HEADS = 4
DEPTH = 3
DROP = .15


class EdgeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH,3*WIDTH)
        self.edge_bias = nn.Linear(14,HEADS)
        self.edge_value = nn.Linear(14,WIDTH)
        self.out = nn.Linear(WIDTH,WIDTH)
        self.ff = nn.Sequential(nn.LayerNorm(WIDTH),nn.Linear(WIDTH,2*WIDTH),
                                nn.GELU(),nn.Dropout(DROP),nn.Linear(2*WIDTH,WIDTH))
        self.dropout = nn.Dropout(DROP)

    def forward(self, x, edge, mask):
        n,k,_ = x.shape
        shape = (n,k,HEADS,WIDTH//HEADS)
        q,kvec,v = [t.reshape(shape).transpose(1,2) for t in self.qkv(self.norm(x)).chunk(3,dim=-1)]
        logits = q@kvec.transpose(-1,-2) * (WIDTH//HEADS)**-.5
        logits = logits + self.edge_bias(edge).permute(0,3,1,2)
        weights = self.dropout(logits.masked_fill(~mask[:,None,None,:],-1.e4).softmax(-1))
        changes = self.edge_value(edge).reshape(n,k,k,HEADS,WIDTH//HEADS)
        interaction = (weights.permute(0,2,3,1)[...,None]*changes).sum(2).flatten(2)
        update = (weights@v).transpose(1,2).reshape(n,k,WIDTH) + interaction
        x = x + self.dropout(self.out(update))
        x = x + self.dropout(self.ff(x))
        return x.masked_fill(~mask[...,None],0.)


class RelativeVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(nn.LayerNorm(46),nn.Linear(46,WIDTH),nn.GELU(),
                                       nn.Dropout(DROP),nn.Linear(WIDTH,WIDTH))
        self.layers = nn.ModuleList(EdgeAttention() for _ in range(DEPTH))
        self.final = nn.LayerNorm(WIDTH)
        self.rank = nn.Sequential(nn.Linear(WIDTH,64),nn.GELU(),nn.Linear(64,1))
        self.gate = nn.Sequential(nn.Linear(3*WIDTH+14,WIDTH),nn.GELU(),nn.Dropout(DROP),nn.Linear(WIDTH,2))

    def forward(self,x,edge,mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x,edge,mask)
        x = self.final(x)
        anchor = x[:,:1].expand_as(x)
        residual = self.gate(torch.cat((x,anchor,x-anchor,edge[:,0]),dim=-1))
        return self.rank(x).squeeze(-1),residual[...,0],residual[...,1]


def objective(scores, gate, improvement, truth, valid):
    s = scores.masked_fill(~valid,-1.e4)
    top = truth.masked_fill(~valid,-1.e4).amax(1) >= .15
    distribution = (truth/.05).masked_fill(~valid,-1.e4).softmax(1)
    listnet = -(distribution*s.log_softmax(1)).sum(1)
    rank_loss = (top*listnet).sum()/top.sum().clamp(min=1)
    delta = truth[:,1:] - truth[:,:1]
    diff = s[:,1:] - s[:,:1]
    mask = valid[:,1:]
    incumbent = truth[:,:1] >= .50
    contender = truth[:,1:] >= .50
    penalty = abs(delta)*(1. + 3.*(incumbent&~contender) + 1.5*(~incumbent&contender))
    asym = penalty*F.softplus(-4.*delta.sign()*diff)
    prob = F.binary_cross_entropy_with_logits(gate[:,1:],(delta>.03).float(),reduction='none')
    gain = F.huber_loss(improvement[:,1:],delta,reduction='none',delta=.1)
    size = mask.sum(1).clamp(min=1)
    asym_loss = ((asym*mask).sum(1)/size).mean()
    gate_loss = (((prob+2.*gain)*mask).sum(1)/size).mean()
    return rank_loss + asym_loss + .5*gate_loss


def top_nonanchor(scores, gate, improvement, valid):
    highest = scores[:,1:].masked_fill(~valid[:,1:],-torch.inf).max(1).indices + 1
    row = torch.arange(len(scores),device=scores.device)
    margin = (scores[row,highest]-scores[:,0]).masked_fill(~valid[row,highest],-torch.inf)
    return highest,margin,gate[row,highest].sigmoid(),improvement[row,highest]
