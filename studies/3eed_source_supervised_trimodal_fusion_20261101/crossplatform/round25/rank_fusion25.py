"""Second finite source-only readout: gated residual ranking logits.
Original matching projection is frozen; a zero-init learned scorer reads the
same language-conditioned RGB/point interaction, preserving source ordering.
"""
import torch,math
from torch import nn
from fusion25 import TrimodalFusion,F
class RankFusion(TrimodalFusion):
 def __init__(self,projection=None):
  super().__init__(projection=projection);self.projection.requires_grad_(False);self.rank=nn.Sequential(nn.Linear(200,64),nn.GELU(),nn.Linear(64,1));nn.init.zeros_(self.rank[-1].weight);nn.init.zeros_(self.rank[-1].bias)
 def forward(self,b,mode='rgb',rgb_dropout=0.):
  h=b['h'].float();t=self.tnorm(b['text_mean'].float());q=self.q(self.hnorm(h))+self.text(t)[:,None];r=b['rgb'].float();glob=b['global'].float()[:,None,None].expand(-1,256,1,-1);r=torch.cat([r,glob],2)
  if mode in ['no_rgb','zero_rgb']:r=torch.zeros_like(r)
  elif mode=='permute_rgb':r=r.roll(17,dims=1)
  elif mode=='global_rgb':r=torch.cat([torch.zeros_like(r[:,:,:4]),r[:,:,4:]],2)
  elif mode!='rgb':raise ValueError(mode)
  if self.training and rgb_dropout:r=r*(torch.rand((len(r),1,1,1),device=r.device)>=rgb_dropout)
  r=self.rnorm(r);k=self.key(r)+self.position[None,None];v=self.value(r);attention=((q[:,:,None]*k).sum(-1)/math.sqrt(64)).softmax(-1);visual=(attention[:,:,:,None]*v).sum(2);joint=torch.cat([q,visual,q*visual],-1);delta=self.delta(joint);alpha=self.gate(torch.cat([q,visual,b['geom'].float()],-1)).sigmoid()*b['valid'][:,:,None];fused=h+alpha*delta;query=F.normalize(self.projection(fused),dim=-1);logits=query@b['tokens'].float().transpose(-1,-2)/.07;logits=logits.masked_fill(b['padding'][:,None],-torch.inf);mass=(logits.softmax(-1)*b['mask'][:,None]).sum(-1);residual=self.rank(torch.cat([joint,b['geom'].float()],-1)).squeeze(-1)*alpha.squeeze(-1);scores=mass.clamp_min(1e-8).log()+residual
  return scores,{'gate':alpha.squeeze(-1),'delta_energy':(alpha*delta).square().mean()+residual.square().mean()*.01,'visual_attention':attention}
