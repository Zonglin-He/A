"""Language-conditioned candidate-local RGB cross-attention before matching.
Zero-init residual exactly preserves source; original boxes are never modified.
"""
import torch,math,numpy as np
from torch import nn
from common25 import *
F=torch.nn.functional

def source_projection():
 m=nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64));s=load_npz(BASE/'source_features_attempt002/projection_source_PRIVATE.npz');m.load_state_dict({k:torch.tensor(s['image.'+k]) for k in m.state_dict()});return m

class TrimodalFusion(nn.Module):
 def __init__(self,projection=None):
  super().__init__();self.hnorm=nn.LayerNorm(288);self.tnorm=nn.LayerNorm(288);self.rnorm=nn.LayerNorm(384);self.q=nn.Linear(288,64);self.text=nn.Linear(288,64);self.key=nn.Linear(384,64);self.value=nn.Linear(384,64);self.position=nn.Parameter(torch.randn(5,64)*.01);self.delta=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,288));self.gate=nn.Sequential(nn.Linear(136,32),nn.GELU(),nn.Linear(32,1));self.projection=source_projection() if projection is None else projection;nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias);nn.init.constant_(self.gate[-1].bias,-2.)
 def forward(self,b,mode='rgb',rgb_dropout=0.):
  h=b['h'].float();t=self.tnorm(b['text_mean'].float());q=self.q(self.hnorm(h))+self.text(t)[:,None];r=b['rgb'].float();glob=b['global'].float()[:,None,None].expand(-1,256,1,-1);r=torch.cat([r,glob],2)
  if mode=='no_rgb':r=torch.zeros_like(r)
  elif mode=='zero_rgb':r=torch.zeros_like(r)
  elif mode=='permute_rgb':r=r.roll(17,dims=1)
  elif mode=='global_rgb':r=torch.cat([torch.zeros_like(r[:,:,:4]),r[:,:,4:]],2)
  elif mode!='rgb':raise ValueError(mode)
  if self.training and rgb_dropout:
   keep=(torch.rand((len(r),1,1,1),device=r.device)>=rgb_dropout);r=r*keep
  r=self.rnorm(r);k=self.key(r)+self.position[None,None];v=self.value(r);attention=(q[:,:,None]*k).sum(-1)/math.sqrt(64);attention=attention.softmax(-1);visual=(attention[:,:,:,None]*v).sum(2);delta=self.delta(torch.cat([q,visual,q*visual],-1));alpha=self.gate(torch.cat([q,visual,b['geom'].float()],-1)).sigmoid()*b['valid'][:,:,None];fused=h+alpha*delta;query=F.normalize(self.projection(fused),dim=-1);logits=query@b['tokens'].float().transpose(-1,-2)/.07;logits=logits.masked_fill(b['padding'][:,None],-torch.inf);scores=(logits.softmax(-1)*b['mask'][:,None]).sum(-1)
  return scores,{'gate':alpha.squeeze(-1),'delta_energy':(alpha*delta).square().mean(),'visual_attention':attention}

def bank(rr,labels=None):
 zz=[feature(r) for r in rr];rgb=[rgb_feature(r) for r in rr];N=len(rr);T=max(len(z['tokens']) for z in zz);tokens=np.zeros((N,T,64),np.float32);padding=np.ones((N,T),bool);mask=np.zeros((N,T),bool)
 for i,z in enumerate(zz):L=len(z['tokens']);tokens[i,:L]=z['tokens'];padding[i,:L]=False;mask[i,:L]=z['token_mask']
 data={'h':np.stack([z['h'] for z in zz]),'tokens':tokens,'padding':padding,'mask':mask,'text_mean':np.stack([z['text_h'][1:-1].mean(0) for z in zz]),'scores':np.stack([z['scores'] for z in zz]),'rgb':np.stack([x['rgb'] for x in rgb]),'global':np.stack([x['global_token'] for x in rgb]),'geom':np.stack([x['geom'] for x in rgb]),'valid':np.stack([x['valid'] for x in rgb])}
 if labels is not None:data['ious']=np.stack([labels[r['id']] for r in rr]).astype(np.float32)
 return {k:torch.from_numpy(v) for k,v in data.items()}

def part(data,ix,device='cuda'):return {k:v[ix].to(device,non_blocking=True) for k,v in data.items()}
