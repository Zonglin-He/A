"""Baseline-only task interfaces. No original checkpoint is modified.

MM-TTA uses two genuine, source-supervised modality heads with shared binary
referent semantics; native candidate IDs are not permanent semantic classes.
MATE uses a source-prepared masked local-patch autoencoder and task head.
"""
import torch
from torch import nn
from torch.nn import functional as F
class PairHead(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(64,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128,2))
 def forward(self,x):return self.net(x)
class RGBHead(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d(1));self.match=nn.Sequential(nn.Linear(96,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128,2))
 def forward(self,image,text):return self.match(torch.cat([self.enc(image).flatten(1),text.expand(len(image),-1)],1))
class MaskedRegion(nn.Module):
 def __init__(self):
  super().__init__();self.patch=nn.Sequential(nn.Linear(24,96),nn.GELU(),nn.Linear(96,96));self.pos=nn.Linear(3,96);layer=nn.TransformerEncoderLayer(96,4,192,dropout=0,batch_first=True,norm_first=True);self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.decoder=nn.Sequential(nn.Linear(192,96),nn.GELU(),nn.Linear(96,24));self.cls=nn.Sequential(nn.Linear(160,128),nn.GELU(),nn.Linear(128,2))
 def features(self,crops,masked=False,generator=None):
  # Geometry-only grouping:16 local patches,8 points each. Decoder receives
  # patch centres but predicts offsets, so coordinates are not direct answers.
  order=crops[:,:,0].argsort(1);xx=crops.gather(1,order[:,:,None].expand(-1,-1,3)).reshape(-1,16,8,3);center=xx.mean(2);offset=xx-center[:,:,None];tok=self.patch(offset.flatten(2))+self.pos(center)
  if masked:
   ranks=torch.rand(tok.shape[:2],device=tok.device,generator=generator).argsort(1);keep=ranks[:,:2];hidden=ranks[:,2:];enc=self.encoder(tok.gather(1,keep[:,:,None].expand(-1,-1,96)));pooled=enc.mean(1);hpos=self.pos(center).gather(1,hidden[:,:,None].expand(-1,-1,96));pred=self.decoder(torch.cat([pooled[:,None].expand_as(hpos),hpos],-1));target=offset.flatten(2).gather(1,hidden[:,:,None].expand(-1,-1,24));loss=F.mse_loss(pred,target)
  else:pooled=self.encoder(tok).mean(1);loss=None
  return pooled,loss
 def forward(self,crops,joint,masked=False,generator=None):
  h,L=self.features(crops,masked,generator);return self.cls(torch.cat([h,joint],1)),L
def tensor(a,device='cuda'):return {k:torch.as_tensor(v,device=device) for k,v in a.items()}
def image_tensor(a):return a['images'].permute(0,3,1,2).float()/255
