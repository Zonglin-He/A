"""Same frozen cosine region propagation, actual finer layer and coarse control.

No trained projection, new expert, classifier fitting or target annotations.
"""
import torch
import torch.nn.functional as F
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
LAYERS=['Coarse','Fine_layer3','Fine_downsampled']

@torch.no_grad()
def propagate_chunked(maps,valid,positions,boxes,temperature=.07,chunk=4):
 n,c,h,w=maps.shape;dev=maps.device
 f=F.normalize(maps.float().flatten(2).transpose(1,2),dim=-1);outputs=[]
 for a,box in zip(positions,boxes):
  mask=cell_overlap(box.to(maps)[None],h,w)[0].float().flatten()*valid[a].flatten();rows=[]
  for lo in range(0,n,chunk):
   logits=(f[lo:lo+chunk]@f[a].T)/temperature;logits[...,~valid[a].flatten()]=-torch.inf
   m=(logits.softmax(-1)@mask).reshape(-1,h,w)*valid[lo:lo+chunk]
   assert torch.isfinite(m).all()and m.min()>=0 and m.max()<=1+1e-6;rows.append(m.cpu())
  outputs.append(torch.cat(rows))
 return torch.stack(outputs)if outputs else torch.empty(0,n,h,w)


def downsample(maps,valid,size):
 return F.adaptive_avg_pool2d(maps.float(),size),F.interpolate(valid[:,None].float(),size=size,mode='nearest')[:,0].bool()


def properties():
 from vg_tta.stvg_soft_correspondence_v1 import propagate
 torch.manual_seed(20261018);x=torch.randn(5,12,6,8);v=torch.ones(5,6,8,dtype=torch.bool);b=[torch.tensor([.5,.5,.4,.5])]
 a=propagate_chunked(x,v,[1],b,chunk=2);z=propagate(x,v,[1],b)
 assert torch.equal(a,z);d,dv=downsample(x,v,(3,4));assert d.shape==(5,12,3,4)and dv.shape==(5,3,4)
 return dict(status='pass',chunked_same_formula_exact_CPU=True,actual_channels_preserved=True,no_trained_projection=True)
