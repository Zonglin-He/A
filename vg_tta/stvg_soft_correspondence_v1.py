"""Frozen image features to soft box-region evidence; no GT/model/data readers."""
import torch
import torch.nn.functional as F

def cell_overlap(boxes,h,w):
 """Fraction of each unit grid cell inside original normalized cxcywh boxes."""
 b=boxes.double();lo=b[...,:2]-b[...,2:]/2;hi=b[...,:2]+b[...,2:]/2;x=torch.arange(w,device=b.device,dtype=b.dtype);y=torch.arange(h,device=b.device,dtype=b.dtype)
 dx=(torch.minimum(hi[...,0,None]*w,x+1)-torch.maximum(lo[...,0,None]*w,x)).clamp(0,1);dy=(torch.minimum(hi[...,1,None]*h,y+1)-torch.maximum(lo[...,1,None]*h,y)).clamp(0,1)
 return dy[..., :,None]*dx[...,None,:]

def soft_iou(boxes,maps):
 h,w=maps.shape[-2:];area=cell_overlap(boxes,h,w);m=maps.double();inter=(area*m).sum((-1,-2));boxarea=boxes[...,2].double()*boxes[...,3].double()*h*w;union=boxarea+m.sum((-1,-2))-inter
 return torch.where(union>0,inter/union,torch.zeros_like(inter))

@torch.no_grad()
def propagate(maps,valid,anchor_positions,anchor_boxes,temperature=.07):
 assert maps.ndim==4 and valid.shape==(maps.shape[0],*maps.shape[-2:]);N,C,H,W=maps.shape;f=F.normalize(maps.float().flatten(2).transpose(1,2),dim=-1);outputs=[]
 for a,box in zip(anchor_positions,anchor_boxes):
  mask=cell_overlap(box.to(maps)[None],H,W)[0].float().flatten()*valid[a].flatten();logits=(f@f[a].T)/temperature;logits[...,~valid[a].flatten()]=-torch.inf;A=torch.softmax(logits,-1);m=(A@mask).reshape(N,H,W)*valid
  assert torch.isfinite(m).all() and m.min()>=0 and m.max()<=1+1e-6;outputs.append(m.cpu())
 return torch.stack(outputs)if outputs else torch.empty(0,N,H,W)

def contract():
 # Exact full-cell region IoU, exterior area, and a one-hot identity transfer.
 b=torch.tensor([[.5,.5,1.,1.],[.25,.25,.5,.5],[1.5,.5,1.,1.]])
 m=torch.zeros(3,2,2);m[:,0,0]=1;r=soft_iou(b,m);assert torch.allclose(r,torch.tensor([.25,1.,0.],dtype=torch.float64))
 f=torch.eye(4).T.reshape(1,4,2,2);v=torch.ones(1,2,2,dtype=torch.bool);p=propagate(f,v,[0],[b[1]],temperature=.01);assert torch.allclose(p[0,0],m[0],atol=1e-6)
 assert soft_iou(b[1:2],p[0]).item()>1-1e-6
 return dict(status='pass',soft_IoU_exact_cases=3,identity_correspondence=True,outside_image_area_retained=True)
