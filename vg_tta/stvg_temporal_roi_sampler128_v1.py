"""Fixed fractional-area pooling before an existing source-trained classifier."""
import torch
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap

def pools(memory,valid,boxes,h,w):
 assert memory.shape[0]==h*w and valid.shape==(memory.shape[1],h*w)
 fmap=memory.permute(1,2,0).reshape(memory.shape[1],memory.shape[2],h,w)
 wg=valid.to(memory.dtype);wg=wg/wg.sum(-1,keepdim=True)
 global_pool=(fmap.flatten(2)*wg[:,None]).sum(-1)[...,None,None]
 weights=[];empty=[]
 for b,v,g in zip(boxes,valid,wg):
  vm=v.reshape(h,w);vh=int(vm.any(1).sum());vw=int(vm.any(0).sum());assert vm[:vh,:vw].all()
  x=torch.zeros(h,w,dtype=torch.float64,device=b.device);x[:vh,:vw]=cell_overlap(b[None],vh,vw)[0];x=x.flatten().to(memory.dtype)*v
  missing=bool(x.sum()<=0);x=g if missing else x/x.sum();weights.append(x);empty.append(missing)
 weights=torch.stack(weights);assert torch.allclose(weights.sum(-1),torch.ones(len(boxes),device=memory.device),atol=2e-6)
 roi=(fmap.flatten(2)*weights[:,None]).sum(-1)[...,None,None]
 return fmap,global_pool,roi,dict(empty=empty,weights=weights.detach().cpu(),valid_weights=wg.detach().cpu())

def properties():
 m=torch.arange(6*2*4,dtype=torch.float32).reshape(6,2,4);v=torch.ones(2,6,dtype=torch.bool);b=torch.tensor([[.5,.5,1.,1.],[2.,2.,.1,.1]],dtype=torch.float64)
 f,g,r,a=pools(m,v,b,2,3);assert torch.equal(g,r) and a['empty']==[False,True]
 b[0]=torch.tensor([.1,.4,.01,.01]);_,_,r,a=pools(m,v,b,2,3);assert not a['empty'][0] and not torch.equal(r[0],g[0])
 return dict(status='pass',full_image_equals_valid_global=True,outside_falls_back=True,subcell_target_nonempty=True)
