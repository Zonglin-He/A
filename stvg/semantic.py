"""Same frozen Source word head on existing full-time target trajectory, context1.5."""
from contextlib import contextmanager
import torch,numpy as np

def footprint(box,h,w,valid):
 box=box.double();yy=torch.arange(h,dtype=torch.float64,device=box.device)/h;xx=torch.arange(w,dtype=torch.float64,device=box.device)/w
 x0,y0=box[:2]-box[2:]/2;x1,y1=box[:2]+box[2:]/2
 dx=(torch.minimum(xx+1/w,x1)-torch.maximum(xx,x0)).clamp_min(0)*w;dy=(torch.minimum(yy+1/h,y1)-torch.maximum(yy,y0)).clamp_min(0)*h
 return (dy[:,None]*dx[None,:]).clamp(0,1)*valid

def keep_region(box,valid):
 b=box.clone().double();b[2:]*=1.5;keep=(footprint(b,*valid.shape,valid)>0)&valid
 fallback=not bool(keep.any())
 if fallback:keep=valid.clone()
 assert bool(keep.any())and not bool((keep&~valid).any())
 return keep,fallback

def restrict_prob(prob,keep):
 assert prob.ndim==4 and keep.ndim==2 and prob.shape[0]==keep.shape[0]and prob.shape[-1]==keep.shape[-1];assert bool(keep.any(1).all())
 if bool(keep.all()):return prob
 x=prob.masked_fill(~keep[:,None,None,:],0);z=x.sum(-1,keepdim=True);assert bool((z>0).all()),'No numerical zero-support fallback allowed';return x/z

@contextmanager
def support_keys(head,keep):
 hooks=[];calls=[]
 def hook(module,args):
  p=args[0];calls.append(dict(keep_mass=p.mul(keep[:,None,None,:]).sum(-1).detach().cpu()))
  if bool(keep.all()):return None
  return (restrict_prob(p,keep),)
 try:
  for layer in head.layer_ca:hooks.append(layer.attention.self.dropout.register_forward_pre_hook(hook))
  yield calls
 finally:
  for h in hooks:h.remove()
 assert len(calls)==len(head.layer_ca)

def masks(aux,offset,n,h,w,valid,kind):
 result=[];stats=[]
 for j in range(n):
  index=offset+2*j;m,fb=keep_region(aux[index],valid[j]);result.append(m);stats.append(dict(frame_index=index,selected_cells=int(m.sum()),valid_cells=int(valid[j].sum()),empty_ROI_fallback=fb))
 return torch.stack(result).flatten(1),stats

def target_semantics(model,cap,aux):
 kind="TargetContext"
 import torch,numpy as np
 from stvg.visual_cache import transfer
 from contextlib import nullcontext
 obs=[];supportstats=[];mass=[]
 with torch.no_grad(),torch.autocast('cuda',enabled=False):
  for off in cap.get('temporal_cache',cap.get('native_hidden'))['offsets']:
   M=transfer(off['inputs']['encoded_feature'],'cuda');h,w=off['grid_size'];n=M.shape[1];assert not off['inputs']['encoded_mask'].any(),'Actualcohortunmasked; noborrowedmaskorderingassumption';count=h*w;fm=M[-count:].permute(1,2,0).reshape(n,256,h,w).detach();ft=M[:-count].mean(1).unsqueeze(0);inputs=[];outputs=[];head=model.t_spatial_clas;handles=[head.head.register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach().cpu().clone())),head.head.register_forward_hook(lambda m,a,o:outputs.append(o.detach().cpu().clone()))]
   if kind in ['Global','AllKeysNoOp']:keep=torch.ones(n,count,dtype=torch.bool);stats=[]
   else:keep,stats=masks(aux,off['offset'],n,h,w,torch.ones(n,h,w,dtype=torch.bool),kind)
   try:
    with (nullcontext([])if kind=='Global'else support_keys(head,keep.cuda()))as calls:head(fm,ft[:,:1])
   finally:
    for handle in handles:handle.remove()
   if kind!='Global':assert len(calls)==2;mass.extend(float(x['keep_mass'].mean())for x in calls)
   assert len(inputs)==len(outputs)==1;obs.append(dict(z=outputs[0][:,0].numpy(),u=inputs[0][:,0].numpy()));supportstats.extend(stats)
 n=len(cap['frame_ids']);z=np.stack([obs[j%2]['z'][j//2]for j in range(n)]).astype(np.float64);prob=np.clip(1/(1+np.exp(-z)),1e-6,1-1e-6);u=np.stack([obs[j%2]['u'][j//2]for j in range(n)]);return prob,u,supportstats,mass
