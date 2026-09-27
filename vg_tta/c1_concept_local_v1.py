"""Original-coordinate Adam, projection after step; frozen complete-query readout."""
import math
import torch
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.objectives import SpatialLoss
QUERY='spatial.query_residual'

@torch.no_grad()
def expression(model,batch,state,text,subject):
 from methods.decota_final_simplified_v1.backbone import inserted_state,query_subject,offset_batch
 b=dict(batch);b['texts']=[text];captured=[];boxes=[]
 with query_subject(model,b,subject),inserted_state(model,state):
  hook=model.ground_decoder.decoder.register_forward_pre_hook(lambda m,a,k:captured.append(detached(k['query_tgt'])),with_kwargs=True)
  try:
   for off in [0,1]:
    v=offset_batch(b,off)
    with torch.autocast('cuda',dtype=torch.float16):out=model(v['videos'],v['texts'],v['targets'],iteration_rate=-1)
    boxes.append(out['pred_boxes'].detach())
  finally:hook.remove()
 assert len(captured)==4
 qq=[captured[i] for i in [1,3]]
 assert all(torch.equal(q,q[:1].expand_as(q)) for q in qq)
 z=torch.stack([q[0,0].float() for q in qq])
 n=sum(len(v) for v in boxes)
 return dict(z=z.mean(0).cpu(),offset_z=z.cpu(),boxes=torch.stack([boxes[i%2][i//2] for i in range(n)]).float().cpu())

def basis(z,z0):
 if not z:return torch.zeros(256,0),torch.zeros(0)
 d=torch.stack([v-z0 for v in z],1).double()
 u,s,_=torch.linalg.svd(d,full_matrices=False)
 rank=min(8,int((s>max(1e-10,float(s[0])*1e-5)).sum()))
 return u[:,:rank].float(),s.float()

def diagnostics(path,initial,scale,radius):
 for v in path:
  q=v['state'][QUERY];ln=torch.cat([(t-initial[n].cpu()).flatten() for n,t in v['state'].items() if n!=QUERY])
  v.update(query_norm=float(q.norm()),query_relative_norm=float(q.norm())/scale,
           LN_displacement=float(ln.norm()),boundary=radius is not None and float(q.norm())>=radius*(1-1e-5))

@torch.enable_grad()
def fit(replay,before,anchors,cfg,U,rho,mult,scale):
 replay.restore(replay.initial);params=dict(replay.named);q=params[QUERY]
 opt=torch.optim.Adam([dict(params=[q],lr=cfg.spatial_lr*mult),dict(params=[v for n,v in replay.named if n!=QUERY],lr=cfg.spatial_lr)],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 lossfn=SpatialLoss(anchors,before['boxes']);path=[];best=math.inf;choice=0;radius=rho*scale
 try:
  for step in range((0 if lossfn.empty else 10)+1):
   opt.zero_grad(set_to_none=True);v=replay.values();loss=lossfn(v['boxes']);lv=float(loss.detach());assert math.isfinite(lv)
   path.append(dict(step=step,loss=lv,state=detached(replay.state(),'cpu'),boxes=detached(v['boxes'],'cpu')))
   if lv<best:best=lv;choice=step
   if lossfn.empty or step==10:break
   loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
   opt.step()
   with torch.no_grad():
    if U is not None:q.copy_(U@(U.T@q))
    q.mul_(min(1.,radius/max(float(q.norm()),1e-20)))
  diagnostics(path,replay.initial,scale,radius)
  return dict(path=path,selected_step=choice,final=dict(boxes=path[choice]['boxes']),state=path[choice]['state'],initial_state=detached(replay.initial,'cpu'),backwards=len(path)-1,skipped=lossfn.empty,failure=None,rho=rho,multiplier=mult,radius=radius)
 finally:replay.restore(replay.initial)
