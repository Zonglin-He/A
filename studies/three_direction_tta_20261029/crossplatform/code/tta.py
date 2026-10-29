"""Episodic, label-free LN adaptation for 3EED. No GT reaches this module."""
import re,copy
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
LEXICON=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')
def text_span(tokenizer,text):
 matches=[m for m in re.finditer(r'\b(?:'+ '|'.join(LEXICON)+r')\b',text,re.I)]
 # Text-only first object noun; never supplied target class or bbox.
 chosen=matches[0] if matches else None
 tok=tokenizer(text,return_offsets_mapping=True,return_tensors='pt')
 off=tok['offset_mapping'][0]
 if chosen:mask=(off[:,1]>chosen.start())&(off[:,0]<chosen.end())
 else:mask=(off[:,1]>off[:,0])
 return mask, chosen.group() if chosen else 'all_text_fallback'
def boxes(o):return torch.cat([o['last_center'],o['last_pred_size']],-1)[0]
def scores(o,mask):
 sim=(o['last_proj_queries']@o['proj_tokens'].transpose(-1,-2))/0.07
 p=sim.softmax(-1)[0]
 mask=mask[:p.shape[-1]].to(p.device)
 return (p*mask[None]).sum(-1)
def iou(a,b):
 lo=torch.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2);hi=torch.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)
 inter=(hi-lo).clamp(min=0).prod();return inter/(a[3:].prod()+b[3:].prod()-inter+1e-8)
def match(a,b):
 scale=a[:,3:].median(0).values.clamp(min=0.5)
 c=torch.cdist(a[:,:3]/scale,b[:,:3]/scale)+torch.cdist(a[:,3:]/scale,b[:,3:]/scale)
 rows,cols=linear_sum_assignment(c.detach().cpu().numpy());return torch.as_tensor(cols,device=a.device)
class Adapter:
 def __init__(self,model,lr=1e-4):
  self.model=model;self.lr=lr
  for p in model.parameters():p.requires_grad_(False)
  self.params=[];self.names=[]
  for name,module in model.named_modules():
   if isinstance(module,torch.nn.LayerNorm) and name.startswith('cross_encoder.'):
    for key,p in module.named_parameters(recurse=False):p.requires_grad_(True);self.params.append(p);self.names.append(name+'.'+key)
  self.initial=[p.detach().clone() for p in self.params]
 def reset(self):
  with torch.no_grad():
   for p,q in zip(self.params,self.initial):p.copy_(q)
  self.model.eval()
 def run(self,inputs,mask,steps=3,mode='consistency',drop=.1,seed=0):
  torch_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
  self.reset();m=self.model
  pc=inputs['point_clouds'];rng=np.random.default_rng(seed)
  n=pc.shape[1]
  unique=np.unique(pc[0].detach().cpu().numpy(),axis=0,return_index=True)[1]
  keep=rng.choice(unique,max(1,int(len(unique)*(1-drop))),replace=False)
  indices=np.concatenate([keep,rng.choice(keep,n-len(keep),replace=True)])
  sparse={**inputs,'point_clouds':pc[:,indices].contiguous()}
  original=m._run_backbones
  with torch.no_grad():
   cache=original(inputs);cache_s=original(sparse)
  m._run_backbones=lambda _:dict(cache)
  try:
   with torch.no_grad():base=m(inputs)
   m._run_backbones=lambda _:dict(cache_s)
   with torch.no_grad():teacher=m(sparse)
   m._run_backbones=lambda _:dict(cache)
   a,b=boxes(base),boxes(teacher);align=match(a,b)
   sb,st=scores(base,mask),scores(teacher,mask)[align]
   agreement=float(iou(a[sb.argmax()],b[scores(teacher,mask).argmax()]))
   # Gate is fixed before scores/labels are read. No online GT selection.
   accept=agreement>=.25
   target=((sb+st)/2/.1).softmax(-1).detach()
   initial_top=int(sb.argmax());out={0:base};trace=[]
   opt=torch.optim.Adam(self.params,lr=self.lr)
   for step in range(1,steps+1):
    pred=m(inputs)
    student_alignment=match(a,boxes(pred))
    s=scores(pred,mask)[student_alignment];p=(s/.1).softmax(-1)
    entropy=-(p*p.clamp_min(1e-8).log()).sum()
    kl=torch.nn.functional.kl_div((s/.1).log_softmax(-1),target,reduction='sum')
    loss=entropy if mode=='entropy' else kl
    update=(mode!='gated' or accept)
    if update:
     opt.zero_grad(set_to_none=True);loss.backward();nonzero_grad=sum(int(p.grad is not None and torch.count_nonzero(p.grad).item()>0) for p in self.params);grad_norm=float(torch.nn.utils.clip_grad_norm_(self.params,1.));opt.step()
    with torch.no_grad():out[step]=m(inputs)
    trace.append({'step':step,'student_slot_rematches':int(torch.count_nonzero(student_alignment-torch.arange(len(student_alignment),device=student_alignment.device))),'loss':float(loss.detach()),'entropy':float(entropy.detach()),'updated':update,'nonzero_grad_params':nonzero_grad if update else 0,'grad_norm':grad_norm if update else 0.0,'parameter_delta_l2':float(sum(((p.detach()-q)**2).sum() for p,q in zip(self.params,self.initial)).sqrt())})
   return out,teacher,{'original_unique_points':len(unique),'sparse_unique_points':len(keep),'agreement':agreement,'gate_accept':accept,'initial_top':initial_top,'trace':trace,'param_names':self.names,'param_count':sum(p.numel() for p in self.params)}
  finally:
   m._run_backbones=original
   self.reset()
   for p in self.params:p.grad=None
   torch.set_rng_state(torch_rng)
   if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
