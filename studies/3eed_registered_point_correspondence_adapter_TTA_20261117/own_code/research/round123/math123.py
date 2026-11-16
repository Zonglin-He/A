"""Registered-point contrastive adapter adaptation; no task-label/teacher/expert target."""
import torch,numpy as np,hashlib
import torch.nn.functional as F
NAMES=['point_adapter.up.weight','point_adapter.up.bias'];SETTINGS=['corr1','corr3','wrong1','wrong3','random1','random3','free3']
def geometry123(xyz,valid,salt,device='cuda'):
 ids=np.flatnonzero(valid);xx=xyz[ids].astype(float);dist2=((xx[:,None]-xx[None])**2).sum(-1);far=dist2>3.2**2;n=len(ids);order=np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16)).permutation(n);perm=np.empty(n,int)
 if n:perm[order]=np.roll(order,1)
 return torch.as_tensor(ids,device=device),torch.as_tensor(far,device=device),torch.as_tensor(perm,device=device)
def parts123(A,B,RA,RB,ids,far,perm=None):
 qa=F.normalize(A,dim=-1);qb=F.normalize(B,dim=-1);q=qa[ids];k=qb[ids];k=k[perm] if perm is not None else k
 if len(ids)>=2 and bool(far.any()):
  s=q@k.T/.07;allow=far|torch.eye(len(ids),device=ids.device,dtype=torch.bool);left=far.any(1);right=far.any(0);l=(s.masked_fill(~allow,-torch.inf).logsumexp(1)-s.diag())[left].mean();r=(s.T.masked_fill(~allow.T,-torch.inf).logsumexp(1)-s.diag())[right].mean();corr=(l+r)/2
 else:corr=A.sum()*0+B.sum()*0
 keep=((qa-RA).square().sum(-1).mean()+(qb-RB).square().sum(-1).mean())/4
 return corr+keep,corr,keep

def flat123(m):return torch.cat([m.state_dict()[k].flatten() for k in NAMES])
def put123(m,S0,delta):
 off=0
 with torch.no_grad():
  for name,p in m.named_parameters():
   if name in NAMES:
    k=p.numel();p.copy_(torch.as_tensor(S0[name],device=p.device)+delta[off:off+k].reshape_as(p));off+=k
 assert off==4896

def fit123(m,S0,PA,PB,ids,far,perm,radius,salt):
 for p in m.parameters():p.requires_grad_(False)
 ps=[dict(m.named_parameters())[k] for k in NAMES];assert sum(p.numel() for p in ps)==4896;states={};traces={};RA=F.normalize(PA,dim=-1).detach();RB=F.normalize(PB,dim=-1).detach();energies=[]
 for kind in ['corr','wrong']:
  delta=torch.zeros(4896,device=PA.device);put123(m,S0,delta);steps=[];prevA,prevB=PA,PB
  for t in range(3):
   for p in ps:p.requires_grad_(True)
   A=m.point_adapter(PA);B=m.point_adapter(PB);L,c,k=parts123(A,B,RA,RB,ids,far,perm if kind=='wrong' else None);gg=torch.autograd.grad(L,ps,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(ps,gg)]);assert torch.isfinite(g).all();step=-radius*g/g.norm() if g.norm()>0 and len(ids)>=2 and far.any() else g*0;delta=delta+step;put123(m,S0,delta)
   for p in ps:p.requires_grad_(False)
   with torch.no_grad():
    A=m.point_adapter(PA);B=m.point_adapter(PB);after,ca,ka=parts123(A,B,RA,RB,ids,far,perm if kind=='wrong' else None);energy=float(torch.sqrt((A-prevA).double().square().sum()+(B-prevB).double().square().sum()));prevA,prevB=A.clone(),B.clone();steps.append({'loss_before':float(L.detach()),'loss_after':float(after),'contrastive_after':float(ca),'keep_after':float(ka),'gradient_L2':float(g.norm()),'step_L2':float(step.norm()),'cumulative_L2':float(delta.norm()),'induced_feature_step_L2':energy})
    if kind=='corr':energies.append(energy)
   if t in [0,2]:states[kind+str(t+1)]=delta.clone()
  traces[kind]=steps
 for t in [1,3]:
  g=torch.as_tensor(np.random.default_rng(int(hashlib.sha256((salt+'|random|'+str(t)).encode()).hexdigest()[:16],16)).normal(size=4896),device=PA.device,dtype=PA.dtype);states['random'+str(t)]=g/g.norm()*states['corr'+str(t)].norm()
 # A stronger same-information no-network-update control: optimize A/B feature arrays
 # with the same loss and match each step's actual adapter-induced feature energy.
 A=PA.detach().clone();B=PB.detach().clone();steps=[]
 for t,energy in enumerate(energies):
  A.requires_grad_(True);B.requires_grad_(True);L,c,k=parts123(A,B,RA,RB,ids,far);ga,gb=torch.autograd.grad(L,[A,B]);norm=torch.sqrt(ga.double().square().sum()+gb.double().square().sum());factor=energy/float(norm) if norm>0 else 0.;A=(A-factor*ga).detach();B=(B-factor*gb).detach();steps.append({'loss_before':float(L.detach()),'matched_feature_step_L2':energy,'feature_gradient_L2':float(norm)})
 common=((A-PA)+(B-PB))/2;mask=torch.zeros(len(PA),device=PA.device,dtype=torch.bool);mask[ids]=True;common=common.masked_fill(~mask[:,None],0);states['free3']=common;traces['free']=steps;put123(m,S0,torch.zeros(4896,device=PA.device));return states,traces,{'no_C_decoded_before_all_states_fixed':True,'raw_feature_source_preserve_weight':1.,'parameter_radius':radius,'fit_point_pairs':len(ids),'directed_geometric_negatives':int(far.sum()),'active':bool(len(ids)>=2 and far.any())}
