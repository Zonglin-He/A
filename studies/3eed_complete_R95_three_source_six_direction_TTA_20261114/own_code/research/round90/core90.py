"""Joint live fusion and candidate-conditioned binding; frozen model stays differentiable."""
from pathlib import Path
import sys,contextlib,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round89'))
from core89 import *
H90=ROOT/'research/round90';B90=ROOT/'crossplatform/round90';BRANCHES90=['continue_joint','query_joint','candidate_headonly','candidate_joint']

class Interface90(CandidateInterface):
 def forward(self,h,local,own,text,reltext,route,variant='full'):
  Q=F.layer_norm(self.q(h.float()),(64,));E=len(local)
  if E==0 or variant=='query_only':
   delta=self.out(torch.cat([Q,Q*0,Q*0,Q*0,h.new_zeros((256,9))],-1)).squeeze(-1);return delta,{'conditional':F.normalize(Q,dim=-1)[:,None],'conditional_valid':torch.ones((256,1),device=h.device,dtype=torch.bool),'subject_weights':Q.new_zeros((256,0))}
  Z=F.layer_norm(self.z(local.float()),(64,));O=F.layer_norm(self.z(own.float()),(64,));T=F.layer_norm(self.t(text.float()),(64,));R=F.layer_norm(self.t(reltext.float()),(64,))*route['relation_tokens'].any(-1)[:,None];valid=route['weights_plain']>0;geo=route['hyp_geo'];own_geo=route['own_geo'];own_valid=route['own_valid'];shared=variant=='shared_mix'
  options=torch.cat([O[:,None],Z[0][None].expand(256,-1,-1)],1);geos=torch.cat([own_geo[:,None],geo[:,0]],1);vm=torch.cat([own_valid[:,None],valid[0][None].expand(256,-1)],1)
  if shared:vm=vm.clone();vm[:,0]=False
  def read(q,k,g,v):
   logits=(q[:,None]*k).sum(-1)/8
   if not shared:logits=logits+self.geo(g).squeeze(-1)
   a=logits.masked_fill(~v,-1e4).softmax(-1)*v;a=a/a.sum(-1,keepdim=True).clamp_min(1e-12);return (a[:,:,None]*k).sum(1),(a[:,:,None]*g).sum(1),a
  qs=T[0][None].expand(256,-1)+(Q if not shared else 0);sub,subgeo,sw=read(qs,options,geos,vm);refs=[];gg=[];rv=[]
  if variant!='no_ref':
   for e in range(1,E):
    query=T[e][None]+R[e][None]+(Q+sub if not shared else Q*0);v=valid[e][None].expand(256,-1);z=Z[e][None].expand(256,-1,-1);rr,rg,_=read(query,z,geo[:,e],v);refs.append(rr);gg.append(rg);rv.append(v.any(-1))
  ref=torch.stack(refs).mean(0) if refs else Q*0;rel=R[1:].mean(0) if E>1 and variant!='no_ref' else T[0]*0;g=torch.stack(gg).mean(0) if gg else subgeo*0;inp=torch.cat([Q,sub,ref,T[0][None].expand(256,-1)+rel[None],g],-1);delta=self.out(inp).squeeze(-1)
  vals=[Q,sub]+refs+[O];masks=[torch.ones(256,device=h.device,dtype=torch.bool),vm.any(-1)]+rv+[own_valid];conditional=F.normalize(torch.stack(vals,1),dim=-1);return delta,{'conditional':conditional,'conditional_valid':torch.stack(masks,1),'subject_weights':sw}

class Fusion90(Fusion89):
 def __init__(self,p):super().__init__(p);self.interface=Interface90();self.pool_mode='strict';self.branch='candidate_joint'
 @contextlib.contextmanager
 def capture(self,base):
  self.local={};hook=None
  if self.route is not None:
   def record(module,args,kw,out):
    rt=self.route;query=kw['query'][:,0];key=kw['key'][:,0];value=kw['value'][:,0];D=module.embed_dim;heads=module.num_heads;hd=D//heads;W=module.in_proj_weight;bb=module.in_proj_bias;Q=F.linear(query,W[:D],None if bb is None else bb[:D]).reshape(-1,heads,hd).transpose(0,1);K=F.linear(key,W[D:2*D],None if bb is None else bb[D:2*D]).reshape(-1,heads,hd).transpose(0,1);V=F.linear(value,W[2*D:],None if bb is None else bb[2*D:]).reshape(-1,heads,hd).transpose(0,1);A=(Q@K.transpose(-1,-2)/math.sqrt(hd)).softmax(-1);sup=rt['supports'];own=rt['own_support'];tm=rt['token_roles'];texts=[];reltexts=[];reads=[];own_z=key.new_zeros((256,288))
    if self.variant=='wrong_space':sup=sup[:,:,rt['point_perm']];own=own[:,rt['point_perm']]
    elif self.variant=='role_swap' and len(sup)>1:sup=sup.roll(1,0)
    def pool(att,mask):
     support=mask[None] if self.pool_mode=='strict' else .05+.95*mask[None];weights=att[:,None]*support;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12);v=(weights@V).transpose(0,1).reshape(len(mask),D);out=module.out_proj(v)
     if self.pool_mode=='strict':out=out*mask.any(-1)[:,None]
     return out
    for e in range(len(tm)):
     mk=tm[e];texts.append(query[mk].mean(0) if mk.any() else query.sum(0)*0);rel=rt['relation_tokens'][e];reltexts.append(query[rel].mean(0) if rel.any() else query.sum(0)*0);att=A[:,mk].mean(1) if mk.any() else A.mean(1)*0+1/len(key);reads.append(pool(att,sup[e]));
     if e==0:own_z=pool(att,own)
    self.local={'local':torch.stack(reads) if reads else key.new_zeros((0,3,288)),'own':own_z,'text':torch.stack(texts) if texts else key.new_zeros((0,288)),'relation_text':torch.stack(reltexts) if reltexts else key.new_zeros((0,288))}
    return None
   hook=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(record,with_kwargs=True)
  try:yield
  finally:
   if hook is not None:hook.remove()
 def main(self,base,z,d,mode,*args,**kw):
  with self.capture(base):p,q,e=Fusion79.main(self,base,z,d,mode,*args,**kw)
  e.update(self.local);e['base_probability']=p
  if self.local:
   valid=self.route['weights_plain']>0;e['local_reads']=F.normalize(self.local['local'][valid].float(),dim=-1)
  else:e['local_reads']=q.new_zeros((0,288))
  if self.use_interface:
   delta,cond=self.interface(e['query_hidden'],e['local'],e['own'],e['text'],e['relation_text'],self.route,self.variant);p=(p.clamp_min(1e-300).log()+delta.double()).softmax(0);e.update(cond);e['residual']=delta
  else:e.update(conditional=q[:,None],conditional_valid=torch.ones((256,1),device=q.device,dtype=torch.bool),subject_weights=q.new_zeros((256,0)))
  return p,q,e
 def theta_parameters(self):return Fusion.tta_parameters(self)
 def source_parameters(self):
  own=list(self.match.named_parameters()) if self.branch=='continue_joint' else list(self.interface.named_parameters());prefix='match.' if self.branch=='continue_joint' else 'interface.';return ([] if self.branch=='candidate_headonly' else self.theta_parameters())+[(prefix+n,p) for n,p in own]
 def tta_parameters(self):
  if self.branch=='continue_joint':extra=[('match.'+n,p) for n,p in self.match.named_parameters()]
  elif self.branch=='query_joint':extra=[('interface.q.'+n,p) for n,p in self.interface.q.named_parameters()]
  else:extra=[('interface.'+n,p) for n,p in self.interface.named_parameters() if not n.startswith('out.')]
  return self.theta_parameters()+extra

def configure90(m,branch,variant='full'):
 m.branch=branch;m.use_interface=branch!='continue_joint';m.variant='query_only' if branch=='query_joint' else variant;m.pool_mode='strict'
def load_R76_90(m):return load_R76(m)
def enable_source90(m):
 m.requires_grad_(False)
 for _,p in m.source_parameters():p.requires_grad_(True)
def enable_tta90(m):
 m.requires_grad_(False)
 for _,p in m.tta_parameters():p.requires_grad_(True)
def data90(r,source=False):
 z0=prefix(r);z=tensor_prefix(z0);d=tensor_data(load(casepath(r)),r);folder=B89/'source_features640_attempt001/routes_PRIVATE' if source else B89/'target_evidence336_attempt001/samples_PRIVATE';rt=load(folder/(r['id']+'.npz'));route=tensor_route89(rt);region={'point':rt['region_point'],'image':rt['region_image']};ab=[mask89(r,d,region,k) for k in ([89001,89002] if source else [88001,88002])];pm=[a for a,b in ab];im=[b for a,b in ab];ws=local_weights89(d,pm,im,region);return z0,z,d,rt,route,pm,im,ws

def conditional_loss90(a,b,valid,perm=None):
 if perm is not None:b=b[perm];valid=valid&valid[perm]
 v=(a-b).square().sum(-1)/2;return (v*valid).sum()/valid.sum().clamp_min(1)
def source_slot90(e,y,rt):
 sw=e['subject_weights']
 if not sw.numel() or not len(rt['hypotheses_PRIVATE']):return e['conditional'].sum()*0
 ids=rt['hypotheses_PRIVATE'][0];yy=y.new_zeros(3);known=ids>=0
 if known.any():yy[torch.tensor(known,device=y.device)]=y[torch.tensor(ids[known],device=y.device)]
 compatible=torch.cat([(y>.25)[:,None],(yy>.25)[None].expand(256,-1)],1);vm=torch.cat([torch.as_tensor(rt['own_valid'],device=y.device)[:,None],torch.as_tensor(rt['weights_plain'][0]>0,device=y.device)[None].expand(256,-1)],1);pos=(y>.25)&(compatible&vm).any(-1);mass=(sw*compatible*vm).sum(-1);w=y*pos;return -(w*mass.clamp_min(1e-12).log()).sum()/w.sum().clamp_min(1e-12)
def source_loss90(m,base,z,d,rt,pm,im,ws,y):
 p,q,e=m.main(base,z,d,'bidirectional');target=e['conditional'].detach();valid=e['conditional_valid'];ground=ground89(p,y);corr=p.sum()*0;slot=source_slot90(e,y,rt) if m.branch.startswith('candidate') else p.sum()*0;masked=[]
 for a,b in zip(pm,im):
  pp,_,ee=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b);masked.append(ground89(pp,y));corr=corr+conditional_loss90(ee['conditional'],target,valid&ee['conditional_valid'])/2
  if m.branch.startswith('candidate'):slot=slot+source_slot90(ee,y,rt)/2
 masked=torch.stack(masked).mean();phys=physical89(m,z,d,pm,im,ws);loss=ground+masked+phys.double()+.1*corr+.1*slot;return loss,{'ground_full':ground,'ground_masked':masked,'physical':phys,'conditional':corr,'target_slot_proxy':slot},p

def objective90(m,base,z,d,refp,refq,refcond,refvalid,reflocal,pm,im,ws,kind='full',perm=None):
 p,q,e=m.main(base,z,d,'bidirectional');keep=(refp*(refp.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum()+.1*conditional_loss90(e['conditional'],refcond,refvalid&e['conditional_valid']);corr=p.sum()*0
 for a,b in zip(pm,im):
  _,_,ee=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b)
  if kind=='bypass':
   if ee['local_reads'].numel():corr=corr+(ee['local_reads']-reflocal).square().sum(-1).mean()/4
  else:corr=corr+conditional_loss90(ee['conditional'],refcond,refvalid&ee['conditional_valid'],perm if kind=='shuffle' else None)/2
 phys=physical89(m,z,d,pm,im,ws);c=.1*corr*(0 if kind=='physical' else 1);return phys.double()+keep+c,p,{'physical':phys,'keep':keep,'conditional':c}

def flat90(grad,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for g,p in zip(grad,ps)])
def apply90(m,old,delta):
 off=0
 with torch.no_grad():
  for (_,p),v in zip(m.tta_parameters(),old):n=p.numel();p.copy_(v+delta[off:off+n].reshape_as(p));off+=n
 assert off==len(delta)
