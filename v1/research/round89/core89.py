"""Source-prepared candidate-conditioned reads. Import reads no target inputs."""
from pathlib import Path
import sys,contextlib,math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round79'))
from core79 import *
sys.path.insert(0,str(ROOT/'research/round88'));from routing88 import token_roles88,build88
H89=ROOT/'research/round89';B89=ROOT/'crossplatform/round89';R76_INIT=ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz'

class CandidateInterface(nn.Module):
 def __init__(self):
  super().__init__();self.q=nn.Linear(288,64);self.z=nn.Linear(288,64);self.t=nn.Linear(288,64);self.geo=nn.Sequential(nn.Linear(9,32),nn.GELU(),nn.Linear(32,1));self.out=nn.Sequential(nn.Linear(265,64),nn.GELU(),nn.Linear(64,1));nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
 def forward(self,h,local,own,text,reltext,route,variant='full'):
  E=local.shape[0];Q=F.layer_norm(self.q(h.float()),(64,))
  if E==0 or variant=='query_only':return self.out(torch.cat([Q,Q*0,Q*0,Q*0,h.new_zeros((256,9))],-1)).squeeze(-1)
  Z=F.layer_norm(self.z(local.float()),(64,));O=F.layer_norm(self.z(own.float()),(64,));T=F.layer_norm(self.t(text.float()),(64,));R=F.layer_norm(self.t(reltext.float()),(64,))*route['relation_tokens'].any(-1)[:,None];valid=route['weights_plain']>0;geo=route['hyp_geo'];own_geo=route['own_geo'];own_valid=route['own_valid'];shared=variant=='shared_mix'
  options=torch.cat([O[:,None],Z[0][None].expand(256,-1,-1)],1);geos=torch.cat([own_geo[:,None],geo[:,0]],1);vm=torch.cat([own_valid[:,None],valid[0][None].expand(256,-1)],1)
  if shared:vm=vm.clone();vm[:,0]=False
  def read(q,k,g,v):
   logits=(q[:,None]*k).sum(-1)/8
   if not shared:logits=logits+self.geo(g).squeeze(-1)
   a=logits.masked_fill(~v,-1e4).softmax(-1)*v;a=a/a.sum(-1,keepdim=True).clamp_min(1e-12);return (a[:,:,None]*k).sum(1),(a[:,:,None]*g).sum(1)
  qs=T[0][None].expand(256,-1)+(Q if not shared else 0);sub,subgeo=read(qs,options,geos,vm);refs=[];gg=[]
  if variant!='no_ref':
   for e in range(1,E):
    query=T[e][None]+R[e][None]+(Q+sub if not shared else Q*0);v=valid[e][None].expand(256,-1);z=Z[e][None].expand(256,-1,-1);rr,rg=read(query,z,geo[:,e],v);refs.append(rr);gg.append(rg)
  ref=torch.stack(refs).mean(0) if refs else Q*0;rel=R[1:].mean(0) if E>1 and variant!='no_ref' else T[0]*0;g=torch.stack(gg).mean(0) if gg else subgeo*0;inp=torch.cat([Q,sub,ref,T[0][None].expand(256,-1)+rel[None],g],-1);return self.out(inp).squeeze(-1)

class Fusion89(Fusion79):
 def __init__(self,p):super().__init__(p);self.interface=CandidateInterface();self.route=None;self.variant='full';self.use_interface=True;self.local={}
 @contextlib.contextmanager
 def capture(self,base):
  self.local={};hook=None
  if self.route is not None:
   def record(module,args,kw,out):
    rt=self.route;query=kw['query'][:,0];key=kw['key'][:,0];value=kw['value'][:,0];D=module.embed_dim;heads=module.num_heads;hd=D//heads;W=module.in_proj_weight;bb=module.in_proj_bias;Q=F.linear(query,W[:D],None if bb is None else bb[:D]).reshape(-1,heads,hd).transpose(0,1);K=F.linear(key,W[D:2*D],None if bb is None else bb[D:2*D]).reshape(-1,heads,hd).transpose(0,1);V=F.linear(value,W[2*D:],None if bb is None else bb[2*D:]).reshape(-1,heads,hd).transpose(0,1);A=(Q@K.transpose(-1,-2)/math.sqrt(hd)).softmax(-1);sup=rt['supports'];own=rt['own_support'];tm=rt['token_roles'];texts=[];reltexts=[];reads=[];own_z=key.new_zeros((256,288))
    if self.variant=='wrong_space':sup=sup[:,:,rt['point_perm']];own=own[:,rt['point_perm']]
    elif self.variant=='role_swap' and len(sup)>1:sup=sup.roll(1,0)
    def pool(att,mask):
     weights=att[:,None]*(.05+.95*mask[None]);weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12);v=(weights@V).transpose(0,1).reshape(len(mask),D);return module.out_proj(v)
    for e in range(len(tm)):
     mask=tm[e];texts.append(query[mask].mean(0) if mask.any() else query.sum(0)*0);rel=rt['relation_tokens'][e];reltexts.append(query[rel].mean(0) if rel.any() else query.sum(0)*0);att=A[:,mask].mean(1) if mask.any() else A.mean(1)*0+1/len(key);reads.append(pool(att,sup[e]));
     if e==0:own_z=pool(att,own)
    self.local={'local':torch.stack(reads) if reads else key.new_zeros((0,3,288)),'own':own_z,'text':torch.stack(texts) if texts else key.new_zeros((0,288)),'relation_text':torch.stack(reltexts) if reltexts else key.new_zeros((0,288))}
    # Observer only: global cross encoder and decoder remain exact R76 paths.
    return None
   hook=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(record,with_kwargs=True)
  try:yield
  finally:
   if hook is not None:hook.remove()
 def main(self,base,z,d,mode,*args,**kw):
  with self.capture(base):p,q,extra=super().main(base,z,d,mode,*args,**kw)
  extra.update(self.local);extra['base_probability']=p
  if self.local:
   valid=self.route['weights_plain']>0;extra['local_reads']=F.normalize(self.local['local'][valid].float(),dim=-1)
   if self.use_interface and self.variant!='no_residual':
    delta=self.interface(extra['query_hidden'],self.local['local'],self.local['own'],self.local['text'],self.local['relation_text'],self.route,self.variant);p=(p.clamp_min(1e-300).log()+delta.double()).softmax(0);extra['residual']=delta
  else:extra['local_reads']=q.new_zeros((0,288))
  return p,q,extra

def load_R76(m):
 S=load(R76_INIT);missing,unexpected=m.load_state_dict({k:torch.as_tensor(v,device='cuda') for k,v in S.items()},strict=False);assert not unexpected and all(k.startswith('interface.') for k in missing)
 return S

def route89(r,z0,g,graph,roles,attention,offsets,doc):
 a,info=build88(r,z0,g,graph,roles,attention);xyz=z0['xyz'][0];boxes=z0['boxes'];own=(np.abs(xyz[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(-1);E=len(roles);geo=np.zeros((256,E,3,9),np.float32);og=np.zeros((256,9),np.float32);visible=g['visible'];uv=g['uv_grid']/[48,32];candidateuv=np.zeros((256,2));candidateknown=np.zeros(256,bool)
 for i in range(256):
  v=own[i]&visible
  if v.any():candidateuv[i]=uv[v].mean(0);candidateknown[i]=True
 def geom(mask,mu,var):
  v=mask&visible;view=uv[v].mean(0) if v.any() else np.zeros(2);known=candidateknown&bool(v.any());out=np.zeros((256,9),np.float32);out[:,:3]=np.clip((mu-boxes[:,:3])/np.maximum(boxes[:,3:],.1),-10,10);out[:,3:6]=np.clip(np.sqrt(np.maximum(var,0))/np.maximum(boxes[:,3:],.1),0,10);out[:,6:8]=(view-candidateuv)*known[:,None];out[:,8]=float(v.sum())/max(1,int(mask.sum()));return out
 for e in range(E):
  for h in range(3):geo[:,e,h]=geom(a['supports'][e,h].astype(bool),a['support_centers_PRIVATE'][e,h],a['support_variances_PRIVATE'][e,h])
 for q in range(256):
  xx=xyz[own[q]];mu=xx.mean(0) if len(xx) else boxes[q,:3];var=xx.var(0) if len(xx) else np.zeros(3);og[q]=geom(own[q],mu,var)[q]
 relation=np.zeros_like(roles)
 for edge in graph['relations']:
  if 'sentence' not in edge or 'governor_word' not in edge:continue
  words=doc['sentences'][edge['sentence']]['words'];w=next((w for w in words if w['id']==edge['governor_word']),None)
  if w is None:continue
  for e,en in enumerate(graph['entities']):
   if en['role'] in [edge['subject'],edge['reference']]:relation[e]|=(offsets[:,1]>offsets[:,0])&(offsets[:,0]<w['end_char'])&(offsets[:,1]>w['start_char'])
 a.update(own_support=own.astype(np.float32),own_valid=own.sum(-1)>=2,hyp_geo=geo,own_geo=og,relation_tokens=relation);return a,info

def tensor_route89(a):
 keys=['token_roles','supports','weights_plain','weights_relation','point_perm','rank_pairs','rank_weights','own_support','own_valid','hyp_geo','own_geo','relation_tokens'];return {k:torch.as_tensor(a[k],device='cuda') for k in keys}

def mask89(r,d,region,repeat):
 seed=int(hashlib.sha256(('R86globalcoverage|'+r['id']+'|'+str(repeat)).encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed);remaining=np.flatnonzero(d['depth_valid'].cpu().numpy());n=len(remaining)//4;orders=[]
 for _ in range(2):
  glob=rng.permutation(remaining)[:int(np.ceil(n/2))];left=remaining[~np.isin(remaining,glob)];loc=rng.permutation(left[region['image'][left]])[:n-len(glob)];left=left[~np.isin(left,loc)];fill=rng.permutation(left)[:n-len(glob)-len(loc)];take=np.r_[glob,loc,fill];orders.append(take);remaining=remaining[~np.isin(remaining,take)]
 im=torch.zeros(1536,device='cuda',dtype=torch.bool);im[torch.tensor(orders[0],device='cuda')]=True;pm=d['visible']&torch.isin(d['cell'],torch.tensor(orders[1],device='cuda'));assert not im[d['cell'][pm]].any();return pm,im

def local_weights89(d,pm,im,region):
 out=[]
 for p,i in zip(pm,im):
  ws={}
  for k,valid,loc in [('P',p&d['visible'],region['point']),('I',i,region['image']),('D',d['depth_valid'],region['image'])]:
   a=valid.float();b=a*torch.as_tensor(loc,device='cuda');ws[k]=.25*a/a.sum().clamp_min(1)+.75*(b/b.sum().clamp_min(1) if b.any() else a/a.sum().clamp_min(1))
  out.append(ws)
 return out

def physical89(m,z,d,pm,im,ws):
 vals=[]
 for p,i,w in zip(pm,im,ws):
  P,I,D=m.local_losses(z,d,p,i);vals.append(.1*(w['P']*P).sum()+.1*(w['I']*I).sum()+.01*(w['D']*D).sum())
 return torch.stack(vals).mean()

def objective89(m,base,z,d,refp,refq,refs,pm,im,ws,physical_only=False):
 p,q,e=m.main(base,z,d,'bidirectional');keep=(refp*(refp.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2;corr=q.sum()*0
 for a,b in zip(pm,im):
  _,_,ee=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b)
  if ee['local_reads'].numel():corr=corr+(ee['local_reads']-refs).square().sum(-1).mean()/4
 pairs=m.route['rank_pairs'];rank=p.sum()*0
 if len(pairs):rank=(F.softplus(-(p[pairs[:,0]].log()-p[pairs[:,1]].log()))*m.route['rank_weights']).sum()/len(pairs)
 physical=physical89(m,z,d,pm,im,ws);C=.1*corr*(0 if physical_only else 1);R=.01*rank*(0 if physical_only else 1);return physical.double()+keep+C+R,p,{'physical':physical,'keep':keep,'correspondence':C,'rank':R}

def cached_score89(m,a,route,branch,variant='full'):
 h=a['h'];u=a['u'];late=a['late'];mask=a['mask'].bool();raw=m.match(h[None])[0]+late;q=F.normalize(raw,dim=-1);ss=q@u.T/.07;p=(torch.logsumexp(ss[:,mask],-1)-torch.logsumexp(ss[:,~mask],-1)).double().softmax(0)
 if branch in ['candidate','query']:p=(p.log()+m.interface(h,a['local'],a['own'],a['text'],a['relation_text'],route,'query_only' if branch=='query' else variant).double()).softmax(0)
 return p

def ground89(p,y):return -(torch.softmax(y.double()/.1,0)*p.clamp_min(1e-300).log()).sum() if float(y.max())>0 else p.sum()*0
