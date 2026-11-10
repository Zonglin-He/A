"""Independent fixed-anchor ablation: native encoder with boundary hooks.
Unlike the producer's encoder-layer loop, this calls native BiEncoder.forward.
No R66 producer imports.
"""
import torch,numpy as np
from scipy.special import softmax,logsumexp
from scipy.sparse import csr_matrix

def probability(query,tokens,mask):
 l=np.einsum('ik,jk->ij',query.astype(float),tokens.astype(float))/.07;return softmax(logsumexp(l[:,mask],1)-logsumexp(l[:,~mask],1))
def overlap(a,b):
 l=np.maximum(a[:,None,:3]-a[:,None,3:]/2,b[None,:,:3]-b[None,:,3:]/2);h=np.minimum(a[:,None,:3]+a[:,None,3:]/2,b[None,:,:3]+b[None,:,3:]/2);i=np.maximum(h-l,0).prod(-1);u=np.maximum(a[:,3:],0).prod(1)[:,None]+np.maximum(b[:,3:],0).prod(1)[None]-i;return np.divide(i,u,out=np.zeros_like(i),where=u>0)
def support(x,b):return np.less_equal(np.abs(x[None]-b[:,None,:3]),b[:,None,3:]/2).all(-1)&(b[:,3:]>0).all(1)[:,None]
def af(a,b,x):
 A=support(x,a).astype(np.float32);B=support(x,b).astype(np.float32);shared=(csr_matrix(A)@csr_matrix(B).T).toarray();u=A.sum(1)[:,None]+B.sum(1)[None]-shared;j=np.divide(shared,u,out=np.zeros_like(u),where=u>0);w=np.sqrt(np.maximum(j*overlap(a,b),0));w[shared<4]=0;return w.astype(np.float32),j,shared

def groups_and_pool(boxes,xyz,p):
 _,j,s=af(boxes,boxes,xyz);io=overlap(boxes,boxes);unused=set(range(256));groups=[]
 for rep in np.argsort(-p,kind='stable'):
  if int(rep) not in unused:continue
  row=np.array([k for k in range(256) if k in unused and io[rep,k]>=.5 and j[rep,k]>=.5 and s[rep,k]>=4],np.int64);row=np.unique(np.r_[int(rep),row]);unused.difference_update(map(int,row));groups.append(row)
 assert not unused;order=sorted(range(len(groups)),key=lambda k:(-float(p[groups[k]].sum()),int(groups[k][0])));return groups,order[:4]

def region_controls(x,b,groups,pool):
 region=np.array([support(x,b[groups[j]]).any(0) for j in pool]);allowed=np.where(~region.any(0))[0];out=[]
 for m in region:
  c=np.zeros(len(x),bool);n=int(m.sum());valid=n>=4 and n<=len(x)-4 and len(allowed)>=n
  if valid:
   center=x[m].mean(0);anchor=allowed[np.argmax(np.linalg.norm(x[allowed]-center,axis=1))];idx=allowed[np.argsort(np.linalg.norm(x[allowed]-x[anchor],axis=1),kind='stable')[:n]];c[idx]=True
  out.append((m,c,valid))
 return out

def mapped(p,W,mass,base):
 q=W@p;return mass*q/max(float(q.sum()),1e-300)+(1-mass)*base

def replay(model,z,mask=None,category=False):
 device=next(model.parameters()).device;T=lambda k:torch.tensor(z[k],device=device);V=T('vis0');P=T('pos0');L=T('class_lang0' if category else 'lang0');pad=T('class_textpad' if category else 'textpad');ids=T('query_points_sample_inds').long()[None];m=torch.zeros(V.shape[:2],dtype=torch.bool,device=device) if mask is None else torch.tensor(mask,dtype=torch.bool,device=device)[None];active=bool(m.any());qm=m.gather(1,ids) if active else None;hooks=[];proof={'max_blocked_attention':0.}
 if active:
  def before_layer(_mod,args,kwargs):return (args[0].masked_fill(m[:,:,None],0),*args[1:]),kwargs
  def after_layer(_mod,args,out):return out[0].masked_fill(m[:,:,None],0),out[1]
  def mask_visual(_mod,args,kwargs):return args,dict(kwargs,key_padding_mask=m)
  def check(_mod,args,kwargs,out):
   expected=kwargs['key_padding_mask'];w=out[1];blocked=w.masked_select(expected[:,None].expand_as(w));proof['max_blocked_attention']=max(proof['max_blocked_attention'],float(blocked.abs().max()) if blocked.numel() else 0.)
  for layer in model.cross_encoder.layers:
   hooks.append(layer.register_forward_pre_hook(before_layer,with_kwargs=True));hooks.append(layer.register_forward_hook(after_layer));hooks.append(layer.self_attention_visual.self_attn.register_forward_hook(check,with_kwargs=True));hooks.append(layer.cross_layer.cross_lv.register_forward_hook(check,with_kwargs=True))
  for dec in model.decoder:hooks.append(dec.cross_v.register_forward_pre_hook(mask_visual,with_kwargs=True));hooks.append(dec.cross_v.register_forward_hook(check,with_kwargs=True));hooks.append(dec.self_attn.register_forward_hook(check,with_kwargs=True))
 try:
  if active:V=V.masked_fill(m[:,:,None],0)
  V,L=model.cross_encoder(vis_feats=V,pos_feats=P,padding_mask=m,text_feats=L,text_padding_mask=pad,end_points={},detected_feats=None,detected_mask=None)
  _,F,_=model.gsample_module(T('xyz'),V.transpose(1,2).contiguous(),ids.int());q=model.decoder_query_proj(F).transpose(1,2).contiguous()
  for j,dec in enumerate(model.decoder):q=dec(q,V,L,T('position'+str(j)),qm,pad,detected_feats=None,detected_mask=None)
  tt=torch.nn.functional.normalize(model.contrastive_align_projection_text(L),dim=-1);qq=torch.nn.functional.normalize(model.contrastive_align_projection_image(q),dim=-1);a=qq@tt.transpose(-1,-2)/.07;tm=T('class_token_mask' if category else 'token_mask').bool()[None,None];odds=(torch.logsumexp(a.masked_fill(~tm,-torch.inf),-1)-torch.logsumexp(a.masked_fill(tm,-torch.inf),-1))[0];p=odds.double().softmax(-1)
  assert proof['max_blocked_attention']==0.;return {'h':q[0].cpu().numpy(),'query':qq[0].cpu().numpy(),'tokens':tt[0].cpu().numpy(),'p':p.cpu().numpy()}
 finally:
  for h in hooks:h.remove()
