"""True decoder-chain edits + unchanged R59 representation fusion, no labels."""
import torch,numpy as np
from common62 import *
def gpu_strict():
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def cpu_head(initial):
 m=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,64));m.load_state_dict({k:torch.tensor(v) for k,v in initial.items()});return m.eval().requires_grad_(False)
def decoder_chain(model,z,view,layer=None,semantic=None,return_attention=False):
 device=next(model.parameters()).device;t=lambda k:torch.tensor(z[view+'_'+k],device=device);query=t('prefix_query');vis=t('prefix_vis');lang=t('prefix_lang');pos=t('prefix_pos');padding=t('prefix_text_padding');anchor=t('query_anchor_xyz')[None];hist=[];attentions={};hooks=[];module=None;W=b=None
 try:
  if layer is not None and semantic is None:
   module=model.decoder[layer].cross_l.out_proj;W=module.weight.detach().clone();b=module.bias.detach().clone();module.weight.copy_(W*1.5);module.bias.copy_(b*1.5)
  if return_attention:
   for i in range(6):
    def save(_m,args,out,index=i):attentions[index]=out[1][0].detach().cpu().numpy()
    hooks.append(model.decoder[i].cross_l.register_forward_hook(save))
  for i in range(6):
   query=model.decoder[i](query,vis,lang,pos,None,padding,detected_feats=None,detected_mask=None)
   endpoint={};center,size=model.prediction_heads[i](query.transpose(1,2).contiguous(),base_xyz=anchor,end_points=endpoint,prefix='audit_');pos=torch.cat([center.detach(),size.detach()],-1);hist.append(pos[0].detach().cpu().numpy())
  return query[0].detach().cpu().numpy(),hist,attentions
 finally:
  for h in hooks:h.remove()
  if module is not None:module.weight.copy_(W);module.bias.copy_(b)
def unit(x,fallback=None):
 n=np.linalg.norm(x,axis=-1,keepdims=True);y=np.divide(x,n,out=np.zeros_like(x),where=n>1e-12);return np.where(n>1e-12,y,fallback) if fallback is not None else y
def score(raw,z,g):
 parts=[]
 for v in 'ABC':
  y=torch.tensor(np.asarray(raw[v],np.float32))[None];q=torch.nn.functional.normalize(y,dim=-1);t=torch.tensor(z[v+'_tokens'])[None];mask=torch.tensor(z[v+'_token_mask']).bool()[None,None];l=q@t.transpose(-1,-2)/.07;odds=(torch.logsumexp(l.masked_fill(~mask,-torch.inf),-1)-torch.logsumexp(l.masked_fill(mask,-torch.inf),-1))[0];p=odds.double().softmax(-1).numpy()
  if v!='A':
   p=g[v+'_W']@p;mass=float(g[v+'_fixed_mass']);p=mass*p/max(float(p.sum()),1e-300)+(1-mass)*g['base']
  parts.append(p)
 return sum(parts)/3
def readouts(head,hs,z,g):
 X={v:head[3](head[2](head[1](head[0](torch.tensor(hs[v])[None]))))[0].numpy() for v in 'ABC'};raw32={v:head[4](torch.tensor(X[v])[None])[0].numpy() for v in 'ABC'};M=np.column_stack([head[4].weight.numpy(),head[4].bias.numpy()]).astype(float);raw64={v:np.column_stack([X[v].astype(float),np.ones(256)])@M.T for v in 'ABC'};q={v:unit(raw64[v]) for v in 'ABC'};W=g['B_W'];rho=g['B_affinity'].max(1);rowmass=W.sum(1);qb=np.divide(W@q['B'],rowmass[:,None],out=np.zeros_like(q['A']),where=rowmass[:,None]>0);targetA=unit(q['A']+rho[:,None]*qb,q['A']);targets={'A':np.linalg.norm(raw64['A'],axis=1)[:,None]*targetA};confidence={'A':rho}
 for v in 'BC':
  W=g[v+'_W'];mass=W.sum(0);u=np.divide(W.T@targetA,mass[:,None],out=np.zeros_like(q[v]),where=mass[:,None]>0);valid=np.linalg.norm(u,axis=1)>1e-12;confidence[v]=g[v+'_affinity'].max(0)*valid;targets[v]=np.linalg.norm(raw64[v],axis=1)[:,None]*unit(u,q[v])
 fused={v:(1-confidence[v][:,None])*raw32[v]+confidence[v][:,None]*targets[v] for v in 'ABC'}
 return score(raw32,z,g),score(fused,z,g),{'raw':raw32,'targets':targets,'confidence':confidence}
def iou(a,b):
 lo=np.maximum(a[:,None,:3]-a[:,None,3:]/2,b[None,:,:3]-b[None,:,3:]/2);hi=np.minimum(a[:,None,:3]+a[:,None,3:]/2,b[None,:,:3]+b[None,:,3:]/2);inter=np.maximum(hi-lo,0).prod(-1);u=np.maximum(a[:,3:],0).prod(-1)[:,None]+np.maximum(b[:,3:],0).prod(-1)[None]-inter;return np.divide(inter,u,out=np.zeros_like(inter),where=u>0)
def lineage(z,boxes,g):
 per={}
 for v in 'ABC':
  old=z[v+'_boxes'].astype(float);new=boxes[v].astype(float);pair=iou(new,old);best=pair.argmax(1);equivalence=iou(old,old);outside=equivalence[np.arange(256),best]<.5;diag=np.diag(pair);delta=np.linalg.norm(new[:,:3]-old[:,:3],axis=1)/np.maximum(np.linalg.norm(old[:,3:],axis=1),1e-12);per[v]={'median_same_slot_internal_box_IoU':float(np.median(diag)),'median_center_shift_over_old_box_diagonal':float(np.median(delta)),'outside_duplicate_group_n':int(outside.sum()),'nonpositive_new_box_sizes':int((new[:,3:]<=0).any(1).sum())}
 for v in 'BC':
  oldW=g[v+'_W'];overlap=iou(boxes['A'].astype(float),boxes[v].astype(float));per[v]['old_transport_mass_with_zero_new_geometric_overlap_fraction']=float(oldW[overlap<=0].sum()/max(float(oldW.sum()),1e-12))
 return per
