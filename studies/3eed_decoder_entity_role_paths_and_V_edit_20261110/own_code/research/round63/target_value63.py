"""Erase one role-specific V direction while preserving protected token values."""
import torch,hashlib
import torch.nn.functional as F
from common63 import *
from paths_heads63 import role_masks,paths,decoder_chain
def permuted(sid,mask,padding):
 valid=np.flatnonzero(~padding.cpu().numpy().reshape(-1));ix=np.arange(len(padding.reshape(-1)));shift=1+int(hashlib.sha256(('R63_value_direction_v2|'+sid).encode()).hexdigest()[:12],16)%(len(valid)-1);ix[valid]=np.roll(valid,shift);return mask[ix]
def patch(model,z,v,cfg):
 layer=cfg['selected_layer'];head=cfg['selected_head'];m=model.decoder[layer].cross_l;sm,rm,valid,meta=role_masks(z['_sid'],len(z[v+'_tokens']));meta=dict(meta);erase=rm if cfg['selected_kind']=='drop_reference' else sm;protect=sm if cfg['selected_kind']=='drop_reference' else rm;lo=576+head*36;hi=lo+36;M=torch.cat([m.in_proj_weight[lo:hi].double(),m.in_proj_bias[lo:hi,None].double()],1);language=torch.tensor(z[v+'_prefix_lang'][0],device=M.device,dtype=torch.float64);X=torch.cat([language,torch.ones(len(language),1,device=M.device,dtype=torch.float64)],1);zero=torch.zeros_like(M)
 if not valid:return zero,zero,meta|{'numeric_UNKNOWN':False,'applied':False},X,protect
 P=X[torch.tensor(protect,device=M.device)];_,s,Vh=torch.linalg.svd(P,full_matrices=False);rank=int((s>s[0]*1e-6).sum());B=Vh[:rank].T
 def basis(mask):
  rows=X[torch.tensor(mask,device=M.device)];residual=rows-(rows@B)@B.T;_,ss,vh=torch.linalg.svd(residual,full_matrices=False);n=float(ss[0]);rr=int((ss>ss[0]*1e-6).sum()) if n>=1e-6 else 0;return vh[:rr].T,n
 U,norm=basis(erase)
 if not U.shape[1]:return zero,zero,meta|{'numeric_UNKNOWN':True,'applied':False,'protected_rank':rank,'erased_residual_norm':norm},X,protect
 change=-(M@U)@U.T;randommask=permuted(z['_sid'],erase,torch.tensor(z[v+'_prefix_text_padding'],device=M.device));RU,rn=basis(randommask);random=zero
 if RU.shape[1]:random=-(M@RU)@RU.T;random=random*change.norm()/random.norm().clamp_min(1e-12)
 meta.update(numeric_UNKNOWN=False,applied=bool(change.norm()>1e-12),protected_rank=rank,erased_rank=U.shape[1],shuffled_erased_rank=RU.shape[1],erased_residual_norm=norm,delta_matrix_Frobenius=float(change.norm()),shuffled_direction_zero=bool(not RU.shape[1]),protected_projection_overlap=float((P@U).abs().max()),erased_direction_output_before=float((M@U).norm()),erased_direction_output_after=float(((M+change)@U).norm()))
 return change,random,meta,X,protect
def source_branch(model,z,v,layer):
 source={}
 def text(_m,args,out):source['text']=out.detach().clone()
 def language(_m,args,out):source['language']=out[0].detach().clone();source['attention']=out[1].detach().clone()
 def visual(_m,args,out):source['visual']=out[0].detach().clone()
 handles=[model.decoder[layer].norm_l.register_forward_hook(text),model.decoder[layer].cross_l.register_forward_hook(language),model.decoder[layer].cross_v.register_forward_hook(visual)]
 try:h,hist,_=decoder_chain(model,z,v)
 finally:
  for hh in handles:hh.remove()
 return h,hist,source
def edit_chain(model,z,v,cfg,mode,source):
 layer=cfg['selected_layer'];head=cfg['selected_head'];m=model.decoder[layer].cross_l;W=m.in_proj_weight.detach().clone();b=m.in_proj_bias.detach().clone();change,random,meta,X,protect=patch(model,z,v,cfg);meta=dict(meta);sm,rm,valid,_=role_masks(z['_sid'],len(z[v+'_tokens']));delta=-change if mode=='opposite' else random if mode=='shuffled' else change;lo=576+head*36;hi=lo+36;newW=W.clone();newb=b.clone();M=torch.cat([W[lo:hi].double(),b[lo:hi,None].double()],1);after=(M+delta).float();newW[lo:hi]=after[:,:288];newb[lo:hi]=after[:,288];physical=mode not in ['functional','random_path'];handles=[];cache={}
 if valid and mode!='random_path':
  P=X[torch.tensor(protect,device=X.device)].float();oldvalues=F.linear(P[:,:288],W[lo:hi],b[lo:hi]);newvalues=F.linear(P[:,:288],newW[lo:hi],newb[lo:hi]);error=float(abs(newvalues-oldvalues).max());assert error<1e-4;meta['protected_value_max_abs_change']=error;meta['actual_changed_parameter_n']=int((newW!=W).sum()+(newb!=b).sum());meta['actual_delta_Frobenius']=float(torch.cat([(newW[lo:hi]-W[lo:hi]).flatten(),newb[lo:hi]-b[lo:hi]]).norm())
 if physical:m.in_proj_weight.copy_(newW);m.in_proj_bias.copy_(newb)
 def before(_m,args,kwargs):cache['kwargs']=kwargs
 def language(module,args,out):
  kw=cache['kwargs'];new=out
  if valid and mode=='functional':new=F.multi_head_attention_forward(kw['query'],kw['key'],kw['value'],288,8,newW,newb,None,None,False,.1,module.out_proj.weight,module.out_proj.bias,training=False,key_padding_mask=kw.get('key_padding_mask'),need_weights=True)
  elif valid and mode=='random_path':
   _,_,cs,cr,project,_=paths(module,kw['query'],kw['key'],kw['value'],kw.get('key_padding_mask'),sm,rm);es=rm if cfg['selected_kind']=='drop_reference' else sm;rr=permuted(z['_sid'],es,kw['key_padding_mask']);_,_,rs,_,_,_=paths(module,kw['query'],kw['key'],kw['value'],kw.get('key_padding_mask'),rr,np.zeros_like(rr));target=torch.zeros_like(cs);other=torch.zeros_like(cs);target[head]=(cr if cfg['selected_kind']=='drop_reference' else cs)[head];other[head]=rs[head];ct=project(target);rc=project(other);norm=rc.norm(dim=-1,keepdim=True);shift=rc*ct.norm(dim=-1,keepdim=True)/norm.clamp_min(1e-12);new=(out[0]-shift,out[1]);meta['random_path_zero_n']=int((norm<=1e-12).sum());meta['random_path_norm_match_max_abs']=float(abs(shift.norm(dim=-1)-ct.norm(dim=-1)).max())
  rel=(new[0]-source['language']).norm(dim=-1)/source['language'].norm(dim=-1).clamp_min(1e-12);meta['median_relative_branch_change']=float(rel.median());meta['max_relative_branch_change']=float(rel.max());meta['selected_attention_max_abs_change']=float(abs(new[1]-source['attention']).max());assert meta['selected_attention_max_abs_change']<1e-6
  return new
 def visual(_m,args,out):
  cache['visual']=out[0]
  if mode=='text_only':return source['visual'],out[1]
 def norm(_m,args):
  if mode=='visual_only':return (source['text']+cache['visual'],)
 handles=[m.register_forward_pre_hook(before,with_kwargs=True),m.register_forward_hook(language)]
 if mode in ['text_only','visual_only']:handles+=[model.decoder[layer].cross_v.register_forward_hook(visual),model.decoder[layer].norm_v.register_forward_pre_hook(norm)]
 try:h,hist,_=decoder_chain(model,z,v)
 finally:
  for hh in handles:hh.remove()
  m.in_proj_weight.copy_(W);m.in_proj_bias.copy_(b)
 return h,hist,meta
