"""Decoder entity-role paths; source outputs and unmodified bias retained."""
import sys,torch,functools
import torch.nn.functional as F
from common63 import *
sys.path.insert(0,str(ROOT/'research/round62'))
from compute62 import decoder_chain,cpu_head,readouts,lineage,gpu_strict
@functools.lru_cache(maxsize=512)
def role_masks(sid,L):
 records=read(ROOT/'research/round62/role_diagnostics_attempt001/MASKS_PRIVATE.json')
 rr=[r for r in records if r['id_PRIVATE']==sid];assert len(rr)==4
 masks={r['role']:np.array(r['mask'],bool) for r in rr};s=masks['subject'];r=masks['reference'];assert len(s)==len(r)==L
 overlap=s&r;valid=bool(s.any() and r.any() and not overlap.any())
 return s,r,valid,{'subject_tokens':int(s.sum()),'reference_tokens':int(r.sum()),'overlap_tokens':int(overlap.sum()),'valid_roles':valid}
def paths(module,q,k,v,padding,sm,rm):
 # B=1, eight 36-dimensional native heads. No normalization of role values.
 Q=F.linear(q,module.in_proj_weight[:288],module.in_proj_bias[:288]).reshape(-1,1,8,36).permute(1,2,0,3)[0]
 K=F.linear(k,module.in_proj_weight[288:576],module.in_proj_bias[288:576]).reshape(-1,1,8,36).permute(1,2,0,3)[0]
 V=F.linear(v,module.in_proj_weight[576:],module.in_proj_bias[576:]).reshape(-1,1,8,36).permute(1,2,0,3)[0]
 logits=(Q/6.)@K.transpose(-1,-2)
 if padding is not None:logits=logits.masked_fill(padding[0][None,None],-torch.inf)
 A=logits.softmax(-1);s=torch.tensor(sm,device=q.device);r=torch.tensor(rm,device=q.device);ms=A[:,:,s].sum(-1);mr=A[:,:,r].sum(-1);cs=A[:,:,s]@V[:,s];cr=A[:,:,r]@V[:,r];allv=A@V
 def project(x):return F.linear(x.permute(1,0,2).reshape(-1,1,288),module.out_proj.weight,None)
 reconstructed=project(allv)+module.out_proj.bias
 return ms,mr,cs,cr,project,reconstructed
def chain(model,z,v,layer=None,kind=None,head=None,trace=False):
 sm,rm,valid,receipt=role_masks(z['_sid'],len(z[v+'_tokens']));handles=[];record={};native_trace={}
 def hook_for(site):
  def pre(m,args,kwargs):record[site]=kwargs
  def post(m,args,out):
   kw=record[site];ms,mr,cs,cr,project,rebuilt=paths(m,kw['query'],kw['key'],kw['value'],kw.get('key_padding_mask'),sm,rm);error=float(abs(rebuilt-out[0]).max());assert error<1e-4,('native_role_path_reconstruction',site,error)
   if trace:
    native_trace[site]=np.stack([ms.cpu().numpy(),mr.cpu().numpy(),cs.norm(dim=-1).cpu().numpy(),cr.norm(dim=-1).cpu().numpy(),F.cosine_similarity(cs,cr,dim=-1).cpu().numpy()]);return
   if not valid:return
   active=torch.ones_like(ms)
   if head is not None:active.zero_();active[head]=1
   if kind=='drop_subject':delta=-cs*active[:,:,None]
   elif kind=='drop_reference':delta=-cr*active[:,:,None]
   elif kind=='swap_roles':
    good=(ms>1e-8)&(mr>1e-8);delta=((mr/ms.clamp_min(1e-8)-1)[:,:,None]*cs+(ms/mr.clamp_min(1e-8)-1)[:,:,None]*cr)*(good*active)[:,:,None]
   else:raise ValueError(kind)
   change=project(delta);new=out[0]+change;receipt.update(reconstruction_error=error,median_relative_branch_change=float((change.norm(dim=-1)/out[0].norm(dim=-1).clamp_min(1e-12)).median()),max_relative_branch_change=float((change.norm(dim=-1)/out[0].norm(dim=-1).clamp_min(1e-12)).max()))
   return new,out[1]
  return pre,post
 for i in (range(6) if trace else [layer]):
  pre,post=hook_for(i);handles.extend([model.decoder[i].cross_l.register_forward_pre_hook(pre,with_kwargs=True),model.decoder[i].cross_l.register_forward_hook(post)])
 try:h,hist,_=decoder_chain(model,z,v)
 finally:
  for hook in handles:hook.remove()
 return h,hist,receipt,native_trace
