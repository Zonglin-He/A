"""Target-only protocol, imported only after source preparation has ended."""
from core89 import *
BASE89=['source','frozen_bi','R59','R87_hard','R88_direct','R88_TTA1','R88_mean3']
BRANCHES89=['continue','query','candidate'];VARIANTS89=['no_ref','shared_mix','wrong_space','role_swap','no_residual']
ARMS89=BASE89+[b+'_'+r for b in BRANCHES89 for r in ['frozen','mean3','tta1']]+['candidate_tta3']+[v+'_'+r for v in VARIANTS89 for r in ['frozen','tta1']]+['candidate_physical1','candidate_no_correspondence1']
def item89(r):
 z0=prefix(r);z=tensor_prefix(z0);g=load(casepath(r));d=tensor_data(g,r);a=load(B89/'target_evidence336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));region={'point':a['region_point'],'image':a['region_image']};ab=[mask89(r,d,region,k) for k in [88001,88002]];pm=[x[0] for x in ab];im=[x[1] for x in ab];ws=local_weights89(d,pm,im,region);return z0,z,d,a,pm,im,ws
def flat89(grad,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for g,p in zip(grad,ps)])
def apply89(m,old,delta):
 off=0
 with torch.no_grad():
  for (_,p),v in zip(m.tta_parameters(),old):n=p.numel();p.copy_(v+delta[off:off+n].reshape_as(p));off+=n
 assert off==len(delta)
def objective_target89(m,base,z,d,refp,refq,refs,pm,im,ws,kind='full'):
 total,p,parts=objective89(m,base,z,d,refp,refq,refs,pm,im,ws,physical_only=kind=='physical')
 if kind=='no_correspondence':total=total-parts['correspondence'];parts={**parts,'correspondence':parts['correspondence']*0}
 return total,p,parts
def patch89(m):return {n:v.detach().cpu().numpy().copy() for n,v in m.tta_parameters()}
def features89(extra):return {k:extra[k].detach().cpu().numpy() for k in ['query_hidden','projected_text','late_RGB_delta','local','own','text','relation_text','base_probability'] if k in extra}
