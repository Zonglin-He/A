"""Label-free single-input update, common to source diagnostics and target evaluation."""
from core90 import *
def update90(m,base,z,d,pm,im,ws,steps=1,kind='full',perm=None):
 m.requires_grad_(False)
 with torch.no_grad():refp,refq,e=m.main(base,z,d,'bidirectional');refcond=e['conditional'].detach();refvalid=e['conditional_valid'];reflocal=e['local_reads'].detach()
 refs=(refp,refq,refcond,refvalid,reflocal);enable_tta90(m);initial=state(m);trace=[];endpoints={};states={};grad_first=None
 for step in range(1,steps+1):
  L,p,parts=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm);named=m.tta_parameters();ps=[p for n,p in named];raw=flat90(torch.autograd.grad(L,ps,allow_unused=True),ps);g=raw*min(1.,1/(float(raw.norm())+1e-6));old=[p.detach().clone() for p in ps];dot=float(raw@g);accepted=False;alpha=0.;trials=[];offset=0;eta2=theta2=0.
  for n,p in named:
   val=float(raw[offset:offset+p.numel()].square().sum());offset+=p.numel()
   if n.startswith('interface.'):eta2+=val
   else:theta2+=val
  if grad_first is None:grad_first=raw.detach().clone()
  if float(g.norm())>1e-12:
   for scale in [1/2**j for j in range(9)]:
    apply90(m,old,-.1*scale*g)
    with torch.no_grad():lv,_,_=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm)
    bound=float(L)-1e-4*.1*scale*max(0,dot);ok=float(lv)<=bound+1e-10;trials.append({'scale':scale,'actual_total':float(lv),'bound':bound,'accepted':ok})
    if ok:accepted=True;alpha=scale;break
  if not accepted:apply90(m,old,torch.zeros_like(g))
  with torch.no_grad():after,pv,aparts=objective90(m,base,z,d,*refs,pm,im,ws,kind,perm)
  assert not accepted or float(after)<=float(L)+1e-10;norm=float(torch.sqrt(sum((p-torch.as_tensor(initial[n],device='cuda')).square().sum() for n,p in named)));trace.append({'step':step,'accepted':accepted,'scale':alpha,'before':float(L),'after':float(after),'parts_before':{k:float(v) for k,v in parts.items()},'parts_after':{k:float(v) for k,v in aparts.items()},'gradient_norm_raw':float(raw.norm()),'gradient_norm_clipped':float(g.norm()),'gradient_eta_router':eta2**.5,'gradient_other_parameters':theta2**.5,'directional_derivative':dot,'parameter_L2':norm,'trials':trials})
  if step in [1,3]:endpoints[step]=pv.detach().cpu().numpy();states[step]={n:p.detach().cpu().numpy().copy() for n,p in named}
 return refp.detach().cpu().numpy(),endpoints,states,trace,refs,grad_first

def put90(m,patch):
 allowed={n for n,p in m.tta_parameters()};assert set(patch)==allowed
 with torch.no_grad():
  for n,p in m.tta_parameters():p.copy_(torch.as_tensor(patch[n],device=p.device))
def shuffle90(r):
 seed=int(hashlib.sha256(('R90conditional_shuffle|'+r['id']).encode()).hexdigest()[:16],16);return torch.tensor(np.random.default_rng(seed).permutation(256),device='cuda')
