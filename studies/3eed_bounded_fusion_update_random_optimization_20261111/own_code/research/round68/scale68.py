"""Actual accepted R66 parameter-delta scaling; same AB-only acceptance rule."""
import sys,time,torch
from common68 import *
sys.path.insert(0,str(ROOT/'research/round66'));import fit_query66 as f
sys.path.insert(0,str(ROOT/'research/round58'));import numerics58 as n
H=ROOT/'research/round68';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');out=ROOT/p['scaled_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.perf_counter();counts={tag(eta):{'eligible':0,'accepted':0,'rejected':0,'nonzero':0} for eta in ETAS}
for i,r in enumerate(rows):
 rel='samples_PRIVATE/'+r['id']+'.npz';ab=load(ROOT/p['AB_root']/rel);e=load(ROOT/p['evidence_root']/rel);fit=load(ROOT/p['fit_root']/rel);records={};arrays={};base=n.head(initial);d=f.data(ab,e);prior={v:n.distribution(base,d,v).clone() for v in 'AB'}
 for eta in ETAS:
  name=tag(eta);model=n.head(initial);changed=bool(any(np.any(fit[k]!=initial[k]) for k in initial));allowed=bool(e['update_allowed']) and changed;before=float(f.objective(model,d,e,prior)) if allowed else 0.;accepted=False;after=before
  if allowed:
   proposal={k:(fit[k].copy() if eta==1. else initial[k]+np.float32(eta)*(fit[k]-initial[k])) for k in initial};model.load_state_dict({k:torch.tensor(v) for k,v in proposal.items()});after=float(f.objective(model,d,e,prior));accepted=bool(n.permitted(before,after));counts[name]['eligible']+=1
   if not accepted:model.load_state_dict(base.state_dict())
   counts[name]['accepted' if accepted else 'rejected']+=1
  norm=float(np.sqrt(sum(float(((v.detach().numpy()-initial[k]).astype(float)**2).sum()) for k,v in model.state_dict().items())));counts[name]['nonzero']+=int(norm>0);records[name]={'eligible':allowed,'accepted':accepted,'loss_before':before,'loss_proposed':after,'loss_final':after if accepted else before,'parameter_delta_L2':norm,'C_used':False,'state_sha256':n.phash(model)}
  if allowed:
   for k,v in n.state(model).items():arrays[name+'_'+k]=v
 if arrays:np.savez_compressed(out/rel,**arrays)
 dump(out/'cases_PRIVATE'/(r['id']+'.json'),records)
 if(i+1)%64==0:print('SCALE68',i+1,336,flush=True)
dump(out/'COSTS.json',{'n':336,'counts':counts,'C_read_or_used':False,'new_gradient_steps':0,'seconds':time.perf_counter()-start});seal(out,'R68_ACTUAL_PARAMETER_SCALE_ACCEPTANCE_ALL_AB_ONLY_SEALED_BEFORE_C',n=336);print(counts)
