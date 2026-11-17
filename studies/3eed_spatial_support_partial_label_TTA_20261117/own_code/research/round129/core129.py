"""Spatial-support conditional partial labels at the unchanged R127 interface.

The complete deployed readout remains differentiable. Candidate sets are frozen
before fitting. Unknown-output KL is checked at ACTUAL line-search endpoints.
No current top1 is injected into a candidate set. No C observation is decoded
until every proposed parameter state has been fixed.
"""
from pathlib import Path
import sys
R129=Path(__file__).resolve().parents[2];H129=R129/'research/round129';B129=R129/'crossplatform/round129'
sys.path.insert(0,str(R129/'research/round127'));from core127 import *
from support129 import glogits,taskloss,collapse,keep_kl,direct
SET129=['conditional','partial','partial_guard','wrong_guard'];PRIMARY129='partial_guard_F'
NEW129=['global_scope_F','global_scope_C','direct_conditional_F','direct_conditional_C','direct_partial_F','direct_partial_C']+[s+'_'+v for s in SET129 for v in ['F','C']]
REF129=['original_source','prepared_A','legacy_ABC_context','zero_F','zero_C','R127_global_F','R127_global_C','R127_ab_native_F','R127_ab_native_C','direct_global_F','direct_global_C'];ARMS129=REF129+NEW129

def get_scope(e):
 p=B129/'scope_PRIVATE'/e['cohort']/(e['source']+'_to_'+e['target'])/(e['row']['id']+'.npz');a=load(p);meta=jread(p.with_suffix('.json'));groups=[np.flatnonzero(a['assignment']==j).tolist() for j in range(int(a['assignment'].max())+1)];return a,meta,groups
def objective129(base,m,z,d,FAB,late,res,groups,S,C,T,kind):
 lp,h,u=body127(m,base,z,d,FAB,late,res,True);task=taskloss(lp,groups,S,C,T,kind);preserve=(F.normalize(m.point_adapter(FAB),dim=-1)-F.normalize(FAB,dim=-1)).square().sum(-1).mean()/2;return task+preserve,task,preserve,lp,h,u
def fit129(base,m,S0,z,d,FAB,late,res,groups,scope,meta,radius,epsilon,engineering):
 states={};traces={};grads={};checks=[];params=[dict(m.named_parameters())[n] for n in NAMES]
 with torch.no_grad():lp0,_,_=body127(m,base,z,d,FAB,late,res,True)
 p0=lp0.exp().detach();err=float(abs(p0-torch.as_tensor(scope['pF'],device='cuda')).max());assert err<1e-11,err
 configurations=[('conditional','conditional','S','C',bool(meta['scope_eligible'])),('partial','partial','S','C',bool(meta['partial_eligible'])),('partial_guard','partial','S','C',bool(meta['partial_eligible'])),('wrong_guard','partial','wrong_S','wrong_C',bool(meta['partial_eligible']))]
 if meta['scope_eligible'] and not meta['old_R127_global_active']:configurations.append(('global_new_eligibility','global','S','C',True))
 for setting,kind,sk,ck,active in configurations:
  restore(m,S0);S=torch.as_tensor(scope[sk],device='cuda');C=torch.as_tensor(scope[ck],device='cuda');T=torch.as_tensor(scope['teacher_global'] if kind=='global' else scope['teacher_conditional'],device='cuda');queryS=S[torch.as_tensor(scope['assignment'],device='cuda')];reference=collapse(p0,queryS);delta=torch.zeros(4896,device='cuda');steps=[]
  if not active:
   states[setting]=delta.cpu().numpy();traces[setting]=[];continue
  for it in range(3):
   put123(m,S0,delta)
   for p in params:p.requires_grad_(True)
   L,task,preserve,lp,_,_=objective129(base,m,z,d,FAB,late,res,groups,S,C,T,kind);before=float(L.detach());gg=torch.autograd.grad(L,params,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(params,gg)]);assert torch.isfinite(g).all();grads[setting+'_step'+str(it+1)]=g.cpu().numpy();old=delta.clone();direction=-radius*g/g.norm() if float(g.norm())>1e-12 and float(task.detach())>1e-12 else g*0
   for p in params:p.requires_grad_(False)
   if engineering and it==0 and setting in ['conditional','partial_guard'] and float(g.norm())>1e-7:
    eps=radius*.001;unit=g/g.norm();vals=[]
    for sign in [-1,1]:
     put123(m,S0,old+sign*eps*unit)
     with torch.no_grad():vals.append(float(objective129(base,m,z,d,FAB,late,res,groups,S,C,T,kind)[0]))
    fd=(vals[1]-vals[0])/(2*eps);relative=abs(fd-float(g.norm()))/max(float(g.norm()),1e-12);assert relative<.2,(setting,fd,float(g.norm()),relative);checks.append({'setting':setting,'finite_difference_relative_error':relative})
   accepted=False;fraction=0.;trials=[]
   if direction.norm()>0:
    for k in range(9):
     candidate=old+direction/(2**k);put123(m,S0,candidate)
     with torch.no_grad():ll,tt,pp,lp,_,_=objective129(base,m,z,d,FAB,late,res,groups,S,C,T,kind);unknown=float(keep_kl(reference,collapse(lp.exp(),queryS)));value=float(ll)
     armijo=value<=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;feasible=unknown<=epsilon+1e-12;guard=setting in ['partial_guard','wrong_guard'];ok=armijo and (feasible or not guard);trials.append({'fraction':2.**(-k),'objective':value,'unknown_KL':unknown,'Armijo':armijo,'unknown_budget_pass':feasible,'accepted':ok})
     if ok:delta=candidate;accepted=True;fraction=2.**(-k);break
   if not accepted:delta=old
   put123(m,S0,delta)
   with torch.no_grad():ll,tt,pp,lp,_,_=objective129(base,m,z,d,FAB,late,res,groups,S,C,T,kind);unknown=float(keep_kl(reference,collapse(lp.exp(),queryS)))
   if setting in ['partial_guard','wrong_guard']:assert unknown<=epsilon+1e-12
   steps.append({'before':before,'after':float(ll),'task_after':float(tt),'feature_keep_after':float(pp),'gradient_L2':float(g.norm()),'parameter_L2':float(delta.norm()),'accepted':accepted,'fraction':fraction,'unknown_KL':unknown,'trials':trials})
  states[setting]=delta.cpu().numpy();traces[setting]=steps
 restore(m,S0);return states,traces,grads,checks

def process129(base,m,S,e,engineering=False):
 restore(m,S);S0,radius=start123(m,S);z0,z,A,B,ids,far,_,present,salt=a_b123(e);scope,meta,groups=get_scope(e);d=geomdata125(e,z0);late,res,provider=source_context127(e)
 with np.load(R129/e['observations']) as f:
  pre='real_B_' if e['cohort']=='DEV' else 'B_';valid=torch.as_tensor(f[pre+'valid'],device='cuda') if present else torch.zeros(1024,device='cuda',dtype=torch.bool)
 assert torch.equal(B[~valid],A[~valid]);FAB=torch.where(valid[:,None],(A+B)/2,A)
 states,traces,grads,checks=fit129(base,m,S0,z,d,FAB,late,res,groups,scope,meta,radius,jread(H129/'SOURCE_CALIBRATION.json')['unknown_KL_epsilon'],engineering)
 # The original global KL states are reused, not refit, on unchanged eligibility.
 if meta['scope_eligible'] and meta['old_R127_global_active']:
  p=R129/f"crossplatform/round127/formal/{e['cohort']}/{e['source']}_to_{e['target']}/states_PRIVATE/{e['row']['id']}.npz"
  with np.load(p) as f:states['global_scope']=f['fab_full'].copy()
 elif meta['scope_eligible']:states['global_scope']=states.pop('global_new_eligibility')
 else:states['global_scope']=np.zeros(4896,np.float32)
 stamp=time.monotonic();assert set(states)==set(SET129+['global_scope'])
 # Only now may held-out C or previous C readouts be decoded.
 with np.load(R129/e['observations']) as f:
  pre='real_C_' if e['cohort']=='DEV' else 'C_';available=pre+'vis' in f;heldout=torch.as_tensor(f[pre+'vis'],device='cuda') if available else A
 scores={};ff={};write=[];epsilon=jread(H129/'SOURCE_CALIBRATION.json')['unknown_KL_epsilon']
 oldroot=R129/f"crossplatform/round127/formal/{e['cohort']}/{e['source']}_to_{e['target']}"
 oldfeat=np.load(oldroot/'features_PRIVATE'/(e['row']['id']+'.npz'))
 oldpred=np.load(oldroot/'samples_PRIVATE'/(e['row']['id']+'.npz'));oldp=dict(zip(oldpred['arm_names'],oldpred['scores']))
 for setting in ['zero']+SET129+['global_scope']:
  restore(m,S0)
  if setting!='zero':put123(m,S0,torch.as_tensor(states[setting],device='cuda'))
  freeze91(m);changed=[k for k,v in state(m).items() if not np.array_equal(v,S0[k])];assert set(changed)<=set(NAMES);write.append({'setting':setting,'actual_parameter_L2':float(flat123(m).double().norm()),'changed_tensors':len(changed)})
  for v,P in [('F',FAB),('C',heldout)]:
   reuse_zero=setting=='zero' or not np.any(states[setting])
   reuse_global=setting=='global_scope' and meta['old_R127_global_active'] and meta['scope_eligible']
   if reuse_zero or reuse_global:
    key=('zero' if reuse_zero else 'fab_full')+'_'+v
    h=torch.as_tensor(oldfeat[key+'_h'],device='cuda');u=torch.as_tensor(oldfeat[key+'_u'],device='cuda');lp=torch.as_tensor(oldp[key],device='cuda').clamp_min(1e-300).log()
   else:
    with torch.no_grad():lp,h,u=body127(m,base,z,d,P,late,res,True)
   p=lp.exp().cpu().numpy();scores[setting+'_'+v]=p;ff[setting+'_'+v+'_h']=h.cpu().numpy();ff[setting+'_'+v+'_u']=u.cpu().numpy()
   if setting=='zero':
    scores['direct_conditional_'+v]=direct(p,groups,scope['S'],scope['C'],scope['teacher_conditional'],'conditional');scores['direct_partial_'+v]=direct(p,groups,scope['S'],scope['C'],scope['teacher_conditional'],'partial')
   if setting=='partial_guard' and v=='F':
    qs=torch.as_tensor(scope['S'][scope['assignment']],device='cuda');ref=torch.as_tensor(scope['pF'],device='cuda');uk=float(keep_kl(collapse(ref,qs),collapse(lp.exp(),qs)));assert uk<=epsilon+1e-11
 oldfeat.close();oldpred.close();restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());return scores,ff,states|{k+'_gradient':v for k,v in grads.items()},{'scope':meta,'traces':traces,'writebacks':write,'finite_difference':checks,'source_A_provider':provider,'parameter_radius':radius,'scope_parameters':4896,'B_available':present,'C_available':available,'all_states_fixed_before_C_monotonic':stamp,'C_used_in_gradient_or_gate':False,'new_source_training':False,'global_same_scope_reused':bool(meta['scope_eligible'] and meta['old_R127_global_active'])}
