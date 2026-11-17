"""Same partial task, fixed S, no unknown-output guard; bounded step factors."""
from pathlib import Path
import sys
R130=Path(__file__).resolve().parents[2];H130=R130/'research/round130';B130=R130/'crossplatform/round130'
sys.path.insert(0,str(R130/'research/round129'));from core129 import *
SET130=['proper_half','proper_double','wrong_half','wrong_base','wrong_double'];FACTORS130={'proper_half':.5,'proper_double':2.,'wrong_half':.5,'wrong_base':1.,'wrong_double':2.}
REF130=['original_source','prepared_A','legacy_ABC_context','zero_F','zero_C','direct_partial_F','direct_partial_C','partial_F','partial_C','R129_guard_F','R129_guard_C'];NEW130=[s+'_'+v for s in SET130 for v in ['F','C']]+['direct_wrong_F','direct_wrong_C'];ARMS130=REF130+NEW130
def control130(e):
 p=B130/'control_PRIVATE'/e['cohort']/(e['source']+'_to_'+e['target'])/(e['row']['id']+'.npz');a=load(p);return a
def fit_one130(base,m,S0,z,d,FAB,late,res,groups,scope,C_np,radius,engineering,setting):
 restore(m,S0);params=[dict(m.named_parameters())[n] for n in NAMES];S=torch.as_tensor(scope['S'],device='cuda');C=torch.as_tensor(C_np,device='cuda');T=torch.as_tensor(scope['teacher_conditional'],device='cuda');delta=torch.zeros(4896,device='cuda');steps=[];grads={};check=None
 for it in range(3):
  put123(m,S0,delta)
  for p in params:p.requires_grad_(True)
  L,task,preserve,lp,_,_=objective129(base,m,z,d,FAB,late,res,groups,S,C,T,'partial');before=float(L.detach());gg=torch.autograd.grad(L,params,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(params,gg)]);assert torch.isfinite(g).all();grads[setting+'_step'+str(it+1)+'_gradient']=g.cpu().numpy();old=delta.clone();direction=-radius*g/g.norm() if float(g.norm())>1e-12 and float(task.detach())>1e-12 else g*0
  for p in params:p.requires_grad_(False)
  if engineering and it==0 and float(g.norm())>1e-7 and setting in ['proper_base_check','wrong_base']:
   eps=radius*.001;unit=g/g.norm();vals=[]
   for sign in [-1,1]:
    put123(m,S0,old+sign*eps*unit)
    with torch.no_grad():vals.append(float(objective129(base,m,z,d,FAB,late,res,groups,S,C,T,'partial')[0]))
   fd=(vals[1]-vals[0])/(2*eps);relative=abs(fd-float(g.norm()))/max(float(g.norm()),1e-12);assert relative<.2;check={'setting':setting,'finite_difference_relative_error':relative}
  trials=[];accepted=False;fraction=0.
  if direction.norm()>0:
   for k in range(9):
    candidate=old+direction/(2**k);put123(m,S0,candidate)
    with torch.no_grad():ll,tt,pp,_,_,_=objective129(base,m,z,d,FAB,late,res,groups,S,C,T,'partial');value=float(ll)
    ok=value<=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;trials.append({'fraction':2.**(-k),'objective':value,'Armijo':ok})
    if ok:delta=candidate;accepted=True;fraction=2.**(-k);break
  if not accepted:delta=old
  put123(m,S0,delta)
  with torch.no_grad():ll,tt,pp,_,_,_=objective129(base,m,z,d,FAB,late,res,groups,S,C,T,'partial')
  steps.append({'before':before,'after':float(ll),'task_after':float(tt),'feature_keep_after':float(pp),'gradient_L2':float(g.norm()),'parameter_L2':float(delta.norm()),'accepted':accepted,'fraction':fraction,'trials':trials})
 restore(m,S0);return delta.cpu().numpy(),steps,grads,check
def process130(base,m,S,e,engineering=False):
 restore(m,S);S0,radius=start123(m,S);z0,z,A,B,ids,far,_,present,salt=a_b123(e);scope,meta,groups=get_scope(e);ctrl=control130(e);assert np.array_equal(ctrl['S'],scope['S']) and np.array_equal(ctrl['C'],scope['C']);effective=not np.array_equal(ctrl['C'],ctrl['wrong_C']);d=geomdata125(e,z0);late,res,provider=source_context127(e)
 with np.load(R130/e['observations']) as f:
  pre='real_B_' if e['cohort']=='DEV' else 'B_';valid=torch.as_tensor(f[pre+'valid'],device='cuda') if present else torch.zeros(1024,device='cuda',dtype=torch.bool)
 assert torch.equal(B[~valid],A[~valid]);FAB=torch.where(valid[:,None],(A+B)/2,A);oldroot=R130/'crossplatform/round129/formal'/e['cohort']/(e['source']+'_to_'+e['target']);sid=e['row']['id'];oldrec=jread(oldroot/'records_PRIVATE'/(sid+'.json'))
 with np.load(oldroot/'states_PRIVATE'/(sid+'.npz')) as f:base_state=f['partial'].copy()
 states={};traces={};grads={};checks=[];reused={}
 if engineering:
  statecheck,_,_,check=fit_one130(base,m,S0,z,d,FAB,late,res,groups,scope,ctrl['C'],radius,True,'proper_base_check');error=float(abs(statecheck-base_state).max());assert error<1e-7,error;checks.append({'R129_original_state_max_error':error});
  if check:checks.append(check)
 for setting in SET130:
  if setting.startswith('wrong') and not effective:
   other={'wrong_half':'proper_half','wrong_base':'partial','wrong_double':'proper_double'}[setting];states[setting]=base_state.copy() if other=='partial' else states[other].copy();traces[setting]=oldrec['traces']['partial'] if other=='partial' else traces[other];reused[setting]=other
  else:
   delta,steps,gg,check=fit_one130(base,m,S0,z,d,FAB,late,res,groups,scope,ctrl['wrong_C'] if setting.startswith('wrong') else ctrl['C'],radius*FACTORS130[setting],engineering,setting);states[setting]=delta;traces[setting]=steps;grads.update(gg)
   if check:checks.append(check)
 stamp=time.monotonic()
 # Every fitted state is fixed before unfitted C or historical C outputs open.
 with np.load(R130/e['observations']) as f:
  pre='real_C_' if e['cohort']=='DEV' else 'C_';available=pre+'vis' in f;heldout=torch.as_tensor(f[pre+'vis'],device='cuda') if available else A
 oldp=np.load(oldroot/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(oldp['arm_names'],oldp['scores']));ff={};scores={};write=[]
 oldfeatures=np.load(oldroot/'features_PRIVATE'/(sid+'.npz'))
 for setting in SET130:
  restore(m,S0);put123(m,S0,torch.as_tensor(states[setting],device='cuda'));freeze91(m);changed=[k for k,v in state(m).items() if not np.array_equal(v,S0[k])];assert set(changed)<=set(NAMES);write.append({'setting':setting,'changed_tensors':len(changed),'parameter_L2':float(flat123(m).double().norm())})
  for v,P in [('F',FAB),('C',heldout)]:
   if setting in reused:
    other=reused[setting]
    if other=='partial':p=po['partial_'+v].copy();h=oldfeatures['partial_'+v+'_h'];u=oldfeatures['partial_'+v+'_u']
    else:p=scores[other+'_'+v].copy();h=ff[other+'_'+v+'_h'];u=ff[other+'_'+v+'_u']
   elif not np.any(states[setting]):p=po['zero_'+v].copy();h=oldfeatures['zero_'+v+'_h'];u=oldfeatures['zero_'+v+'_u']
   else:
    with torch.no_grad():lp,hh,uu=body127(m,base,z,d,P,late,res,True)
    p=lp.exp().cpu().numpy();h=hh.cpu().numpy();u=uu.cpu().numpy()
   scores[setting+'_'+v]=p;ff[setting+'_'+v+'_h']=h;ff[setting+'_'+v+'_u']=u
 for v in ['F','C']:scores['direct_wrong_'+v]=direct(po['zero_'+v],groups,ctrl['S'],ctrl['wrong_C'],scope['teacher_conditional'],'partial')
 oldp.close();oldfeatures.close();restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items())
 return scores,ff,states|grads,{'traces':traces,'writebacks':write,'engineering_checks':checks,'effective_shuffle':effective,'identity_control_reuses_same_update':reused,'base_parameter_radius':radius,'step_factors':FACTORS130,'scope_parameters':4896,'source_A_provider':provider,'all_states_fixed_before_C_monotonic':stamp,'C_used_for_fit_or_gate':False,'unknown_KL_guard':False,'feature_keep_weight':1.,'R129_original_reused':True}
