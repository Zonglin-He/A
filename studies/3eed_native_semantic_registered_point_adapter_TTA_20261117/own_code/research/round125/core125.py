"""Task-native registered-view consistency at existing pointUp4896, no task labels."""
from pathlib import Path
import sys
R125=Path(__file__).resolve().parents[2];H125=R125/'research/round125';B125=R125/'crossplatform/round125'
sys.path.insert(0,str(R125/'research/round123'));from core123 import *
SET125=['corr1','corr3','wrong1','wrong3','random1','random3','free3'];NEW125=[s+'_'+v for s in SET125 for v in ['F','C']]+['direct_F','direct_C'];ARMS125=REF123+NEW125;PRIMARY125='corr3_F';assert len(ARMS125)==27
def body125(m,base,z,d,P):
 m.route=None;m.use_interface=False;m.variant='full';_,_,e=m.main(base,dict(z,vis0=P[None]),d,'bidirectional');h,u=e['query_hidden'],e['projected_text'];q=F.normalize(m.match(h[None])[0],dim=-1);return native_scores(q,u,z['token_mask'].bool()).double().log_softmax(0),h,u
def objective125(m,base,z,d,A,B,T,RA,RB,free=False):
 la,ha,ua=body125(m,base,z,d,A);lb,hb,ub=body125(m,base,z,d,B);kl=(T*(T.clamp_min(1e-300).log()-(la+lb)/2)).sum();aa,bb=(A,B) if free else (m.point_adapter(A),m.point_adapter(B));keep=((F.normalize(aa,dim=-1)-RA).square().sum(-1).mean()+(F.normalize(bb,dim=-1)-RB).square().sum(-1).mean())/4;return kl+keep,kl,keep,(ha,ua,hb,ub)
def geomdata125(e,z0):
 key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();f=B97/'shared_physical_geometry_PRIVATE'/(e['row']['id']+'_'+key[:20]+'.npz');assert jread(f.with_suffix('.json'))['geometry_seed_and_sample_sha256']==key;g=load(f);g['query_anchor_indices']=z0['query_points_sample_inds'];return tensor_data(g,e['row'])
def fit125(base,m,S0,z,d,A,B,active,radius,salt,engineering):
 restore(m,S0);RA=F.normalize(A,dim=-1).detach();RB=F.normalize(B,dim=-1).detach()
 with torch.no_grad():
  la,ha,ua=body125(m,base,z,d,A);lb,hb,ub=body125(m,base,z,d,B);T=((la.exp()+lb.exp())/2).detach();teacher={'A_h':ha.cpu().numpy(),'A_u':ua.cpu().numpy(),'B_h':hb.cpu().numpy(),'B_u':ub.cpu().numpy(),'probability':T.cpu().numpy()}
 cycle=np.random.default_rng(int(hashlib.sha256((salt+'|querywrong').encode()).hexdigest()[:16],16)).permutation(256);perm=np.empty(256,int);perm[cycle]=np.roll(cycle,1);wrong=T[torch.as_tensor(perm,device='cuda')];states={};grads={};traces={};energies=[];ps=[dict(m.named_parameters())[n] for n in NAMES];finite=None
 if not active:
  zero=torch.zeros(4896,device='cuda');field=torch.zeros_like(A)
  for k in ['corr1','corr3','wrong1','wrong3','random1','random3']:states[k]=zero.clone()
  for k in ['free3','free_A_delta','free_B_delta']:states[k]=field.clone()
  for kind,target in [('corr',T),('wrong',wrong)]:
   before=float((target*(target.clamp_min(1e-300).log()-(la+lb)/2)).sum());traces[kind]=[{'before':before,'after':before,'KL_after':before,'keep_after':0.,'accepted':False,'fraction':0.,'gradient_L2':0.,'cumulative_L2':0.,'induced_feature_step_L2':0.,'trials':[]} for _ in range(3)]
   for t in range(3):grads[kind+str(t+1)]=zero.clone()
  traces['free']=[{'before':traces['corr'][0]['before'],'accepted':False,'fraction':0.,'feature_gradient_L2':0.,'feature_step_energy_cap':0.,'trials':[]} for _ in range(3)]
  for t in range(3):
   for v in ['A','B']:grads['free'+str(t+1)+'_'+v]=field.clone()
  return states,grads,traces,teacher,perm,finite
 for kind,target in [('corr',T),('wrong',wrong)]:
  delta=torch.zeros(4896,device='cuda');steps=[];restore(m,S0);prevA,prevB=A,B
  for t in range(3):
   put123(m,S0,delta)
   for p in ps:p.requires_grad_(True)
   L,k,preserve,_=objective125(m,base,z,d,A,B,target,RA,RB);gg=torch.autograd.grad(L,ps,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(ps,gg)]);assert torch.isfinite(g).all();grads[kind+str(t+1)]=g.clone();direction=-radius*g/g.norm() if active and g.norm()>0 else g*0;before=float(L.detach())
   for p in ps:p.requires_grad_(False)
   if engineering and kind=='corr' and t==0 and active and g.norm()>1e-7:
    eps=radius*.001;unit=g/g.norm();vals=[]
    for sign in [-1,1]:
     put123(m,S0,sign*eps*unit)
     with torch.no_grad():ll,_,_,_=objective125(m,base,z,d,A,B,target,RA,RB);vals.append(float(ll))
    fd=(vals[1]-vals[0])/(2*eps);rel=abs(fd-float(g.norm()))/float(g.norm());assert rel<.2,(fd,float(g.norm()),rel);finite={'directional_derivative':fd,'autograd_directional':float(g.norm()),'relative_error':rel,'epsilon':eps}
   old=delta.clone();accepted=False;trials=[];fraction=0.
   if active and g.norm()>0:
    for j in range(9):
     candidate=old+direction/(2**j);put123(m,S0,candidate)
     with torch.no_grad():ll,kk,pp,_=objective125(m,base,z,d,A,B,target,RA,RB);val=float(ll)
     ok=val<=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;trials.append({'fraction':2.**(-j),'objective':val,'accepted':ok})
     if ok:delta=candidate;accepted=True;fraction=2.**(-j);break
   if not accepted:delta=old
   put123(m,S0,delta)
   with torch.no_grad():
    aa=m.point_adapter(A);bb=m.point_adapter(B);energy=float(torch.sqrt((aa-prevA).double().square().sum()+(bb-prevB).double().square().sum()));prevA,prevB=aa.clone(),bb.clone();ll,kk,pp,_=objective125(m,base,z,d,A,B,target,RA,RB)
   steps.append({'before':before,'after':float(ll),'KL_after':float(kk),'keep_after':float(pp),'accepted':accepted,'fraction':fraction,'gradient_L2':float(g.norm()),'cumulative_L2':float(delta.norm()),'induced_feature_step_L2':energy,'trials':trials})
   if kind=='corr':energies.append(energy)
   if t in [0,2]:states[kind+str(t+1)]=delta.clone()
  traces[kind]=steps
 for t in [1,3]:
  v=torch.as_tensor(np.random.default_rng(int(hashlib.sha256((salt+'|random|'+str(t)).encode()).hexdigest()[:16],16)).normal(size=4896),device='cuda',dtype=torch.float32);states['random'+str(t)]=v/v.norm()*states['corr'+str(t)].norm()
 restore(m,S0);XA=A.clone();XB=B.clone();steps=[]
 for t,energy in enumerate(energies):
  XA.requires_grad_(True);XB.requires_grad_(True);L,kl,keep,_=objective125(m,base,z,d,XA,XB,T,RA,RB,True);ga,gb=torch.autograd.grad(L,[XA,XB]);grads['free'+str(t+1)+'_A']=ga.detach().clone();grads['free'+str(t+1)+'_B']=gb.detach().clone();norm=float(torch.sqrt(ga.double().square().sum()+gb.double().square().sum()));factor=energy/norm if norm else 0.;oldA,oldB=XA.detach(),XB.detach();accepted=False;fraction=0.;trials=[];before=float(L.detach())
  if energy>0 and norm>0:
   for j in range(9):
    ca=oldA-factor*ga.detach()/(2**j);cb=oldB-factor*gb.detach()/(2**j)
    with torch.no_grad():ll,_,_,_=objective125(m,base,z,d,ca,cb,T,RA,RB,True);val=float(ll)
    rhs=before+1e-4*float((ga.double()*(ca-oldA).double()).sum()+(gb.double()*(cb-oldB).double()).sum())+1e-12;ok=val<=rhs;trials.append({'fraction':2.**(-j),'objective':val,'accepted':ok})
    if ok:XA,XB=ca.detach(),cb.detach();accepted=True;fraction=2.**(-j);break
  if not accepted:XA,XB=oldA,oldB
  steps.append({'before':before,'accepted':accepted,'fraction':fraction,'feature_gradient_L2':norm,'feature_step_energy_cap':energy,'trials':trials})
 states['free3']=((XA-A)+(XB-B))/2;states['free_A_delta']=XA-A;states['free_B_delta']=XB-B;traces['free']=steps;restore(m,S0);return states,grads,traces,teacher,perm,finite
def process125(base,m,S,e,engineering=False):
 restore(m,S);S0,radius=start123(m,S);z0,z,A,B,ids,far,_,present,salt=a_b123(e);active=bool(len(ids)>=2 and far.any());d=geomdata125(e,z0);states,grads,traces,teacher,perm,finite=fit125(base,m,S0,z,d,A,B,active,radius,salt.replace('R123','R125'),engineering);stamp=time.monotonic();d,zz,N,SS,obs,proof=after123(e,z0,z);assert np.array_equal(teacher['A_h'],N['A']['h']) and np.array_equal(teacher['A_u'],N['A']['u']) and np.array_equal(teacher['B_h'],N['B']['h']) and np.array_equal(teacher['B_u'],N['B']['u']);late=torch.as_tensor(SS['A']['late'],device='cuda');res=torch.as_tensor(SS['fusedABC']['r'],device='cuda');T=torch.as_tensor(teacher['probability'],device='cuda');ff={'fixed_late':SS['A']['late'],'fixed_r':SS['fusedABC']['r'],'teacher_probability':teacher['probability'],'teacher_A_h':teacher['A_h'],'teacher_A_u':teacher['A_u'],'teacher_B_h':teacher['B_h'],'teacher_B_u':teacher['B_u'],'teacher_permutation':perm};scores={};aux={};write=[];RA=F.normalize(A,dim=-1);RB=F.normalize(B,dim=-1)
 zero_scores={}
 for setting,delta in [('zero',None)]+[(k,states[k]) for k in SET125]:
  restore(m,S0)
  if setting not in ['zero','free3']:put123(m,S0,delta)
  freeze91(m);actual=state(m);assert {k for k in S0 if not np.array_equal(S0[k],actual[k])}<=set(NAMES);write.append({'setting':setting,'actual_parameter_L2':float(flat123(m).double().norm()),'field_L2':float(delta.double().norm()) if setting=='free3' else 0.})
  if not active and not engineering and setting!='zero':
   aux[setting]=dict(aux['zero'])
   for k in ['A_h','A_u','B_h','B_u']:ff[setting+'_fit_'+k]=ff['zero_fit_'+k]
   for v in ['F','C']:
    for k in ['h','u','mask']:ff[setting+'_'+v+'_'+k]=ff['zero_'+v+'_'+k]
    scores[setting+'_'+v]=zero_scores[v].copy()
   continue
  with torch.no_grad():
   aa=A+states['free_A_delta'] if setting=='free3' else A;bb=B+states['free_B_delta'] if setting=='free3' else B;L,kl,keep,aaout=objective125(m,base,z,d,aa,bb,T,RA,RB,setting=='free3');aux[setting]={'fit_objective':float(L),'fit_KL':float(kl),'fit_keep':float(keep)}
   for k,v in zip(['A_h','A_u','B_h','B_u'],aaout):ff[setting+'_fit_'+k]=v.cpu().numpy()
   if 'C' in obs:
    cc=obs['C']['vis']+(delta if setting=='free3' else 0);lc,_,_=body125(m,base,z,d,cc);aux[setting]['heldout_C_KL']=float((T*(T.clamp_min(1e-300).log()-lc)).sum())
   for v,vz in zz.items():
    if setting=='free3':vz=dict(vz,vis0=vz['vis0']+delta[None])
    h,u=native123(base,m,vz,d);q=F.normalize(m.match(h[None])[0]+late,dim=-1);logits=native_scores(q,u,z['token_mask'].bool()).double()+res.double();p=logits.softmax(0);ff[setting+'_'+v+'_h']=h.cpu().numpy();ff[setting+'_'+v+'_u']=u.cpu().numpy();ff[setting+'_'+v+'_mask']=z['token_mask'].cpu().numpy()
    if setting=='zero':
     zero_scores[v]=p.cpu().numpy()
     assert np.array_equal(h.cpu().numpy(),N['fusedABC' if v=='F' else 'C']['h']) and np.array_equal(u.cpu().numpy(),N['fusedABC' if v=='F' else 'C']['u']);native=native_scores(F.normalize(m.match(h[None])[0],dim=-1),u,z['token_mask'].bool()).double();scores['direct_'+v]=(T.clamp_min(1e-300).log()+logits-native).softmax(0).cpu().numpy()
    else:scores[setting+'_'+v]=p.cpu().numpy()
 restore(m,S);freeze91(m);assert all(np.array_equal(state(m)[k],v) for k,v in S.items());return scores,ff,{k:v.cpu().numpy() for k,v in (states|{k+'_gradient':v for k,v in grads.items()}).items()},dict(proof,fit={'active':active,'fit_point_pairs':len(ids),'parameter_radius':radius,'no_C_decoded_before_all_states_fixed':True},traces=traces,auxiliary=aux,writebacks=write,finite_difference=finite,all_states_before_C_monotonic=stamp,scope_parameters=4896,no_source_GT_used=True)
