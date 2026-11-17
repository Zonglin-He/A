"""Two-factor fit-input/readout intervention; same actual FAB full readout serves.

Existing source-A late RGB and competition values are fixed in ALL full readers.
Their addition, normalization and native text matching remain in the gradient.
No source GT, external expert, new boxes, C fitting or support-rule additions.
"""
from pathlib import Path
import sys
R127=Path(__file__).resolve().parents[2];H127=R127/'research/round127';B127=R127/'crossplatform/round127'
sys.path.insert(0,str(R127/'research/round125'));from core125 import *
SET127=['ab_native','ab_full','fab_native','fab_full','wrong_fab_full','random']
NEW127=['zero_F','zero_C','direct_full_F','direct_full_C','direct_native_bridge_F','direct_native_bridge_C']+[s+'_'+v for s in SET127 for v in ['F','C']]
REF127=['original_source','prepared_A','legacy_ABC_context'];ARMS127=REF127+NEW127;PRIMARY127='fab_full_F'

def source_context127(e):
 co=e['cohort'];name=e['source']+'_to_'+e['target'];sid=e['row']['id']
 p=R127/('crossplatform/round113' if co=='DEV' else 'crossplatform/round109')/name/'features_PRIVATE'/(sid+'.npz')
 with np.load(p) as f:
  pre='frozen_full_A_' if co=='DEV' else 'real_A_'
  late=f[pre+'late'];late=late[0] if co=='DEV' else late
  res=f[pre+'r'].copy()
 return torch.as_tensor(late.copy(),device='cuda'),torch.as_tensor(res,device='cuda'),{'path':str(p.relative_to(R127)),'sha256':sha(p),'provider':'existing_source_prepared_A_only_no_B_or_C'}

def body127(m,base,z,d,P,late,res,full):
 m.route=None;m.use_interface=False;m.variant='full'
 _,_,e=m.main(base,dict(z,vis0=P[None]),d,'bidirectional')
 h,u=e['query_hidden'],e['projected_text'];raw=m.match(h[None])[0]
 q=F.normalize(raw+(late if full else 0),dim=-1)
 logits=native_scores(q,u,z['token_mask'].bool()).double()+(res.double() if full else 0)
 return logits.log_softmax(0),h,u

def loss127(m,base,z,d,views,target,late,res,full):
 outputs=[body127(m,base,z,d,V,late,res,full) for V in views]
 avglog=torch.stack([x[0] for x in outputs]).mean(0)
 kl=(target*(target.clamp_min(1e-300).log()-avglog)).sum()
 keep=torch.stack([(F.normalize(m.point_adapter(V),dim=-1)-F.normalize(V,dim=-1)).square().sum(-1).mean()/2 for V in views]).mean()
 return kl+keep,kl,keep,outputs

def fit127(base,m,S0,z,d,A,B,FAB,late,res,active,radius,salt,engineering):
 restore(m,S0);saved={};teachers={}
 with torch.no_grad():
  for mode,full in [('native',False),('full',True)]:
   lp=[]
   for v,P in [('A',A),('B',B)]:
    l,h,u=body127(m,base,z,d,P,late,res,full);lp.append(l.exp())
    saved['teacher_'+v+'_h']=h.cpu().numpy();saved['teacher_'+v+'_u']=u.cpu().numpy()
   teachers[mode]=torch.stack(lp).mean(0).detach();saved['teacher_'+mode]=teachers[mode].cpu().numpy()
 cycle=np.random.default_rng(int(hashlib.sha256((salt+'|wrong_teacher').encode()).hexdigest()[:16],16)).permutation(256)
 perm=np.empty(256,int);perm[cycle]=np.roll(cycle,1);saved['teacher_permutation']=perm
 states={};grads={};traces={};finite=[];fitout={};ps=[dict(m.named_parameters())[n] for n in NAMES]
 for kind in SET127[:-1]:
  views=[A,B] if kind.startswith('ab_') else [FAB];full=not kind.endswith('native')
  target=teachers['full' if full else 'native'];target=target[torch.as_tensor(perm,device='cuda')] if kind.startswith('wrong_') else target
  delta=torch.zeros(4896,device='cuda');restore(m,S0);steps=[]
  for t in range(3):
   put123(m,S0,delta)
   for p in ps:p.requires_grad_(active)
   L,kl,keep,outs=loss127(m,base,z,d,views,target,late,res,full)
   if active:
    gg=torch.autograd.grad(L,ps,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(ps,gg)])
   else:g=torch.zeros_like(delta)
   assert torch.isfinite(g).all();grads[kind+'_step'+str(t+1)+'_gradient']=g.cpu().numpy()
   before=float(L.detach());old=delta.clone();direction=-radius*g/g.norm() if g.norm()>0 else g*0
   for p in ps:p.requires_grad_(False)
   if engineering and active and t==0 and kind=='fab_full' and g.norm()>1e-7:
    eps=radius*.001;unit=g/g.norm();vals=[]
    for sign in [-1,1]:
     put123(m,S0,old+sign*eps*unit)
     with torch.no_grad():vals.append(float(loss127(m,base,z,d,views,target,late,res,full)[0]))
    fd=(vals[1]-vals[0])/(2*eps);rel=abs(fd-float(g.norm()))/float(g.norm());assert rel<.2,(fd,float(g.norm()),rel)
    finite.append({'kind':kind,'directional_derivative':fd,'autograd_directional':float(g.norm()),'relative_error':rel,'epsilon':eps})
   accepted=False;fraction=0.;trials=[]
   if active and g.norm()>0:
    for j in range(9):
     candidate=old+direction/(2**j);put123(m,S0,candidate)
     with torch.no_grad():ll,kk,pp,_=loss127(m,base,z,d,views,target,late,res,full);value=float(ll)
     rhs=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;ok=value<=rhs
     trials.append({'fraction':2.**(-j),'objective':value,'bound':rhs,'accepted':ok})
     if ok:delta=candidate;accepted=True;fraction=2.**(-j);break
   if not accepted:delta=old
   put123(m,S0,delta)
   with torch.no_grad():ll,kk,pp,out=loss127(m,base,z,d,views,target,late,res,full)
   steps.append({'before':before,'after':float(ll),'KL_after':float(kk),'keep_after':float(pp),'accepted':accepted,'fraction':fraction,'gradient_L2':float(g.norm()),'cumulative_L2':float(delta.norm()),'trials':trials})
  states[kind]=delta.cpu().numpy();traces[kind]=steps
  for v,(_,h,u) in zip(['A','B'] if len(views)==2 else ['F'],out):
   fitout[kind+'_fit_'+v+'_h']=h.cpu().numpy();fitout[kind+'_fit_'+v+'_u']=u.cpu().numpy()
 random=torch.as_tensor(np.random.default_rng(int(hashlib.sha256((salt+'|norm_random').encode()).hexdigest()[:16],16)).normal(size=4896),device='cuda',dtype=torch.float32)
 states['random']=(random/random.norm()*torch.as_tensor(states['fab_full'],device='cuda').norm()).cpu().numpy()
 restore(m,S0);return states,grads,traces,saved|fitout,finite

def process127(base,m,S,e,engineering=False):
 restore(m,S);S0,radius=start123(m,S);z0,z,A,B,ids,far,_,present,salt=a_b123(e)
 active=bool(len(ids)>=2 and far.any());d=geomdata125(e,z0);late,res,provider=source_context127(e)
 # B unsupported locations already exactly retain A. Validate this, not merely assume.
 with np.load(R127/e['observations']) as ob:
  pre='real_B_' if e['cohort']=='DEV' else 'B_';valid=torch.as_tensor(ob[pre+'valid'],device='cuda') if present else torch.zeros(1024,device='cuda',dtype=torch.bool)
 assert torch.equal(B[~valid],A[~valid]);FAB=torch.where(valid[:,None],(A+B)/2,A)
 states,grads,traces,ff,finite=fit127(base,m,S0,z,d,A,B,FAB,late,res,active,radius,salt.replace('R123','R127'),engineering)
 stamp=time.monotonic();assert set(states)==set(SET127)
 # All edits are now fixed: C is opened only here, never through old ABC providers.
 with np.load(R127/e['observations']) as ob:
  pre='real_C_' if e['cohort']=='DEV' else 'C_';available=pre+'vis' in ob;C=torch.as_tensor(ob[pre+'vis'],device='cuda') if available else A
 ff.update(fixed_late=late.cpu().numpy(),fixed_r=res.cpu().numpy(),mask=z['token_mask'].cpu().numpy())
 scores={};write=[]
 for setting in ['zero']+SET127:
  restore(m,S0)
  if setting!='zero':put123(m,S0,torch.as_tensor(states[setting],device='cuda'))
  freeze91(m);actual=state(m);assert {k for k in S0 if not np.array_equal(S0[k],actual[k])}<=set(NAMES)
  write.append({'setting':setting,'parameter_L2':float(flat123(m).double().norm())})
  for v,P in [('F',FAB),('C',C)]:
   with torch.no_grad():lp,h,u=body127(m,base,z,d,P,late,res,True)
   ff[setting+'_'+v+'_h']=h.cpu().numpy();ff[setting+'_'+v+'_u']=u.cpu().numpy();scores[('zero' if setting=='zero' else setting)+'_'+v]=lp.exp().cpu().numpy()
   if setting=='zero':
    raw=m.match(h[None])[0];native=native_scores(F.normalize(raw,dim=-1),u,z['token_mask'].bool()).double().log_softmax(0)
    scores['direct_native_bridge_'+v]=(torch.as_tensor(ff['teacher_native'],device='cuda').clamp_min(1e-300).log()+lp-native).softmax(0).detach().cpu().numpy()
    scores['direct_full_'+v]=ff['teacher_full'].copy()
   if engineering and setting=='fab_full' and v=='F':
    with torch.no_grad():injected=m.point_adapter(P).detach()
    restore(m,S0)
    with torch.no_grad():_,hh,uu=body127(m,base,z,d,injected,late,res,True)
    assert torch.equal(h,hh) and torch.equal(u,uu),'actual weight edit versus equivalent point injection'
    put123(m,S0,torch.as_tensor(states[setting],device='cuda'))
 restore(m,S);freeze91(m);assert all(np.array_equal(state(m)[k],v) for k,v in S.items())
 return scores,ff,states|grads,{'fit':{'active':active,'fit_point_pairs':len(ids),'parameter_radius':radius},'B_available':present,'C_available':available,'source_A_provider':provider,'traces':traces,'writebacks':write,'finite_difference':finite,'all_states_before_C_monotonic':stamp,'C_arrays_used_in_gradient':False,'final_input':'registered FAB; no C in F','full_readout_context':'fixed existing source-A late and r, both in differentiable logit computation','scope_parameters':4896,'no_source_GT_used':True}
