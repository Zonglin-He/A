"""One frozen LOG trajectory: decode1, inherit3, append unchanged steps4/5."""
from common133 import *
import sys
sys.path.insert(0,str(R133/'research/round132'));from core132 import *
from common133 import R133,H133,B133,rd,dump,sha,utc,basecheck
ARMS133=['zero_F','direct_F','partial1_F','partial3_F','partial5_F','wrong1_F','wrong3_F','wrong5_F','R131_partial_F']
def reference133(e):
 name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=R133/'crossplatform/round132/formal'/name
 with np.load(D/'samples_PRIVATE'/(sid+'.npz')) as a:p=dict(zip(a['arm_names'],a['scores']));boxes=a['source_boxes'].copy()
 return {'zero_F':p['LOG_zero'],'direct_F':p['LOG_direct'],'partial3_F':p['LOG_partial'],'wrong3_F':p['LOG_wrong'],'R131_partial_F':p['partial_F']},boxes
def append133(base,m,S0,z,d,FAB,late,res,groups,support,C,T,radius,start):
 params=[dict(m.named_parameters())[n] for n in NAMES];Sg=torch.as_tensor(support,device='cuda');Cg=torch.as_tensor(C,device='cuda');Tg=torch.as_tensor(T,device='cuda');delta=torch.as_tensor(start.copy(),device='cuda');traces=[];gradients={}
 for it in [4,5]:
  put123(m,S0,delta)
  for p in params:p.requires_grad_(True)
  L,task,keep,_,_,_=objective129(base,m,z,d,FAB,late,res,groups,Sg,Cg,Tg,'partial');before=float(L.detach());gg=torch.autograd.grad(L,params,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(params,gg)]);assert torch.isfinite(g).all();gradients['step'+str(it)+'_gradient']=g.cpu().numpy();old=delta.clone();direction=-radius*g/g.norm() if float(g.norm())>1e-12 and float(task.detach())>1e-12 else g*0
  for p in params:p.requires_grad_(False)
  trials=[];accepted=False;fraction=0.
  if direction.norm()>0:
   for k in range(9):
    candidate=old+direction/(2**k);put123(m,S0,candidate)
    with torch.no_grad():ll,tt,pp,_,_,_=objective129(base,m,z,d,FAB,late,res,groups,Sg,Cg,Tg,'partial');value=float(ll)
    ok=value<=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;trials.append({'fraction':2.**(-k),'objective':value,'Armijo':ok})
    if ok:delta=candidate;accepted=True;fraction=2.**(-k);break
  if not accepted:delta=old
  put123(m,S0,delta)
  with torch.no_grad():ll,tt,pp,_,_,_=objective129(base,m,z,d,FAB,late,res,groups,Sg,Cg,Tg,'partial')
  traces.append({'step':it,'before':before,'after':float(ll),'task_after':float(tt),'feature_keep_after':float(pp),'gradient_L2':float(g.norm()),'parameter_L2':float(delta.norm()),'accepted':accepted,'fraction':fraction,'trials':trials})
 return delta.cpu().numpy(),traces,gradients
def process133(base,m,S,e,anchors,pairs,engineering=False):
 scores,boxes=reference133(e);sid=e['row']['id'];name=e['source']+'_to_'+e['target']
 if not e['active']:
  for kind in ['partial','wrong']:
   for step in [1,5]:scores[kind+str(step)+'_F']=scores['zero_F'].copy()
  return scores,None,None,{'active':False,'no_new_model_compute':True,'reference_zero':True,'trajectories':{}},boxes
 assert sha(R133/e['provider_states'])==e['provider_states_sha256'] and sha(R133/e['provider_features'])==e['provider_features_sha256'] and sha(R133/e['reconstructed'])==e['reconstructed_sha256'];tick=time.perf_counter();restore(m,S);D=R133/'crossplatform/round131/formal'/name;z0=load(D/'prefix_PRIVATE'/(sid+'.npz'));assert np.array_equal(z0['boxes'],boxes);z=tensor_prefix(z0);d=geomdata125(e,z0);oldf=load(R133/e['provider_features']);a=anchors[e['frame_PRIVATE']];late=torch.as_tensor(oldf['fixed_late'],device='cuda');res=torch.as_tensor(oldf['fixed_r'],device='cuda');valid=torch.as_tensor(oldf['valid_B'],device='cuda');A=z['vis0'][0]
 if 'B_vis' in oldf:B=torch.as_tensor(oldf['B_vis'],device='cuda');provider='cached_R132_B_vis'
 else:
  pair=pairs[a['B_pair']];regpath=str((R133/'crossplatform/round131/registration_PRIVATE'/(a['B_pair']+'.json')).relative_to(R133));obs,proof=extra132(base,a['A'],pair,z0,regpath);assert obs is not None and torch.equal(obs['valid'],valid);B=obs['vis'];provider='recomputed_original_B_identical_sampling'
 assert torch.equal(B[~valid],A[~valid]);FAB=torch.where(valid[:,None],(A+B)/2,A);S0,radius=start123(m,S);record=e['provider_record'];assert radius==record['parameter_radius'];ass=oldf['assignment'];groups=[np.flatnonzero(ass==j) for j in range(ass.max()+1)];withstate=load(R133/e['reconstructed']);scope=np.load(R133/'crossplatform/round132/log_scopes_PRIVATE'/name/(sid+'.npz'));assert np.array_equal(scope['S'],oldf['S']) and np.array_equal(scope['C'],oldf['C']);ff={'fixed_late':oldf['fixed_late'],'fixed_r':oldf['fixed_r'],'mask':oldf['mask']};states={};traces={};timing={'prefix_and_B_seconds':time.perf_counter()-tick};bridges={}
 for label in ['partial','wrong']:
  key='wrong_base' if e['reference_original_R131'] and label=='wrong' else label;oldtr=record['traces'][key];states[label+'1']=withstate[label+'1'];states[label+'3']=withstate[label+'3'];traces[label]=oldtr.copy()
  if label=='wrong' and record['identity_control']:
   for step in [1,5]:states[label+str(step)]=states['partial'+str(step)].copy();scores[label+str(step)+'_F']=scores['partial'+str(step)+'_F'];ff[label+str(step)+'_h']=ff['partial'+str(step)+'_h'];ff[label+str(step)+'_u']=ff['partial'+str(step)+'_u']
   traces[label]=traces['partial'];timing[label]={'identity_reused':True};continue
  ti=time.perf_counter();restore(m,S0);put123(m,S0,torch.as_tensor(states[label+'1'],device='cuda'));freeze91(m)
  with torch.no_grad():lp,h,u=body127(m,base,z,d,FAB,late,res,True)
  scores[label+'1_F']=lp.exp().cpu().numpy();ff[label+'1_h']=h.cpu().numpy();ff[label+'1_u']=u.cpu().numpy();t1=time.perf_counter()-ti
  if engineering:
   restore(m,S0);put123(m,S0,torch.as_tensor(states[label+'3'],device='cuda'));freeze91(m)
   with torch.no_grad():lp,_,_=body127(m,base,z,d,FAB,late,res,True)
   error=float(abs(lp.exp().cpu().numpy()-scores[label+'3_F']).max());assert error<1e-7;assert lp.argmax().item()==int(scores[label+'3_F'].argmax());bridges[label]=error
  ti=time.perf_counter();restore(m,S0);C=oldf['C'] if label=='partial' else oldf['wrong_C'];delta,tr,gg=append133(base,m,S0,z,d,FAB,late,res,groups,oldf['S'],C,oldf['teacher_conditional'],radius,states[label+'3']);t45=time.perf_counter()-ti;states[label+'5']=delta;traces[label]+=tr;states.update({label+'_'+k:v for k,v in gg.items()});ti=time.perf_counter();restore(m,S0);put123(m,S0,torch.as_tensor(delta,device='cuda'));freeze91(m)
  with torch.no_grad():lp,h,u=body127(m,base,z,d,FAB,late,res,True)
  scores[label+'5_F']=lp.exp().cpu().numpy();ff[label+'5_h']=h.cpu().numpy();ff[label+'5_u']=u.cpu().numpy();timing[label]={'decode1_seconds':t1,'append4and5_seconds':t45,'decode5_seconds':time.perf_counter()-ti};assert {k for k,v in state(m).items() if not np.array_equal(v,S0[k])}<=set(NAMES)
 restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());return scores,ff,states,{'active':True,'identity_control':record['identity_control'],'radius':radius,'B_provider':provider,'trajectories':traces,'timing':timing,'engineering_step3_bridge':bridges,'no_new_GT_or_observation':True},boxes
