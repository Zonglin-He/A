"""Retained-candidate ordinal repair through the unchanged full FAB reader."""
from common134 import *
import sys
sys.path.insert(0,str(R134/'research/round133'));from core133 import *
from common134 import R134,H134,B134,rd,dump,sha,utc,basecheck
ARMS134=['zero_F','partial_direct_F','partial3_F','pair_direct_F','partial3_pair_direct_F','wrongpair_direct_F','pair_only_F','partial_pair_F','partial_wrongpair_F']
FITS134=['pair_only','partial_pair','partial_wrongpair']
def reference134(e,ev):
 assert sha(R134/e['original_prediction_path'])==e['original_prediction_sha256']
 with np.load(R134/e['original_prediction_path']) as f:ps=dict(zip(f['arm_names'],f['scores']));boxes=f['source_boxes'].copy()
 out={k:ps[v].copy() for k,v in [('zero_F','zero_F'),('partial_direct_F','direct_F'),('partial3_F','partial3_F')]}
 for k in ['pair_direct_F','partial3_pair_direct_F','wrongpair_direct_F']:out[k]=ev[k].copy()
 return out,boxes
def ordinal_loss134(lp,groups,win,lose,margin):
 gs=torch.stack([lp[torch.as_tensor(g,device=lp.device)].max() for g in groups])
 loss=F.relu(margin-gs[win]+gs[lose]).mean()
 return loss,gs
def objective134(base,m,z,d,FAB,late,res,groups,S,C,win,lose,margin,kind):
 lp,h,u=body127(m,base,z,d,FAB,late,res,True)
 partial=taskloss(lp,groups,S,C,None,'partial') if kind!='pair_only' else lp.sum()*0
 ordinal,gs=ordinal_loss134(lp,groups,win,lose,margin)
 preserve=(F.normalize(m.point_adapter(FAB),dim=-1)-F.normalize(FAB,dim=-1)).square().sum(-1).mean()/2
 return partial+ordinal+preserve,partial,ordinal,preserve,lp,h,u,gs
def fit134(base,m,S0,z,d,FAB,late,res,groups,ev,radius,kind,engineering):
 params=[dict(m.named_parameters())[k] for k in NAMES];scope=torch.as_tensor(ev['S'],device='cuda');C=torch.as_tensor(ev['C'],device='cuda');prefix='wrong_' if kind=='partial_wrongpair' else ''
 win=torch.as_tensor(ev[prefix+'win'],device='cuda');lose=torch.as_tensor(ev[prefix+'lose'],device='cuda');margin=torch.as_tensor(ev['margin'],device='cuda');delta=torch.zeros(4896,device='cuda');traces=[];grads={};finite=[]
 def obj():return objective134(base,m,z,d,FAB,late,res,groups,scope,C,win,lose,margin,kind)
 for step in range(1,4):
  put123(m,S0,delta)
  for p in params:p.requires_grad_(True)
  L,pt,ol,kp,*_=obj();before=float(L.detach());gg=torch.autograd.grad(L,params,allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(params,gg)]);assert torch.isfinite(g).all();grads[kind+'_step'+str(step)+'_gradient']=g.cpu().numpy();old=delta.clone();direction=-radius*g/g.norm() if float(g.norm())>1e-12 and float((pt+ol).detach())>1e-12 else g*0
  for p in params:p.requires_grad_(False)
  if engineering and step==1 and kind=='partial_pair' and float(g.norm())>1e-7:
   eps=radius*.001;vals=[]
   for sign in [-1,1]:
    put123(m,S0,old+sign*eps*g/g.norm())
    with torch.no_grad():vals.append(float(obj()[0]))
   fd=(vals[1]-vals[0])/(2*eps);err=abs(fd-float(g.norm()))/max(float(g.norm()),1e-12);assert err<.25,(fd,float(g.norm()),err);finite.append({'step':1,'relative_error':err,'gradient_L2':float(g.norm())})
  trials=[];accepted=False;fraction=0.
  if direction.norm()>0:
   for j in range(9):
    candidate=old+direction/(2**j);put123(m,S0,candidate)
    with torch.no_grad():value=float(obj()[0])
    bound=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;ok=value<=bound;trials.append({'fraction':2.**(-j),'objective':value,'bound':bound,'accepted':ok})
    if ok:delta=candidate;accepted=True;fraction=2.**(-j);break
  if not accepted:delta=old
  put123(m,S0,delta)
  with torch.no_grad():ll,pt,ol,kp,*_=obj()
  traces.append({'step':step,'before':before,'after':float(ll),'partial_after':float(pt),'ordinal_after':float(ol),'keep_after':float(kp),'gradient_L2':float(g.norm()),'parameter_L2':float(delta.norm()),'accepted':accepted,'fraction':fraction,'trials':trials})
 restore(m,S0);return delta.cpu().numpy(),traces,grads,finite
def process134(base,m,S,e,anchors,pairs,engineering=False):
 assert sha(R134/e['evidence_path'])==e['evidence_sha256'];ev=load(R134/e['evidence_path']);scores,boxes=reference134(e,ev)
 if not e['pair_active']:
  scores.update(pair_only_F=scores['zero_F'].copy(),partial_pair_F=scores['partial3_F'].copy(),partial_wrongpair_F=scores['partial3_F'].copy())
  return scores,None,None,{'pair_active':False,'old_partial_active':e['active'],'all_new_fits_reuse_reference':True},boxes
 tick=time.perf_counter();restore(m,S);sid=e['row']['id'];name=e['source']+'_to_'+e['target'];D=R134/'crossplatform/round131/formal'/name;z0=load(D/'prefix_PRIVATE'/(sid+'.npz'));assert np.array_equal(z0['boxes'],boxes);z=tensor_prefix(z0);d=geomdata125(e,z0);old=load(D/'features_PRIVATE'/(sid+'.npz'));late=torch.as_tensor(old['fixed_late'],device='cuda');res=torch.as_tensor(old['fixed_r'],device='cuda');valid=torch.as_tensor(old['valid_B'],device='cuda');A=z['vis0'][0]
 cached=load(R134/e['provider_features']) if e['active'] else {}
 if 'B_vis' in cached:B=torch.as_tensor(cached['B_vis'],device='cuda');provider='cached_original_B'
 else:
  a=anchors[e['frame_PRIVATE']];pair=pairs[a['B_pair']];regpath=str((R134/'crossplatform/round131/registration_PRIVATE'/(a['B_pair']+'.json')).relative_to(R134));obs,proof=extra132(base,a['A'],pair,z0,regpath);assert obs is not None and torch.equal(obs['valid'],valid);B=obs['vis'];provider='reconstructed_same_original_B'
 assert torch.equal(B[~valid],A[~valid]);FAB=torch.where(valid[:,None],(A+B)/2,A);S0,radius=start123(m,S);ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))]
 if e['active']:assert radius==e['provider_record']['parameter_radius']
 with torch.no_grad():lp,h,u=body127(m,base,z,d,FAB,late,res,True)
 bridge=float(abs(lp.exp().cpu().numpy()-scores['zero_F']).max());assert bridge<1e-7,bridge
 ff={'fixed_late':old['fixed_late'],'fixed_r':old['fixed_r'],'mask':old['mask'],'FAB':FAB.cpu().numpy(),'zero_h':h.cpu().numpy(),'zero_u':u.cpu().numpy()};states={};traces={};grads={};checks=[];cost={};initial={};pathnodes={}
 with torch.no_grad():
  for kind in FITS134:
   pre='wrong_' if kind=='partial_wrongpair' else '';win=torch.as_tensor(ev[pre+'win'],device='cuda');lose=torch.as_tensor(ev[pre+'lose'],device='cuda');margin=torch.as_tensor(ev['margin'],device='cuda');ol,gs=ordinal_loss134(lp,groups,win,lose,margin);initial[kind]={'ordinal_loss':float(ol),'violated_pairs':int(((gs[win]-gs[lose])<margin-1e-10).sum())}
 for kind in FITS134:
  ti=time.perf_counter();delta,tr,gg,finite=fit134(base,m,S0,z,d,FAB,late,res,groups,ev,radius,kind,engineering);states[kind]=delta;traces[kind]=tr;grads.update(gg);checks+=finite;restore(m,S0);put123(m,S0,torch.as_tensor(delta,device='cuda'));freeze91(m)
  with torch.no_grad():ll,pt,ol,kp,lp,h,u,gs=objective134(base,m,z,d,FAB,late,res,groups,torch.as_tensor(ev['S'],device='cuda'),torch.as_tensor(ev['C'],device='cuda'),torch.as_tensor(ev['wrong_win' if kind=='partial_wrongpair' else 'win'],device='cuda'),torch.as_tensor(ev['wrong_lose' if kind=='partial_wrongpair' else 'lose'],device='cuda'),torch.as_tensor(ev['margin'],device='cuda'),kind)
  scores[kind+'_F']=lp.exp().cpu().numpy();ff[kind+'_h']=h.cpu().numpy();ff[kind+'_u']=u.cpu().numpy();ff[kind+'_adapted_FAB']=m.point_adapter(FAB).detach().cpu().numpy();cost[kind]=time.perf_counter()-ti;assert {k for k,v in state(m).items() if not np.array_equal(v,S0[k])}<=set(NAMES)
 restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items())
 return scores,ff,states|grads,{'pair_active':True,'old_partial_active':e['active'],'pair_count':len(ev['win']),'effective_wrong_pair':e['effective_wrong_pair'],'B_provider':provider,'zero_reader_max_error':bridge,'radius':radius,'traces':traces,'finite_difference':checks,'initial_ordinal':initial,'fitting_seconds':cost,'seconds_joint':time.perf_counter()-tick,'new_GT_or_observation':False},boxes
