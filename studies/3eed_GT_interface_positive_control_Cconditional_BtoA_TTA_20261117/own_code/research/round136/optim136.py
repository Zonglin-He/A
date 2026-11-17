"""Shared source-centered Up4896 optimizer; objectives supplied by separate stages."""
from common136 import *
import sys
sys.path.insert(0,str(R136/'research/round135'));from core135 import *
from common136 import R136,H136,B136,rd,dump,sha,utc,check
def inputs136(e):
 name=e['source']+'_to_'+e['target'];sid=e['row']['id'];D=R136/'crossplatform/round135/formal'/name;z0=load(R136/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));z=tensor_prefix(z0);_,d=routegeom135(e,z0);rt=load(D/'route_PRIVATE'/(sid+'.npz'));route=tensor_route89(rt);ev=load(B136/'evidence_PRIVATE'/name/(sid+'.npz'));ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];old=load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(old['arm_names'],old['scores']));return z0,z,d,route,ev,groups,ps
def fit136(base,m,S,z,d,route,task,engineering=False,steps=3):
 restore(m,S);freeze91(m);P=z['vis0'][0];params=[dict(m.named_parameters())[k] for k in NAMES];delta=torch.zeros(4896,device='cuda');radius=radius135(m,S);trace=[];states={};fd=[]
 with torch.no_grad():ref=F.normalize(m.point_adapter(P),dim=-1).detach();lp0,a0,_=live135(m,base,z,d,P,route)
 def objective():
  lp,a,_=live135(m,base,z,d,P,route);t,parts=task(lp);keep=(F.normalize(m.point_adapter(P),dim=-1)-ref).square().sum(-1).mean()/2;return t+keep,t,keep,parts,lp,a
 for step in range(1,steps+1):
  put123(m,S,delta)
  for p in params:p.requires_grad_(True)
  L,t,k,parts,*_=objective();before=float(L.detach());assert step!=1 or float(k)==0.;gg=torch.autograd.grad(L,params);g=torch.cat([x.detach().flatten() for x in gg]);assert torch.isfinite(g).all();states['gradient'+str(step)]=g.cpu().numpy();old=delta.clone();direction=-radius*g/g.norm() if float(g.norm())>1e-12 and float(t.detach())>1e-12 else g*0
  for p in params:p.requires_grad_(False)
  if engineering and step==1 and float(g.norm())>1e-7:
   vals=[];eps=radius*.001
   for sign in [-1,1]:
    put123(m,S,old+sign*eps*g/g.norm())
    with torch.no_grad():vals.append(float(objective()[0]))
   derivative=(vals[1]-vals[0])/(2*eps);err=abs(derivative-float(g.norm()))/float(g.norm());assert err<.25,(derivative,float(g.norm()),err);fd.append({'relative_error':err})
  trials=[];accepted=False;fraction=0.
  if float(direction.norm())>0:
   for j in range(9):
    candidate=old+direction/(2**j);put123(m,S,candidate)
    with torch.no_grad():value=float(objective()[0])
    bound=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;ok=value<=bound;trials.append({'fraction':2.**(-j),'objective':value,'bound':bound,'accepted':ok})
    if ok:delta=candidate;accepted=True;fraction=2.**(-j);break
  if not accepted:delta=old
  put123(m,S,delta)
  with torch.no_grad():L,t,k,parts,*_=objective()
  trace.append({'step':step,'before':before,'after':float(L),'task_after':float(t),'keep_after':float(k),'parts_after':{n:float(v) for n,v in parts.items()},'gradient_L2':float(g.norm()),'parameter_L2':float(delta.norm()),'accepted':accepted,'fraction':fraction,'trials':trials})
 with torch.no_grad():_,_,_,_,lp,a=objective();adapt=m.point_adapter(P).cpu().numpy()
 assert {k for k,v in state(m).items() if not np.array_equal(v,S[k])}<=set(NAMES);states['delta']=delta.cpu().numpy();feat={'adapted_A':adapt}
 npfeatures(feat,'final',a);restore(m,S);freeze91(m);return lp0.exp().cpu().numpy(),lp.exp().cpu().numpy(),feat,states,{'trace':trace,'radius':radius,'finite_difference':fd,'scope_only_Up':True}
