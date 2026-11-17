"""Restore intact source adaptation state and live source-prepared competition.

Only point-adapter Up receives episodic deltas; source gates, matching and
competition parameters stay frozen. The keep anchor is the intact adapter.
"""
from common135 import *
import sys
sys.path.insert(0,str(R135/'research/round134'));from core134 import *
import ordinal134 as O134
sys.path.insert(0,str(R135/'research/round129'));import support129 as U129
from common135 import R135,H135,B135,rd,dump,sha,utc,basecheck
FEATURES=['h','u','late','local','own','text','relation_text','mask']
ZERO_ARMS=[s+'_'+r+'_'+v for s in ['intact','reset'] for r in ['live','fixed'] for v in ['A','F']]
ARMS135=ZERO_ARMS+['AB_mean','direct_A','direct_F','tta_A','tta_F','wrong_A','wrong_F','legacy_zero_F','legacy_R134']
FITS135=['tta_A','tta_F','wrong_A','wrong_F']
def radius135(m,S):return .01*float(torch.sqrt(sum(torch.as_tensor(S['point_adapter.'+n],device='cuda').double().square().sum() for n,p in m.point_adapter.named_parameters())))
def live135(m,base,z,d,P,route):
 m.route=route;m.use_interface=False;m.variant='full';a=native_feature91(m,base,dict(z,vis0=P[None]),d);p,detail,score=cached91(m,a,route,'full_disc');return score.log_softmax(0),a,detail
def fixed135(m,a,late,res):
 q=F.normalize(m.match(a['h'][None])[0]+late,dim=-1);return (native_scores(q,a['u'],a['mask'].bool()).double()+res.double()).log_softmax(0)
def npfeatures(out,prefix,a):
 for k in FEATURES:out[prefix+'_'+k]=a[k].detach().cpu().numpy()
def routegeom135(e,z0):
 key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();p=B97/'shared_physical_geometry_PRIVATE'/(e['row']['id']+'_'+key[:20]+'.npz');assert jread(p.with_suffix('.json'))['geometry_seed_and_sample_sha256']==key;g=load(p);g['query_anchor_indices']=z0['query_points_sample_inds'];return g,tensor_data(g,e['row'])
def prepare135(base,m,S,e,anchors,pairs):
 tick=time.perf_counter();restore(m,S);freeze91(m);r=e['row'];sid=r['id'];name=e['source']+'_to_'+e['target'];D=R135/'crossplatform/round131/formal'/name;z0=load(D/'prefix_PRIVATE'/(sid+'.npz'));z=tensor_prefix(z0);A=z['vis0'][0];g,d=routegeom135(e,z0);rt,info=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);old=load(D/'features_PRIVATE'/(sid+'.npz'));valid=torch.as_tensor(old['valid_B'],device='cuda');late=torch.as_tensor(old['fixed_late'],device='cuda');res=torch.as_tensor(old['fixed_r'],device='cuda');B=A;provider='no_B'
 if bool(valid.any()):
  cached=load(R135/e['provider_features']) if e['active'] else {}
  if 'B_vis' in cached:B=torch.as_tensor(cached['B_vis'],device='cuda');provider='cached_original_B'
  else:
   an=anchors[e['frame_PRIVATE']];pair=pairs[an['B_pair']];rp=str((R135/'crossplatform/round131/registration_PRIVATE'/(an['B_pair']+'.json')).relative_to(R135));obs,proof=extra132(base,an['A'],pair,z0,rp);assert obs is not None and torch.equal(obs['valid'],valid);B=obs['vis'];provider='reconstructed_original_B'
 assert torch.equal(B[~valid],A[~valid]);FAB=torch.where(valid[:,None],(A+B)/2,A);ff={'B_vis':B.cpu().numpy(),'valid_B':valid.cpu().numpy(),'fixed_late':old['fixed_late'],'fixed_r':old['fixed_r']};ps={};aa={}
 with torch.no_grad():
  for statekind in ['intact','reset']:
   if statekind=='intact':restore(m,S)
   else:start123(m,S)
   for view,P in [('A',A),('F',FAB)]:
    if view=='F' and not bool(valid.any()):
     a=aa[statekind+'_A'];lp=aa[statekind+'_A_lp']
    else:lp,a,_=live135(m,base,z,d,P,route)
    aa[statekind+'_'+view]=a;aa[statekind+'_'+view+'_lp']=lp
    ps[statekind+'_live_'+view]=lp.exp().cpu().numpy();ps[statekind+'_fixed_'+view]=fixed135(m,a,late,res).exp().cpu().numpy();npfeatures(ff,statekind+'_'+view,a)
  restore(m,S)
  if bool(valid.any()):lb,ab,_=live135(m,base,z,d,B,route)
  else:lb=aa['intact_A_lp'];ab=aa['intact_A']
  npfeatures(ff,'teacher_B',ab);pA=ps['intact_live_A'];pB=lb.exp().cpu().numpy();pF=ps['intact_live_F'];ps['AB_mean']=(pA+pB)/2
 oldpred=load(B97/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(oldpred['arm_names'],oldpred['scores']));err=float(abs(pA-op['prepared_frozen']).max());assert err<1e-7 and pA.argmax()==op['prepared_frozen'].argmax(),(name,sid,err)
 prior=load(B134/'formal'/name/'samples_PRIVATE'/(sid+'.npz'));priorp=dict(zip(prior['arm_names'],prior['scores']));assert np.array_equal(prior['source_boxes'],z0['boxes']);bridge=float(abs(ps['reset_fixed_F']-priorp['zero_F']).max());assert bridge<1e-7,bridge
 geo=U129.geometry(z0['xyz'][0],z0['boxes'],valid.cpu().numpy());groups=geo['groups'];assert np.array_equal(geo['assignment'],old['assignment']);scope=U129.qualify(geo,4,.5);C=np.zeros_like(scope);kappa=rd(R135/'research/round132/SOURCE_LOG_CALIBRATION.json')['kappa']
 if scope.sum()>=2:
  for p in [pA,pB]:
   x=U129.glogits_np(p,groups)[scope];C[scope]|=x.max()-x<=kappa
 win,lose,margin=O134.evidence(pA,pB,pF,groups,C,kappa);ww,ll,perm=O134.wrong_pairs(win,lose,groups,C,e['source']+'|'+sid);active=bool(valid.any() and scope.sum()>=2 and C.any() and ((scope&~C).any() or len(win)))
 ev={'pA':pA,'pB':pB,'S':scope,'C':C,'assignment':geo['assignment'],'win':win,'lose':lose,'margin':margin,'wrong_win':ww,'wrong_lose':ll,'wrong_perm':perm}
 projections={}
 for v in ['A','F']:
  p=U129.direct(ps['intact_live_'+v],groups,scope,C,None,'partial');ps['direct_'+v],projections[v]=O134.project(p,groups,win,lose,margin)
  if not bool(valid.any()):assert np.array_equal(ps['direct_'+v],pA)
 ps['legacy_zero_F']=priorp['zero_F'];ps['legacy_R134']=priorp['partial_pair_F'];restore(m,S);freeze91(m)
 assert all(np.array_equal(v,state(m)[k]) for k,v in S.items())
 return ps,ff,rt,ev,{'full_A_probability_error':err,'legacy_F_bridge_error':bridge,'has_B':bool(valid.any()),'active':active,'partial_active':bool((scope&~C).any() and C.any()),'pair_count':len(win),'B_provider':provider,'seconds':time.perf_counter()-tick,'projection':projections},z0['boxes']
def objective135(base,m,z,d,P,route,groups,ev,wrong,reference):
 lp,a,_=live135(m,base,z,d,P,route);S=torch.as_tensor(ev['S'],device='cuda');C=torch.as_tensor(ev['C'],device='cuda');pt=taskloss(lp,groups,S,C,None,'partial');prefix='wrong_' if wrong else '';w=torch.as_tensor(ev[prefix+'win'],device='cuda');l=torch.as_tensor(ev[prefix+'lose'],device='cuda');margin=torch.as_tensor(ev['margin'],device='cuda')
 if len(w):ol,_=ordinal_loss134(lp,groups,w,l,margin)
 else:ol=lp.sum()*0
 kp=(F.normalize(m.point_adapter(P),dim=-1)-reference).square().sum(-1).mean()/2
 return pt+ol+kp,pt,ol,kp,lp,a
def fit135(base,m,S,z,d,P,route,groups,ev,wrong,engineering):
 restore(m,S);freeze91(m);params=[dict(m.named_parameters())[k] for k in NAMES];radius=radius135(m,S);delta=torch.zeros(4896,device='cuda');tr=[];ss={};fd=[]
 with torch.no_grad():reference=F.normalize(m.point_adapter(P),dim=-1).detach()
 def obj():return objective135(base,m,z,d,P,route,groups,ev,wrong,reference)
 for step in range(1,4):
  put123(m,S,delta)
  for p in params:p.requires_grad_(True)
  L,pt,ol,kp,*_=obj();assert step!=1 or float(kp)==0.;before=float(L.detach());gg=torch.autograd.grad(L,params);g=torch.cat([v.detach().flatten() for v in gg]);assert torch.isfinite(g).all();ss['gradient'+str(step)]=g.cpu().numpy();old=delta.clone();direction=-radius*g/g.norm() if float(g.norm())>1e-12 and float((pt+ol).detach())>1e-12 else g*0
  for p in params:p.requires_grad_(False)
  if engineering and step==1 and float(g.norm())>1e-7:
   vals=[];eps=radius*.001
   for sign in [-1,1]:
    put123(m,S,old+sign*eps*g/g.norm())
    with torch.no_grad():vals.append(float(obj()[0]))
   f=(vals[1]-vals[0])/(2*eps);err=abs(f-float(g.norm()))/float(g.norm());assert err<.25,(f,float(g.norm()),err);fd.append({'relative_error':err})
  accepted=False;fraction=0.;trials=[]
  if float(direction.norm())>0:
   for j in range(9):
    candidate=old+direction/(2**j);put123(m,S,candidate)
    with torch.no_grad():value=float(obj()[0])
    bound=before+1e-4*float(g.double()@(candidate-old).double())+1e-12;ok=value<=bound;trials.append({'fraction':2.**(-j),'objective':value,'bound':bound,'accepted':ok})
    if ok:delta=candidate;accepted=True;fraction=2.**(-j);break
  if not accepted:delta=old
  put123(m,S,delta)
  with torch.no_grad():val,pt,ol,kp,*_=obj()
  tr.append({'step':step,'before':before,'after':float(val),'partial_after':float(pt),'ordinal_after':float(ol),'keep_after':float(kp),'gradient_L2':float(g.norm()),'parameter_L2':float(delta.norm()),'accepted':accepted,'fraction':fraction,'trials':trials})
 with torch.no_grad():val,pt,ol,kp,lp,a=obj();adapt=m.point_adapter(P).detach().cpu().numpy()
 assert {k for k,v in state(m).items() if not np.array_equal(v,S[k])}<=set(NAMES);ss['delta']=delta.cpu().numpy();restore(m,S);return lp.exp().cpu().numpy(),a,ss,{'trace':tr,'radius':radius,'finite_difference':fd,'scope_only_Up':True},adapt
