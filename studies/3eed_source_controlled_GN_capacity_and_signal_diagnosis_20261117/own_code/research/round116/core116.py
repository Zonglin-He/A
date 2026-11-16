"""Measured sparse-depth adaptation through existing RGB GroupNorm affine parameters."""
from pathlib import Path
import sys
R116=Path(__file__).resolve().parents[2];H116=R116/'research/round116';B116=R116/'crossplatform/round116'
sys.path.insert(0,str(R116/'research/round114'));from core114 import *
REF116=['original_source','prepared_frozen','native_real_fusedABC','direct_three_geometric','neither_correct_A','neither_correct_fusedABC','anchor_frozen_C','anchor_noLate_fusedABC','native_gate_only_fusedABC','gateI0_frozen_fusedABC']
SETS116=['depth_trust','depth_natural','wrong_depth_trust','random_GN_trust']
STATES116=[(s,t) for s in SETS116 for t in (['raw'] if s=='random_GN_trust' else ['raw','bt'])]
NEW116=[s+'_'+t+'_'+v for s,t in STATES116 for v in V114];ARMS116=REF116+NEW116;PRIMARY116='depth_trust_raw_fusedABC'

def gn116(m):
 out=[]
 for path,mod in m.rgb.named_modules():
  if isinstance(mod,torch.nn.GroupNorm):
   for name,p in mod.named_parameters(recurse=False):out.append(('rgb.'+path+'.'+name,p))
 assert len(out)==6 and sum(p.numel() for _,p in out)==224;return out

def flat116(m):return torch.cat([p.detach().flatten() for _,p in gn116(m)])
def put116(m,S,delta):
 off=0
 with torch.no_grad():
  for n,p in gn116(m):k=p.numel();p.copy_(torch.as_tensor(S[n],device=p.device)+delta[off:off+k].reshape_as(p));off+=k
 assert off==224

def enable116(m):
 freeze91(m)
 for _,p in gn116(m):p.requires_grad_(True)

def split116(r,d):
 ids=torch.nonzero(d['depth_valid'],as_tuple=True)[0];seed=int(hashlib.sha256(('R116_physical_holdout|'+r['id']).encode()).hexdigest()[:16],16);rg=np.random.default_rng(seed);order=rg.permutation(len(ids));n=len(ids)//4;hold=ids[torch.as_tensor(order[:n],device=ids.device)];train=ids[torch.as_tensor(order[n:],device=ids.device)];perm=rg.permutation(len(train))
 while len(train)>1 and np.any(perm==np.arange(len(train))):perm=rg.permutation(len(train))
 target=d['depth_target'].clone();target[train]=target[train[torch.as_tensor(perm,device=ids.device)]];return train,hold,target

def depth116(m,z,d,pm,im,ids,target):
 losses=[]
 for a,b in zip(pm,im):
  # Original source-prepared I-only depth head. No point/depth feature is an input.
  _,_,I=m.features(z['vis0'][0],d,'bidirectional',point_mask=a,image_mask=b);v=m.pred_depth(I).squeeze(-1);losses.append(F.smooth_l1_loss(v[ids],target[ids]) if len(ids) else v.sum()*0)
 return torch.stack(losses).mean()

def aonly116(src,r):
 z0,g,z,proof=current113(src,r);d=tensor_data(g,r);pm,im=maskpair79(r,d);prefix='frozen_anchor_A_'
 with np.load(B113/(src+'_to_'+r['platform'])/'features_PRIVATE'/(r['id']+'.npz')) as old:a={k:old[prefix+k] for k in ['h','u','mask','r']}
 return z0,z,d,pm,im,a,proof

def fit116(base,m,S,src,r,engineering=False):
 anchor=initial113(m,S,True);z0,z,d,pm,im,a,proof=aonly116(src,r);train,hold,wrongtarget=split116(r,d);restore(m,anchor);freeze91(m)
 with torch.no_grad():p0,l0=fast114(m,z,d,a);h=torch.as_tensor(a['h'][0],device='cuda');q0=F.normalize(m.match(h[None])[0]+l0,dim=-1);refs=(p0.detach(),q0.detach());norm0=float(flat116(m).double().norm());radius=.01*norm0
 grads={};before={}
 for key,target in [('correct',d['depth_target']),('wrong',wrongtarget)]:
  restore(m,anchor);enable116(m);loss=depth116(m,z,d,pm,im,train,target);gg=torch.autograd.grad(loss,[p for _,p in gn116(m)],allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for (_,p),v in zip(gn116(m),gg)]);assert torch.isfinite(g).all();grads[key]=g;before[key]=float(loss.detach())
 g=grads['correct'];gw=grads['wrong'];natural=-.5*torch.clamp(1/(g.norm()+1e-6),max=1)*g;trust=-radius*g/g.norm() if float(g.norm())>0 else g*0;wrong=-radius*gw/gw.norm() if float(gw.norm())>0 else gw*0;rg=np.random.default_rng(int(hashlib.sha256(('R116_GN_random_direction|'+src+'|'+r['id']).encode()).hexdigest()[:16],16));random=torch.as_tensor(rg.normal(size=224),device='cuda',dtype=torch.float32);random=random*float(trust.norm())/random.norm();proposals={'depth_trust':trust,'depth_natural':natural,'wrong_depth_trust':wrong,'random_GN_trust':random};states={};records={}
 for setting,raw in proposals.items():
  target=wrongtarget if setting=='wrong_depth_trust' else d['depth_target'];which='wrong' if setting=='wrong_depth_trust' else 'correct';gradient=grads[which];trials=[];accepted=False
  for k in range(1 if setting=='random_GN_trust' else 9):
   restore(m,anchor);delta=raw/(2**k);put116(m,anchor,delta);freeze91(m)
   with torch.no_grad():
    pp,ll=fast114(m,z,d,a);qq=F.normalize(m.match(h[None])[0]+ll,dim=-1);aux=depth116(m,z,d,pm,im,train,target);keep=keep91(pp,qq,*refs);objective=aux.double()+keep
   bound=before[which]+1e-4*float(gradient.double()@delta.double());ok=bool(torch.isfinite(objective) and float(objective)<=bound+1e-10);trials.append({'fraction_of_raw':1/2**k,'depth_fit':float(aux),'keep':float(keep),'objective':float(objective),'bound':bound,'accepted':ok,'proposal_L2':float(delta.norm())})
   if k==0:states[(setting,'raw')]=delta.clone()
   if setting!='random_GN_trust' and ok:states[(setting,'bt')]=delta.clone();accepted=True;break
  if setting!='random_GN_trust' and not accepted:states[(setting,'bt')]=raw*0
  records[setting]={'scope_parameters':224,'train_depth_cells':len(train),'physical_holdout_cells':len(hold),'gradient_L2':float(gradient.norm()),'raw_L2':float(raw.norm()),'source_GN_L2':norm0,'trust_radius':radius,'accepted':accepted if setting!='random_GN_trust' else False,'trials':trials,'C_before_fit':False,'I_only_depth_input':True}
 # No C arrays/neural states or heldout target loss evaluated above.
 restore(m,anchor);freeze91(m);return {'anchor':anchor,'z0':z0,'z':z,'d':d,'pm':pm,'im':im,'a':a,'train':train,'hold':hold,'states':states,'records':records,'gradient':g,'wrong_gradient':gw,'proof':proof,'refs':refs}

def evaluate116(base,m,S,src,r,c,engineering=False):
 assert set(c['states'])==set(STATES116);z,d,anchor=c['z'],c['d'],c['anchor'];zz,proof=heldout113(src,r,z);out={};features={};hold={};activity=[];engine=[];rt=None
 if engineering:
  restore(m,S);z0,geom,_,_=current113(src,r);rt,_=role97(base,m,r,z0,geom,z,d);rt=tensor_route89(rt)
 with np.load(B113/(src+'_to_'+r['platform'])/'features_PRIVATE'/(r['id']+'.npz')) as ff,torch.no_grad():
  for (setting,tag),delta in c['states'].items():
   restore(m,anchor);put116(m,anchor,delta);freeze91(m);actual=state(m);changed=[n for n in anchor if not np.array_equal(actual[n],anchor[n])];allowed={n for n,_ in gn116(m)};assert set(changed)<=allowed;norm=float(torch.sqrt(sum((p-torch.as_tensor(anchor[n],device='cuda')).double().square().sum() for n,p in gn116(m))));activity.append({'setting':setting,'state':tag,'actual_changed_tensors':len(changed),'actual_writeback_L2':norm,'proposal_L2':float(delta.norm())})
   hold[setting+'_'+tag]=float(depth116(m,z,d,c['pm'],c['im'],c['hold'],d['depth_target']))
   for view,vz in zz.items():
    a={k:ff['frozen_anchor_'+view+'_'+k] for k in ['h','u','mask','r']};p,late=fast114(m,vz,d,a);label=setting+'_'+tag+'_'+view;out[label]=p.cpu().numpy();features[label+'_late']=late.cpu().numpy()
    if engineering and view=='A':
     f=feature91(m,base,vz,d,rt,anchor);pn,_,_=scorer91(m,f,rt,anchor,'R76','full_disc');assert torch.equal(pn,p) and np.array_equal(f['h'].cpu().numpy(),a['h'][0]) and np.array_equal(f['u'].cpu().numpy(),a['u'][0]);engine.append({'setting':setting,'state':tag,'native_fast_probability_error':0.,'native_foundation_unchanged':True})
  restore(m,anchor);freeze91(m);hold['frozen']=float(depth116(m,z,d,c['pm'],c['im'],c['hold'],d['depth_target']))
 restore(m,S);freeze91(m);return out,features,dict(proof,physical_holdout_loss=hold,writeback=activity,engineering=engine)
