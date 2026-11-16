"""Source-faithful RGB keys/auxiliary graph with native frozen grounding geometry."""
from pathlib import Path
import sys,contextlib,types
R114=Path(__file__).resolve().parents[2];H114=R114/'research/round114';B114=R114/'crossplatform/round114'
sys.path.insert(0,str(R114/'research/round113'));from core113 import *
V114=['A','C','fusedABC']
SET114=['sourcekey_sourceaux','nativekey_sourceaux','sourcekey_nativeaux','sourcekey_sourceaux_wrong','native_gate_only','native_bridge_no_gate']
STATIC114=['sourcekey_frozen','gateI0_frozen']
NEW114=[s+'_'+v for s in SET114+STATIC114 for v in V114]
REF114=['original_source','prepared_frozen','real_frozen_fusedABC','native_real_fusedABC','direct_three_geometric','neither_correct_A','neither_correct_fusedABC','anchor_frozen_C','anchor_noLate_fusedABC','anchor_bridge_natural_raw_A','anchor_bridge_natural_raw_C','anchor_bridge_natural_raw_fusedABC']
ARMS114=REF114+NEW114;PRIMARY114='sourcekey_sourceaux_fusedABC'

def raw114(m,g,only=None):
 a=[];offset=0
 for name,p in m.tta_parameters():
  n=p.numel();keep=scope113(name,'bridge') and (only is None or (name=='gate_I')==(only=='gate'));a.append(g[offset:offset+n] if keep else torch.zeros_like(g[offset:offset+n]));offset+=n
 gg=torch.cat(a);return -.5*torch.clamp(1/(gg.norm()+1e-6),max=1)*gg

@contextlib.contextmanager
def native_main114(m):
 """Keep original prepared RGB keys, but never inject prepared PP into grounding."""
 original=m.features
 def separated(self,P,d,mode,*args,**kw):
  PP,II,I=original(P,d,mode,*args,**kw);pm=kw.get('point_mask',args[1] if len(args)>1 else None)
  return P.masked_fill(pm[:,None],0) if pm is not None else P,II,I
 m.features=types.MethodType(separated,m)
 try:yield
 finally:m.features=original

def input114(src,r):
 z0,g,z,proof=current113(src,r);d=tensor_data(g,r);pm,im=maskpair79(r,d);ws=weight_sets({'id':'UNUSED_UNIFORM'},d,pm,im,np.zeros(1024),np.zeros(1536))[0]['uniform'];dw=bound94(d,permutation94(r,d,0));return z0,z,d,dw,pm,im,ws,proof

def proposals114(m,S,anchor,src,r,z,d,dw,pm,im,ws,engineering=False):
 # Fit files only contain previous current-A gradients. No old hidden view or task score.
 f=B113/(src+'_to_'+r['platform'])/'fits_PRIVATE'/(r['id']+'.npz')
 with np.load(f) as fs:gS=torch.as_tensor(fs['gradient_full'],device='cuda');gN=torch.as_tensor(fs['gradient_anchor'],device='cuda')
 restore(m,S);enable_theta91(m);loss=uniform_aux91(m,z,dw,pm,im,ws);pars=[p for _,p in m.tta_parameters()];gg=torch.autograd.grad(loss,pars,allow_unused=True);gW=torch.cat([(torch.zeros_like(p) if a is None else a).detach().flatten() for p,a in zip(pars,gg)])
 if engineering:
  restore(m,S);enable_theta91(m);lc=uniform_aux91(m,z,d,pm,im,ws);g0=torch.autograd.grad(lc,pars,allow_unused=True);g0=torch.cat([(torch.zeros_like(p) if a is None else a).detach().flatten() for p,a in zip(pars,g0)]);assert float((g0-gS).abs().max())<1e-12,float((g0-gS).abs().max());assert torch.equal(raw114(m,g0),raw114(m,gS)), 'actual bridge proposal differs'
 dc=raw114(m,gS);dn=raw114(m,gN);dw0=raw114(m,gW);dwm=dw0*(dc.norm()/dw0.norm()) if float(dw0.norm()) else dw0
 dirs={'sourcekey_sourceaux':dc,'nativekey_sourceaux':dc,'sourcekey_nativeaux':dn,'sourcekey_sourceaux_wrong':dwm,'native_gate_only':raw114(m,gN,'gate'),'native_bridge_no_gate':raw114(m,gN,'notgate')}
 meta={'gradient_source_AUX_L2':float(gS.norm()),'gradient_native_AUX_L2':float(gN.norm()),'gradient_wrong_source_AUX_L2':float(gW.norm()),'source_wrong_aux_before':float(loss.detach()),'C_loaded_before_all_proposals':False,'backprop_parameter_count':102498,'update_scope_parameters':45313,'proposals':{}}
 for k,v in dirs.items():meta['proposals'][k]={'parameter_L2':float(v.norm()),'gate_delta':float(v[0]) if next(iter(m.tta_parameters()))[0]=='gate_I' else None,'wrong_same_L2':k.endswith('_wrong')}
 restore(m,S);freeze91(m);return dirs,meta

def fast114(m,z,d,a):
 ids=z['query_points_sample_inds'].long();known=(d['p2i'][ids]>=0).any(-1);_,II,_=m.features(z['vis0'][0],d,'bidirectional');late=torch.tanh(m.gate_late)*m.late(m.point_image(II,d)[ids].float())*known[:,None];h=torch.as_tensor(a['h'][0],device='cuda');u=torch.as_tensor(a['u'][0],device='cuda');mask=torch.as_tensor(a['mask'],device='cuda').bool();r=torch.as_tensor(a['r'],device='cuda');q=F.normalize(m.match(h[None])[0]+late,dim=-1);p=(native_scores(q,u,mask).double()+r.double()).softmax(0);return p,late

def evaluate114(base,m,S,anchor,src,r,z0,z,d,pm,im,dirs,engineering=False):
 # All six proposals fixed before extra-view arrays or hidden features are opened.
 zz,proof=heldout113(src,r,z);ff=B113/(src+'_to_'+r['platform'])/'features_PRIVATE'/(r['id']+'.npz');scores={};features={};ev={'actual_native_main_max_error':0.,'source_key_frozen_late_max_error':0.,'old_natural_max_error':0.};rt=None
 if engineering:
  restore(m,S);gkey=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();geom=load(B97/'shared_physical_geometry_PRIVATE'/(r['id']+'_'+gkey[:20]+'.npz'));geom['query_anchor_indices']=np.asarray(z0['query_points_sample_inds'],np.int64);rt,_=role97(base,m,r,z0,geom,z,d);rt=tensor_route89(rt)
 with np.load(ff) as old,torch.no_grad():
  for setting in SET114+STATIC114:
   sourcekeys=setting.startswith('sourcekey');st=S if sourcekeys else anchor;restore(m,st)
   if setting in dirs:theta_step91(m,st,dirs[setting])
   if setting=='gateI0_frozen':m.gate_I.zero_()
   freeze91(m)
   for view,vz in zz.items():
    pref='frozen_anchor_'+view+'_';a={k:old[pref+k] for k in ['h','u','mask','r']};p,late=fast114(m,vz,d,a);label=setting+'_'+view;scores[label]=p.cpu().numpy();features[label+'_late']=late.cpu().numpy()
    if setting=='sourcekey_frozen':
     er=float((late.cpu()-torch.from_numpy(old['frozen_full_'+view+'_late'][0])).abs().max());assert er==0;ev['source_key_frozen_late_max_error']=max(ev['source_key_frozen_late_max_error'],er)
    if engineering and view=='A' and (setting in ['sourcekey_frozen','sourcekey_sourceaux','nativekey_sourceaux','gateI0_frozen']):
     with native_main114(m) if sourcekeys else contextlib.nullcontext():f=feature91(m,base,vz,d,rt,st);pn,_,_=scorer91(m,f,rt,st,'R76','full_disc')
     er=float((pn-p).abs().max());assert er<1e-12 and np.array_equal(f['h'].cpu().numpy(),a['h'][0]) and np.array_equal(f['u'].cpu().numpy(),a['u'][0]);assert np.array_equal(f['late'].cpu().numpy(),features[label+'_late']);ev['actual_native_main_max_error']=max(ev['actual_native_main_max_error'],er)
 restore(m,anchor)
 # Existing natural step reproduced after formal new states; reference parity only.
 with np.load(B113/(src+'_to_'+r['platform'])/'fits_PRIVATE'/(r['id']+'.npz')) as fs:gn=torch.as_tensor(fs['gradient_anchor'],device='cuda');theta_step91(m,anchor,raw114(m,gn))
 with np.load(ff) as old,torch.no_grad():
  for v,vz in zz.items():
   a={k:old['frozen_anchor_'+v+'_'+k] for k in ['h','u','mask','r']};p,_=fast114(m,vz,d,a);want=old['anchor_bridge_natural_raw_'+v+'_late'][0];assert np.array_equal(_.cpu().numpy(),want)
 restore(m,S);freeze91(m);return scores,features,dict(proof,**ev)
