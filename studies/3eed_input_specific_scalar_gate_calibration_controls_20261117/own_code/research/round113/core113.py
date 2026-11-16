"""Native-geometry anchored physical RGB bridge adaptation, fixed evidence and loss."""
from pathlib import Path
import sys,contextlib,copy
R113=Path(__file__).resolve().parents[2];H113=R113/'research/round113';B113=R113/'crossplatform/round113'
sys.path.insert(0,str(R113/'research/round112'));from core112 import *
REFS113=['original_source','original_source_matchedcoverage','prepared_frozen','real_frozen_C','real_frozen_fusedABC','native_real_fusedABC','native_real_ensemble','real_TTA_C','real_TTA_fusedABC','direct_three_geometric','neither_correct_A','neither_correct_fusedABC','neither_wrong_fusedABC','noPA_correct_fusedABC','noEarlyRGB_correct_fusedABC']
# Seven fixed directions; no layer/strength search or target-label selection.
SETTINGS113=['full_all','full_bridge_nm','anchor_all','anchor_bridge_nm','anchor_rgb_nm','anchor_bridge_natural','anchor_bridge_nm_wrong']
READS113=['A','C','fusedABC','feature3_fusedABC']
TTA113=[s+'_'+t+'_'+v for s in SETTINGS113 for t in ['raw','bt'] for v in READS113]
FROZEN113=[p+'_frozen_'+k+'_'+v for p in ['full','anchor'] for k in ['mean3','feature3'] for v in ['A','C','fusedABC']]+['anchor_frozen_C','anchor_noLate_feature3_fusedABC','anchor_noLate_fusedABC','anchor_lateOnly_feature3_fusedABC','anchor_lateOnly_fusedABC','anchor_native_feature3_fusedABC']
NEW113=TTA113+FROZEN113;ARMS113=REFS113+NEW113;PRIMARY113='anchor_bridge_nm_bt_feature3_fusedABC'
assert len(SETTINGS113)==7 and len(NEW113)==74 and len(ARMS113)==89 and len(set(ARMS113))==89

def current113(src,r):
 """A only: never loads aligned B/C arrays or their scores before update selection."""
 name=src+'_to_'+r['platform'];f=B111/name/'prefix_PRIVATE'/(r['id']+'.npz');z0=load(f);key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();gf=B97/'shared_physical_geometry_PRIVATE'/(r['id']+'_'+key[:20]+'.npz');proof=jread(gf.with_suffix('.json'));assert proof['geometry_seed_and_sample_sha256']==key;g=load(gf);g['query_anchor_indices']=np.asarray(z0['query_points_sample_inds'],np.int64)
 return z0,g,tensor_prefix(z0),{'R111_prefix_sha256':sha(f),'geometry_cache_sha256':sha(gf),'no_new_sensor_or_PointNet_forward':True,'B_C_arrays_not_read_before_all_fits':True}

def heldout113(src,r,z):
 f=B111/(src+'_to_'+r['platform'])/'observations_PRIVATE'/(r['id']+'.npz');ob=load(f);oo=[];cc=[]
 for view in ['B','C']:
  if 'real_'+view+'_vis' in ob:
   a={k:torch.as_tensor(ob['real_'+view+'_'+k],device='cuda') for k in ['vis','valid']};oo.append(a)
   if view=='C':cc.append(a)
 return {'A':z,'C':prefix106(z,cc,False) if cc else z,'fusedABC':prefix106(z,oo,True)}, {'observations_sha256':sha(f),'C_available':bool(cc),'aligned_observations':len(oo),'C_loaded_only_after_all_fit_acceptance':True}

def initial113(m,S,anchored):
 edit112(m,S,'neither' if anchored else 'full');freeze91(m);return state(m)

def scope113(name,kind):
 if kind=='all':return True
 if name=='gate_I' or name.startswith('point_to_image.'):return True
 return kind=='rgb' and name.startswith('image_adapter.')

def project113(m,g,kind):
 a=[];off=0
 for n,p in m.tta_parameters():
  k=p.numel();a.append(g[off:off+k] if scope113(n,kind) else torch.zeros_like(g[off:off+k]));off+=k
 assert off==len(g);return torch.cat(a)

def native_views113(m,base,z,d,route,S,pm,im):
 return [feature91(m,base,z,d,route,S,a,b) for a,b in [(None,None)]+list(zip(pm,im))]

def fast_views113(m,z,d,cached,pm,im):
 """Exact functional replay after real bridge/RGB weight writes, foundation fixed.
 Cached native h/u/local/text are valid because point_adapter identity and gate_P=0.
 This is tested against the complete model on actual inputs before formal use.
 """
 assert float(m.gate_P)==0 and not torch.count_nonzero(m.point_adapter.up.weight) and not torch.count_nonzero(m.point_adapter.up.bias)
 out=[];ids=z['query_points_sample_inds'].long();known=(d['p2i'][ids]>=0).any(-1)
 for f,(a,b) in zip(cached,[(None,None)]+list(zip(pm,im))):
  _,II,_=m.features(z['vis0'][0],d,'bidirectional',point_mask=a,image_mask=b)
  late=torch.tanh(m.gate_late)*m.late(m.point_image(II,d)[ids].float())*known[:,None];out.append(dict(f,late=late))
 return out

def read113(m,vv,route,S,no_late=False,no_r=False):
 ps=[];qq=[];uu=[];inputs=[];rr=[]
 for f in vv:
  f=dict(f,late=f['late']*0) if no_late else f
  hook=m.interface.out[0].register_forward_pre_hook(lambda _m,args:inputs.append(args[0]))
  try:p,e,s=scorer91(m,f,route,S,'R76','full_disc')
  finally:hook.remove()
  if no_r:p=e['native_scores'].softmax(0)
  ps.append(p);qq.append(e['projected_query']);uu.append(f['u']);rr.append(e['residual'])
 q=F.normalize(torch.stack(qq).mean(0),dim=-1);u=F.normalize(torch.stack(uu).mean(0),dim=-1);ss=native_scores(q,u,vv[0]['mask'].bool()).double();ii=torch.stack(inputs).mean(0) if inputs else None;r=m.interface.out(ii).squeeze(-1) if ii is not None else ss*0
 if not no_r:ss=ss+r.double()
 return {'single':ps[0],'mean3':torch.stack(ps).mean(0),'feature3':ss.softmax(0),'q':qq[0],'qmean':q,'umean':u,'r':rr[0],'rs':torch.stack(rr),'rmean':r,'inputmean':ii,'projected':qq}

def arrays113(vv,rd):
 out={k:torch.stack([a[k] for a in vv]).detach().cpu().numpy() for k in ['h','u','late']};out['mask']=vv[0]['mask'].detach().cpu().numpy();out.update({k:rd[k].detach().cpu().numpy() for k in ['r','rs','rmean','qmean','umean']});out['inputmean']=rd['inputmean'].detach().cpu().numpy() if rd['inputmean'] is not None else np.zeros((256,265),np.float32);out['interface_enabled']=np.array(rd['inputmean'] is not None);return out

def objective113(m,base,z,d,fitd,route,S,refs,pm,im,ws,cached=None):
 m.route=route;m.use_interface=True;m.variant='full';m.pool_mode='strict'
 if cached is None:
  f=feature91(m,base,z,d,route,S);p,e,_=scorer91(m,f,route,S,'R76','full_disc');q=e['projected_query']
 else:
  # Only the unmasked cached foundation is needed for the preservation term.
  vv=fast_views113(m,z,d,cached,pm,im);p,e,_=scorer91(m,vv[0],route,S,'R76','full_disc');q=e['projected_query']
 aux=uniform_aux91(m,z,fitd,pm,im,ws);keep=keep91(p,q,*refs);return aux.double()+keep,aux,keep,p,q

def fit113(m,base,z,d,dw,route,Sinit,kind,g,parentraw,pm,im,ws,cached,wrong=False,natural=False):
 """One source-reset proposal, fixed nine-scale loss backtracking, no task GT."""
 restore(m,Sinit);freeze91(m)
 with torch.no_grad():f=feature91(m,base,z,d,route,Sinit);p,e,_=scorer91(m,f,route,Sinit,'R76','full_disc');refs=(p.detach(),e['projected_query'].detach())
 gg=project113(m,g,kind);raw=-.5*torch.clamp(1/(gg.norm()+1e-6),max=1)*gg
 if not natural and kind!='all' and float(raw.norm())>0:raw=raw*(parentraw.norm()/raw.norm())
 fitd=dw if wrong else d;cc=cached if kind!='all' and Sinit['gate_P']==0 and not np.any(Sinit['point_adapter.up.weight']) else None
 restore(m,Sinit)
 with torch.no_grad():before=float(uniform_aux91(m,z,fitd,pm,im,ws))
 trials=[];states={};accepted=False;reference_delta_norm=float(parentraw.norm());start=time.perf_counter()
 for k,alpha in enumerate(GRID91):
  delta=raw/(2**k);theta_step91(m,Sinit,delta);freeze91(m)
  with torch.no_grad():L,aux,keep,p,q=objective113(m,base,z,d,fitd,route,Sinit,refs,pm,im,ws,cc)
  bound=before+1e-4*float(g.double()@delta.double());ok=bool(torch.isfinite(L) and float(L)<=bound+1e-10);trial={'alpha':alpha,'fraction_of_raw':1/2**k,'physical':float(aux),'keep':float(keep),'objective':float(L),'bound':bound,'accepted':ok,'proposal_parameter_L2':float(delta.norm())};trials.append(trial)
  if k==0:states['raw']=theta_patch91(m)
  if ok:states['bt']=theta_patch91(m);accepted=True;break
 if not accepted:states['bt']={n:Sinit[n].copy() for n,_ in m.tta_parameters()}
 chosen=state(m);meta={'scope':kind,'wrong_fit_binding':wrong,'natural_scope_clip':natural,'auxiliary_before':before,'full_gradient_L2':float(g.norm()),'scope_gradient_L2':float(gg.norm()),'raw_proposal_parameter_L2':float(raw.norm()),'parent_all_proposal_parameter_L2':reference_delta_norm,'accepted':accepted,'accepted_alpha':trials[-1]['alpha'] if accepted else 0.,'trials':trials,'fit_seconds':time.perf_counter()-start,'scope_parameters':sum(p.numel() for n,p in m.tta_parameters() if scope113(n,kind)),'full_parent_gradient_computed_parameters':int(len(g)),'no_C_or_fused_features_read':True}
 for tag,th in states.items():
  for n,p in m.tta_parameters():
   if not scope113(n,kind):assert np.array_equal(th[n],Sinit[n]),n
  if kind!='all' and Sinit['gate_P']==0:assert float(th['gate_P'])==0 and not np.any(th['point_adapter.up.weight']) and not np.any(th['point_adapter.up.bias'])
 restore(m,Sinit);return states,meta

def cpu113(a,S,no_late=False,no_r=False):
 """Independent CPU full/three-view readout, native ReLU and interface GELU."""
 import torch.nn.functional as fn
 hs=torch.from_numpy(a['h']);us=torch.from_numpy(a['u']);ls=torch.from_numpy(a['late'])*int(not no_late);qq=[]
 for h,u,late in zip(hs,us,ls):
  x=h
  for j in [0,2]:x=fn.relu(fn.linear(x,torch.from_numpy(S['match.'+str(j)+'.weight']),torch.from_numpy(S['match.'+str(j)+'.bias'])))
  x=fn.linear(x,torch.from_numpy(S['match.4.weight']),torch.from_numpy(S['match.4.bias']));qq.append(fn.normalize(x+late,dim=-1))
 mask=torch.from_numpy(a['mask']).bool()
 def score(q,u,r):
  ss=q@u.T/.07;ss=ss[:,mask].logsumexp(-1)-ss[:,~mask].logsumexp(-1);return (ss.double()+r.double()*int(not no_r)).softmax(0)
 ps=[score(q,u,torch.from_numpy(r)) for q,u,r in zip(qq,us,a['rs'])];qm=fn.normalize(torch.stack(qq).mean(0),dim=-1);um=fn.normalize(us.mean(0),dim=-1);x=torch.from_numpy(a['inputmean']);x=fn.gelu(fn.linear(x,torch.from_numpy(S['interface.out.0.weight']),torch.from_numpy(S['interface.out.0.bias'])));rr=fn.linear(x,torch.from_numpy(S['interface.out.2.weight']),torch.from_numpy(S['interface.out.2.bias'])).squeeze(-1) if bool(a['interface_enabled']) else torch.zeros(256)
 return {'single':ps[0].numpy(),'mean3':torch.stack(ps).mean(0).numpy(),'feature3':score(qm,um,rr).numpy()},{'qmean_error':0. if no_late else float(abs(qm-torch.from_numpy(a['qmean'])).max()),'umean_error':float(abs(um-torch.from_numpy(a['umean'])).max()),'residual_mean_error':float(abs(rr-torch.from_numpy(a['rmean'])).max())}
