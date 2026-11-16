"""Physical RGB-GN updates with live source-compatible semantic context.
All task anchors/native point-text foundation immutable; no expert/source training.
"""
from pathlib import Path
import sys
R120=Path(__file__).resolve().parents[2];H120=R120/'research/round120';B120=R120/'crossplatform/round120'
sys.path.insert(0,str(R120/'research/round116'));from core116 import *
sys.path.insert(0,str(R120/'research/round118'));import core118 as C118
ST120=['depth_trust','depth_natural','wrong_depth_trust','random_GN_trust']
REFMAP120={'original_source':'original_source','prepared_A':'prepared_A','prepared_F':'prepared_F','native_F':'native_F','repaired_F':'repaired_F','lateOnly_F':'lateOnly_F','context_zero_F':'F_lSA_rSF','native_ABC':'ABC_native','context_zero_noR_F':'F_lSA_rzero'}
NEW120=['context_zero_C','context_zero_ABC','context_zero_rshuffle_F']+[s+'_'+v for s in ST120 for v in ['live_F','frozenR_F','frozenLate_F','shuffleR_F','live_C','live_ABC']]
ARMS120=list(REFMAP120)+NEW120;PRIMARY120='depth_trust_live_F';assert len(NEW120)==27 and len(ARMS120)==36

def current120(co,src,r):
 name=src+'_to_'+r['platform'];f=R120/('crossplatform/round111' if co=='DEV' else 'crossplatform/round117/formal')/name/'prefix_PRIVATE'/(r['id']+'.npz');z0=load(f);key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();gf=B97/'shared_physical_geometry_PRIVATE'/(r['id']+'_'+key[:20]+'.npz');assert jread(gf.with_suffix('.json'))['geometry_seed_and_sample_sha256']==key;g=load(gf);g['query_anchor_indices']=np.asarray(z0['query_points_sample_inds'],np.int64);z=tensor_prefix(z0);d=tensor_data(g,r);return z0,g,z,d,{'current_A_prefix_sha256':sha(f),'physical_geometry_sha256':sha(gf),'no_media_or_PointNet_recompute':True,'B_C_loaded_before_proposals':False}

def proposals120(m,S,co,src,r,z,d,pm,im,engineering):
 train,hold,wrongtarget=split116(r,d);restore(m,S);freeze91(m);radius=.01*float(flat116(m).double().norm());saved=None
 if co=='DEV':
  f=B116/'formal'/(src+'_to_'+r['platform'])/'states_PRIVATE'/(r['id']+'.npz');saved=load(f);proposals={s:torch.as_tensor(saved[s+'_raw'],device='cuda') for s in ST120};grad={k:torch.as_tensor(saved[n],device='cuda') for k,n in [('correct','gradient'),('wrong','wrong_gradient')]};proof={'reused_R116_RAW_states_sha256':sha(f),'no_old_BT_reuse':True}
 else:proposals={};grad={};proof={'fresh_current_A_GN_gradient':True,'no_old_fit_repeated':True}
 before={}
 for key,target in [('correct',d['depth_target']),('wrong',wrongtarget)]:
  restore(m,S)
  if co=='EXP' or engineering:
   enable116(m);L=depth116(m,z,d,pm,im,train,target);gg=torch.autograd.grad(L,[p for _,p in gn116(m)],allow_unused=True);g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for (_,p),v in zip(gn116(m),gg)]);assert torch.isfinite(g).all()
   if co=='DEV':assert torch.equal(g,grad[key]),float(abs(g-grad[key]).max())
   grad[key]=g;before[key]=float(L.detach())
  else:
   freeze91(m)
   with torch.no_grad():before[key]=float(depth116(m,z,d,pm,im,train,target))
 if co=='EXP':
  g,gw=grad['correct'],grad['wrong'];trust=-radius*g/g.norm() if g.norm()>0 else g*0;wrong=-radius*gw/gw.norm() if gw.norm()>0 else gw*0;natural=-.5*torch.clamp(1/(g.norm()+1e-6),max=1)*g;rg=np.random.default_rng(int(hashlib.sha256(('R116_GN_random_direction|'+src+'|'+r['id']).encode()).hexdigest()[:16],16));random=torch.as_tensor(rg.normal(size=224),device='cuda',dtype=torch.float32);random=random*float(trust.norm())/random.norm();proposals=dict(zip(ST120,[trust,natural,wrong,random]))
 assert set(proposals)==set(ST120);restore(m,S);freeze91(m);stamp={'all_four_RAW_states_fixed':True,'before_B_C_arrays_or_hidden_reads':True,'monotonic_seconds':time.monotonic()};proof.update(source_GN_L2=radius*100,trust_radius=radius,fit_depth_before=before,train_cells=len(train),holdout_cells=len(hold),GN_parameters=224,actual_loss='existing_image_only_SmoothL1_measured_logdepth',stamp=stamp);return proposals,grad,train,hold,proof

def after120(co,src,r,z):
 name=src+'_to_'+r['platform'];f=R120/('crossplatform/round111' if co=='DEV' else 'crossplatform/round117/formal')/name/'observations_PRIVATE'/(r['id']+'.npz');ob=load(f);oo={}
 for v in ['B','C']:
  prefix='real_'+v+'_' if co=='DEV' else v+'_'
  if prefix+'vis' in ob:oo[v]={k:torch.as_tensor(ob[prefix+k],device='cuda') for k in ['vis','valid']}
 zz={'A':z,'fusedABC':prefix106(z,list(oo.values()),True)};N,_,hashes=C118.components(co,name,r['id']);return zz,N,{'observations_sha256':sha(f),'C_available':'C' in oo,'supported_B_C':len(oo),'native_cached_provider_sha256':hashes,'after_all_RAW_states_fixed':True}

def context120(m,base,z,d,route,S):
 f=feature91(m,base,z,d,route,S);ins=[];hk=m.interface.out[0].register_forward_pre_hook(lambda _m,args:ins.append(args[0]))
 try:p,e,_=scorer91(m,f,route,S,'R76','full_disc')
 finally:hk.remove()
 return {'late':f['late'].detach(),'r':e['residual'].detach(),'r_input':ins[0].detach() if ins else torch.zeros((256,265),device='cuda'),'r_enabled':bool(ins),'h':f['h'].detach(),'u':f['u'].detach(),'p':p.detach()}

def read120(raw,z,late,res):
 q=F.normalize(raw+late,dim=-1);return (native_scores(q,torch.as_tensor(z['u'],device='cuda'),torch.as_tensor(z['mask'],device='cuda').bool()).double()+res.double()).softmax(0)

def evaluate120(base,m,S,co,src,r,z0,z,d,route,pm,im,proposals,train,hold,engineering):
 zz,N,ev=after120(co,src,r,z);ff={};out={};gnmeta=[];losses={};frozen=None;allchecks=[];name=src+'_to_'+r['platform'];seed=int(hashlib.sha256(('R120_context_query_derangement|'+co+'|'+name+'|'+r['id']).encode()).hexdigest()[:16],16);order=np.random.default_rng(seed).permutation(256);perm=np.empty(256,int);perm[order]=np.roll(order,1);assert np.all(perm!=np.arange(256));pi=torch.as_tensor(perm,device='cuda');ff['residual_derangement']=perm
 restore(m,S);freeze91(m)
 with torch.no_grad():raw={v:m.match(torch.as_tensor(a['h'],device='cuda')[None])[0] for v,a in N.items()}
 for setting,delta in [('zero',None)]+list(proposals.items()):
  restore(m,S)
  if delta is not None:put116(m,S,delta)
  freeze91(m);actual=state(m);changed=[n for n in S if not np.array_equal(actual[n],S[n])];allowed={n for n,p in gn116(m)};assert set(changed)<=allowed;actnorm=float(torch.sqrt(sum((p-torch.as_tensor(S[n],device='cuda')).double().square().sum() for n,p in gn116(m))));gnmeta.append({'setting':setting,'changed_tensors':len(changed),'actual_writeback_L2':actnorm,'proposal_L2':float(delta.norm()) if delta is not None else 0.})
  with torch.no_grad():
   ca=context120(m,base,zz['A'],d,route,S);cf=context120(m,base,zz['fusedABC'],d,route,S)
   for v,c in [('SA',ca),('SF',cf)]:
    for k in ['late','r','r_input']:ff[setting+'_'+v+'_'+k]=c[k].cpu().numpy()
    ff[setting+'_'+v+'_r_enabled']=np.array(c['r_enabled'])
   losses[setting]={'fit_depth':float(depth116(m,z,d,pm,im,train,d['depth_target'])),'heldout_depth':float(depth116(m,z,d,pm,im,hold,d['depth_target']))}
   if setting=='zero':
    frozen=(ca['late'].clone(),cf['r'].clone());ps={v:read120(raw[v],N[v],*frozen) for v in N};out['context_zero_C']=ps['C'].cpu().numpy();out['context_zero_ABC']=torch.stack([ps[v] for v in ['A','B','C']]).mean(0).cpu().numpy();out['context_zero_rshuffle_F']=read120(raw['fusedABC'],N['fusedABC'],frozen[0],frozen[1][pi]).cpu().numpy();old=np.load(R120/'crossplatform/round118/formal'/co/name/'samples_PRIVATE'/(r['id']+'.npz'));po=dict(zip(old['arm_names'],old['scores']));er=float(abs(ps['fusedABC'].cpu().numpy()-po['F_lSA_rSF']).max());assert er<5e-6;ev['old_context_zero_probability_error']=er
    # Actual source-compatible providers must reproduce historical hidden/late/residual.
    if co=='DEV':
     fpath=B113/name/'features_PRIVATE'/(r['id']+'.npz');oldpre='frozen_full_'
    else:fpath=R120/'crossplatform/round109'/name/'features_PRIVATE'/(r['id']+'.npz');oldpre='real_'
    with np.load(fpath) as fs:
     for view,c in [('A',ca),('fusedABC',cf)]:
      for k in ['h','u','late','r']:
       oldv=fs[oldpre+view+'_'+k];oldv=oldv[0] if co=='DEV' and k!='r' else oldv;assert np.array_equal(c[k].cpu().numpy(),oldv),(co,src,r['id'],view,k)
    ev['old_source_A_F_h_u_late_r_bitexact']=True
   else:
    live={v:read120(raw[v],N[v],ca['late'],cf['r']) for v in N};out[setting+'_live_F']=live['fusedABC'].cpu().numpy();out[setting+'_live_C']=live['C'].cpu().numpy();out[setting+'_live_ABC']=torch.stack([live[v] for v in ['A','B','C']]).mean(0).cpu().numpy();out[setting+'_frozenR_F']=read120(raw['fusedABC'],N['fusedABC'],ca['late'],frozen[1]).cpu().numpy();out[setting+'_frozenLate_F']=read120(raw['fusedABC'],N['fusedABC'],frozen[0],cf['r']).cpu().numpy();out[setting+'_shuffleR_F']=read120(raw['fusedABC'],N['fusedABC'],ca['late'],cf['r'][pi]).cpu().numpy()
   if engineering:
    # Native point/text base is unaffected by GN under actual hard foundation bypass.
    with functional112(m,'neither'):fn=feature91(m,base,z,d,route,S)
    assert np.array_equal(fn['h'].cpu().numpy(),N['A']['h']) and np.array_equal(fn['u'].cpu().numpy(),N['A']['u']);allchecks.append({'setting':setting,'native_foundation_h_u_bitexact':True,'only_GN224_changed':True,'interface_input265_actual':cf['r_enabled'],'source_context_residual_change_L2':float((cf['r']-frozen[1]).double().norm())})
 restore(m,S);freeze91(m);assert set(out)==set(NEW120);return out,ff,dict(ev,writeback=gnmeta,physical_losses=losses,engineering=allchecks)
