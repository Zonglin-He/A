"""Independent CPU physical-GN writebacks, depth recovery, interface+matching replay."""
from pathlib import Path
import json,hashlib,sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
R=Path(__file__).resolve().parents[2];H=R/'research/round120';B=R/'crossplatform/round120';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round118'));import core118 as C
sys.path.insert(0,str(R/'research/round112'));from core112 import cpu112
sys.path.insert(0,str(R/'research/round116'));from audit116 import cpudepth,split,Fusion91,tensor_data,masks
ST=['depth_trust','depth_natural','wrong_depth_trust','random_GN_trust'];REF={'original_source':'original_source','prepared_A':'prepared_A','prepared_F':'prepared_F','native_F':'native_F','repaired_F':'repaired_F','lateOnly_F':'lateOnly_F','context_zero_F':'F_lSA_rSF','native_ABC':'ABC_native','context_zero_noR_F':'F_lSA_rzero'}
def residue(inp,enabled,S):
 if not enabled:return np.zeros(256,np.float32)
 a=F.gelu(F.linear(torch.from_numpy(inp),torch.from_numpy(S['interface.out.0.weight']),torch.from_numpy(S['interface.out.0.bias'])));return F.linear(a,torch.from_numpy(S['interface.out.2.weight']),torch.from_numpy(S['interface.out.2.bias'])).squeeze(-1).numpy()
def main():
 torch.set_num_threads(4);mat=rd(B/'MATRIX_SEALED.json');assert mat['n']==2304 and len(mat['new_arms'])==27 and rd(H/'ENGINEERING.json')['n']==24;lock=rd(H/'experiment/DESIGN_LOCK.json')
 for p,h in lock['files'].items():assert sha(R/p)==h
 for p,h in lock['input_seals'].items():assert sha(R/p)==h
 reg=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];n=aliases=heads=states=ties=0;maxp=maxr=maxdepth=maxnorm=maxdelta=0.
 for src in ['waymo','drone','quad']:
  assert sha(R/reg[src]['state_path'])==reg[src]['state_sha256']
  with np.load(R/reg[src]['state_path']) as z:S={k:z[k] for k in z.files}
  head={k[6:]:v for k,v in S.items() if k.startswith('match.')};model=Fusion91(nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64))).cpu().eval().requires_grad_(False);model.load_state_dict({k:torch.from_numpy(np.asarray(v).copy()) for k,v in S.items()},strict=True);names=[('rgb.'+p+'.'+k,v) for p,m in model.rgb.named_modules() if isinstance(m,nn.GroupNorm) for k,v in m.named_parameters(recurse=False)];assert sum(p.numel() for _,p in names)==224;flat=torch.cat([torch.from_numpy(S[k]).flatten() for k,_ in names])
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if tgt==src:continue
    name=src+'_to_'+tgt;D=B/'formal'/co/name;assert sha(D/'SEALED.json')==mat['directions'][co+'/'+name]
    for p,h in rd(D/'SEALED.json')['files'].items():assert sha(D/p)==h
    for ix,r in enumerate(C.rows(co,tgt)):
     sid=r['id'];meta=rd(D/'records_PRIVATE'/(sid+'.json'));a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));po=dict(zip(a['arm_names'],a['scores']));old=np.load(R/'crossplatform/round118/formal'/co/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));n+=1
     for k,v in REF.items():assert np.array_equal(po[k],op[v])
     if meta['protected_alias_fallback']:aliases+=1;assert all(np.array_equal(po[k],op['prepared_A']) for k in mat['new_arms']);continue
     N,_,hashes=C.components(co,name,sid);assert hashes==meta['evaluation']['native_cached_provider_sha256'];assert meta['fit']['stamp']['all_four_RAW_states_fixed'] and meta['evaluation']['after_all_RAW_states_fixed'];ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));rr={}
     for st in ['zero']+ST:
      rr[st]=residue(ff[st+'_SF_r_input'],bool(ff[st+'_SF_r_enabled']),S);er=float(abs(rr[st]-ff[st+'_SF_r']).max());maxr=max(maxr,er);assert er<5e-6
     perm=ff['residual_derangement'];assert np.all(np.sort(perm)==np.arange(256)) and np.all(perm!=np.arange(256))
     def p(view,st='zero',rstate=None,lstate=None,shuffle=False):
      z=N[view];late=ff[(lstate or st)+'_SA_late'];res=rr[rstate or st];return cpu112(z['h'],z['u'],z['mask'],late,res[perm] if shuffle else res,head)
     expected={'context_zero_C':p('C'),'context_zero_ABC':np.mean([p(v) for v in ['A','B','C']],axis=0),'context_zero_rshuffle_F':p('fusedABC',shuffle=True)}
     for st in ST:expected.update({st+'_live_F':p('fusedABC',st),st+'_live_C':p('C',st),st+'_live_ABC':np.mean([p(v,st) for v in ['A','B','C']],axis=0),st+'_frozenR_F':p('fusedABC',st,rstate='zero'),st+'_frozenLate_F':p('fusedABC',st,lstate='zero'),st+'_shuffleR_F':p('fusedABC',st,shuffle=True)})
     assert set(expected)==set(mat['new_arms'])
     for k,pp in expected.items():
      er=float(abs(pp-po[k]).max());maxp=max(maxp,er);assert er<5e-6
      if pp.argmax()!=po[k].argmax():assert po[k].max()-po[k][pp.argmax()]<=2*er+1e-12;ties+=1
      heads+=1
     ff.close();pref=R/('crossplatform/round111' if co=='DEV' else 'crossplatform/round117/formal')/name/'prefix_PRIVATE'/(sid+'.npz');assert sha(pref)==meta['inputs']['current_A_prefix_sha256']
     with np.load(pref) as z:key=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest()
     geompath=R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+key[:20]+'.npz');assert sha(geompath)==meta['inputs']['physical_geometry_sha256']
     with np.load(geompath) as z:geom={k:z[k] for k in z.files}
     d=tensor_data(geom,r,device='cpu');train,hold,_=split(r,d);ims=[masks(r,d,k)[1] for k in [76001,76002]];theta=np.load(D/'states_PRIVATE'/(sid+'.npz'));g=torch.from_numpy(theta['gradient']);gw=torch.from_numpy(theta['wrong_gradient']);radius=.01*float(flat.double().norm());expected_delta={'depth_trust':-radius*g/g.norm() if g.norm()>0 else g*0,'depth_natural':-.5*torch.clamp(1/(g.norm()+1e-6),max=1)*g,'wrong_depth_trust':-radius*gw/gw.norm() if gw.norm()>0 else gw*0};rg=np.random.default_rng(int(hashlib.sha256(('R116_GN_random_direction|'+src+'|'+sid).encode()).hexdigest()[:16],16));random=torch.tensor(rg.normal(size=224),dtype=torch.float32);expected_delta['random_GN_trust']=random*float(expected_delta['depth_trust'].norm())/random.norm()
     for st,v in expected_delta.items():er=float(abs(v-torch.from_numpy(theta[st])).max());maxdelta=max(maxdelta,er);assert er<1e-6
     records={x['setting']:x for x in meta['evaluation']['writeback']}
     for st in ['zero']+ST:
      delta=torch.zeros(224) if st=='zero' else torch.from_numpy(theta[st]);off=0
      with torch.no_grad():
       for k,p0 in names:nn0=p0.numel();p0.copy_(torch.from_numpy(S[k])+delta[off:off+nn0].reshape_as(p0));off+=nn0
       norm=float(torch.sqrt(sum((p0-torch.from_numpy(S[k])).double().square().sum() for k,p0 in names)));err=abs(norm-records[st]['actual_writeback_L2']);maxnorm=max(maxnorm,err);assert err<1e-10
       fit=cpudepth(model,None,d,ims,train,d['depth_target']);held=cpudepth(model,None,d,ims,hold,d['depth_target']);want=meta['evaluation']['physical_losses'][st];er=max(abs(fit-want['fit_depth']),abs(held-want['heldout_depth']));maxdepth=max(maxdepth,er);assert er<5e-6
      states+=1
     theta.close()
     if (ix+1)%32==0:print('AUDIT120',co,name,ix+1,flush=True)
 assert n==2304 and aliases==154 and heads==58050 and states==10750;out={'state':'PASS_ALL27_NEW_CPU_CONTEXT_MATCHING_GN_WRITEBACK_AND_PHYSICAL_FIT_HOLDOUT','n':n,'aliases':aliases,'CPU_endpoints':heads,'GN_states':states,'max_probability_error':maxp,'max_interface_residual_error':maxr,'CPU_near_ties':ties,'max_depth_error':maxdepth,'max_actual_norm_error':maxnorm,'max_proposal_error':maxdelta,'GT_used':False,'FINAL_access':False};(H/'NATIVE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
