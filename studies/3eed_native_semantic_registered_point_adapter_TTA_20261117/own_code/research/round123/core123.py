"""Actual early point-adapter TTA from registered descriptors, fixed RGB context."""
from pathlib import Path
import sys
R123=Path(__file__).resolve().parents[2];H123=R123/'research/round123';B123=R123/'crossplatform/round123'
sys.path.insert(0,str(R123/'research/round120'));from core120 import *
from math123 import *
REF123=['original_source','prepared_A','prepared_F','native_F','native_ABC','lateOnly_F','mean_context','headmean_context','robust_context','weight_mean_context','zero_C']
NEW123=[s+'_'+v for s in SETTINGS for v in ['F','C']];ARMS123=REF123+NEW123;PRIMARY123='corr3_F';assert len(ARMS123)==25

def start123(m,S):
 restore(m,S)
 with torch.no_grad():m.point_adapter.up.weight.zero_();m.point_adapter.up.bias.zero_();m.gate_P.zero_()
 freeze91(m);S0=state(m);allowed={'point_adapter.up.weight','point_adapter.up.bias','gate_P'};assert {k for k in S if not np.array_equal(S[k],S0[k])}<=allowed
 radius=.01*float(torch.sqrt(sum(torch.as_tensor(S['point_adapter.'+n],device='cuda').double().square().sum() for n,p in m.point_adapter.named_parameters())));return S0,radius

def a_b123(e):
 r=e['row'];z0=load(R123/e['prefix']);assert sha(R123/e['prefix'])==e['prefix_sha256'];z=tensor_prefix(z0);f=R123/e['observations'];oo=np.load(f);pre='real_B_' if e['cohort']=='DEV' else 'B_';present=pre+'vis' in oo;A=z['vis0'][0];B=torch.as_tensor(oo[pre+'vis'],device='cuda') if present else A;valid=oo[pre+'valid'] if present else np.zeros(1024,bool);oo.close();salt='R123|'+e['source']+'|'+r['id'];ids,far,perm=geometry123(z0['xyz'][0],valid,salt+'|wrong',device='cuda');return z0,z,A,B,ids,far,perm,present,salt

def native123(base,m,z,d):
 m.route=None;m.use_interface=False;m.variant='full'
 with torch.no_grad():p,q,e=m.main(base,z,d,'bidirectional')
 return e['query_hidden'],e['projected_text']

def after123(e,z0,z):
 co,src,r=e['cohort'],e['source'],e['row'];name=src+'_to_'+r['platform'];f=R123/e['observations'];oo=np.load(f);observations={};pre0='real_' if co=='DEV' else ''
 for v in ['B','C']:
  pre=pre0+v+'_'
  if pre+'vis' in oo:observations[v]={k:torch.as_tensor(oo[pre+k],device='cuda') for k in ['vis','valid']}
 oo.close();key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();gfile=B97/'shared_physical_geometry_PRIVATE'/(r['id']+'_'+key[:20]+'.npz');assert jread(gfile.with_suffix('.json'))['geometry_seed_and_sample_sha256']==key;g=load(gfile);g['query_anchor_indices']=z0['query_points_sample_inds'];d=tensor_data(g,r);zz={'F':prefix106(z,list(observations.values()),True),'C':prefix106(z,[observations['C']],False) if 'C' in observations else z};N,SS,proof=C118.components(co,name,r['id']);return d,zz,N,SS,observations,{'C_available':'C' in observations,'B_available':'B' in observations,'providers':proof,'geometry_sha256':sha(gfile),'observations_sha256':sha(f),'C_arrays_used_in_gradient':False}

def process123(base,m,S,e,engineering=False):
 restore(m,S);S0,radius=start123(m,S);z0,z,A,B,ids,far,perm,present,salt=a_b123(e);states,traces,fit=fit123(m,S0,A,B,ids,far,perm,radius,salt);stamp=time.monotonic();d,zz,N,SS,obs,proof=after123(e,z0,z);late=torch.as_tensor(SS['A']['late'],device='cuda');res=torch.as_tensor(SS['fusedABC']['r'],device='cuda');scores={};ff={'fixed_late':SS['A']['late'],'fixed_r':SS['fusedABC']['r'],'fit_ids':ids.cpu().numpy(),'fit_far':far.cpu().numpy(),'fit_permutation':perm.cpu().numpy()};params=[];aux={};checks=[]
 def read(h,u):q=F.normalize(m.match(h[None])[0]+late,dim=-1);return (native_scores(q,u,z['token_mask'].bool()).double()+res.double()).softmax(0)
 for setting,delta in [('zero',None)]+list(states.items()):
  restore(m,S0)
  if setting not in ['zero','free3']:put123(m,S0,delta)
  freeze91(m);now=state(m);changed=[k for k in S0 if not np.array_equal(now[k],S0[k])];assert set(changed)<=set(NAMES);params.append({'setting':setting,'changed_tensors':len(changed),'actual_parameter_L2':float(flat123(m).double().norm()),'actual_feature_field_L2':float(delta.double().norm()) if setting=='free3' else 0.})
  with torch.no_grad():
   aa=m.point_adapter(A);bb=m.point_adapter(B)
   if setting=='free3':aa=A+delta;bb=B+delta
   L,c,k=parts123(aa,bb,F.normalize(A,dim=-1),F.normalize(B,dim=-1),ids,far);aux[setting]={'fit_objective':float(L),'fit_corr':float(c),'fit_keep':float(k)}
   if 'C' in obs:
    C=obs['C']['vis'];hc,hfar,_=geometry123(z0['xyz'][0],obs['C']['valid'].cpu().numpy(),salt+'|heldout',device='cuda');cc=m.point_adapter(C) if setting!='free3' else C+delta;hl,hh,hk=parts123(aa,cc,F.normalize(A,dim=-1),F.normalize(C,dim=-1),hc,hfar);qa=F.normalize(aa[hc],dim=-1);qc=F.normalize(cc[hc],dim=-1)
    if len(hc)>=2 and hfar.any():sim=qa@qc.T;allowed=hfar|torch.eye(len(hc),device='cuda',dtype=torch.bool);valid=hfar.any(1);retr=(sim.masked_fill(~allowed,-torch.inf).argmax(1)==torch.arange(len(hc),device='cuda'))[valid].float().mean();aux[setting].update(heldout_corr=float(hh),heldout_objective=float(hl),heldout_retrieval=float(retr),heldout_pairs=len(hc))
   for v,vz in zz.items():
    if setting=='free3':vz=dict(vz,vis0=vz['vis0']+delta[None])
    h,u=native123(base,m,vz,d);p=read(h,u);ff[setting+'_'+v+'_h']=h.cpu().numpy();ff[setting+'_'+v+'_u']=u.cpu().numpy();ff[setting+'_'+v+'_mask']=z['token_mask'].cpu().numpy()
    if setting=='zero':assert np.array_equal(h.cpu().numpy(),N['fusedABC' if v=='F' else 'C']['h']) and np.array_equal(u.cpu().numpy(),N['fusedABC' if v=='F' else 'C']['u'])
    else:scores[setting+'_'+v]=p.cpu().numpy()
    if engineering and setting=='corr1':
     patched=vz['vis0'].clone();actual=m.point_adapter(patched[0]);restore(m,S0);hc0,uc0=native123(base,m,dict(vz,vis0=actual[None]),d);assert torch.equal(h,hc0) and torch.equal(u,uc0);put123(m,S0,delta);checks.append({'view':v,'parameter_write_functional_point_input_bitexact':True,'updated_h_change_L2':float((h-torch.as_tensor(N['fusedABC' if v=='F' else 'C']['h'],device='cuda')).norm())})
 restore(m,S);freeze91(m);assert all(np.array_equal(state(m)[k],v) for k,v in S.items());return scores,ff,{k:v.cpu().numpy() for k,v in states.items()},dict(proof,fit=fit,traces=traces,auxiliary=aux,writebacks=params,engineering=checks,all_proposals_fixed_before_decoding_C_monotonic=stamp,B_for_fit=present,supervision='approximate_registered_spatial_seed_correspondence_NOT_instance_GT',scope_parameters=4896)
