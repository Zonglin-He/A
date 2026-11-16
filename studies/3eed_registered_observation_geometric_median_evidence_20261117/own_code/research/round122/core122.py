"""Change aligned-seed fusion only. Frozen 3EED, no newsource/expert/modelupdate."""
from pathlib import Path
import sys
R122=Path(__file__).resolve().parents[2];H122=R122/'research/round122';B122=R122/'crossplatform/round122'
sys.path.insert(0,str(R122/'research/round120'));from core120 import *
from aggregate122 import solve122,VARS
REF122={'original_source':'original_source','prepared_A':'prepared_A','prepared_F':'prepared_F','native_F':'native_F','repaired_F':'repaired_F','lateOnly_F':'lateOnly_F','mean_context':'F_lSA_rSF','native_ABC':'ABC_native','context_ABC':'ABC_lSA_rSA','context_noR_ABC':'ABC_lSA_rzero'}
NEW122=[v+'_'+r for v in VARS for r in ['native','context']];ARMS122=list(REF122)+NEW122;PRIMARY122='robust_context';assert len(ARMS122)==20

def cache122(e):
 r=e['row'];z0=load(R122/e['prefix']);assert sha(R122/e['prefix'])==e['prefix_sha256'];ob=load(R122/e['observations']);assert sha(R122/e['observations'])==e['observations_sha256'];A=z0['vis0'][0];obs=[]
 for v in ['B','C']:
  p=('real_' if e['cohort']=='DEV' else '')+v+'_'
  if p+'vis' in ob:obs.append({k:ob[p+k] for k in ['vis','valid']})
 key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();gf=B97/'shared_physical_geometry_PRIVATE'/(r['id']+'_'+key[:20]+'.npz');assert jread(gf.with_suffix('.json'))['geometry_seed_and_sample_sha256']==key;g=load(gf);g['query_anchor_indices']=z0['query_points_sample_inds'];return z0,tensor_prefix(z0),tensor_data(g,r),obs,{'geometry_sha256':sha(gf),'no_raw_media_or_PointNet':True}

def variants122(e,A,obs):
 robust,w,tri,hist=solve122(A,obs);X=np.stack([A]+[x['vis'] for x in obs]).astype(float);meanw=np.stack([np.ones(1024,bool)]+[x['valid'] for x in obs]);meanw=meanw/meanw.sum(0)[None];Vmean=(X*meanw[:,:,None]).sum(0).astype(np.float32);out={k:Vmean.copy() for k in VARS};out['robust']=robust;ff={'robust_weights':w,'triple':tri};info={'three_valid_seeds':int(tri.sum()),'geometric_objective_means':hist};ids=np.flatnonzero(tri)
 if len(ids):
  seed=int(hashlib.sha256(('R122_spatial_weight_derangement|'+e['source']+'|'+e['row']['id']).encode()).hexdigest()[:16],16);rg=np.random.default_rng(seed);order=rg.permutation(len(ids));perm=np.empty(len(ids),int);perm[order]=np.roll(order,1);ws=w.copy();ws[:,ids]=w[:,ids[perm]];wa=w.copy();wa[:,ids]=w[:,ids].mean(1)[:,None];out['weight_shuffle']=(X*ws[:,:,None]).sum(0).astype(np.float32);out['weight_mean']=(X*wa[:,:,None]).sum(0).astype(np.float32);wrong=[];perms={}
  for j,o in enumerate(obs):
   sd=int(hashlib.sha256(('R122_correspondence_derangement|'+e['source']+'|'+e['row']['id']+'|'+str(j)).encode()).hexdigest()[:16],16);order=np.random.default_rng(sd).permutation(len(ids));p=np.empty(len(ids),int);p[order]=np.roll(order,1);v=o['vis'].copy();v[ids]=v[ids[p]];wrong.append({'vis':v,'valid':o['valid']});perms[str(j)]=p
  vr,wr,tr,hh=solve122(A,wrong);assert np.array_equal(tr,tri);WX=np.stack([A]+[x['vis'] for x in wrong]).astype(float);out['wrong_robust']=vr;out['wrong_mean']=(WX*meanw[:,:,None]).sum(0).astype(np.float32);ff.update(weight_shuffle=ws,weight_mean=wa,wrong_robust_weights=wr,weight_permutation=perm,**{'wrong_permutation_'+k:v for k,v in perms.items()});info['wrong_objective_means']=hh
 else:info['no_triple_no_change']=True
 for k,v in out.items():assert np.array_equal(v[~tri],Vmean[~tri]);assert np.isfinite(v).all()
 return out,ff,info

def native122(base,m,S,z,d):
 m.route=None;m.use_interface=False;m.variant='full'
 with functional112(m,'neither'),torch.no_grad():p,q,e=m.main(base,z,d,'bidirectional')
 return e['query_hidden'],e['projected_text']

def process122(base,m,S,e,engineering=False):
 z0,z,d,obs,proof=cache122(e);views,ff,info=variants122(e,z0['vis0'][0],obs);co,src,r=e['cohort'],e['source'],e['row'];name=src+'_to_'+r['platform'];N,SS,provider=C118.components(co,name,r['id']);F0=N['fusedABC'];sa=SS['A'];sf=SS['fusedABC'];out={};nchecks=0
 with torch.no_grad():
  raw0=m.match(torch.as_tensor(F0['h'],device='cuda')[None])[0];basep=(native_scores(F.normalize(raw0+torch.as_tensor(sa['late'],device='cuda'),dim=-1),torch.as_tensor(F0['u'],device='cuda'),torch.as_tensor(F0['mask'],device='cuda').bool()).double()+torch.as_tensor(sf['r'],device='cuda').double()).softmax(0);old=np.load(R122/f'crossplatform/round118/formal/{co}/{name}/samples_PRIVATE/{r["id"]}.npz');po=dict(zip(old['arm_names'],old['scores']));assert float(abs(basep.cpu().numpy()-po['F_lSA_rSF']).max())<5e-6
  if engineering:
   zz=prefix106(z,[{k:torch.as_tensor(v,device='cuda') for k,v in o.items()} for o in obs],True);hh,uu=native122(base,m,S,zz,d);assert np.array_equal(hh.cpu().numpy(),F0['h']) and np.array_equal(uu.cpu().numpy(),F0['u']);nchecks+=1
  # Without three observations there is no robust/fixed/shuffled distinction; reuse exact mean h/U.
  for v in VARS:
   if info['three_valid_seeds']:
    vz=dict(z,vis0=torch.as_tensor(views[v],device='cuda')[None]);hh,uu=native122(base,m,S,vz,d)
   else:hh=torch.as_tensor(F0['h'],device='cuda');uu=torch.as_tensor(F0['u'],device='cuda')
   raw=m.match(hh[None])[0];mask=torch.as_tensor(F0['mask'],device='cuda').bool();native=native_scores(F.normalize(raw,dim=-1),uu,mask).double().softmax(0);p=(native_scores(F.normalize(raw+torch.as_tensor(sa['late'],device='cuda'),dim=-1),uu,mask).double()+torch.as_tensor(sf['r'],device='cuda').double()).softmax(0);out[v+'_native']=native.cpu().numpy();out[v+'_context']=p.cpu().numpy();ff[v+'_h']=hh.cpu().numpy();ff[v+'_u']=uu.cpu().numpy()
  ff.update(mask=F0['mask'],fixed_late=sa['late'],fixed_residual=sf['r']);info.update(proof,provider_sha256=provider,originalA256_six_positions_FIXED=True,source_model_updated=False,engineering_old_mean_h_U_bitexact=bool(nchecks));return out,ff,info
