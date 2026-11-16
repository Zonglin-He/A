"""Frozen PointNet observations aligned onto canonical A seeds, then native network."""
from pathlib import Path
import sys,hashlib
R106=Path(__file__).resolve().parents[2];H106=R106/'research/round106';B106=R106/'crossplatform/round106';sys.path.insert(0,str(R106/'research/round105'));from core105 import *
from scipy.spatial import cKDTree
def interpolation106(xyzA,xyzB,featuresB,rawBmodel):
 dist,ind=cKDTree(np.asarray(xyzB,float)).query(np.asarray(xyzA,float),k=3);near=cKDTree(rawBmodel).query(xyzA)[0];valid=(near<=.25)&(dist[:,-1]<=1.6);w=1/np.maximum(dist,1e-8);w/=w.sum(1,keepdims=True);zero=dist[:,0]<=1e-8;w[zero]=[1.,0.,0.];ix=torch.as_tensor(ind,device=featuresB.device);wt=torch.as_tensor(w,device=featuresB.device,dtype=featuresB.dtype);out=(featuresB[ix]*wt[:,:,None]).sum(1);return out,torch.as_tensor(valid,device=featuresB.device),{'static_raw_supported':int((near<=.25).sum()),'threeNN_supported':int((dist[:,-1]<=1.6).sum()),'aligned_seed_n':int(valid.sum()),'third_neighbor_median':float(np.median(dist[:,-1])),'identity_zero_distance_rows':int(zero.sum())}
def extra106(base,A,pair,z0,kind,slot):
 regpath=R106/'crossplatform/round105/registration_full_PRIVATE'/(pair['pair_PRIVATE']+'.json');reg=jread(regpath)
 if not reg['geometry_cycle_supported']:return None,{'status':reg['status']}
 sensor=pair['B'] if kind=='real' else A;assert sensor['eligible_for_extra_media'];actual=sha(sensor['pointpath']);expected=reg['B_sensor' if kind=='real' else 'A_sensor']['point_sha256'];assert actual==expected;raw=geo.points(sensor);assert np.isfinite(raw).all();rng=np.random.default_rng(int(hashlib.sha256(('R106sample|'+pair['pair_PRIVATE']+'|'+kind+'|'+slot).encode()).hexdigest()[:16],16));ids=rng.choice(len(raw),16384,replace=len(raw)<16384);pts=np.asarray(raw[ids],float)
 if kind=='real':pts=geo.transform_points(pts,np.asarray(reg['forward']['T_raw_B_to_A']));fullraw=geo.transform_points(np.asarray(raw,float),np.asarray(reg['forward']['T_raw_B_to_A']))
 else:fullraw=np.asarray(raw,float)
 M=geo.raw_to_model_transform(geo.camera(A));pts=np.ascontiguousarray(geo.transform_points(pts,M),np.float32);full=geo.transform_points(fullraw,M)
 with torch.no_grad():out=base.backbone_net(torch.as_tensor(pts,device='cuda')[None]);v=out['fp2_features'][0].T;xyz=out['fp2_xyz'][0].cpu().numpy();aligned,valid,proof=interpolation106(z0['xyz'][0],xyz,v,full)
 # Unsupported canonical points retain the immutable source A feature.
 aa=torch.as_tensor(z0['vis0'][0],device='cuda');aligned=torch.where(valid[:,None],aligned,aa);proof.update(status='usable' if valid.any() else 'no_seed_alignment',kind=kind,sample_points=16384,point_sha256=actual,sample_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),registration_sha256=sha(regpath));return ({'vis':aligned,'valid':valid} if valid.any() else None),proof
def wrong106(v,r,slot):
 if v is None:return None
 ids=torch.nonzero(v['valid'],as_tuple=True)[0];perm=np.random.default_rng(int(hashlib.sha256(('R106wrong|'+r['id']+'|'+slot).encode()).hexdigest()[:16],16)).permutation(len(ids));out=v['vis'].clone();out[ids]=out[ids[torch.as_tensor(perm,device='cuda')]];return {'vis':out,'valid':v['valid']}
def prefix106(z,observations,fused=False):
 if not observations:return z
 A=z['vis0'][0];num=A.clone() if fused else torch.zeros_like(A);den=torch.ones((len(A),1),device=A.device) if fused else torch.zeros((len(A),1),device=A.device)
 for v in observations:
  valid=v['valid'][:,None];num+=v['vis']*valid;den+=valid
 V=torch.where(den>0,num/den.clamp_min(1),A);return dict(z,vis0=V[None])
def feature106(m,base,z,d,route,S,observations,fused=False):
 return native105(m,base,prefix106(z,observations,fused),d,route,S,[])
