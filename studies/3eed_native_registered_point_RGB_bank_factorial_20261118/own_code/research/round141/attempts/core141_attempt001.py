"""Native spatial-token/physical-image-bank prerequisite helpers.

Original learned modules and A256 identities stay fixed. The route's original
A geometric PRIOR stays frozen; region support arrays explicitly extend to
extra points in the SAME original hypothesized boxes. No new object identity.
"""
from pathlib import Path
import sys,hashlib,contextlib
ROOT141=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT141/'research/round136'))
from optim136 import *
sys.path.insert(0,str(ROOT141/'research/round105'))
import geometry105 as geo141
from PIL import Image
from scipy.spatial import cKDTree
MAX_EXTRA141=512

def native_observation141(base,A,pair,z0,regpath,kind):
 reg=rd(ROOT141/regpath);assert reg['geometry_cycle_supported'];assert A['eligible_for_extra_media'] and not A['protected_image_content_alias'] and sha(A['pointpath'])==reg['A_sensor']['point_sha256'];sensor=pair['B'] if kind=='real' else A
 assert sensor['eligible_for_extra_media'] and not sensor['protected_image_content_alias']
 expected=reg['B_sensor' if kind=='real' else 'A_sensor']['point_sha256'];assert sha(sensor['pointpath'])==expected
 raw=geo141.points(sensor);assert np.isfinite(raw).all();salt=('R106sample|'+pair['pair_PRIVATE']+'|real|B') if kind=='real' else ('R141sample|'+pair['pair_PRIVATE']+'|sameA|B');rng=np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16));ids=rng.choice(len(raw),16384,replace=len(raw)<16384);pts=np.asarray(raw[ids],float)
 if kind=='real':pts=geo141.transform_points(pts,np.asarray(reg['forward']['T_raw_B_to_A']))
 M=geo141.raw_to_model_transform(geo141.camera(A));pts=np.ascontiguousarray(geo141.transform_points(pts,M),np.float32)
 with torch.no_grad():
  out=base.backbone_net(torch.as_tensor(pts,device='cuda')[None]);v=out['fp2_features'][0].T.detach();xyz=out['fp2_xyz'][0].detach().cpu().numpy()
 rawA=geo141.points(A);assert np.isfinite(rawA).all();fullA=geo141.transform_points(np.asarray(rawA,float),M);static=cKDTree(fullA).query(xyz)[0]<=.25;new=cKDTree(np.asarray(z0['xyz'][0],float)).query(xyz)[0]>1e-4;eligible=np.flatnonzero(static&new);chosen=eligible[:MAX_EXTRA141]
 # The real observation must also reproduce the historical interpolation provider.
 bridge=None
 if kind=='real':
  fullB=geo141.transform_points(geo141.transform_points(np.asarray(raw,float),np.asarray(reg['forward']['T_raw_B_to_A'])),M);aligned,valid,proof=interpolation106(z0['xyz'][0],xyz,v,fullB);Afeatures=torch.as_tensor(z0['vis0'][0],device='cuda');aligned=torch.where(valid[:,None],aligned,Afeatures);bridge={'vis':aligned,'valid':valid,'proof':proof}
 return {'xyz':xyz,'features':v,'eligible_indices':eligible,'selected_indices':chosen,'full_A_model':fullA,'bridge':bridge,'sample_sha256':hashlib.sha256(pts.tobytes()).hexdigest(),'static_count':int(static.sum()),'novel_count':len(eligible)}

def prefix_bank141(base,z0,anchor_features,xyz_extra,features_extra):
 z=tensor_prefix(z0);k=len(xyz_extra);assert len(anchor_features)==1024 and len(features_extra)==k and k<=MAX_EXTRA141
 if not k:return dict(z,vis0=anchor_features[None])
 xyz=torch.as_tensor(np.ascontiguousarray(xyz_extra),device='cuda',dtype=z['xyz'].dtype)[None];pos=base.pos_embed(xyz).transpose(1,2).contiguous();zn=dict(z,xyz=torch.cat([z['xyz'],xyz],1),vis0=torch.cat([anchor_features,features_extra],0)[None],pos0=torch.cat([z['pos0'],pos],1));assert torch.equal(zn['xyz'][:,:1024],z['xyz']) and torch.equal(zn['pos0'][:,:1024],z['pos0']);assert int(zn['query_points_sample_inds'].max())<1024
 for j in range(6):assert torch.equal(zn['position'+str(j)],z['position'+str(j)])
 return zn

def route_bank141(rt,boxes,extra_xyz):
 """Only extend region supports; explicitly retain original A physical priors."""
 k=len(extra_xyz);out={name:val.copy() for name,val in rt.items()}
 if not k:return out
 own=(np.abs(np.asarray(extra_xyz)[None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(-1);ids=rt['hypotheses_PRIVATE'];more=np.zeros(ids.shape+(k,),np.float32)
 for e in range(ids.shape[0]):
  for h in range(ids.shape[1]):
   if ids[e,h]>=0:more[e,h]=own[int(ids[e,h])]
 out['supports']=np.concatenate([rt['supports'],more],-1);out['own_support']=np.concatenate([rt['own_support'],own.astype(rt['own_support'].dtype)],-1);out['point_perm']=np.r_[rt['point_perm'],np.arange(1024,1024+k)]
 for name in ['hyp_geo','own_geo','own_valid','weights_plain','weights_relation','token_roles','relation_tokens']:assert np.array_equal(out[name],rt[name])
 assert np.array_equal(out['supports'][...,:1024],rt['supports']) and np.array_equal(out['own_support'][:,:1024],rt['own_support']);return out

def graph_A_bank141(A,z0,gA,xyz_extra,fullA):
 k=len(xyz_extra)
 if not k:return {n:v.copy() for n,v in gA.items()}
 assert A['eligible_for_extra_media'] and not A['protected_image_content_alias'];assert sha(A['imagepath'])==A['image_sha256'];wh=Image.open(A['imagepath']).size;cam=geo141.camera(A);xyz=np.concatenate([z0['xyz'][0],xyz_extra],0);pr=geo141.project_model_points(xyz,cam,wh);back=geo141.project_model_points(fullA,cam,wh);vis=geo141.visibility(pr,back,wh,True);vis[:1024]=gA['visible'];gg=geo141.graph_kernel(pr,wh,vis)
 for name in ['p2i','cell']:assert np.array_equal(gg[name][:1024],gA[name]),name
 assert np.max(abs(gg['p2i_weight'][:1024]-gA['p2i_weight']))<1e-5
 # Keep original forward A edges numerically exact; reverse edges use all points.
 for name in ['p2i','p2i_weight','visible','cell','uv_grid','point_depth']:gg[name][:1024]=gA[name]
 gg['image']=gA['image'];gg['query_anchor_indices']=z0['query_points_sample_inds'];return gg

def merge_graphs141(gs):
 n=len(gs[0]['p2i']);assert all(len(g['p2i'])==n for g in gs);out={k:v.copy() for k,v in gs[0].items()};out['p2i']=np.concatenate([np.where(g['p2i']>=0,g['p2i']+i*1536,-1) for i,g in enumerate(gs)],axis=1);out['p2i_weight']=np.concatenate([g['p2i_weight'] for g in gs],axis=1);out['i2p']=np.concatenate([g['i2p'] for g in gs]);out['i2p_weight']=np.concatenate([g['i2p_weight'] for g in gs]);out['visible']=np.logical_or.reduce([g['visible'] for g in gs]);assert out['p2i'].max()<len(gs)*1536 and out['i2p'].max()<n;return out

def wrong_extra_graph141(g,salt):
 out={k:v.copy() for k,v in g.items()};n=len(g['p2i']);ids=np.flatnonzero((g['p2i']>=0).any(1));perm=np.arange(n);rng=np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16));order=rng.permutation(ids)
 if len(order)>1:perm[order]=np.roll(order,1)
 inverse=np.argsort(perm);out['p2i']=g['p2i'][perm];out['p2i_weight']=g['p2i_weight'][perm];out['i2p']=np.where(g['i2p']>=0,inverse[g['i2p'].clip(0)],-1);assert np.array_equal((out['p2i']>=0).any(1),(g['p2i']>=0).any(1));return out

@contextlib.contextmanager
def rgb_banks141(m,extra_images):
 # Same already-audited R105 strategy: independent GroupNorm per image.
 extra=[m.rgb((x-.5)*2) for x in extra_images];hook=m.rgb.register_forward_hook(lambda _m,_a,o:torch.cat([o]+extra,dim=2)) if extra else None
 try:yield
 finally:
  if hook is not None:hook.remove()

def read_bank141(base,m,z,graphs,rt,r):
 g=merge_graphs141(graphs);d=tensor_data(g,r);n=z['vis0'].shape[1];d['wrong_permutation']=torch.arange(n,device='cuda');route=tensor_route89(rt);extra=[torch.as_tensor(q['image'].copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255 for q in graphs[1:]]
 with torch.no_grad(),rgb_banks141(m,extra):lp,a,_=live135(m,base,z,d,z['vis0'][0],route)
 return lp.exp().cpu().numpy(),{k:a[k].cpu().numpy() for k in FEATURES},g
