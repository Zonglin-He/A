"""Canonical A seeds -> additional sensor camera, with measured surface checks.

No annotation/semantic model. A graph is retained exactly; extra unsupported
points are UNKNOWN. A candidate/region identity is not an instance label.
"""
from pathlib import Path
import sys, numpy as np
from PIL import Image
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R/'crossplatform/round7'),str(R/'crossplatform/round12')]
from camera_projection import CameraSpec,read_camera_allowlist,project_model_points,raw_to_model_points
from temporal_association import raw_to_model_transform,inverse_se3,transform_points
GW,GH=48,32
def camera(r):return CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),waymo_view_suffix=int(r.get('suffix') or 0),missing_distortion_policy='assume_zero_declared')
def points(r):
 p=Path(r['pointpath']);a=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);return np.asarray(a[:,:3])
def graph_kernel(p,wh,visible):
 uv=p['uv'];uvr=np.where(np.isfinite(uv),uv,0)*np.array([384/wh[0],256/wh[1]]);coord=uvr/8;grid=np.floor(coord).astype(int);cell=np.clip(grid[:,1],0,GH-1)*GW+np.clip(grid[:,0],0,GW-1);n=len(visible);p2i=np.full((n,9),-1,int);iw=np.zeros((n,9),np.float32);offs=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
 for i in np.flatnonzero(visible):
  cc=grid[i]+offs;ok=(cc[:,0]>=0)&(cc[:,0]<GW)&(cc[:,1]>=0)&(cc[:,1]<GH);ids=cc[:,1]*GW+cc[:,0];p2i[i,ok]=ids[ok];iw[i,ok]=np.exp(-np.square(cc[ok]+.5-coord[i]).sum(1)/2).astype(np.float32)
 rev=np.full((GH*GW,8),-1,int);rw=np.zeros((GH*GW,8),np.float32);vis=np.flatnonzero(visible)
 if len(vis):
  yy,xx=np.indices((GH,GW));centers=np.c_[xx.reshape(-1)+.5,yy.reshape(-1)+.5];dist,ix=cKDTree(coord[vis]).query(centers,k=min(8,len(vis)));dist=np.asarray(dist).reshape(GH*GW,-1);ix=np.asarray(ix).reshape(GH*GW,-1);keys=vis[ix];dep=p['depth'][keys];mindep=np.where(dist<=1.5,dep,np.inf).min(1)
  for j in range(len(centers)):
   good=(dist[j]<=1.5)&(dep[j]<=mindep[j]+.5+.03*mindep[j]);kk=keys[j,good];rev[j,:len(kk)]=kk;rw[j,:len(kk)]=np.exp(-dist[j,good]**2/2)
 depth=np.zeros(GH*GW,np.float32);valid=np.zeros(GH*GW,bool)
 if len(vis):
  ds=np.full(GH*GW,np.inf);np.minimum.at(ds,cell[vis],p['depth'][vis]);valid=np.isfinite(ds);depth[valid]=np.log1p(ds[valid]).astype(np.float32)
 return {'p2i':p2i,'p2i_weight':iw,'i2p':rev,'i2p_weight':rw,'visible':visible,'cell':cell,'depth_target':depth,'depth_valid':valid,'uv_grid':coord.astype(np.float32),'point_depth':p['depth'].astype(np.float32)}
def visibility(p,background,wh,symmetric=False):
 ii=np.flatnonzero(background['in_image']);ap=np.floor(background['uv'][ii]/4).astype(int);bw=(wh[0]+3)//4;bc=ap[:,1]*bw+ap[:,0];buf=np.full(bw*((wh[1]+3)//4),np.inf);np.minimum.at(buf,bc,background['depth'][ii]);visible=p['in_image'].copy();pi=np.flatnonzero(visible);uvq=np.floor(p['uv'][pi]/4).astype(int);nearest=buf[uvq[:,1]*bw+uvq[:,0]];tol=.3+.02*p['depth'][pi]
 visible[pi]&=np.isfinite(nearest)&((np.abs(p['depth'][pi]-nearest)<=tol) if symmetric else (p['depth'][pi]<=nearest+tol));return visible
def original_rebuild(r,z):
 cam=camera(r);wh=Image.open(r['imagepath']).size;raw=points(r);xyz=raw_to_model_points(raw[np.asarray(z['point_indices'],int)],r['platform'],int(r.get('suffix') or 0));seeds=z['xyz'][0].astype(float);assert cKDTree(xyz).query(seeds)[0].max()<1e-4;p=project_model_points(seeds,cam,wh);return graph_kernel(p,wh,visibility(p,project_model_points(xyz,cam,wh),wh))
def additional(A,B,z,registration):
 assert registration['geometry_cycle_supported'];ca,cb=camera(A),camera(B);im=Image.open(B['imagepath']).convert('RGB');wh=im.size;seeds=z['xyz'][0].astype(float);rawA=transform_points(seeds,inverse_se3(raw_to_model_transform(ca)));rawB=transform_points(rawA,inverse_se3(np.asarray(registration['forward']['T_raw_B_to_A'])));modelB=raw_to_model_points(rawB,B['platform'],int(B.get('suffix') or 0));pb=points(B);pb=pb[np.isfinite(pb).all(1)];p=project_model_points(modelB,cb,wh);bg=project_model_points(raw_to_model_points(pb,B['platform'],int(B.get('suffix') or 0)),cb,wh);vis=visibility(p,bg,wh,True);near=cKDTree(pb).query(rawB)[0];vis&=near<=.25;g=graph_kernel(p,wh,vis);g['image']=np.array(im.resize((384,256),Image.Resampling.BILINEAR));g['query_anchor_indices']=np.asarray(z['query_points_sample_inds'],np.int64);proof={'registration_supported':True,'visible_seeds':int(vis.sum()),'mapped_queries':int(vis[np.asarray(z['query_points_sample_inds'],int)].sum()),'nearest_return_limit_m':.25,'symmetric_depth_consistency':True,'canonical_A_geometry_retained':True};return g,proof
def merge_graphs(gs):
 assert gs;out={k:v.copy() for k,v in gs[0].items()};out['p2i']=np.concatenate([np.where(g['p2i']>=0,g['p2i']+i*1536,-1) for i,g in enumerate(gs)],axis=1);out['p2i_weight']=np.concatenate([g['p2i_weight'] for g in gs],axis=1);out['i2p']=np.concatenate([g['i2p'] for g in gs],axis=0);out['i2p_weight']=np.concatenate([g['i2p_weight'] for g in gs],axis=0);out['visible']=np.logical_or.reduce([g['visible'] for g in gs]);assert out['p2i'].max()<len(gs)*1536 and out['i2p'].max()<1024;return out
def shuffled_extra(g,seed):
 out={k:v.copy() for k,v in g.items()};ids=np.flatnonzero((g['p2i']>=0).any(1));perm=np.arange(1024);perm[ids]=np.random.default_rng(seed).permutation(ids);inverse=np.argsort(perm);out['p2i']=g['p2i'][perm];out['p2i_weight']=g['p2i_weight'][perm];out['i2p']=np.where(g['i2p']>=0,inverse[g['i2p'].clip(0)],-1);assert np.array_equal((out['p2i']>=0).any(1),(g['p2i']>=0).any(1));return out
