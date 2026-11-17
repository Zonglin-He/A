"""Full-frame native 2x image grids, with exact same physical visibility."""
from common152 import *
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0,str(R/'research/round151'))
from crop151 import target_geometry
from common152 import R,H,B,rd,sha,dump,check,utc
WIDTH,HEIGHT=768,512
def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def grid_graph(p,wh,visible,width=WIDTH,height=HEIGHT):
 assert width%8==0 and height%8==0
 gw,gh=width//8,height//8;uv=p['uv'];coord=np.where(np.isfinite(uv),uv,0)*np.array([gw/wh[0],gh/wh[1]]);ij=np.floor(coord).astype(int);cell=np.clip(ij[:,1],0,gh-1)*gw+np.clip(ij[:,0],0,gw-1);n=len(visible);edge=np.full((n,9),-1,int);weight=np.zeros((n,9),np.float32);offs=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
 for i in np.flatnonzero(visible):
  pxy=ij[i]+offs;ok=(pxy[:,0]>=0)&(pxy[:,0]<gw)&(pxy[:,1]>=0)&(pxy[:,1]<gh);edge[i,ok]=pxy[ok,1]*gw+pxy[ok,0];weight[i,ok]=np.exp(-np.square(pxy[ok]+.5-coord[i]).sum(1)/2)
 rev=np.full((gw*gh,8),-1,int);rw=np.zeros((gw*gh,8),np.float32);vi=np.flatnonzero(visible)
 if len(vi):
  yy,xx=np.indices((gh,gw));centers=np.c_[xx.ravel()+.5,yy.ravel()+.5];dist,ii=cKDTree(coord[vi]).query(centers,k=min(8,len(vi)));dist=np.asarray(dist).reshape(gw*gh,-1);ii=np.asarray(ii).reshape(gw*gh,-1);keys=vi[ii];dep=p['depth'][keys];front=np.where(dist<=1.5,dep,np.inf).min(1)
  for j in range(gw*gh):
   good=(dist[j]<=1.5)&(dep[j]<=front[j]+.5+.03*front[j]);kk=keys[j,good];rev[j,:len(kk)]=kk;rw[j,:len(kk)]=np.exp(-dist[j,good]**2/2)
 depth=np.zeros(gw*gh,np.float32);valid=np.zeros(gw*gh,bool)
 if len(vi):
  ds=np.full(gw*gh,np.inf);np.minimum.at(ds,cell[vi],p['depth'][vi]);valid=np.isfinite(ds);depth[valid]=np.log1p(ds[valid])
 return dict(p2i=edge,p2i_weight=weight,i2p=rev,i2p_weight=rw,visible=visible.copy(),cell=cell,depth_target=depth,depth_valid=valid,uv_grid=coord.astype(np.float32),point_depth=p['depth'].astype(np.float32))
def wrong_graph(g,salt):
 out={k:v.copy() for k,v in g.items()};perm=np.arange(len(g['p2i']));rng=np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16));degree=(g['p2i']>=0).sum(1)
 for d in range(1,10):
  order=rng.permutation(np.flatnonzero(g['visible']&(degree==d)))
  if len(order)>1:perm[order]=np.roll(order,1)
 inv=np.argsort(perm);out['p2i']=g['p2i'][perm];out['p2i_weight']=g['p2i_weight'][perm];out['i2p']=np.where(g['i2p']>=0,inv[g['i2p'].clip(0)],-1);assert np.array_equal((out['p2i']>=0).sum(1),degree) and np.array_equal(out['visible'],g['visible']);return out
def specs(e,z):
 native,projections,images=target_geometry(e,z);high=[];low=[];wrong=[]
 for v,(old,p,im) in enumerate(zip(native,projections,images)):
  assert im.width>=WIDTH and im.height>=HEIGHT
  g=grid_graph(p,im.size,old['visible']);g['query_anchor_indices']=z['query_points_sample_inds'];g['image']=np.asarray(im.resize((WIDTH,HEIGHT),Image.Resampling.BILINEAR));l={k:a.copy() for k,a in g.items()};l['image']=np.asarray(Image.fromarray(old['image']).resize((WIDTH,HEIGHT),Image.Resampling.BILINEAR));high.append(g);low.append(l);wrong.append(wrong_graph(g,e['source']+'|'+e['row']['id']+'|R152|'+str(v)))
 return dict(full_high=high,full_low=low,full_wrong=wrong)
def merged(gs):
 out={k:v.copy() for k,v in gs[0].items()};sizes=[len(g['i2p']) for g in gs];offset=np.cumsum([0]+sizes[:-1]);out.update(p2i=np.concatenate([np.where(g['p2i']>=0,g['p2i']+o,-1) for g,o in zip(gs,offset)],1),p2i_weight=np.concatenate([g['p2i_weight'] for g in gs],1),i2p=np.concatenate([g['i2p'] for g in gs]),i2p_weight=np.concatenate([g['i2p_weight'] for g in gs]),visible=np.logical_or.reduce([g['visible'] for g in gs]));assert out['p2i'].max()<sum(sizes);return out
