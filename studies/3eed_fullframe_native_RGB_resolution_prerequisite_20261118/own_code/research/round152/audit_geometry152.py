"""Independent variable-resolution physical graph and native reader audit."""
from common152 import *
import numpy as np,torch
from PIL import Image
from scipy.spatial import cKDTree
argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round141'));from audit141 import cam,tx,raw_to_model_transform,project_model_points,cnn,read135,tensors;sys.argv=argv
from common152 import R,H,B,rd,sha,dump,check,utc
def reconstruct_grid(uv,depth,visible,wh,width,height):
 gw,gh=width//8,height//8;q=(np.nan_to_num(uv,nan=0,posinf=0,neginf=0)/np.asarray(wh))*[gw,gh];floor=np.floor(q).astype(int);N=len(q);edges=np.full((N,9),-1,int);pw=np.zeros((N,9),np.float32)
 for yy in range(-1,2):
  for xx in range(-1,2):
   slot=(yy+1)*3+xx+1;cell=floor+[xx,yy];ok=visible&(cell[:,0]>=0)&(cell[:,0]<gw)&(cell[:,1]>=0)&(cell[:,1]<gh);edges[ok,slot]=cell[ok,1]*gw+cell[ok,0];pw[ok,slot]=np.exp(-np.sum((cell[ok]+.5-q[ok])**2,axis=1)/2)
 vv=np.flatnonzero(visible);reverse=np.full((gw*gh,8),-1,int);rw=np.zeros((gw*gh,8),np.float32)
 if vv.size:
  centers=np.array([[x+.5,y+.5] for y in range(gh) for x in range(gw)]);distance,nearest=cKDTree(q[vv]).query(centers,k=min(8,len(vv)));distance=np.asarray(distance).reshape(len(centers),-1);nearest=vv[np.asarray(nearest).reshape(len(centers),-1)];dep=depth[nearest];closest=np.min(np.where(distance<=1.5,dep,np.inf),axis=1);ok=(distance<=1.5)&(dep<=closest[:,None]+.5+.03*closest[:,None])
  for row in np.flatnonzero(ok.any(1)):
   use=np.flatnonzero(ok[row]);reverse[row,:len(use)]=nearest[row,use];rw[row,:len(use)]=np.exp(-distance[row,use]**2/2)
 return dict(p2i=edges,p2i_weight=pw,i2p=reverse,i2p_weight=rw,visible=visible.copy())
def independent_specs(e,z,decode=True):
 src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;bb=np.load(R/'crossplatform/round145/image'/name/'banks_PRIVATE'/(sid+'.npz'));sensors=[e['A_sensor'],e['pair_PRIVATE']['B']];images=[]
 for ob in sensors:
  assert ob['eligible_for_extra_media'] and not ob['protected_image_content_alias'];assert sha(ob['imagepath'])==ob['image_sha256'];im=Image.open(ob['imagepath']);images.append(im.convert('RGB') if decode else im)
 ca,cb=[cam(x) for x in sensors];Ma,Mb=[raw_to_model_transform(x) for x in [ca,cb]];rr=rd(R/e['registration_path']);assert sha(R/e['registration_path'])==e['registration_sha256'];rawB=tx(tx(z['xyz'][0],np.linalg.inv(Ma)),np.linalg.inv(np.asarray(rr['forward']['T_raw_B_to_A'])));projections=[project_model_points(z['xyz'][0],ca,images[0].size),project_model_points(tx(rawB,Mb),cb,images[1].size)];hi=[];lo=[];wrong=[]
 for v,(im,p) in enumerate(zip(images,projections)):
  assert im.width>=768 and im.height>=512;vis=bb['norm_'+['A','B'][v]+'image_visible'];g=reconstruct_grid(p['uv'],p['depth'],vis,im.size,768,512);native=reconstruct_grid(p['uv'],p['depth'],vis,im.size,384,256)
  for k,a in native.items():
   old=bb['norm_'+['A','B'][v]+'image_'+k]
   if k.endswith('weight'):assert abs(a-old).max(initial=0)<1e-5
   else:assert np.array_equal(a,old),(name,k)
  if decode:g['image']=np.asarray(im.resize((768,512),Image.Resampling.BILINEAR))
  low={k:a.copy() for k,a in g.items()}
  if decode:low['image']=np.asarray(im.resize((384,256),Image.Resampling.BILINEAR).resize((768,512),Image.Resampling.BILINEAR))
  bad={k:a.copy() for k,a in g.items()};degree=np.count_nonzero(g['p2i']>=0,axis=1);perm=np.arange(1024);salt=src+'|'+sid+'|R152|'+str(v);rng=np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16))
  for deg in range(1,10):
   ids=np.flatnonzero(vis&(degree==deg));ids=rng.permutation(ids)
   if len(ids)>1:perm[ids]=np.r_[ids[-1],ids[:-1]]
  inverse=np.argsort(perm);bad['p2i']=g['p2i'][perm];bad['p2i_weight']=g['p2i_weight'][perm];bad['i2p']=np.where(g['i2p']>=0,inverse[np.maximum(g['i2p'],0)],-1);assert np.array_equal((bad['p2i']>=0).sum(1),degree);hi.append(g);lo.append(low);wrong.append(bad)
 return dict(full_high=hi,full_low=lo,full_wrong=wrong)
def independent_merge(gs):
 offsets=np.r_[0,np.cumsum([len(g['i2p']) for g in gs])];return dict(p2i=np.concatenate([np.where(g['p2i']>=0,g['p2i']+offsets[i],-1) for i,g in enumerate(gs)],1),p2i_weight=np.concatenate([g['p2i_weight'] for g in gs],1),i2p=np.vstack([g['i2p'] for g in gs]),i2p_weight=np.vstack([g['i2p_weight'] for g in gs]),visible=np.any(np.stack([g['visible'] for g in gs]),axis=0))
