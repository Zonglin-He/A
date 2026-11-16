"""Calibrated seed/pixel sparse graphs and raw RGB, no target labels or experts."""
from common76 import *
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points,raw_to_model_points
import time,argparse
W,HIMG=384,256;GW,GH=48,32

def geometry(r,z):
 im=Image.open(r['imagepath']).convert('RGB');wh=im.size;rgb=np.array(im.resize((W,HIMG),Image.Resampling.BILINEAR));fields=read_camera_allowlist(r['metapath']);cam=CameraSpec.from_camera_fields(r['platform'],fields,waymo_view_suffix=int(r.get('suffix') or 0) if r['platform']=='waymo' else 0,missing_distortion_policy='assume_zero_declared');raw=(np.load(r['pointpath'],allow_pickle=False) if Path(r['pointpath']).suffix=='.npy' else np.fromfile(r['pointpath'],np.float32).reshape(-1,4))[:,:3];xyz=raw_to_model_points(raw[np.asarray(z['point_indices'],int)],r['platform'],int(r.get('suffix') or 0));seeds=z['xyz'][0].astype(float);near=cKDTree(xyz).query(seeds,k=1)[0];assert near.max()<1e-4,('seed_raw_coordinate_error',near.max());p=project_model_points(seeds,cam,wh);allp=project_model_points(xyz,cam,wh);ii=np.flatnonzero(allp['in_image']);ap=np.floor(allp['uv'][ii]/4).astype(int);bw=(wh[0]+3)//4;bc=ap[:,1]*bw+ap[:,0];buf=np.full(bw*((wh[1]+3)//4),np.inf);np.minimum.at(buf,bc,allp['depth'][ii]);visible=p['in_image'].copy();uv=p['uv'].copy();scale=np.array([W/wh[0],HIMG/wh[1]]);uvr=np.where(np.isfinite(uv),uv,0)*scale;coord=uvr/8;grid=np.floor(coord).astype(int);cell=np.clip(grid[:,1],0,GH-1)*GW+np.clip(grid[:,0],0,GW-1);pi=np.flatnonzero(visible);uvq=np.floor(uv[pi]/4).astype(int);nearest=buf[uvq[:,1]*bw+uvq[:,0]];tol=.3+.02*p['depth'][pi];visible[pi]&=p['depth'][pi]<=nearest+tol
 n=len(seeds);i2p=np.full((n,9),-1,int);iw=np.zeros((n,9),np.float32);offs=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
 for i in np.flatnonzero(visible):
  cc=grid[i]+offs;ok=(cc[:,0]>=0)&(cc[:,0]<GW)&(cc[:,1]>=0)&(cc[:,1]<GH);ids=cc[:,1]*GW+cc[:,0];i2p[i,ok]=ids[ok];iw[i,ok]=np.exp(-np.square(cc[ok]+.5-coord[i]).sum(1)/2).astype(np.float32)
 # Reverse local keys, visibility/front-depth constrained, never global attention.
 rev=np.full((GH*GW,8),-1,int);rw=np.zeros((GH*GW,8),np.float32);vis=np.flatnonzero(visible)
 if len(vis):
  yy,xx=np.indices((GH,GW));centers=np.c_[xx.reshape(-1)+.5,yy.reshape(-1)+.5];dist,ix=cKDTree(coord[vis]).query(centers,k=min(8,len(vis)));dist=np.asarray(dist).reshape(GH*GW,-1);ix=np.asarray(ix).reshape(GH*GW,-1);keys=vis[ix];dep=p['depth'][keys];mindep=np.where(dist<=1.5,dep,np.inf).min(1)
  for j in range(len(centers)):
   good=(dist[j]<=1.5)&(dep[j]<=mindep[j]+.5+.03*mindep[j]);kk=keys[j,good];rev[j,:len(kk)]=kk;rw[j,:len(kk)]=np.exp(-dist[j,good]**2/2)
 depth=np.zeros(GH*GW,np.float32);valid=np.zeros(GH*GW,bool)
 if len(vis):
  ds=np.full(GH*GW,np.inf);np.minimum.at(ds,cell[vis],p['depth'][vis]);valid=np.isfinite(ds);depth[valid]=np.log1p(ds[valid]).astype(np.float32)
 return {'image':rgb,'p2i':i2p,'p2i_weight':iw,'i2p':rev,'i2p_weight':rw,'visible':visible,'cell':cell,'depth_target':depth,'depth_valid':valid,'uv_grid':coord.astype(np.float32),'point_depth':p['depth'].astype(np.float32),'query_anchor_indices':np.asarray(z['query_points_sample_inds'],np.int64)}, {'seed_raw_max_error':float(near.max()),'visible_seeds':int(visible.sum()),'RGB_cells_with_depth':int(valid.sum()),'reverse_RGB_tokens':int((rev>=0).any(1).sum()),'image_sha256':sha(r['imagepath']),'point_sha256':sha(r['pointpath']),'camera_fields_only':True,'missing_distortion_assumed_zero':cam.distortion is None}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args();rr=source_rows('TRAIN512')+source_rows('VAL128')+target_rows()
 if a.engineering:rr=[next(r for r in rr if r['platform']=='waymo' and r['suffix']==s) for s in range(5)]+[next(r for r in rr if r['platform']==pl) for pl in ['drone','quad']]
 out=H/'engineering_geometry_attempt004' if a.engineering else BASE/'prepared976_attempt001';assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();start=time.perf_counter();counts=[]
 for i,r in enumerate(rr):
  if a.engineering and r['platform']=='waymo':
   z=load(H/'engineering_capture_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
  else:z=prefix(r)
  data,proof=geometry(r,z);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),**data);dump(out/'cases_PRIVATE'/(r['id']+'.json'),proof);counts.append({'platform':r['platform'],**{k:proof[k] for k in ['visible_seeds','RGB_cells_with_depth','reverse_RGB_tokens','seed_raw_max_error']}})
  if(i+1)%64==0:print('PREPARE76',i+1,len(rr),flush=True)
 rec={'n':len(rr),'no_target_GT_or_expert_read':True,'visibility_coverage':counts,'seconds':time.perf_counter()-start};dump(out/'COUNTS.json',rec);seal(out,'R76_RAW_RGB_AND_LOCAL_BIDIRECTIONAL_PHYSICAL_CORRESPONDENCES_NO_TARGET_GT',n=len(rr));print({'n':len(rr),'seconds':rec['seconds'],'zero_visible_inputs':sum(x['visible_seeds']==0 for x in counts)})
if __name__=='__main__':main()
