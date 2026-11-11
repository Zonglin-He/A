"""Independent projection algebra and local graph invariants on all inputs."""
from common76 import *
from scipy.spatial import cKDTree
import time
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import read_camera_allowlist
rr=source_rows('TRAIN512')+source_rows('VAL128')+target_rows();ss=read(BASE/'prepared976_attempt001/SEALED.json');maxerror=0;start=time.perf_counter();proof=[]
for i,r in enumerate(rr):
 f=casepath(r);assert sha(f)==ss['files'][f.relative_to(BASE/'prepared976_attempt001').as_posix()];g=load(f);z=prefix(r);xyz=z['xyz'][0].astype(float);fields=read_camera_allowlist(r['metapath']);E=np.asarray(fields['image_extrinsic']);K=np.asarray(fields['image_intrinsic'])[:,:3];T=np.eye(4)
 if r['platform']=='waymo':
  deg=[0,-45,45,-90,90][r['suffix']];a=np.deg2rad(-deg);R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]);T[:3,:3]=R;axis=np.eye(4);axis[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]];T=axis@np.linalg.inv(E)@T
 else:
  if r['platform']=='drone':T[2,3]=-1.8
  T=E@T
 X=(np.c_[xyz,np.ones(len(xyz))]@T.T)[:,:3];front=X[:,2]>.1
 from PIL import Image
 wh=Image.open(r['imagepath']).size;D=fields.get('camera_distortion');D=np.zeros(5) if D is None else np.asarray(D).reshape(-1)
 if np.any(D):
  import cv2
  uv=cv2.projectPoints(X[front],np.zeros(3),np.zeros(3),K,D)[0][:,0]
 else:
  t=X[front]@K.T;uv=t[:,:2]/t[:,2:]
 expected=np.zeros((1024,2));expected[front]=uv*np.array([384/wh[0],256/wh[1]])/8;ok=front&np.isfinite(g['uv_grid']).all(1);err=float(abs(g['uv_grid'][ok]-expected[ok]).max()) if ok.any() else 0.;maxerror=max(maxerror,err);assert err<2e-4
 assert g['p2i'].shape==(1024,9) and g['i2p'].shape==(1536,8);assert not (g['p2i'][~g['visible']]>=0).any();assert np.all((g['p2i_weight']>0)==(g['p2i']>=0));assert np.all((g['i2p_weight']>0)==(g['i2p']>=0))
 jj,kk=np.where(g['i2p']>=0);ids=g['i2p'][jj,kk];cent=np.c_[jj%48+.5,jj//48+.5];dist=np.linalg.norm(g['uv_grid'][ids]-cent,axis=1);assert (dist<=1.50001).all() and g['visible'][ids].all();assert np.max(abs(g['point_depth']-X[:,2]))<1e-4
 proof.append({'row':f'INPUT_{i+1:04d}','platform':r['platform'],'visible_seeds':int(g['visible'].sum()),'reverse_tokens':int((g['i2p']>=0).any(1).sum()),'known_depth_cells':int(g['depth_valid'].sum()),'projection_max_error_feature_cells':err})
 if(i+1)%128==0:print('GRAPH_AUDIT76',i+1,len(rr),flush=True)
out=H/'independent_graph_audit_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R76_ALL976_INDEPENDENT_CAMERA_ALGEBRA_VISIBILITY_AND_SPARSE_GRAPH_INVARIANTS','n':976,'max_projection_error_feature_cells':maxerror,'new_target_GT_or_models':False,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);dump(H/'ALL_GEOMETRY_COVERAGE_ROWS.json',proof);print(rec)
