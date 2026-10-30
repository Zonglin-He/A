"""Finite allowlist camera extraction. CPU only; no target annotations indexed.
Official JSON containers are parsed, but only three camera keys are requested.
PIL reads image header sizes only; no pixels, learned features or geometry labels.
"""
from pathlib import Path
import json,hashlib,sys,datetime
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;S=R.parent;P=S.parents[1];sys.path.insert(0,str(P/'upstream'))
from utils.box_util import project_points_to_2d
ALLOWED=('image_intrinsic','image_extrinsic','camera_distortion')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pathlock=S/'PATH_LOCK.json';assert sha(pathlock)==(S/'PATH_LOCK.sha256').read_text().strip();cohorts=json.loads(pathlock.read_text())['cohorts'];dest=R/'arrays';dest.mkdir(exist_ok=False);rows={};checks=[]
 for part in ['dev','confirm']:
  samples={s['id']:s for s in json.loads((S/'inputs'/(part+'_manifest.json')).read_text())};out=[]
  for s in cohorts[part]:
   path=Path(s['meta_path']);container=json.loads(path.read_text());camera={k:container[k] for k in ALLOWED if k in container};del container
   image=path.with_name('image.jpg');status='available'
   if not image.is_file():status='missing_local_image_header'
   if 'image_intrinsic' not in camera or 'image_extrinsic' not in camera:status='missing_camera_matrix'
   if status!='available':out.append({'id':s['id'],'platform':s['platform'],'status':status});continue
   with Image.open(image) as im:width,height=im.size # Header-only, no .load()/convert()/pixel access.
   K=np.asarray(camera['image_intrinsic'],np.float64);E=np.asarray(camera['image_extrinsic'],np.float64);D=np.asarray(camera.get('camera_distortion',np.zeros(4)),np.float64);assert K.ndim==2 and K.shape[0]>=3 and K.shape[1]>=3 and E.shape==(4,4) and np.isfinite(K).all() and np.isfinite(E).all() and np.isfinite(D).all()
   T=np.eye(4);angle=0
   if s['platform']=='waymo':
    view_index=int(path.parent.name.split('_')[-1]);angle=[0,-45,45,-90,90][view_index];rad=np.deg2rad(-angle);c,ss=np.cos(rad),np.sin(rad);T[:3,:3]=[[c,-ss,0],[ss,c,0],[0,0,1]]
   elif s['platform']=='drone':T[2,3]=-1.8
   E_source=E@T;fp=dest/(s['id']+'.npz');np.savez_compressed(fp,image_intrinsic=K,image_extrinsic=E,camera_distortion=D,distortion_present=np.array('camera_distortion' in camera),image_size_wh=np.array([width,height],np.int64),source_to_raw=T,source_to_camera=E_source,official_front_rotation_degrees=np.array(angle,np.int64))
   lp=Path(s['lidar_path']);raw=np.load(lp,allow_pickle=False) if lp.suffix=='.npy' else np.fromfile(lp,np.float32).reshape(-1,4);xyz=raw[:,:3]
   projected,depth,valid=project_points_to_2d(xyz,(height,width),camera)
   item=samples[s['id']];vp=item['views'][0];pc=np.load(vp['point_path'],allow_pickle=False);ix=np.load(vp['raw_indices_path'],allow_pickle=False);assert sha(Path(vp['point_path']))==vp['point_sha256'] and sha(Path(vp['raw_indices_path']))==vp['raw_indices_sha256']
   # Equivalent direct source-to-camera route, checked against original raw points.
   inv=(np.c_[pc,np.ones(len(pc))]@T.T)[:,:3];err=float(np.max(np.abs(inv-xyz[ix])))
   p2,d2,v2=project_points_to_2d(pc,(height,width),{'image_intrinsic':K,'image_extrinsic':E_source,'camera_distortion':D})
   pr,dr,vr=project_points_to_2d(xyz[ix],(height,width),camera);good=np.isfinite(p2).all(1)&np.isfinite(pr).all(1)&(dr>.1)
   report={'id':s['id'],'platform':s['platform'],'split':s['split'],'image_size_wh':[width,height],'raw_points':len(xyz),'raw_positive_depth_fraction':float(np.mean(depth>0)),'raw_depth_over_point1_fraction':float(np.mean(depth>.1)),'raw_FOV_and_depth_over_point1_fraction':float(np.mean(valid)),'raw_finite_projected_fraction':float(np.isfinite(projected).all(1).mean()),'sample_positive_depth_fraction':float(np.mean(d2>0)),'sample_FOV_depth_fraction':float(np.mean(v2)),'source_to_raw_max_abs_roundoff_m':err,'sample_projection_delta_max_pixels_positive_depth':float(np.max(np.abs(p2[good]-pr[good]))) if good.any() else None,'sample_FOV_mask_disagreement_count':int(np.count_nonzero(v2!=vr)),'distortion_present':'camera_distortion' in camera,'intrinsic_shape':list(K.shape),'pose_used':False,'status':status};checks.append(report)
   out.append({'id':s['id'],'platform':s['platform'],'split':s['split'],'status':status,'camera_path':str(fp),'camera_sha256':sha(fp),'image_size_wh':[width,height],'pose_used':False})
  mf=R/(part+'_manifest.json');mf.write_text(json.dumps(out,indent=2)+'\n');rows[part]={'manifest_sha256':sha(mf),'n':len(out),'camera_available':sum(x['status']=='available' for x in out)}
 (R/'projection_checks_PRIVATE.json').write_text(json.dumps(checks,indent=2)+'\n')
 seal={'state':'sealed_camera_metadata_only','prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path_lock_sha256':sha(pathlock),'input_manifest_sha256':{p:sha(S/'inputs'/(p+'_manifest.json')) for p in ['dev','confirm']},'extract_code_sha256':sha(Path(__file__)),'official_projection_code_sha256':sha(P/'upstream/utils/box_util.py'),'allowlist':list(ALLOWED),'JSON_containers_with_annotations_opened':True,'annotation_fields_indexed':False,'no_pixel_features':True,'image_dimensions':'PIL.Image.open header size only; no guessed 2cx width','pose_used':False,'official_projection':'raw homogeneous row points @ image_extrinsic.T; cv2.projectPoints float32 camera XYZ with K[:3,:3] and D; FOV within W/H and depth>.1','source_coordinates':'source_to_raw is inverse suffix-dependent XY front rotation for Waymo; Drone z-=1.8; Quad identity. source_to_camera=E_raw@source_to_raw. Unused official translation constants are not introduced.','missing_camera_policy':'explicit unavailable; never invent dimensions or camera transform','cohorts':rows,'files':{str(p.relative_to(R)):sha(p) for p in sorted(dest.glob('*.npz'))},'projection_checks_sha256':sha(R/'projection_checks_PRIVATE.json'),'claim_limit':'Projection/FOV fractions and coordinate round-trip support coordinate consistency, not semantic object association or a GT validated independent teacher.'}
 (R/'sealed.json').write_text(json.dumps(seal,indent=2)+'\n');print(json.dumps({'cohorts':rows,'projection_summary':{p:{'n':sum(x['platform']==p for x in checks),'raw_FOV_depth_fraction_mean':float(np.mean([x['raw_FOV_and_depth_over_point1_fraction'] for x in checks if x['platform']==p])),'max_roundtrip_m':max(x['source_to_raw_max_abs_roundoff_m'] for x in checks if x['platform']==p)} for p in ['waymo','drone','quad']},'seal_sha256':sha(R/'sealed.json')},indent=2))
if __name__=='__main__':main()
