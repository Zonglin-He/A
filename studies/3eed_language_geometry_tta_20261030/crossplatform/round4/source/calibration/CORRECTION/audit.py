from pathlib import Path
import json,hashlib,sys
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;C=R.parent;S=C.parent;P=S.parents[1];sys.path.insert(0,str(P/'upstream'))
from utils.box_util import project_points_to_2d
ALLOWED=('image_intrinsic','image_extrinsic','camera_distortion','pose')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=json.loads((R/'PLAN.json').read_text());base=json.loads((S/'PATH_LOCK.json').read_text())['cohorts'];out=R/'arrays';out.mkdir(exist_ok=False);A=np.eye(4);A[:3,:3]=plan['A_3x3'];checks=[];manifests={}
 for part in ['dev','confirm']:
  rows=[]
  for s in base[part]:
   if s['platform']!='waymo':continue
   path=Path(s['meta_path']);container=json.loads(path.read_text());fields={k:container[k] for k in ALLOWED if k in container};del container
   K=np.asarray(fields['image_intrinsic'],float);E=np.asarray(fields['image_extrinsic'],float);pose=np.asarray(fields['pose'],float);D=np.asarray(fields.get('camera_distortion',np.zeros(4)),float);assert E.shape==pose.shape==(4,4)
   with Image.open(path.with_name('image.jpg')) as im:w,h=im.size
   raw=np.load(s['lidar_path'],allow_pickle=False)[:,:3];iE=np.linalg.inv(E);iP=np.linalg.inv(pose)
   transforms={'E':E,'invE':iE,'A_E':A@E,'A_invE':A@iE,'E_invPose':E@iP,'invE_invPose':iE@iP,'A_E_invPose':A@E@iP,'A_invE_invPose':A@iE@iP};assert list(transforms)==plan['fixed_candidates'];table={}
   for name,M in transforms.items():
    uv,depth,valid=project_points_to_2d(raw,(h,w),{'image_intrinsic':K,'image_extrinsic':M,'camera_distortion':D});cam=(np.c_[raw,np.ones(len(raw))]@M.T)[:,:3];den=cam[:,2];pinhole=(cam@K[:3,:3].T);finite=valid&np.isfinite(pinhole).all(1);res=None
    if np.count_nonzero(D)==0 and finite.any():
     expected=pinhole[finite,:2]/pinhole[finite,2:3];res=float(np.max(np.abs(expected-uv[finite])))
    table[name]={'positive_depth_fraction':float(np.mean(depth>0)),'FOV_depth_over_point1_fraction':float(np.mean(valid)),'pinhole_vs_cv2_max_abs_pixels_inFOV':res,'n_inFOV':int(valid.sum())}
   initial=np.load(C/'arrays'/(s['id']+'.npz'),allow_pickle=False);T=initial['source_to_raw'];fp=out/(s['id']+'.npz');np.savez_compressed(fp,image_intrinsic=K,image_extrinsic_raw=E,pose_vehicle_to_global=pose,camera_distortion=D,image_size_wh=np.array([w,h],np.int64),waymo_camera_to_opencv=A,source_to_raw=T,raw_vehicle_to_camera=A@iE,source_to_camera=A@iE@T)
   rec={'id':s['id'],'platform':'waymo','image_size_wh':[w,h],'image_principal_point_fraction_wh':[float(K[0,2]/w),float(K[1,2]/h)],'intrinsic_focal_positive':bool(K[0,0]>0 and K[1,1]>0),'raw_coordinate_abs_max_m':np.max(np.abs(raw),axis=0).astype(float).tolist(),'raw_coordinate_range_m':np.ptp(raw,axis=0).astype(float).tolist(),'E_rotation_det':float(np.linalg.det(E[:3,:3])),'E_rotation_orthogonality_max_error':float(np.max(np.abs(E[:3,:3].T@E[:3,:3]-np.eye(3)))),'E_translation_norm_m':float(np.linalg.norm(E[:3,3])),'pose_rotation_det':float(np.linalg.det(pose[:3,:3])),'pose_rotation_orthogonality_max_error':float(np.max(np.abs(pose[:3,:3].T@pose[:3,:3]-np.eye(3)))),'pose_translation_norm_m':float(np.linalg.norm(pose[:3,3])),'candidates':table};checks.append(rec)
   rows.append({'id':s['id'],'platform':'waymo','camera_path':str(fp),'camera_sha256':sha(fp),'mapping':'A @ inverse(E_camera_to_vehicle) @ source_to_raw','metadata_container_sha256':sha(path),'image_size_wh':[w,h]})
  mf=R/(part+'_manifest.json');mf.write_text(json.dumps(rows,indent=2)+'\n');manifests[part]={'n':len(rows),'sha256':sha(mf)}
 (R/'CHECKS_PRIVATE.json').write_text(json.dumps(checks,indent=2)+'\n');summary={name:{key:float(np.mean([x['candidates'][name][key] for x in checks])) for key in ['positive_depth_fraction','FOV_depth_over_point1_fraction']} for name in plan['fixed_candidates']}
 summary['coordinate_diagnostics']={key:[min(x[key] for x in checks),max(x[key] for x in checks)] for key in ['E_rotation_det','E_rotation_orthogonality_max_error','E_translation_norm_m','pose_rotation_det','pose_rotation_orthogonality_max_error','pose_translation_norm_m']}
 summary['pinhole_vs_cv2_A_invE_max_pixels_inFOV']=max(x['candidates']['A_invE']['pinhole_vs_cv2_max_abs_pixels_inFOV'] for x in checks if x['candidates']['A_invE']['pinhole_vs_cv2_max_abs_pixels_inFOV'] is not None)
 summary['n']=len(checks);(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');seal={'state':'finite_camera_audit_sealed','plan_sha256':sha(R/'PLAN.json'),'code_sha256':sha(Path(__file__)),'old_camera_seal_sha256':sha(C/'sealed.json'),'allowlist':list(ALLOWED),'annotation_fields_indexed':False,'no_caption_or_image_pixels':True,'no_GT_numerical_scores':True,'cohorts':manifests,'files':{str(p.relative_to(R)):sha(p) for p in out.glob('*.npz')},'checks_sha256':sha(R/'CHECKS_PRIVATE.json'),'summary_sha256':sha(R/'SUMMARY.json')};(R/'sealed.json').write_text(json.dumps(seal,indent=2)+'\n');print(json.dumps(summary,indent=2));print('SEAL',sha(R/'sealed.json'))
if __name__=='__main__':main()
