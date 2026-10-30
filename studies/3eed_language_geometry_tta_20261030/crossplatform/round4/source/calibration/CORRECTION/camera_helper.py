"""Camera-only projection API. No annotation/caption/image reads or learned model."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
R=Path(__file__).resolve().parent;P=R.parents[3];sys.path.insert(0,str(P/'upstream'))
from utils.box_util import project_points_to_2d

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_camera(row):
 p=Path(row['camera_path']);assert sha(p)==row['camera_sha256']
 with np.load(p,allow_pickle=False) as z:
  return {k:z[k].copy() for k in ['image_intrinsic','camera_distortion','image_size_wh','source_to_camera']}
def project_source_points(points,camera):
 """Project source/model XYZ; valid requires FOV and depth>0.1m. No clipping.
 These pixels are geometric predictions, not measured 2D correspondences.
 """
 xyz=np.asarray(points);assert xyz.ndim==2 and xyz.shape[1]==3 and np.isfinite(xyz).all()
 w,h=map(int,camera['image_size_wh']);return project_points_to_2d(xyz,(h,w),{'image_intrinsic':camera['image_intrinsic'],'image_extrinsic':camera['source_to_camera'],'camera_distortion':camera['camera_distortion']})
