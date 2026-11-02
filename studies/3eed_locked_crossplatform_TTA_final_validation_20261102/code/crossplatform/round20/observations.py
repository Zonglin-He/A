"""Build independent masks and measured observations without task annotations."""
import sys
import numpy as np
from common import *
sys.path[:0]=[str(ROOT/'crossplatform/round15'),str(ROOT/'crossplatform/round13'),str(ROOT/'crossplatform/round7')]
from input_io import load_case
from scene_observation import build_scene_observation
from camera_projection import CameraSpec
def load(r):
 if r['stage']=='A':
  row={k:v for k,v in r.items() if k not in ('stage','alias')};cases={x['id']:x for x in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')};idx=read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');frame,_,binding=load_case(row,cases[r['id']],idx)
 else:
  from PIL import Image
  fr=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json')[r['frame_alias']]
  for k,h in [('imagepath','image_sha256'),('pointpath','point_sha256'),('metapath','metadata_sha256')]:assert sha(r[k])==fr[h]
  rgb=np.array(Image.open(r['imagepath']).convert('RGB'));p=Path(r['pointpath']);raw=np.load(p) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);raw=np.ascontiguousarray(raw[:,:3],np.float32);fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
  if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
  camera=CameraSpec.from_camera_fields(r['platform'],fields,missing_distortion_policy='assume_zero_declared');mp=ROOT/'crossplatform/round12/masks501_attempt001'/(r['frame_alias']+'.npz');ms=read(mp.parent/'SEALED.json');assert ms['files'][mp.name]==sha(mp)
  with np.load(mp) as z:ins={k:z[k].copy() for k in ('boxes_xyxy','class_ids','scores','masks_bool','instance_indices')};assert tuple(z['original_hw'])==rgb.shape[:2]
  frame={'rgb':rgb,'raw_points':raw,'camera':camera,'instances':ins};binding={'image_sha256':fr['image_sha256'],'point_sha256':fr['point_sha256'],'mask_cache_sha256':sha(mp)}
 h,w=frame['rgb'].shape[:2];scene=build_scene_observation(raw_points=frame['raw_points'],camera_spec=frame['camera'],image_size_wh=(w,h),instances=frame['instances'],image_sha256=binding['image_sha256']);return frame,scene,binding
