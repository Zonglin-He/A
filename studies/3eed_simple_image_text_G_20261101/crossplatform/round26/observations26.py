"""Target observations only; camera/mask correspondence contains no task labels."""
import numpy as np
from common26 import *
import observations as legacy
from scene_observation import build_scene_observation
from camera_projection import CameraSpec
def load(r):
 if r['stage']!='P':return legacy.load(r)
 from PIL import Image
 fr=read(EXP.parent/'input/P_FRAME_INDEX_PRIVATE.json')[r['frame_alias']]
 for k,h in [('imagepath','image_sha256'),('pointpath','point_sha256'),('metapath','metadata_sha256')]:assert sha(r[k])==fr[h]
 rgb=np.array(Image.open(r['imagepath']).convert('RGB'));p=Path(r['pointpath']);raw=np.load(p) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);raw=np.ascontiguousarray(raw[:,:3],np.float32);fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
 if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
 camera=CameraSpec.from_camera_fields(r['platform'],fields,missing_distortion_policy='assume_zero_declared');mp=BASE/'masksP64_attempt001'/(r['frame_alias']+'.npz');assert read(mp.parent/'SEALED.json')['files'][mp.name]==sha(mp)
 with np.load(mp) as z:ins={k:z[k].copy() for k in ('boxes_xyxy','class_ids','scores','masks_bool','instance_indices')};assert tuple(z['original_hw'])==rgb.shape[:2]
 binding={'image_sha256':fr['image_sha256'],'point_sha256':fr['point_sha256'],'mask_cache_sha256':sha(mp)};frame={'rgb':rgb,'raw_points':raw,'camera':camera,'instances':ins};h,w=rgb.shape[:2]
 scene=build_scene_observation(raw_points=raw,camera_spec=camera,image_size_wh=(w,h),instances=ins,image_sha256=binding['image_sha256']);return frame,scene,binding
