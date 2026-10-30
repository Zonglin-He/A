"""Calibration/header allowlist inventory, no model or annotation value consumers.
Streaming parser discards non-allowlisted subtrees, notably ground_info. No target
box/class/positive_map, captions, or GT data structures are materialized here.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math
import ijson
from ijson.common import ObjectBuilder
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[2];ROOT=R/'crossplatform/dataset_full/3eed_dataset';OUT=R/'research/round7'
ALLOWED={'image_intrinsic','image_extrinsic','camera_distortion','pose','image_path','lidar_path_proj','timestamp'}
def camera_allowlist(path):
 builders={};keys=[]
 with open(path,'rb') as f:
  for prefix,event,value in ijson.parse(f):
   if prefix=='' and event=='map_key':keys.append(value);continue
   first=prefix.split('.',1)[0]
   if first not in ALLOWED:continue
   builder=builders.setdefault(first,ObjectBuilder());builder.event(event,value)
 return {key:builder.value for key,builder in builders.items()},keys

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def record_platform(platform):
 paths=sorted((ROOT/platform).glob('*/*/meta_info.json'));counts=Counter();dims=Counter();kshapes=Counter();dshapes=Counter();topkeys=Counter();exceptions=Counter();scenes=set();fourth_max=0.;ortho_max=0.;det_max=0.;bottom_max=0.;intrinsics_outside=0;rows=[]
 for p in paths:
  camera,keys=camera_allowlist(p);topkeys.update(keys);counts['metadata_containers']+=1;scenes.add(p.parent.parent.name);impath=p.with_name('image.jpg');lp=p.with_name('lidar.npy' if platform=='waymo' else 'lidar.bin');row={'metadata_relative_path_hash':hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest(),'platform':platform,'local_image_exists':impath.is_file(),'local_lidar_exists':lp.is_file()};counts['image_exists']+=row['local_image_exists'];counts['lidar_exists']+=row['local_lidar_exists']
  if impath.is_file():
   try:
    with Image.open(impath) as im:wh=im.size;fmt=im.format;mode=im.mode
    dims[str(wh)]+=1;counts['image_header_readable']+=1;counts['image_mode_'+mode]+=1;counts['image_format_'+str(fmt)]+=1
    with open(impath,'rb') as f:f.seek(-2,2);ends=f.read()==b'\xff\xd9'
    counts['image_has_terminal_JPEG_EOI']+=ends;row['image_size_wh']=list(wh)
   except Exception as e:exceptions['image_'+type(e).__name__]+=1;row['image_header_error']=type(e).__name__;wh=None
  else:wh=None
  if 'image_intrinsic' in camera and 'image_extrinsic' in camera:
   K=np.asarray(camera['image_intrinsic'],np.float64);E=np.asarray(camera['image_extrinsic'],np.float64);counts['has_K_E']+=1;kshapes[str(K.shape)]+=1
   assert K.shape in [(3,3),(3,4)] and E.shape==(4,4) and np.isfinite(K).all() and np.isfinite(E).all()
   if K.shape==(3,4):fourth_max=max(fourth_max,float(np.abs(K[:,3]).max()))
   Q=E[:3,:3];ortho_max=max(ortho_max,float(np.abs(Q.T@Q-np.eye(3)).max()));det_max=max(det_max,abs(float(np.linalg.det(Q))-1));bottom_max=max(bottom_max,float(np.abs(E[3]-[0,0,0,1]).max()))
   if wh is not None:intrinsics_outside+=int(not(0<=K[0,2]<wh[0] and 0<=K[1,2]<wh[1]))
   counts['pose_present']+=('pose' in camera)
  else:counts['missing_K_or_E']+=1
  if 'camera_distortion' in camera:
   d=np.asarray(camera['camera_distortion'],float);assert np.isfinite(d).all();counts['distortion_present']+=1;dshapes[str(d.shape)]+=1;counts['distortion_nonzero']+=int(bool(np.any(d)))
  rows.append(row)
 return {'counts':dict(counts),'scenes':len(scenes),'image_size_histogram':dict(dims),'intrinsic_shape_histogram':dict(kshapes),'distortion_shape_histogram':dict(dshapes),'top_level_key_presence_counts':dict(topkeys),'intrinsic_fourth_column_max_abs_if_present':fourth_max,'extrinsic_rotation_orthogonality_max_abs':ortho_max,'extrinsic_rotation_det_delta_from_one_max_abs':det_max,'extrinsic_bottom_row_max_abs_error':bottom_max,'principal_point_outside_local_image_count':intrinsics_outside,'exceptions':dict(exceptions)},rows

def main():
 stats={};private=[]
 for p in ['waymo','drone','quad']:stats[p],rows=record_platform(p);private.extend(rows)
 inventory={'state':'completed_GT_free_camera_and_image_header_inventory','auditor_sha256':sha(Path(__file__)),'metadata_allowlist':sorted(ALLOWED),'streaming_parser':'ijson.parse; only allowlisted subtrees are materialized; ground_info/class/box/positive_map/caption values never indexed, emitted, or materialized as structures','metadata_containers_with_GT_bytes_parsed':True,'target_annotation_consumers':False,'media_access':'Image headers, mode/size/format, terminal JPEG EOI only. No image decode or semantic analysis; readable header+EOI is not a guarantee of all-pixel integrity.','no_pointcloud_contents_read':True,'no_model_GPU_or_new_GT_scoring':True,'platforms':stats}
 (OUT/'RGB_CAMERA_HEADER_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n');(OUT/'RGB_CAMERA_HEADER_ROWS_PRIVATE.json').write_text(json.dumps(private,indent=2)+'\n');print(json.dumps(inventory,indent=2))
if __name__=='__main__':main()
