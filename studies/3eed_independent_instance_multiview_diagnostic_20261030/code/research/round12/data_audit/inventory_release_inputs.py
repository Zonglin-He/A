"""Input-only 3EED schema/sequence availability audit; no labels or model.
Stream only explicit camera/pose/time/source metadata whitelist. Never build
annotation objects or consume GT class/bbox/instance values. No pixel/point load.
"""
from pathlib import Path
import collections,hashlib,json,sys
import numpy as np
from PIL import Image
import ijson
from ijson.common import ObjectBuilder
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;ROOT=B/'__DATASET_ROOT__'
ALLOW={'image_intrinsic','image_extrinsic','camera_distortion','pose','ego_pose','world_pose','lidar_pose','sensor_pose','sensor2ego','ego2global','lidar_to_world','sensor_to_world','camera_to_world','cam2world','translation_vector','rotation_matrix','timestamp','timestamp_micros','timestamp_ns','time_stamp','frame_timestamp','image_timestamp','lidar_timestamp','frame_id','frame_name','scene_name','context_name','segment_name','camera_name','lidar_name','original_path','original_index','source_frame_id','sequence_name'}
FORBIDDEN={'ground_info','bbox_3d','gt_bbox','class','instance_id','target_id','objects','annotations','instances','bbox_2d','caption','description'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def white_read(p):
 builders={};keys=[]
 with Path(p).open('rb') as f:
  for prefix,event,value in ijson.parse(f,use_float=True):
   if prefix=='' and event=='map_key':keys.append(value)
   first=prefix.split('.',1)[0]
   if first in ALLOW:builders.setdefault(first,ObjectBuilder()).event(event,value)
 return keys,{k:b.value for k,b in builders.items()}
def serial(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def matrix_summary(x):
 try:a=np.asarray(x,float)
 except (TypeError,ValueError):return {'type':type(x).__name__,'numeric':False}
 r={'shape':list(a.shape),'finite':bool(np.isfinite(a).all())}
 if a.shape==(4,4) and r['finite']:
  R=a[:3,:3];r.update(rotation_det=float(np.linalg.det(R)),rotation_orthonormal_error_Linf=float(abs(R.T@R-np.eye(3)).max()),translation_norm=float(np.linalg.norm(a[:3,3])),homogeneous_bottom_row=a[3].tolist())
 return r

def main():
 assert not (D/'RELEASE_INPUT_INVENTORY.json').exists();splitstats={};domain_inventory={};examples=[];pairs=[];selection=[]
 for platform in ('waymo','drone','quad'):
  root=ROOT/platform;scenedirs={p.name:p for p in root.iterdir() if p.is_dir()};domain_inventory[platform]={'release_scene_dirs':len(scenedirs),'all_frame_dirs':sum(sum(1 for f in p.iterdir() if f.is_dir()) for p in scenedirs.values()),'top_nondirectory_files':[p.name for p in root.iterdir() if p.is_file()]}
  for split in ('train','val'):
   p=ROOT/'splits'/f'{platform}_{split}.txt';names=[x.strip() for x in p.read_text().splitlines() if x.strip()];available=[x for x in names if x in scenedirs];entries=[]
   for scene in available:
    fs=sorted([x for x in scenedirs[scene].iterdir() if x.is_dir()],key=lambda x:x.name);suffix=collections.Counter();ints=[]
    for f in fs:
     parts=f.name.split('_');suffix[parts[1] if len(parts)==2 else 'none']+=1
     if parts[0].isdigit():ints.append(int(parts[0]))
    uniq=sorted(set(ints));gaps=collections.Counter(b-a for a,b in zip(uniq,uniq[1:]));files=collections.Counter(p.name for f in fs for p in f.iterdir() if p.is_file())
    entries.append({'scene_PRIVATE':scene,'frame_dirs':len(fs),'unique_numeric_prefixes':len(uniq),'prefix_min':min(uniq,default=None),'prefix_max':max(uniq,default=None),'numeric_prefix_gap_histogram':dict(sorted(gaps.items())),'folder_suffix_counts':dict(suffix),'per_frame_filename_counts':dict(files)})
   splitstats[platform+'_'+split]={'split_path_PRIVATE':str(p),'split_sha256':sha(p),'listed_scenes':len(names),'available_scenes':len(available),'missing_scene_names_PRIVATE':[x for x in names if x not in scenedirs],'available_frame_dirs':sum(x['frame_dirs'] for x in entries),'entries_PRIVATE':entries}
  # Fixed schema examples: first lexicographic available VAL scene that has2
  # numeric folders in one suffix group; then first2 numeric IDs in that group.
  val=sorted(x.strip() for x in (ROOT/'splits'/f'{platform}_val.txt').read_text().splitlines() if x.strip() in scenedirs)
  chosen=None
  for scene in val:
   grouped=collections.defaultdict(list)
   for f in scenedirs[scene].iterdir():
    if f.is_dir():
     bits=f.name.split('_')
     if bits[0].isdigit():grouped[bits[1] if len(bits)==2 else 'none'].append((int(bits[0]),f))
   for suffix,seq in sorted(grouped.items()):
    if len(seq)>=2:chosen=(scene,suffix,sorted(seq,key=lambda x:(x[0],x[1].name))[:2]);break
   if chosen:break
  assert chosen;scene,suffix,seq=chosen;pair_fields=[]
  selection.append({'platform':platform,'scene_PRIVATE':scene,'suffix':suffix,'numeric_prefixes':[x[0] for x in seq],'selection':'First lexicographic available officialVAL scene and suffix with>=2numeric folders, first2 ascending numeric prefixes; no GT/pixels/taskscore selection.'})
  for prefix,folder in seq:
   meta=folder/'meta_info.json';keys,fields=white_read(meta);assert not FORBIDDEN&fields.keys()
   point=folder/('lidar.npy' if platform=='waymo' else 'lidar.bin');image=folder/'image.jpg';assert all(p.is_file() for p in (meta,point,image))
   with Image.open(image) as im:
    header={'format':im.format,'size_wh':list(im.size),'mode':im.mode,'DateTime':im.getexif().get(306),'DateTimeOriginal':im.getexif().get(36867)}
   if point.suffix=='.npy':
    with point.open('rb') as f:
     version=np.lib.format.read_magic(f)
     shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
    ph={'format':'NPY_header_only','version':list(version),'shape':list(shape),'fortran_order':fortran,'dtype':str(dtype)}
   else:ph={'format':'BIN_file_stat_only','record_format_assumption':'officialloader_float32x4','size_multiple16':point.stat().st_size%16==0,'records_if_float32x4':point.stat().st_size//16}
   summaries={k:matrix_summary(v) for k,v in fields.items() if k in ('pose','ego_pose','world_pose','lidar_pose','sensor_pose','image_extrinsic','image_intrinsic','camera_distortion','translation_vector','rotation_matrix')}
   examples.append({'platform':platform,'scene_PRIVATE':scene,'frame_folder_PRIVATE':folder.name,'source_paths_PRIVATE':{'meta':str(meta),'point':str(point),'image':str(image)},'source_file_sha256':{'meta_container':sha(meta),'point_bytes_only_no_array_load':sha(point),'image_bytes_only_no_pixels_decode':sha(image)},'source_file_sizes':{p.name:p.stat().st_size for p in (meta,point,image)},'top_level_schema_keys_only':keys,'whitelisted_input_fields_PRIVATE':fields,'input_field_summaries':summaries,'image_header_only':header,'point_header_or_stat_only':ph,'no_GT_class_bbox_instance_caption_values_consumed':True})
   pair_fields.append(fields)
  comparisons={}
  for field in sorted(set(pair_fields[0])&set(pair_fields[1])):
   try:
    a,b=np.asarray(pair_fields[0][field],float),np.asarray(pair_fields[1][field],float)
    if a.shape==b.shape and a.size:comparisons[field]={'same_shape':list(a.shape),'bitwise_equal_numeric':bool(np.array_equal(a,b)),'max_absolute_difference':float(abs(a-b).max())}
   except (TypeError,ValueError):pass
  pairs.append({'platform':platform,'scene_PRIVATE':scene,'suffix':suffix,'numeric_prefix_gap':seq[1][0]-seq[0][0],'input_field_pair_comparisons':comparisons,'temporal_adjacency_sync_world_motion_not_inferred_from_numeric_prefix':True})
 out={'state':'R12_inputonly_release_schema_and_6frame_camera_pose_time_availability_audit','program_sha256':sha(__file__),'dataset_root_PRIVATE':str(ROOT),'allowlist_fields':sorted(ALLOW),'forbidden_payloads_never_built':sorted(FORBIDDEN),'domain_directory_inventory':domain_inventory,'official_split_directory_inventory':splitstats,'example_selection_PRIVATE':selection,'examples_PRIVATE':examples,'pairs_PRIVATE':pairs,'no_pointarrays_pixels_GT_annotations_oldtaskmetrics_models_weights_or_GPU_used':True,'file_hash_scope':'Full bytes hashed without decoding image/point payload. JSON stream builds only allowlisted top-level input metadata; GTobjects/values discarded and never used.','limitations':['Sixframe examples are schema/calibration availability checks only, not all-release field coverage.','Numerically adjacent folder names do not establish adjacent sensor timestamps or image-lidar synchronization.','A 4x4 pose shape does not establish coordinate meaning/metric units or an ego-to-world convention.','Extrinsic matrices alone relate a chosen sensor/camera convention; they do not imply cross-frame worldmotion.']}
 (D/'RELEASE_INPUT_INVENTORY.json').write_text(json.dumps(out,indent=2,default=serial)+'\n');print(json.dumps({'domains':domain_inventory,'splits':{k:{x:v[x] for x in ('listed_scenes','available_scenes','available_frame_dirs')} for k,v in splitstats.items()},'examples':[{k:e[k] for k in ('platform','frame_folder_PRIVATE','top_level_schema_keys_only','input_field_summaries','image_header_only','point_header_or_stat_only')} for e in examples],'pairs':pairs,'receipt_sha256':sha(D/'RELEASE_INPUT_INVENTORY.json')},default=serial))
if __name__=='__main__':main()
