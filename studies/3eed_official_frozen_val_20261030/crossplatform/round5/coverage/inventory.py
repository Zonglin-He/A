"""Filesystem/ZIP/header/allowlist camera audit, never decode numeric targetlabels."""
from pathlib import Path
import json,hashlib,zipfile,collections,time
import numpy as np
from PIL import Image
P=Path(__file__).resolve().parents[2];R=P/'round5/coverage';D=P/'dataset_full/3eed_dataset';DEC=json.JSONDecoder()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def end_value(s,pos):
 i=pos
 while i<len(s) and s[i].isspace():i+=1
 if s[i]=='"':return DEC.raw_decode(s,i)[1],None
 if s[i] not in '[{':
  j=i
  while j<len(s) and s[j] not in ',}]':j+=1
  return j,None
 start=s[i];stack=[];inside=False;escape=False;objects=0
 for j in range(i,len(s)):
  c=s[j]
  if inside:
   if escape:escape=False
   elif c=='\\':escape=True
   elif c=='"':inside=False
   continue
  if c=='"':inside=True
  elif c in '[{':
   if start=='[' and len(stack)==1 and c=='{':objects+=1
   stack.append(c)
  elif c in ']}':
   assert stack and ((stack.pop()=='[')==(c==']')),'JSONcontainer mismatch'
   if not stack:return j+1,objects if start=='[' else None
 raise ValueError('unterminatedJSONcontainer')
def extract(s):
 # Only camera numeric values are decoded; ground_info is lexically skipped/countonly.
 i=0
 while s[i].isspace():i+=1
 assert s[i]=='{';i+=1;out={};count=None
 while True:
  while s[i].isspace() or s[i]==',':i+=1
  if s[i]=='}':break
  k,j=DEC.raw_decode(s,i);assert isinstance(k,str);i=j
  while s[i].isspace():i+=1
  assert s[i]==':';i+=1
  while s[i].isspace():i+=1
  j,n=end_value(s,i)
  if k in ('image_intrinsic','image_extrinsic','camera_distortion'):out[k]=DEC.raw_decode(s,i)[0]
  if k=='ground_info':assert s[i]=='[';count=n
  i=j
 return out,count

def main():
 # Hostile string/number label fixtures confirm labels never decoded or returned.
 f='{"ground_info":[{"bbox_3d":[999,888],"caption":"brace} quote\\\""},{"bbox_3d":[123]}],"image_intrinsic":[[1,0,2],[0,1,3],[0,0,1]],"image_extrinsic":'+json.dumps(np.eye(4).tolist())+'}'
 x,n=extract(f);assert n==2 and set(x)=={'image_intrinsic','image_extrinsic'}
 began=time.perf_counter();archive=P/'downloads/3eed_dataset.zip'
 with zipfile.ZipFile(archive) as z:names=set(z.namelist())
 rows=[];summary={};splits={}
 for platform in ['waymo','drone','quad']:
  sp=D/'splits'/f'{platform}_val.txt';up=P/'upstream/data/splits'/sp.name
  scenes=[x.strip() for x in sp.read_text().splitlines() if x.strip()];assert len(scenes)==len(set(scenes));train={x.strip() for x in (D/'splits'/f'{platform}_train.txt').read_text().splitlines() if x.strip()};missing=[];present=[]
  for scene in scenes:
   sd=D/platform/scene
   if not sd.is_dir():missing.append(scene);continue
   present.append(scene)
   for frame in sorted(x for x in sd.iterdir() if x.is_dir()):
    point=frame/('lidar.npy' if platform=='waymo' else 'lidar.bin');image=frame/'image.jpg';meta=frame/'meta_info.json';r={'platform':platform,'scene':scene,'frame':frame.name,'split':'val','point_present':point.is_file(),'image_present':image.is_file(),'metadata_present':meta.is_file(),'problems':[]}
    for key,fp in [('point',point),('image',image),('metadata',meta)]:
     member='3eed_dataset/'+str(fp.relative_to(D));r[key+'_in_official_ZIP']=member in names
     if fp.is_file()!= (member in names):r['problems'].append(key+'_ZIP_disk_membership_diff')
     if not fp.is_file():r['problems'].append('missing_'+key)
    if point.is_file():
     if platform=='waymo':
      with point.open('rb') as pf:
       v=np.lib.format.read_magic(pf);shape,fortran,dtype=np.lib.format._read_array_header(pf,v);offset=pf.tell()
      r['point_shape']=list(shape);r['point_dtype']=str(dtype);r['point_body_bytes_expected']=int(np.prod(shape))*dtype.itemsize;r['point_size_consistent']=point.stat().st_size-offset==r['point_body_bytes_expected']
     else:
      r['point_size_consistent']=point.stat().st_size%16==0;r['point_shape']=[point.stat().st_size//16,4];r['point_dtype']='float32'
     if not r['point_size_consistent']:r['problems'].append('point_bytecount_inconsistent')
    if image.is_file():
     try:
      with Image.open(image) as im:r['image_size_wh']=list(im.size);r['image_format']=im.format
     except Exception as e:r['problems'].append('image_header_'+type(e).__name__)
    if meta.is_file():
     try:
      camera,n=extract(meta.read_text());r['raw_ground_info_object_count']=n;r['camera_fields']=sorted(camera);r['camera_valid_shapes_finite']=False
      if all(k in camera for k in ('image_intrinsic','image_extrinsic')):
       k=np.asarray(camera['image_intrinsic'],float);e=np.asarray(camera['image_extrinsic'],float);d=np.asarray(camera.get('camera_distortion',[0,0,0,0]),float)
       r['intrinsic_shape']=list(k.shape);r['extrinsic_shape']=list(e.shape);r['camera_valid_shapes_finite']=bool(k.ndim==2 and k.shape[0]>=3 and k.shape[1]>=3 and e.shape==(4,4) and np.isfinite(k).all() and np.isfinite(e).all() and np.isfinite(d).all())
      if not r['camera_valid_shapes_finite']:r['problems'].append('missing_or_invalid_camera')
      if n is None:r['problems'].append('ground_info_not_found')
     except Exception as exc:r['problems'].append('metadata_lexical_audit_'+type(exc).__name__)
    rows.append(r)
  rr=[r for r in rows if r['platform']==platform];summary[platform]={'official_val_scenes':len(scenes),'available_val_scenes':len(present),'missing_val_scenes':missing,'frames':len(rr),'raw_ground_info_objects':sum(r.get('raw_ground_info_object_count') or 0 for r in rr),'missing_scene_frame_count':'unknown: absent from official archive, no frame manifest for missing scenes','image_headers':sum('image_size_wh' in r for r in rr),'camera_valid_shapes_finite':sum(r.get('camera_valid_shapes_finite',False) for r in rr),'complete_frames':sum(not r['problems'] for r in rr),'problem_counts':dict(collections.Counter(p for r in rr for p in r['problems'])),'image_dimensions_histogram':dict(collections.Counter('x'.join(map(str,r['image_size_wh'])) for r in rr if 'image_size_wh' in r)),'train_val_scene_overlap':sorted(set(scenes)&train),'split_file_sha256':sha(sp),'upstream_split_sha256':sha(up),'dataset_and_upstream_split_bitexact':sha(sp)==sha(up)}
  print(platform,summary[platform],flush=True)
 fp=R/'FRAMES_PRIVATE.json';fp.write_text(json.dumps(rows,indent=2));out={'scope':'all actual on-disk frames referenced by unmodified three officialVALscene lists; file/header + camera-allowlist only','dataset_integrity_receipt_sha256':sha(P/'full_dataset_verified.json'),'official_archive_bytes':archive.stat().st_size,'source_code_sha256':sha(Path(__file__)),'GT_numeric_values_decoded_or_indexed':False,'ground_info_scope':'lexical list structural object count only; skip all object payloads, captions/classes/boxes never decoded','RGB_pixels_read':False,'point_coordinate_arrays_loaded':False,'camera_values_decoded_allowlist':['image_intrinsic','image_extrinsic','camera_distortion'],'unknown_membership_missing_scenes_in_official_ZIP':False,'summary':summary,'frames_manifest_sha256':sha(fp),'seconds':time.perf_counter()-began}
 (R/'COVERAGE.json').write_text(json.dumps(out,indent=2));print('TOTAL',sum(v['frames'] for v in summary.values()),sum(v['raw_ground_info_objects'] for v in summary.values()))
if __name__=='__main__':main()
