"""Classical sensor-only new-view eligibility. No masks, semantic network or GT."""
from pathlib import Path
import sys,json,hashlib,time,functools,collections,numpy as np,ijson
from PIL import Image
import cv2
R=Path(__file__).resolve().parents[2];H=R/'research/round109';O=R/'crossplatform/round109/registration_PRIVATE';sys.path.insert(0,str(R/'crossplatform/round12'));import sensor_pnp as backend
from temporal_association import _visible_raw,spatial_split,transform_points,DEFAULT_CONFIG as TC
from camera_projection import CameraSpec,read_camera_allowlist
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 return x
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def geometric(x):
 cfg=backend.DEFAULT_CONFIG;p=x.get('RGB_validation',{});c=x.get('holdout_residuals',{})
 return bool(x.get('T_raw_B_to_A') is not None and p.get('n',0)>=cfg.minimum_holdout and p.get('median_px',float('inf'))<=cfg.maximum_pixel_median and p.get('inlier_fraction',0)>=cfg.minimum_pixel_inlier_fraction and c.get('n',0)>=cfg.minimum_cloud_holdout and c.get('median_m',float('inf'))<=cfg.maximum_cloud_median_m and c.get('inlier_fraction',0)>=cfg.minimum_cloud_inlier_fraction)
def main():
 cv2.setNumThreads(1);plan=rd(H/'experiment/REGISTRATION_LOCK.json');assert sha(H/'PAIRS_PRIVATE.json')==plan['pair_manifest_sha256'];pairs=rd(H/'PAIRS_PRIVATE.json');protected=set();oldframes=rd(R/'research/round105/FRAMES_PRIVATE.json');deny={x['image_sha256'] for x in oldframes if x['protected_image_content_alias']};assert all(x['A']['eligible_for_extra_media'] and x['B']['eligible_for_extra_media'] for x in pairs);frames={f['frame_PRIVATE']:f for x in pairs for f in [x['A'],x['B']]};O.mkdir(exist_ok=False);records=[];start=time.perf_counter()
 @functools.lru_cache(maxsize=8)
 def frame(alias):
  x=frames[alias];assert x['eligible_for_extra_media'] and x['metapath'] not in protected;imghash=sha(x['imagepath']);assert imghash not in deny,'unexpected protected image alias: stop before decode'
  if x['image_sha256'] is not None:assert imghash==x['image_sha256']
  pointsha=sha(x['pointpath'])
  if x['point_sha256'] is not None:assert pointsha==x['point_sha256']
  rgb=np.asarray(Image.open(x['imagepath']).convert('RGB'));raw=np.load(x['pointpath'],mmap_mode='r') if Path(x['pointpath']).suffix=='.npy' else np.fromfile(x['pointpath'],np.float32).reshape(-1,4);pts=np.asarray(raw[:,:3]);pts=pts[np.isfinite(pts).all(1)];cam=CameraSpec.from_camera_fields(x['platform'],read_camera_allowlist(x['metapath']),waymo_view_suffix=x['suffix'],missing_distortion_policy='assume_zero_declared');cam.validated();return rgb,pts,cam,{'image_sha256':imghash,'point_sha256':pointsha,'decoded_RGB_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'image_hw':list(rgb.shape[:2]),'finite_raw_points':len(pts),'foreground_exclusion_available':False}
 for i,pair in enumerate(pairs):
  A,B=pair['A'],pair['B'];rgbA,pa,ca,da=frame(A['frame_PRIVATE']);rgbB,pb,cb,db=frame(B['frame_PRIVATE']);seed=int(hashlib.sha256(('R105PnP|'+pair['pair_PRIVATE']).encode()).hexdigest()[:8],16);rec={'pair_PRIVATE':pair['pair_PRIVATE'],'platform':A['platform'],'kind':pair['kind'],'A_PRIVATE':A['frame_PRIVATE'],'B_PRIVATE':B['frame_PRIVATE'],'A_sensor':da,'B_sensor':db,'geometry_cycle_supported':False,'no_foreground_or_identity_certification':True,'provided_pose_used':False};tick=time.perf_counter()
  if da['decoded_RGB_sha256']==db['decoded_RGB_sha256']:rec['status']='decoded_duplicate_RGB_no_new_visual_observation'
  else:
   f=backend.estimate_pair_registration(rgb_A=rgbA,raw_points_A=pa,camera_A=ca,rgb_B=rgbB,raw_points_B=pb,camera_B=cb,seed=seed);rec['forward']=f;rec['forward_geometry_only_gate']=geometric(f)
   if not geometric(f):rec['status']='forward_sensor_geometry_unsupported'
   else:
    b=backend.estimate_pair_registration(rgb_A=rgbB,raw_points_A=pb,camera_A=cb,rgb_B=rgbA,raw_points_B=pa,camera_B=ca,seed=seed^0x4f391a2);rec['reverse']=b;rec['reverse_geometry_only_gate']=geometric(b)
    if not geometric(b):rec['status']='reverse_sensor_geometry_unsupported'
    else:
     vis,_,_=_visible_raw(pa,ca,rgbA.shape[:2],np.zeros(rgbA.shape[:2],bool),TC);held=vis[spatial_split(vis,seed,TC)];held=held[::max(1,len(held)//5000)];cycle=f['T_raw_B_to_A']@b['T_raw_B_to_A'];err=np.linalg.norm(transform_points(held,cycle)-held,axis=1);ok=bool(len(err)>=20 and np.median(err)<=TC.maximum_holdout_median_m and np.quantile(err,.9)<=TC.maximum_holdout_p90_m);rec['cycle']={'n':len(err),'median_m':float(np.median(err)) if len(err) else None,'p90_m':float(np.quantile(err,.9)) if len(err) else None,'not_used_to_fit_either_transform':True};rec['geometry_cycle_supported']=ok;rec['status']='mask_free_sensor_geometry_and_cycle_supported' if ok else 'sensor_cycle_unsupported'
  rec['seconds']=time.perf_counter()-tick;dump(O/(pair['pair_PRIVATE']+'.json'),rec);records.append(rec);print('PAIR109',i+1,len(pairs),rec['platform'],rec['kind'],rec['status'],round(rec['seconds'],2),flush=True)
 groups=[]
 for pl in ['waymo','drone','quad']:
  for kind in ['same_scene_distinct_sensor_candidate','wrong_scene_control']:
   rr=[x for x in records if x['platform']==pl and x['kind']==kind];groups.append({'platform':pl,'kind':kind,'attempts':len(rr),'forward_numeric_supported':sum(x.get('forward_geometry_only_gate',False) for x in rr),'bidirectional_cycle_supported':sum(x['geometry_cycle_supported'] for x in rr),'statuses':dict(collections.Counter(x['status'] for x in rr))})
 dump(H/'REGISTRATION_SUMMARY.json',{'state':'R109_SENSOR_ONLY_REGISTRATION_PREREQUISITE_COMPLETE','n':len(records),'groups':groups,'seconds':time.perf_counter()-start,'model_or_GT_used':False,'foreground_masks_used':False,'old_selected_pose_cache_used':False,'protected_media_decoded':False,'sensor_support_not_identity_truth':True});dump(O/'SEALED.json',{'n':len(records),'files':{f.name:sha(f) for f in sorted(O.glob('*.json'))},'plan_sha256':sha(H/'experiment/REGISTRATION_LOCK.json'),'no_model_or_GT':True});print('REG109_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
