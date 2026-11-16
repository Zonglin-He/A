"""Same sealed sensor protocol, complete development coverage; no neural model/GT."""
from register_views105 import *
from temporal_association import _visible_raw
def main():
 cv2.setNumThreads(1);plan=rd(H/'experiment/FULL_REGISTRATION_LOCK.json');assert sha(H/'FULL_PAIRS_PRIVATE.json')==plan['pair_sha256'];pairs=rd(H/'FULL_PAIRS_PRIVATE.json');out=R/'crossplatform/round105/registration_full_PRIVATE';out.mkdir(exist_ok=False);protected=set()
 with (R/'research/NIGHT_20261101/input/FINAL384_PRIVATE.json').open('rb') as f:
  for pre,event,value in ijson.parse(f):
   if pre=='item.metapath' and event=='string':protected.add(str(Path(value).resolve()))
 old=rd(R/'research/round12/data_audit/PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json')['rows_PRIVATE'];ix={str(Path(x['meta_path_PRIVATE']).resolve()):x for x in old};deny={ix[x]['image_sha256'] for x in protected};frames={f['frame_PRIVATE']:f for x in pairs for f in [x['A'],x['B']]};probe={x['pair_PRIVATE']:x for x in rd(H/'PAIR_CANDIDATES_PRIVATE.json')};records=[];start=time.perf_counter()
 @functools.lru_cache(maxsize=8)
 def frame(alias):
  x=frames[alias];assert x['eligible_for_extra_media'] and x['metapath'] not in protected;imghash=sha(x['imagepath']);assert imghash not in deny and imghash==x['image_sha256'];pointsha=sha(x['pointpath'])
  if x['point_sha256'] is not None:assert pointsha==x['point_sha256']
  rgb=np.asarray(Image.open(x['imagepath']).convert('RGB'));raw=np.load(x['pointpath'],mmap_mode='r') if Path(x['pointpath']).suffix=='.npy' else np.fromfile(x['pointpath'],np.float32).reshape(-1,4);pts=np.asarray(raw[:,:3]);pts=pts[np.isfinite(pts).all(1)];cam=CameraSpec.from_camera_fields(x['platform'],read_camera_allowlist(x['metapath']),waymo_view_suffix=x['suffix'],missing_distortion_policy='assume_zero_declared');cam.validated();return rgb,pts,cam,{'image_sha256':imghash,'point_sha256':pointsha,'decoded_RGB_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'image_hw':list(rgb.shape[:2]),'finite_raw_points':len(pts),'foreground_exclusion_available':False}
 for i,pair in enumerate(pairs):
  A,B=pair['A'],pair['B'];pid=pair['pair_PRIVATE'];oldpair=probe.get(pid);oldrec=O/(pid+'.json')
  if oldpair is not None and oldrec.is_file() and all(oldpair[k]['frame_PRIVATE']==pair[k]['frame_PRIVATE'] for k in ['A','B']):
   rec=rd(oldrec)
   for k in ['A','B']:
    assert rec[k+'_sensor']['image_sha256']==pair[k]['image_sha256'];assert sha(pair[k]['pointpath'])==rec[k+'_sensor']['point_sha256']
   rec['reused_exact_probe_sha256']=sha(oldrec)
  else:
   rgbA,pa,ca,da=frame(A['frame_PRIVATE']);rgbB,pb,cb,db=frame(B['frame_PRIVATE']);seed=int(hashlib.sha256(('R105PnP|'+pid).encode()).hexdigest()[:8],16);rec={'pair_PRIVATE':pid,'platform':A['platform'],'kind':pair['kind'],'A_PRIVATE':A['frame_PRIVATE'],'B_PRIVATE':B['frame_PRIVATE'],'A_sensor':da,'B_sensor':db,'geometry_cycle_supported':False,'no_foreground_or_identity_certification':True,'provided_pose_used':False};tick=time.perf_counter()
   if da['decoded_RGB_sha256']==db['decoded_RGB_sha256']:rec['status']='decoded_duplicate_RGB_no_new_visual_observation'
   else:
    f=backend.estimate_pair_registration(rgb_A=rgbA,raw_points_A=pa,camera_A=ca,rgb_B=rgbB,raw_points_B=pb,camera_B=cb,seed=seed);rec['forward']=f;rec['forward_geometry_only_gate']=geometric(f)
    if not geometric(f):rec['status']='forward_sensor_geometry_unsupported'
    else:
     b=backend.estimate_pair_registration(rgb_A=rgbB,raw_points_A=pb,camera_A=cb,rgb_B=rgbA,raw_points_B=pa,camera_B=ca,seed=seed^0x4f391a2);rec['reverse']=b;rec['reverse_geometry_only_gate']=geometric(b)
     if not geometric(b):rec['status']='reverse_sensor_geometry_unsupported'
     else:
      vis,_,_=_visible_raw(pa,ca,rgbA.shape[:2],np.zeros(rgbA.shape[:2],bool),TC);held=vis[spatial_split(vis,seed,TC)];held=held[::max(1,len(held)//5000)];cycle=np.asarray(f['T_raw_B_to_A'])@np.asarray(b['T_raw_B_to_A']);err=np.linalg.norm(transform_points(held,cycle)-held,axis=1);ok=bool(len(err)>=20 and np.median(err)<=TC.maximum_holdout_median_m and np.quantile(err,.9)<=TC.maximum_holdout_p90_m);rec['cycle']={'n':len(err),'median_m':float(np.median(err)) if len(err) else None,'p90_m':float(np.quantile(err,.9)) if len(err) else None,'not_used_to_fit_either_transform':True};rec['geometry_cycle_supported']=ok;rec['status']='mask_free_sensor_geometry_and_cycle_supported' if ok else 'sensor_cycle_unsupported'
   rec['seconds']=time.perf_counter()-tick
  dump(out/(pid+'.json'),rec);records.append(rec)
  if (i+1)%32==0:print('FULL_REG105',i+1,len(pairs),flush=True)
 groups=[]
 for pl in ['waymo','drone','quad']:
  rr=[x for x in records if x['platform']==pl];groups.append({'platform':pl,'attempts':len(rr),'geometry_cycle_supported':sum(x['geometry_cycle_supported'] for x in rr),'statuses':dict(collections.Counter(x['status'] for x in rr))})
 dump(H/'FULL_REGISTRATION_SUMMARY.json',{'n':len(records),'groups':groups,'reused_exact_probe':sum('reused_exact_probe_sha256' in x for x in records),'seconds':time.perf_counter()-start,'no_model_or_GT':True});dump(out/'SEALED.json',{'n':len(records),'files':{f.name:sha(f) for f in sorted(out.glob('*.json'))},'lock_sha256':sha(H/'experiment/FULL_REGISTRATION_LOCK.json')});print('FULL_REG105_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
