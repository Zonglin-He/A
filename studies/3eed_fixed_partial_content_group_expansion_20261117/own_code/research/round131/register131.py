"""Same sensor-only PnP/reverse/cycle qualification as R109; fixed B candidates."""
from pathlib import Path
import sys,json,hashlib,time,collections,concurrent.futures,multiprocessing
import numpy as np,cv2
from PIL import Image
R=Path(__file__).resolve().parents[2];H=R/'research/round131';O=R/'crossplatform/round131/registration_PRIVATE';sys.path.insert(0,str(R/'research/round109'));from register109 import backend,geometric,clean,_visible_raw,spatial_split,transform_points,TC,CameraSpec,read_camera_allowlist
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def worker(pair):
 cv2.setNumThreads(1);A,B=pair['A'],pair['B'];tick=time.perf_counter();deny={f['image_sha256'] for f in rd(R/'research/round105/FRAMES_PRIVATE.json') if f['protected_image_content_alias']};frames=[]
 for x in [A,B]:
  assert x['eligible_for_extra_media'];ih=sha(x['imagepath']);assert ih==x['image_sha256'] and ih not in deny;ph=sha(x['pointpath'])
  if x['point_sha256'] is not None:assert ph==x['point_sha256']
  rgb=np.asarray(Image.open(x['imagepath']).convert('RGB'));raw=np.load(x['pointpath'],mmap_mode='r') if Path(x['pointpath']).suffix=='.npy' else np.fromfile(x['pointpath'],np.float32).reshape(-1,4);p=np.asarray(raw[:,:3]);p=p[np.isfinite(p).all(1)];cam=CameraSpec.from_camera_fields(x['platform'],read_camera_allowlist(x['metapath']),waymo_view_suffix=x['suffix'],missing_distortion_policy='assume_zero_declared');cam.validated();frames.append((rgb,p,cam,{'image_sha256':ih,'point_sha256':ph,'decoded_RGB_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'image_hw':list(rgb.shape[:2]),'finite_raw_points':len(p)}))
 (rgbA,pa,ca,da),(rgbB,pb,cb,db)=frames;seed=int(hashlib.sha256(('R105PnP|'+pair['pair_PRIVATE']).encode()).hexdigest()[:8],16);rec={'pair_PRIVATE':pair['pair_PRIVATE'],'platform':A['platform'],'A_sensor':da,'B_sensor':db,'geometry_cycle_supported':False,'provided_pose_used':False,'no_foreground_or_identity_certification':True}
 if da['decoded_RGB_sha256']==db['decoded_RGB_sha256']:rec['status']='decoded_duplicate_RGB_no_new_visual_observation'
 else:
  f=backend.estimate_pair_registration(rgb_A=rgbA,raw_points_A=pa,camera_A=ca,rgb_B=rgbB,raw_points_B=pb,camera_B=cb,seed=seed);rec['forward']=f;rec['forward_geometry_only_gate']=geometric(f)
  if not geometric(f):rec['status']='forward_sensor_geometry_unsupported'
  else:
   b=backend.estimate_pair_registration(rgb_A=rgbB,raw_points_A=pb,camera_A=cb,rgb_B=rgbA,raw_points_B=pa,camera_B=ca,seed=seed^0x4f391a2);rec['reverse']=b;rec['reverse_geometry_only_gate']=geometric(b)
   if not geometric(b):rec['status']='reverse_sensor_geometry_unsupported'
   else:
    vis,_,_=_visible_raw(pa,ca,rgbA.shape[:2],np.zeros(rgbA.shape[:2],bool),TC);held=vis[spatial_split(vis,seed,TC)];held=held[::max(1,len(held)//5000)];err=np.linalg.norm(transform_points(held,f['T_raw_B_to_A']@b['T_raw_B_to_A'])-held,axis=1);ok=bool(len(err)>=20 and np.median(err)<=TC.maximum_holdout_median_m and np.quantile(err,.9)<=TC.maximum_holdout_p90_m);rec['cycle']={'n':len(err),'median_m':float(np.median(err)) if len(err) else None,'p90_m':float(np.quantile(err,.9)) if len(err) else None,'not_used_to_fit_either_transform':True};rec['geometry_cycle_supported']=ok;rec['status']='mask_free_sensor_geometry_and_cycle_supported' if ok else 'sensor_cycle_unsupported'
 rec['seconds']=time.perf_counter()-tick;return clean(rec)
def main():
 lock=rd(H/'experiment/INPUT_LOCK.json')
 for n,v in lock['files'].items():assert sha(H/n)==v
 assert sha(H/lock.get('selection_code_path','select131.py'))==lock['selection_code_sha256'];pairs=rd(H/'PAIRS_PRIVATE.json');O.mkdir(parents=True,exist_ok=False);rows=[];start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for i,rec in enumerate(pool.map(worker,pairs,chunksize=1)):
   (O/(rec['pair_PRIVATE']+'.json')).write_text(json.dumps(rec,indent=2)+'\n');rows.append(rec)
   if (i+1)%32==0:print('REGISTER131',i+1,len(pairs),round(time.perf_counter()-start,1),flush=True)
 out={'n':len(rows),'geometry_cycle_supported':sum(r['geometry_cycle_supported'] for r in rows),'groups':{p:{'n':sum(r['platform']==p for r in rows),'supported':sum(r['geometry_cycle_supported'] for r in rows if r['platform']==p)} for p in ['waymo','drone','quad']},'statuses':dict(collections.Counter(r['status'] for r in rows)),'seconds':time.perf_counter()-start,'GT_or_semantic_model_used':False,'no_candidate_replacement_after_gate':True,'four_processes_same_fixed_registration_math':True};(H/'REGISTRATION_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n');(O/'SEALED.json').write_text(json.dumps({'n':len(rows),'files':{p.name:sha(p) for p in O.glob('*.json')}},indent=2)+'\n');print(out)
if __name__=='__main__':main()
