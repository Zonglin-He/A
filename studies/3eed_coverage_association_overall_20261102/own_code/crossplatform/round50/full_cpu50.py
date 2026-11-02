"""Run-local R50 literal text/whole-scene geometry, fixed R8 candidate pools.

Only CPU stages. No VLM, detector, grounder, label, box fitting or update.
Old-entry source/class/pair rules remain historical; wide entry only adds
valid native crops with exclusive visible support in an original source box.
"""
import sys,time,functools
from pathlib import Path
import numpy as np
from full_common50 import *
from full_frontend50 import source_arrays,legacy_category_without_features,legacy_entry
sys.path[:0]=[str(ROOT/'crossplatform/round23'),str(ROOT/'crossplatform/round20'),str(ROOT/'crossplatform/round15'),str(ROOT/'crossplatform/round7')]
from pool_weights50 import legacy_geometry_from_visible,legacy_weights,wide_entry,native_crop_valid

@functools.lru_cache(maxsize=4)
def frame_index(runroot):
 folder=Path(runroot)/'frames';s=read(folder/'SEALED.json')
 assert s['state']=='R50_FULL5912_ORIGINAL_R12_MASKS_CAMERA_COMPLETE' and s['n']==5912
 return s

def full_observed(row,runroot,with_rgb=True):
 """Original R15 whole-scene observation; source/text never affect ownership.

 Point format and Drone +1.8 float32 bridge are unchanged. Only one complete
 scene is live per row. All reused/fresh masks have the same native IDs/order.
 """
 runroot=Path(runroot);folder=runroot/'frames';s=frame_index(str(runroot))
 rel='cases_PRIVATE/'+row['id']+'.json';assert sha(folder/rel)==s['files'][rel];b=read(folder/rel)
 mp=folder/'masks_PRIVATE'/(row['id']+'.npz');cp=folder/'camera_PRIVATE'/(row['id']+'.json')
 assert sha(mp)==b['mask_cache_sha256'] and sha(cp)==b['camera_cache_sha256']
 assert sha(row['pointpath'])==b['point_sha256'] and sha(row['imagepath'])==b['image_sha256'] and sha(row['metapath'])==b['metadata_sha256']
 return observation_from_files(row,mp,cp,b,with_rgb)

def observation_from_files(row,maskpath,camerapath,binding_record,with_rgb=True):
 """Shared checked-R15 builder; also usable by separately locked oldDEV2 audit.

 The caller must bind the actual files to its immutable input plan. This pure
 route does not find data, choose rows, grant access, or silently load caches.
 """
 from camera_projection import CameraSpec
 from scene_observation import build_scene_observation
 b=binding_record;mp=Path(maskpath);cp=Path(camerapath)
 assert sha(mp)==b['mask_cache_sha256'] and sha(cp)==b['camera_cache_sha256']
 assert sha(row['pointpath'])==b['point_sha256'] and sha(row['imagepath'])==b['image_sha256'] and sha(row['metapath'])==b['metadata_sha256']
 ins=npz(mp);H,W=map(int,ins['original_hw']);p=Path(row['pointpath'])
 raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4)
 assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all();raw=np.ascontiguousarray(raw[:,:3],np.float32)
 fields=read(cp);camera=CameraSpec.from_camera_fields(row['platform'],fields,missing_distortion_policy='assume_zero_declared')
 instances={k:ins[k] for k in ('boxes_xyxy','class_ids','scores','masks_bool','instance_indices')}
 frame={'raw_points':raw,'camera':camera,'instances':instances,'original_hw':np.array([H,W])}
 if with_rgb:
  from PIL import Image
  frame['rgb']=np.array(Image.open(row['imagepath']).convert('RGB'));assert frame['rgb'].shape[:2]==(H,W)
 scene=build_scene_observation(raw_points=raw,camera_spec=camera,image_size_wh=(W,H),instances=instances,image_sha256=b['image_sha256'])
 binding={k:b[k] for k in ('image_sha256','point_sha256','mask_cache_sha256')}
 return frame,scene,binding

def project_boxes(boxes,camera,size_wh):
 from camera_projection import project_model_points
 sys.path.insert(0,str(ROOT/'crossplatform/round13'));from association import box_corners
 rect=np.zeros((256,4));valid=np.zeros(256,bool)
 for q,b in enumerate(boxes):
  if (b[3:]<=0).any():continue
  p=project_model_points(box_corners(b),camera,size_wh);valid[q]=bool(np.all(p['depth']>.1) and np.isfinite(p['uv']).all())
  if valid[q]:rect[q]=np.r_[p['uv'].min(0),p['uv'].max(0)]
 return rect,valid

def geometry_one(row,runroot,source_seal):
 """All geometry for evidence/readout; original R17 matrix and choose."""
 source=source_arrays(row,source_seal);frame,scene,binding=full_observed(row,runroot,with_rgb=False)
 textpath=Path(runroot)/'text/cases_PRIVATE'/(row['id']+'.json');g=read(textpath)
 return geometry_arrays(row,source,frame,scene,binding,g,textpath)

def geometry_arrays(row,source,frame,scene,binding,g,textpath):
 """One canonical array route, shared by full and no-GT oldDEV engineering."""
 from prepareZ import choose
 from regression_adapter import LEXICON
 sys.path.insert(0,str(ROOT/'crossplatform/round49'));from ownership49 import ownership_counts
 textpath=Path(textpath)
 boxes=source['boxes'];scores=source['scores'];W,H=scene['image_size_wh'];instances=frame['instances'];M=len(instances['class_ids'])
 rectangles,pvalid=project_boxes(boxes,frame['camera'],(W,H))
 raw=legacy_geometry_from_visible(scene['visible_points_model'],scene['point_owner'],scene['depth_camera'],boxes,rectangles,pvalid,instances['boxes_xyxy'])
 category=legacy_category_without_features(row['caption'],g,LEXICON)
 entry=legacy_entry(scene,source,raw,category,choose);old=entry['old_pool']
 counts=ownership_counts(scene['visible_points_model'],scene['point_owner'],boxes,M)
 validcrop=native_crop_valid(instances['boxes_xyxy'],(W,H),[o['mask_geometry_defined'] for o in scene['objects']])
 wide=wide_entry(old,counts['mask_counts'],validcrop);pool=wide['pool']
 ow=legacy_weights(old,raw['point_support_count'],raw['point_support_fraction'],raw['projection_iou'],boxes)
 ww=legacy_weights(pool,raw['point_support_count'],raw['point_support_fraction'],raw['projection_iou'],boxes)
 flags=entry['taxonomy_flags'];masksha=np.array([o['mask_sha256'] for o in scene['objects']])
 arr=dict(source_boxes=boxes,source_scores=scores,old_pool=old,wide_pool=pool,
  old_weights=ow['weights'],wide_old_weights=ww['weights'],mask_sha256=masksha,
  detector_boxes=instances['boxes_xyxy'],native_detector_class=instances['class_ids'],crop_valid=validcrop,
  strict_counts=counts['mask_counts'],mask_counts=counts['mask_counts'],allvisible=counts['allvisible'],
  unassigned=counts['unassigned'],ambiguous=counts['ambiguous'],legacy_support=raw['point_support_count'],
  legacy_coverage=raw['point_support_fraction'],legacy_projection_iou=raw['projection_iou'],
  legacy_depth_trim_count=raw['point_count_after_depth_trim'],legacy_quality=ow['quality'],legacy_valid=ow['valid'],
  projected_rectangles=rectangles,projection_valid=pvalid,
  historical_taxonomy_compatible=flags['historical_taxonomy_compatible'],corrected_taxonomy_compatible=flags['corrected_taxonomy_compatible'])
 case=dict(id_PRIVATE=row['id'],platform=row['platform'],category=category,mask_pool_PRIVATE=old.tolist(),
  wide_pool_PRIVATE=pool.tolist(),old_pool_count=len(old),wide_pool_count=len(pool),additional_mask_count=len(wide['additional_masks']),
  legacy_preparation_PRIVATE=entry['legacy_pairs'],source_mask_PRIVATE=entry['source_mask'],objects=len(old),bindings=binding,
  actual_text_source_PRIVATE=textpath.relative_to(ROOT).as_posix(),actual_text_sha256=sha(textpath),
  actual_subject_query_sha256=__import__('hashlib').sha256(g['subject_query'].encode()).hexdigest(),
  visibility_sha256=scene['visibility_sha256'],source_R8_file_sha256=source['R8_file_sha256'],
  entry_reasons=wide['entry_reasons'],reference_DINO_and_numeric_gate_original_oldpool_nonempty=bool(len(old)),
  no_new_box_or_score_or_token_features=True,token_delta_not_required_literal_gamma_zero=True,
  target_numeric_forward_upper_bound=4*len(pool),DINO_forward_upper_bound=2*len(g['references']) if len(old) else 0,
  reference_numeric_forward_upper_bound=6*len(g['references']) if len(old) else 0)
 return arr,case

def old_text_registry():
 """Only the exact known 336 original graph files are eligible for reuse.

 No protected registration or class/box annotation is read. Current captions
 must equal the historical input byte-for-byte, else the original parser runs.
 """
 rr=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');out={}
 for r in rr:
  suffix='M64' if r['stage']=='M' else '272';base=ROOT/'crossplatform'/('round24' if r['stage']=='M' else 'round23')
  folder=base/('text'+suffix+'_attempt001');rel='cases_PRIVATE/'+r['id']+'.json';s=read(folder/'SEALED.json')
  assert sha(folder/rel)==s['files'][rel]
  key=tuple(r[k] for k in ('imagepath','pointpath','metapath','caption'));assert key not in out
  out[key]={'old_id_PRIVATE':r['id'],'caption':r['caption'],'path':folder/rel,'sha256':s['files'][rel]}
 return out

def main():
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['text','geometry'],required=True)
 for k in ('plan','authorization'):p.add_argument('--'+k,required=True)
 p.add_argument('--chunk',type=int);p.add_argument('--finalize',action='store_true');a=p.parse_args()
 plan,rows,out=context(a,a.stage);folder=out/('text' if a.stage=='text' else 'prepared')
 if a.finalize:
  assert a.chunk is None
  for r in rows:
   assert (folder/'cases_PRIVATE'/(r['id']+'.json')).exists()
   if a.stage=='geometry':assert (folder/'arrays_PRIVATE'/(r['id']+'.npz')).exists()
  seal(folder,'R50_FULL5912_LITERAL_TEXT_COMPLETE' if a.stage=='text' else 'R50_FULL5912_FIXED_SOURCE_LEGACY_W_WIDE_ENTRY_GEOMETRY_COMPLETE',n=5912,
   plan_sha256=sha(a.plan),new_model_forwards=0,no_GT=True);return
 folder.mkdir(parents=True,exist_ok=True)
 for d in ('cases_PRIVATE','arrays_PRIVATE','chunks_PRIVATE'):(folder/d).mkdir(exist_ok=True)
 name='all' if a.chunk is None else f'{a.chunk:04d}';receipt=folder/'chunks_PRIVATE'/(name+'.json');assert not receipt.exists()
 start=time.perf_counter();source_seal=None;nlp=None;reused=0;done=[]
 if a.stage=='text':
  from parse23 import parse
  from textZ import pipeline
  cache=old_text_registry()
 else:
  source_seal=read(ROOT/'crossplatform/round8/full5912_attempt001/SEALED.json')
  ts=verify(out/'text');assert ts['n']==5912
  frame_index(str(out))
 try:
  for i,r in enumerate(rows):
   cp=folder/'cases_PRIVATE'/(r['id']+'.json');assert not cp.exists(),'No implicit overwrite or case retry'
   if a.stage=='text':
    prior=cache.get(tuple(r[k] for k in ('imagepath','pointpath','metapath','caption')))
    if prior:
     import shutil
     assert sha(prior['path'])==prior['sha256'];shutil.copyfile(prior['path'],cp);reused+=1
    else:
     if nlp is None:nlp=pipeline()
     dump(cp,parse(r,nlp))
   else:
    ar,c=geometry_one(r,out,source_seal);ap=folder/'arrays_PRIVATE'/(r['id']+'.npz');np.savez_compressed(ap,**ar)
    c['array_sha256']=sha(ap);dump(cp,c)
   done.append(r['id'])
   if (i+1)%32==0:print('R50_FULL_CPU',a.stage,name,i+1,len(rows),flush=True)
 except BaseException as e:
  dump(folder/('FAILED_'+name+'.json'),{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_automatic_retry':True});raise
 dump(receipt,{'state':'R50_FULL_CPU_CHUNK_COMPLETE','stage':a.stage,'n':len(done),'exact_old_text_reused':reused,
  'seconds':time.perf_counter()-start,'plan_sha256':sha(a.plan),'files':{r['id']:sha(folder/'cases_PRIVATE'/(r['id']+'.json')) for r in rows}})
 print('R50_FULL_CPU_CHUNK_SEALED',a.stage,name,len(done),flush=True)
if __name__=='__main__':main()
