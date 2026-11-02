"""Known candidate pairs with explicit 2D/3D evidence, never box refinement.
Unused LiDAR points are unknown ownership, not demonstrated foreign objects.
"""
import sys,hashlib,itertools
import numpy as np
from common import ROOT,arrsha
sys.path[:0]=[str(ROOT/'crossplatform/round16'),str(ROOT/'crossplatform/round13')]
from preferences import component_mapping
from association import box_corners,rect_iou
from camera_projection import project_model_points

def prepare(scene,boxes,case_id):
 old=component_mapping(scene,boxes);objects=scene['objects'];points=scene['visible_points_model'];depth=scene['depth_camera'];masks=[]
 for o in objects:
  ids=np.asarray(o['exclusive_visible_indices'],int)
  if len(ids)>=4:
   d=depth[ids];q1,q3=np.quantile(d,[.25,.75]);spread=max(q3-q1,.1);ids=ids[(d>=q1-1.5*spread)&(d<=q3+1.5*spread)]
  masks.append(ids)
 Q=len(boxes);M=len(objects);support=np.zeros((Q,M),int);coverage=np.zeros((Q,M));project_iou=np.zeros((Q,M))
 for q,b in enumerate(boxes):
  if (b[3:]<=0).any():continue
  proj=project_model_points(box_corners(b),scene['camera_spec'],scene['image_size_wh'])
  valid=bool(np.all(proj['depth']>.1) and np.isfinite(proj['uv']).all());rect=np.r_[proj['uv'].min(0),proj['uv'].max(0)] if valid else None
  for m,ids in enumerate(masks):
   support[q,m]=np.count_nonzero((np.abs(points[ids]-b[:3])<=b[3:]/2+.1).all(1));coverage[q,m]=support[q,m]/max(len(ids),1)
   if valid:project_iou[q,m]=rect_iou(rect,objects[m]['detector_box_xyxy'])
 pairs=[]
 for m,n in itertools.combinations(range(M),2):
  if objects[m]['predicted_class_id']!=objects[n]['predicted_class_id']:continue
  mask_inter=np.count_nonzero(objects[m]['mask_bool']&objects[n]['mask_bool']);mask_union=np.count_nonzero(objects[m]['mask_bool']|objects[n]['mask_bool']);mask_iou=mask_inter/max(mask_union,1)
  if mask_iou>=.2:continue
  groups=[];relaxed=[]
  for a,b in ((m,n),(n,m)):
   # Pair contrast is measurable; other/unassigned points are not negative truth.
   okay=(support[:,a]>=3)&(coverage[:,a]>=.15)&(project_iou[:,a]>=.1)&(support[:,b]<=.2*np.maximum(support[:,a],1))
   groups.append(np.flatnonzero(okay).tolist())
   weak=(support[:,a]>=1)&(project_iou[:,a]>=.1)&(support[:,b]<support[:,a]);relaxed.append(np.flatnonzero(weak).tolist())
  if len(masks[m]) and len(masks[n]):sep=float(np.linalg.norm(np.median(points[masks[m]],0)-np.median(points[masks[n]],0)))
  else:sep=None
  geometry_separated=bool(sep is not None and sep>=.5)
  oldgroups=[np.flatnonzero(old['assignment']==k).tolist() for k in (m,n)]
  token=hashlib.sha256((case_id+'|'+arrsha(objects[m]['mask_bool'])+'|'+arrsha(objects[n]['mask_bool'])).encode()).hexdigest()
  pairs.append({'mask_indices':[m,n],'query_groups':groups,'relaxed_query_groups':relaxed,'old_strict_query_groups':oldgroups,'geometry_separated':geometry_separated,'component_median_separation_m':sep,'mask_iou':mask_iou,'main_mapping_both':bool(all(groups) and geometry_separated),'old_mapping_both':bool(all(oldgroups) and geometry_separated),'relaxed_mapping_both':bool(all(relaxed) and geometry_separated),'order_hash':token})
 # Finite, caption-independent coverage: up to4 usable pairs +1 unadmitted probe.
 usable=sorted([p for p in pairs if p['main_mapping_both']],key=lambda p:p['order_hash'])[:4]
 probes=sorted([p for p in pairs if not p['main_mapping_both']],key=lambda p:p['order_hash'])[:1]
 selected=usable+probes
 for i,p in enumerate(selected):p['pair_alias']=f'pair_{i:02d}'
 return {'pairs':selected,'all_sameclass_pair_count':len(pairs),'all_main_mapping_pair_count':sum(p['main_mapping_both'] for p in pairs),'all_old_mapping_pair_count':sum(p['old_mapping_both'] for p in pairs),'all_relaxed_mapping_pair_count':sum(p['relaxed_mapping_both'] for p in pairs),'point_support_count':support.tolist(),'point_support_fraction':coverage.tolist(),'projection_iou':project_iou.tolist(),'old_component_diagnostics':old['component_diagnostics'],'point_count_after_depth_trim':[len(x) for x in masks]}
