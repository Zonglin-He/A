"""Fixed, parameter-free DINO readout and float-valued burst coverage.

No label access, no temporal selection, and no model construction.
"""
import numpy as np

def dino_refine(native_boxes,frame_ids,expert):
 boxes=np.asarray(native_boxes,dtype=np.float32).copy();ids=np.asarray(frame_ids,dtype=np.float64)
 assert boxes.shape==(len(ids),4) and np.isfinite(boxes).all() and np.all(np.diff(ids)>0)
 anchors=expert['anchors']['single4'];points=[]
 for v in anchors:
  # Ours retains one admitted Top1 box per observed physical position.
  points.append((int(v['position']),np.asarray(v['box'],dtype=np.float32)))
 if not points:return boxes,dict(admitted_frames=0,fallback='Frozen',expert_calls=0,parameter_updates=0)
 points.sort(key=lambda x:x[0]);assert len({x[0] for x in points})==len(points)
 pos=np.asarray([x[0] for x in points],dtype=np.int64);ev=np.stack([x[1] for x in points])
 assert pos.min()>=0 and pos.max()<len(ids) and ev.shape==(len(pos),4) and np.isfinite(ev).all()
 for d in range(4):boxes[:,d]=np.interp(ids,ids[pos],ev[:,d]).astype(np.float32)
 assert np.array_equal(boxes[pos],ev)
 return boxes,dict(admitted_frames=len(points),interpolation='absolute cxcywh in physical frames; nearest endpoint extension',expert_calls=0,parameter_updates=0)

def severity_observation(row,family,frames,coverage):
 from vg_tta.tastvg_deployment_corruption_v2 import burst_spec,apply_burst
 from scripts.c1_controlled_corruption_v1 import pixelhash
 if family=='clean':
  return np.ascontiguousarray(frames),pixelhash(frames),dict(kind='clean',percentage=0.,positions=[])
 assert family in ('frame_drop','frame_freeze','motion_blur','occlusion','exposure') and float(coverage) in (2.5,5.,10.)
 # Do not round 2.5 to 2; percentages control physical affected-frame fraction.
 spec=burst_spec(row['source'],row['input']['frame_count'],row['frame_ids'],float(coverage))
 shifted=apply_burst(frames,row['input'],spec,family,row['source'])
 return shifted,pixelhash(shifted),dict(spec,kind=family,percentage=float(coverage))
