"""R9 pure CPU metrics on supplied arrays. No annotation/model file access.
Annotation conditioning is descriptive post-seal, never G input or online selection.
"""
from pathlib import Path
import sys, math, collections
import numpy as np
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import candidate_ious, transform_gt

MODES=('rgb_fulltext','pc_only')
KINDS=('direct_G','param1','param3','outputspace1','outputspace3')
ARMS=('source',)+tuple(m+'__'+k for m in MODES for k in KINDS)

def eval_box(raw):
 x=np.asarray(raw).copy();x[...,3:6]=np.maximum(x[...,3:6],np.asarray(1e-6,dtype=x.dtype));return x.astype(float)

def coordinate_errors(pred6,target7):
 p=np.asarray(pred6,float);t=np.asarray(target7,float)
 if p.shape!=(6,) or t.shape!=(7,) or not np.isfinite(p).all() or not np.isfinite(t).all() or (t[3:6]<=0).any():raise ValueError('Invalid coordinate error input')
 d=p-t[:6]
 return {'center_L2_m':float(np.linalg.norm(d[:3])),'size_L2_m':float(np.linalg.norm(d[3:])),'coordinate_Linf_m':float(abs(d).max()),'center_abs_error_m':abs(d[:3]).tolist(),'size_abs_error_m':abs(d[3:]).tolist(),'center_L2_gt_dimension_normalized':float(np.linalg.norm(d[:3]/t[3:6])),'size_relative_abs_mean':float(np.mean(abs(d[3:])/t[3:6])),'native_yaw0_vs_GT_yaw_not_maximalIoU_oracle':True}

def box_iou7(a,b):
 return _box_iou7_cached(np.asarray(a,dtype=np.float64).tobytes(),np.asarray(b,dtype=np.float64).tobytes())

@lru_cache(maxsize=8192)
def _box_iou7_cached(abytes,bbytes):
 a=np.frombuffer(abytes,np.float64);b=np.frombuffer(bbytes,np.float64)
 # General3D supports yaw in both boxes; independent of target rank/matching.
 from obb_iou import oriented_iou
 from independent_core_v2 import yaw_matrix
 a=np.asarray(a,float);b=np.asarray(b,float)
 return float(oriented_iou(a[:3],a[3:6],yaw_matrix(a[6]),b[:3],b[3:6],yaw_matrix(b[6])))

def dedup_inventory(boxes7,target_record_index):
 """Complete-link geometric equivalent groups; no class-based target selection.
 boxes7 must come from the SAME metadata/frame. References maintained by caller.
 """
 boxes=np.asarray(boxes7,float)
 if boxes.ndim!=2 or boxes.shape[1]!=7 or len(boxes)==0 or not 0<=target_record_index<len(boxes):raise ValueError('Need frame-local boxes and target record index')
 if not np.isfinite(boxes).all() or (boxes[:,3:6]<=0).any():raise ValueError('Invalid annotated boxes explicitly fail inventory; no silent skip')
 def same(i,j):return np.array_equal(boxes[i],boxes[j]) or np.max(abs(boxes[i,:3]-boxes[j,:3]))<=1e-5 and box_iou7(boxes[i],boxes[j])>1-1e-7
 groups=[]
 for i in range(len(boxes)):
  match=[g for g in groups if all(same(i,j) for j in g)]
  if match:match[0].append(i)
  else:groups.append([i])
 target_group=next(i for i,g in enumerate(groups) if target_record_index in g)
 return groups,target_group

def spatial_confirmation(q0_raw6,target7,inventory7,target_record_index):
 groups,tg=dedup_inventory(inventory7,target_record_index);v,_=candidate_ious(eval_box(np.asarray(q0_raw6)[None]),np.asarray(target7));target_iou=float(v[0]);other=[]
 for gi,g in enumerate(groups):
  if gi!=tg:
   # Annotation-equivalent members may differ within declared tolerance: keep max.
   other.append(max(float(candidate_ious(eval_box(np.asarray(q0_raw6)[None]),np.asarray(inventory7[j]))[0][0]) for j in g))
 best=max(other) if other else None
 target_represented=box_iou7(target7,inventory7[target_record_index])>1-1e-7
 # No other annotation => cannot establish unique match vs distractors.
 confirmed=bool(target_represented and best is not None and target_iou>.25 and target_iou-best>.1)
 return {'annotated_spatially_confirmed':confirmed,'target_iou':target_iou,'best_unique_other_iou':best,'margin_vs_annotated_others':target_iou-best if best is not None else None,'target_record_group':tg,'unique_annotated_groups':len(groups),'raw_annotated_records':len(inventory7),'geometric_duplicate_records':len(inventory7)-len(groups),'no_other_annotation':best is None,'annotation_inventory_exhaustiveness_unknown':True,'not_semantic_instance_identity_proof':True}

def frame_inventory_from_metadata(metadata,object_index,platform,view_suffix=None):
 """ONLY post-release caller may pass annotation object; no file reader here.
 Ground_info is an expression list. Targets+others union avoids treating repeated
 language records as independent objects. Class/caption/positive-map unused.
 """
 entries=metadata['ground_info'];boxes=[];refs=[];target_idx=None
 for i,entry in enumerate(entries):
  if i==object_index:target_idx=len(boxes)
  boxes.append(transform_gt(entry['bbox_3d'],platform,view_suffix));refs.append(['target_expression',i])
  for j,other in enumerate(entry.get('others',[])):
   raw=np.asarray(other['bbox_3d_other']);expected=7 if platform=='waymo' else 9
   if raw.shape not in ((7,),(expected,)):raise ValueError('Unknown other-box schema')
   if raw.shape==(7,) and platform!='waymo':
    # Existing annotation code uses first7 for others; extra fields unnamed.
    raw=np.r_[raw,np.zeros(2,dtype=raw.dtype)]
   boxes.append(transform_gt(raw,platform,view_suffix));refs.append(['other_annotation',i,j])
 if target_idx is None:raise ValueError('Target record missing')
 return np.asarray(boxes,float),target_idx,refs

def aggregate(rows,arm_names=ARMS):
 out=[]
 for platform in ('drone','quad'):
  pp=[r for r in rows if r['platform']==platform]
  for arm in arm_names:
   for pop in ('all','annotated_spatially_confirmed','accepted_any_axis','actual_pseudo_nonzero','common_eligible_all_modes'):
    mode=arm.split('__')[0] if '__' in arm else None
    def eligible(r):
     if pop=='all':return True
     if pop=='annotated_spatially_confirmed':return r['identity_diagnostic']['annotated_spatially_confirmed']
     if pop=='common_eligible_all_modes':return all(x['accepted_any_axis'] for x in r['G'].values())
     if mode is None:return all(x[pop] for x in r['G'].values())
     return r['G'][mode][pop]
    rr=[r for r in pp if eligible(r)]
    vals=[r['arms'][arm]['iou'] for r in rr];g=collections.defaultdict(list)
    for r in rr:g[r['scene']].append(r['arms'][arm]['iou']-r['arms']['source']['iou'])
    gm=np.array([math.fsum(g[s])/len(g[s]) for s in sorted(g)])
    ci=np.quantile(np.random.default_rng(901).choice(gm,(2000,len(gm))).mean(1),[.025,.975]).tolist() if len(gm)>1 else None
    out.append({'platform':platform,'arm':arm,'population':pop,'source_eligibility_rule':'all modes satisfy requested eligibility population' if mode is None and pop in ('accepted_any_axis','actual_pseudo_nonzero') else None,'n':len(rr),'scene_clusters':len(g),'mean_iou':math.fsum(vals)/len(vals) if vals else None,'Acc25':sum(v>.25 for v in vals)/len(vals) if vals else None,'Acc50':sum(v>.5 for v in vals)/len(vals) if vals else None,'micro_delta_vs_source':math.fsum([r['arms'][arm]['iou']-r['arms']['source']['iou'] for r in rr])/len(rr) if rr else None,'equal_scene_delta_vs_source':float(gm.mean()) if len(gm) else None,'descriptive_scene_CI95':ci,'negative_vs_source_atol1e8':sum(r['arms'][arm]['iou']-r['arms']['source']['iou']< -1e-8 for r in rr),'center_L2_m_mean':math.fsum(r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in rr)/len(rr) if rr else None,'size_L2_m_mean':math.fsum(r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in rr)/len(rr) if rr else None})
 return out
