"""Independent round5 CPU geometry/ranking primitives. No data/model/GT file access.
Primary geometry is axis-aligned prediction vs dataset's transformed yaw-only GT.
"""
from pathlib import Path
import sys,math
import numpy as np
B=Path(__file__).resolve().parents[3];sys.path.insert(0,str(B/'tools'))
from obb_iou import oriented_iou
KS=(1,5,10,256);THRESHOLDS=(.25,.5)

def yaw_matrix(yaw):
 c,s=math.cos(float(yaw)),math.sin(float(yaw));return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
def transform_gt(raw_box,platform,view_suffix=None):
 """Independently mimic actual dataset float64 transform -> float32 7D tensor."""
 raw=np.asarray(raw_box);expected=7 if platform=='waymo' else 9
 if raw.shape!=(expected,):raise ValueError('RawGT schema must be7 Waymo,9 Drone/Quad')
 raw=raw.copy()
 if platform=='drone':raw[2]+=1.8
 b=raw[:7].astype(np.float64).copy()
 if platform=='waymo':
  if view_suffix not in range(5):raise ValueError('Need audited Waymo suffix0..4')
  a=math.radians((0,-45,45,-90,90)[view_suffix]);b[:3]=yaw_matrix(a)@b[:3];b[6]+=a
 elif platform not in ['quad','drone']:raise ValueError('unknownplatform')
 b=b.astype(np.float32)
 if not np.isfinite(b).all() or (b[3:6]<=0).any():raise ValueError('InvalidGT; do not silently skip')
 return b

def candidate_ious(boxes,target):
 """All256 independent general3D convex; AABB broadphase only proves exactzeros."""
 boxes=np.asarray(boxes);t=np.asarray(target)
 if boxes.ndim!=2 or boxes.shape[1]!=6 or t.shape!=(7,):raise ValueError('boxesQx6 target7 expected')
 if not np.isfinite(boxes).all() or not np.isfinite(t).all() or (t[3:6]<=0).any():raise ValueError('Nonfinitepredictions/invalidGT require explicitfailure')
 valid=(boxes[:,3:6]>0).all(-1);R=yaw_matrix(t[6]);gt_half=np.abs(R)@(t[3:6].astype(float)/2);distance=np.abs(boxes[:,:3].astype(float)-t[:3].astype(float));bounds=boxes[:,3:6].astype(float)/2+gt_half
 # Strict disjoint broadphase; contactzero remains delegated to exactconvex routine.
 disjoint=np.any(distance>bounds,axis=1);values=np.zeros(len(boxes),float)
 for j in np.flatnonzero(valid&~disjoint):values[j]=oriented_iou(boxes[j,:3],boxes[j,3:6],np.eye(3),t[:3],t[3:6],R)
 return values,{'invalid_size_count':int((~valid).sum()),'broadphase_proven_zero_count':int((valid&disjoint).sum()),'convex_intersection_calls':int((valid&~disjoint).sum())}

def validate_native_top10(scores,top10):
 s=np.asarray(scores);top=np.asarray(top10)
 if s.ndim!=1 or not np.isfinite(s).all() or top.shape!=(10,) or not np.issubdtype(top.dtype,np.integer):raise ValueError('scoresQ finite +nativeinteger10')
 if len(np.unique(top))!=10 or (top<0).any() or (top>=len(s)).any():raise ValueError('invalidtop10indices')
 if np.any(s[top[:-1]]<s[top[1:]]):raise ValueError('nativetop10notdescending')
 unselected=np.ones(len(s),bool);unselected[top]=False
 if unselected.any() and np.max(s[unselected])>s[top[-1]]:raise ValueError('nativetop10omitshigher score')
 # Tied nativeindices are preserved, not silently stable-resorted.
 return top.astype(np.int64)

def ranking_metrics(values,scores,top10):
 v=np.asarray(values,float);s=np.asarray(scores);top=validate_native_top10(s,top10)
 if v.shape!=s.shape or len(s)!=256 or not np.isfinite(v).all() or np.any((v<0)|(v>1+1e-12)):raise ValueError('Need finite256 IoUs')
 out={};best=float(v.max())
 for k in KS:
  x=float(v[top[:k]].max()) if k<=10 else best
  out[str(k)]={'best_iou_within_retrieved_candidates':x,'acc25':bool(x>.25),'acc50':bool(x>.5),'GTassisted_selection':k>1}
 out['top1_query']=int(top[0]);out['candidate_best_query_GTassisted']=int(v.argmax());out['candidate_coverage_failure25']=best<=.25;out['candidate_coverage_failure50']=best<=.5;out['rank_recoverable25']=bool(v[top[0]]<=.25 and best>.25);out['rank_recoverable50']=bool(v[top[0]]<=.5 and best>.5);out['native_top10_tie_count']=int(np.sum(s==s[top[0]]));return out

def validate_official_clamp(raw,clamped):
 raw=np.asarray(raw);clamped=np.asarray(clamped)
 if raw.shape!=clamped.shape or raw.shape[-1]!=6 or not np.isfinite(raw).all():raise ValueError('Raw/clampedboxes mismatch')
 expected=raw.copy();expected[...,3:6]=np.maximum(expected[...,3:6],np.asarray(1e-6,dtype=raw.dtype))
 if not np.array_equal(clamped,expected):raise ValueError('Officialpredsizeclamp mismatch')
 return {'raw_nonpositive_dimension_count':int((raw[...,3:6]<=0).sum()),'clamped_dimension_count':int((raw[...,3:6]<1e-6).sum()),'clamped_candidate_count':int(np.any(raw[...,3:6]<1e-6,axis=-1).sum())}

def fixtures():
 target=np.array([0.,0.,0.,2.,2.,2.,0.],np.float32);boxes=np.tile([100.,0.,0.,2.,2.,2.],(256,1));boxes[0]=[0,0,0,1,1,2];boxes[1]=[0,0,0,1,2,2];boxes[6]=target[:6];values,g=candidate_ious(boxes,target);scores=1-np.arange(256)*.001;m=ranking_metrics(values,scores,np.arange(10));assert values[0]==.25 and values[1]==.5 and values[6]==1.;assert not m['1']['acc25'] and m['5']['acc25'] and not m['5']['acc50'] and m['10']['acc50'] and m['256']['acc50']
 tied=np.ones(256);native=np.array([6,0,1,2,3,4,5,7,8,9]);assert ranking_metrics(values,tied,native)['top1_query']==6
 b=target.copy();b[0]=4.;assert candidate_ious(boxes[[6]],b)[0][0]==0
 raw=np.array([[[0.,0,0,-1.,0,2]]],np.float32);clamp=raw.copy();clamp[...,3:6]=np.maximum(clamp[...,3:6],np.float32(1e-6));assert validate_official_clamp(raw,clamp)['clamped_dimension_count']==2
 drone=np.array([1.,2.,3.,4.,5.,6.,.7,.8,.9]);t=transform_gt(drone,'drone');assert t.shape==(7,) and t.dtype==np.float32 and t[2]==np.float32(4.8) and t[6]==np.float32(.7)
 waymo=np.array([1.,0.,0.,2.,2.,2.,.1]);t=transform_gt(waymo,'waymo',1);assert np.allclose(t[:2],[2**-.5,-2**-.5]) and abs(t[6]-np.float32(.1-np.pi/4))<1e-7
 return {'synthetic_only':True,'strict25_50_thresholds_rank7_oracle':True,'native_nonstable_tie_preserved':True,'officialclamp_exact':True,'independent_Drone7D_truncation_and_shift':True,'independent_Waymo_rotation_andfloat32':True,'broadphase_uses_only_proven_disjoint_zeros':True,'no_dataset_model_GPU_or_newGTread':True}
if __name__=='__main__':
 import json;print(json.dumps(fixtures(),indent=2))
