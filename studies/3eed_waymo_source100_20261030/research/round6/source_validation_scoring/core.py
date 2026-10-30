"""PureCPU sourceVAL metrics and independent selected3D crosscheck, no file I/O."""
import numpy as np
from batch_yaw_iou import many_iou
from obb_iou import oriented_iou
HEADS=['last_','proposal_'];MODES=['official_bbf','official_bbs','text_bbf','text_bbs'];KS=(1,5,10,256);THRESHOLDS=(.25,.5)
INDEPENDENT_ABS_TOL=1e-8

def native_top(scores,top):
 s=np.asarray(scores);p=np.asarray(top)
 if s.shape!=(256,) or not np.isfinite(s).all() or p.shape!=(10,) or not np.issubdtype(p.dtype,np.integer):raise ValueError('finite256scores and nativeintegerTop10 expected')
 if len(np.unique(p))!=10 or (p<0).any() or (p>=256).any():raise ValueError('nativeTop10 invalidindices')
 if np.any(s[p[:-1]]<s[p[1:]]):raise ValueError('nativeTop10 notdescending')
 left=np.ones(256,bool);left[p]=False
 if np.max(s[left])>s[p[-1]]:raise ValueError('nativeTop10 excludes higher score')
 return p.astype(np.int64)

def validate_boxes(raw,boxes):
 a=np.asarray(raw);b=np.asarray(boxes)
 if a.shape!=(2,256,6) or b.shape!=a.shape or a.dtype!=np.float32 or b.dtype!=np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('finiteFP32 raw/clamped[2,256,6] expected')
 expected=a.copy();expected[...,3:6]=np.maximum(expected[...,3:6],np.float32(1e-6))
 if not np.array_equal(expected,b):raise ValueError('officialdimensionsclamp mismatch orcenter changed')
 return {'raw_negative_candidates_by_head':np.any(a[...,3:6]<0,axis=-1).sum(-1).tolist(),'raw_nonpositive_candidates_by_head':np.any(a[...,3:6]<=0,axis=-1).sum(-1).tolist(),'raw_below_clamp_candidates_by_head':np.any(a[...,3:6]<1e-6,axis=-1).sum(-1).tolist()}

def metrics(values,scores,top):
 v=np.asarray(values,dtype=np.float64);s=np.asarray(scores);p=native_top(s,top)
 if v.shape!=(256,) or not np.isfinite(v).all() or (v<0).any() or (v>1+1e-12).any():raise ValueError('InvalidcandidateIoU')
 a={'query':int(p[0]),'iou':float(v[p[0]]),'oracle_iou_GT_assisted':float(v.max()),'oracle_query_GT_assisted':int(v.argmax()),'topk':{},'top_score_tied_count':int(np.sum(s==s[p[0]]))}
 for k in KS:
  x=float(v[p[:k]].max()) if k<=10 else float(v.max());a['topk'][str(k)]={'max_retrieved_iou_GT_assisted_if_k_gt_1':x,'acc25':bool(x>.25),'acc50':bool(x>.5)}
 a['rank_recoverable50']=a['iou']<=.5 and a['oracle_iou_GT_assisted']>.5;a['candidate_coverage_failure50']=a['oracle_iou_GT_assisted']<=.5;return a

def independent_selected_iou(box6,gt7):
 b=np.asarray(box6,np.float64);g=np.asarray(gt7,np.float64)
 if b.shape!=(6,) or g.shape!=(7,) or not np.isfinite(b).all() or not np.isfinite(g).all() or (b[3:6]<=0).any() or (g[3:6]<=0).any():raise ValueError('Invalid3Dgeometry')
 c,s=np.cos(g[6]),np.sin(g[6]);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
 # Strict broadphase only skips geometrically proven disjoint boxes. Contact goes to hull.
 if np.any(np.abs(b[:3]-g[:3])>b[3:6]/2+np.abs(rotation)@(g[3:6]/2)):return 0.,True
 return float(oriented_iou(b[:3],b[3:6],np.eye(3),g[:3],g[3:6],rotation)),False

def score_arrays(z,gt7):
 if z['head_names'].tolist()!=HEADS:raise ValueError('Wrongheadorder')
 invalid=validate_boxes(z['boxes_raw'],z['boxes']);g=np.asarray(gt7,np.float32).astype(np.float64)
 if g.shape!=(7,) or not np.isfinite(g).all() or np.any(g[3:6]<=0):raise ValueError('SourceGTmustbe7Dpositivefloat32')
 arms={};values={};independent={};maxdiff=0.;calls=0;zero=0;bools=0
 for h,head in enumerate(HEADS):
  b=z['boxes'][h].astype(np.float64);iou=many_iou(np.c_[b,np.zeros(256)],g);values[head]=iou
  for mode in MODES:
   name=('official_full_' if mode.startswith('official') else 'text_full_')+head+mode.split('_')[-1];a=metrics(iou,z[mode][h],z[mode+'_top10'][h]);q=a['query'];key=(h,q)
   if key not in independent:
    independent[key],skip=independent_selected_iou(b[q],g);calls+=not skip;zero+=skip
   other=independent[key];difference=abs(other-a['iou']);maxdiff=max(maxdiff,difference)
   if difference>INDEPENDENT_ABS_TOL:raise AssertionError(('primary2D_vs_independent3Dselected',h,q,a['iou'],other,difference))
   for t in THRESHOLDS:
    if (a['iou']>t)!=(other>t):raise AssertionError(('independentselected_threshold_disagreement',h,q,t,a['iou'],other))
    bools+=1
   a['independent_3D_selected_iou']=other;arms[name]=a
 return arms,values,{'max_abs_selected2D_vs3D_difference':maxdiff,'unique_selected_boxes_checked':len(independent),'independent_convex_calls':calls,'proven_disjoint_zero_checks':zero,'strict_threshold_booleans_checked':bools},invalid
