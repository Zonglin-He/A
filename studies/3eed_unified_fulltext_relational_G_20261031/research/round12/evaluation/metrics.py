"""R12 pure selected-box geometry and independent paired reductions.
No dataset/GT file IO or model/GPU. All input GT supplied by post-seal caller.
"""
from pathlib import Path
import sys,collections,math
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import candidate_ious
BOOTSTRAP_SEED=1201;BOOTSTRAP_DRAWS=2000

def mean(x):return math.fsum(x)/len(x) if x else None
def eval_boxes(x):
 b=np.array(x,copy=True);b[...,3:6]=np.maximum(b[...,3:6],np.asarray(1e-6,dtype=b.dtype));return b.astype(float)
def coordinate_errors(box6,target7):
 b=np.asarray(box6,float);t=np.asarray(target7,float);e=b-t[:6]
 return {'center_L2_m':float(np.linalg.norm(e[:3])),'size_L2_m':float(np.linalg.norm(e[3:])),'abs_error6_m':abs(e).tolist(),'size_parameter_not_global_yaw0_enclosure':True}
def evaluate_case(*,sid,platform,scene,boxes_raw6,arm_names,accepted_axes,target7,statuses):
 boxes=np.asarray(boxes_raw6);t=np.asarray(target7,float);axes=np.asarray(accepted_axes,bool)
 assert boxes.shape==(len(arm_names),6) and axes.shape==boxes.shape and len(statuses)==len(arm_names) and np.isfinite(boxes).all() and t.shape==(7,) and np.isfinite(t).all() and (t[3:6]>0).all()
 values,cost=candidate_ious(eval_boxes(boxes),t);assert np.isfinite(values).all()
 row={'id':sid,'platform':platform,'scene':scene,'arms':{}}
 source=boxes[0]
 for j,name in enumerate(arm_names):
  changed=boxes[j]!=source
  assert np.array_equal(boxes[j,~axes[j]],source[~axes[j]])
  row['arms'][name]={'iou':float(values[j]),'coordinate_errors':coordinate_errors(boxes[j],t),'accepted_axes':axes[j].tolist(),'accepted_any':bool(axes[j].any()),'actual_changed_axes':changed.tolist(),'actual_changed':bool(changed.any()),'raw_nonpositive_sizes':int((boxes[j,3:]<=0).sum()),'status':str(statuses[j]),'center_shift_L2_m':float(np.linalg.norm(boxes[j,:3].astype(float)-source[:3].astype(float))),'size_shift_L2_m':float(np.linalg.norm(boxes[j,3:].astype(float)-source[3:].astype(float)))}
 return row,cost

def signed_stats(values,tol):
 x=list(map(float,values));p=[v for v in x if v>tol];n=[v for v in x if v< -tol]
 return {'n':len(x),'mean':mean(x),'sum':math.fsum(x),'positive_count':len(p),'negative_count':len(n),'ties_within_tol':sum(abs(v)<=tol for v in x),'positive_sum':math.fsum(p),'negative_sum':math.fsum(n),'gain_sum_exceeds_abs_harm_sum':math.fsum(p)>abs(math.fsum(n)),'min':min(x,default=None),'max':max(x,default=None),'tolerance':tol}
def paired_scene(rows,values):
 g=collections.defaultdict(list)
 for r,v in zip(rows,values):g[r['scene']].append(float(v))
 gm=np.array([mean(g[k]) for k in sorted(g)])
 ci=np.quantile(np.random.default_rng(BOOTSTRAP_SEED).choice(gm,(BOOTSTRAP_DRAWS,len(gm))).mean(1),[.025,.975]).tolist() if len(gm)>1 else None
 return {'n':len(values),'scene_clusters':len(g),'micro_mean':mean(list(values)),'equal_scene_mean':mean(gm.tolist()),'equal_scene_descriptive_CI95':ci,'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_draws':BOOTSTRAP_DRAWS,'few_scene_not_significance_or_population_proof':True}

def summaries(rows):
 names=list(rows[0]['arms']);out=[];contrasts=[]
 pairs=[(a,'source') for a in names if a!='source']+[('single_instance_direct','grabcut_direct'),('multiview_instance_direct','single_instance_direct'),('multiview_instance_direct','multiview_instance_half'),('single_instance_direct','single_instance_half'),('multiview_instance_direct','multiview_maskmatch_only_direct'),('multiview_instance_direct','wrongview_instance_direct')]
 for platform in ('drone','quad'):
  pp=[r for r in rows if r['platform']==platform]
  populations={'all':lambda r:True,'single_MV_union_changed':lambda r:r['arms']['single_instance_direct']['actual_changed'] or r['arms']['multiview_instance_direct']['actual_changed'],'single_MV_common_eligible':lambda r:r['arms']['single_instance_direct']['accepted_any'] and r['arms']['multiview_instance_direct']['accepted_any']}
  for name in names:
   for pop,test in list(populations.items())+[('own_changed_descriptive',lambda r:r['arms'][name]['actual_changed'])]:
    rr=[r for r in pp if test(r)];a=[r['arms'][name] for r in rr];n=len(a)
    vals=[x['iou'] for x in a];ce=[x['coordinate_errors']['center_L2_m'] for x in a];se=[x['coordinate_errors']['size_L2_m'] for x in a]
    out.append({'platform':platform,'arm':name,'population':pop,'n':n,'scene_clusters':len({r['scene'] for r in rr}),'mean_iou':mean(vals),'equal_scene_mean_iou':paired_scene(rr,vals)['equal_scene_mean'],'Acc25':sum(v>.25 for v in vals)/n if n else None,'Acc50':sum(v>.5 for v in vals)/n if n else None,'equal_scene_Acc25':paired_scene(rr,[float(v>.25) for v in vals])['equal_scene_mean'],'equal_scene_Acc50':paired_scene(rr,[float(v>.5) for v in vals])['equal_scene_mean'],'center_L2_m_mean':mean(ce),'equal_scene_center_L2_m_mean':paired_scene(rr,ce)['equal_scene_mean'],'size_L2_m_mean':mean(se),'equal_scene_size_L2_m_mean':paired_scene(rr,se)['equal_scene_mean'],'accepted_any_n':sum(x['accepted_any'] for x in a),'actual_changed_n':sum(x['actual_changed'] for x in a),'actual_changed_axis_counts':[sum(x['actual_changed_axes'][j] for x in a) for j in range(6)],'status_histogram':dict(collections.Counter(x['status'] for x in a))})
  for arm,ref in pairs:
   for pop,test in populations.items():
    rr=[r for r in pp if test(r)];iou=[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in rr];center=[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in rr];size=[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in rr]
    contrasts.append({'platform':platform,'arm':arm,'reference':ref,'population':pop,'n':len(rr),'IoU_gain':signed_stats(iou,1e-8),'center_error_reduction_m':signed_stats(center,1e-6),'size_error_reduction_m':signed_stats(size,1e-6),'IoU_scene':paired_scene(rr,iou),'center_scene':paired_scene(rr,center),'size_scene':paired_scene(rr,size),'all_positive_direction_means_improvement':True})
 return out,contrasts
