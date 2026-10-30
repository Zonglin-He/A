"""R11 pure selected-box/atomic-unit metrics. Caller gates GT after all seals.
No file, model or dataset IO. Shared source, physical object and scene dependence
remain explicit; an atomic coordinate replacement is not causal face deletion.
"""
from pathlib import Path
import sys,math,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import candidate_ious
from discrimination import assay,risk_summary
ARMS=('source','unconditional_full','no_rgb_coordinate','weighted_constraints','uniform_weighted_constraints','reliable_50','hash_random_50','source_agreement_50','low_plane_residual_50','point_count_50','stability_only_50')
GATES=ARMS[5:]
BOOTSTRAP_SEED=1101
BOOTSTRAP_DRAWS=2000

def mean(x):
 x=list(x);return math.fsum(x)/len(x) if x else None

def eval_boxes(x):
 b=np.array(x,copy=True);b[...,3:6]=np.maximum(b[...,3:6],np.asarray(1e-6,dtype=b.dtype));return b.astype(float)

def errors(b,t):
 e=np.asarray(b,float)-np.asarray(t,float)[:6]
 return {'center_L2_m':float(np.linalg.norm(e[:3])),'size_L2_m':float(np.linalg.norm(e[3:])),'abs_error6_m':abs(e).tolist()}

def signed(values,tol=1e-8):
 x=list(map(float,values));assert all(math.isfinite(v) for v in x)
 p=[v for v in x if v>tol];n=[v for v in x if v< -tol]
 return {'n':len(x),'mean':mean(x),'sum':math.fsum(x),'positive_count':len(p),'negative_count':len(n),'tie_count':sum(abs(v)<=tol for v in x),'positive_sum':math.fsum(p),'negative_sum':math.fsum(n),'gain_sum_exceeds_absolute_harm_sum':math.fsum(p)>abs(math.fsum(n)),'min':min(x,default=None),'max':max(x,default=None),'tolerance':tol}

def scene_stats(rows,values):
 g=collections.defaultdict(list)
 for row,v in zip(rows,values):g[row['scene']].append(float(v))
 m=np.array([mean(g[k]) for k in sorted(g)])
 ci=np.quantile(np.random.default_rng(BOOTSTRAP_SEED).choice(m,(BOOTSTRAP_DRAWS,len(m))).mean(1),[.025,.975]).tolist() if len(m)>1 else None
 return {'n':len(values),'defined_scene_clusters':len(g),'micro_mean':mean(values),'equal_scene_mean':mean(m),'descriptive_scene_CI95':ci,'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_draws':BOOTSTRAP_DRAWS,'few_scene_not_significance_or_population_proof':True}

def evaluate_case(*,sid,platform,scene,boxes_raw6,atomic_boxes_raw6,target7,statuses,atomic_active6,coordinate_scores6,coordinate_defined6,RGB_score,RGB_defined,case_score,case_defined,gate_selected6,control_names,control_scores,control_defined):
 b=np.asarray(boxes_raw6);atomic=np.asarray(atomic_boxes_raw6);t=np.asarray(target7,float)
 assert b.shape==(len(ARMS),6) and atomic.shape==(6,6) and t.shape==(7,)
 assert np.isfinite(b).all() and np.isfinite(atomic).all() and np.isfinite(t).all() and (t[3:6]>0).all()
 v,cost=candidate_ious(eval_boxes(np.r_[b,atomic]),t)
 source=b[0];full=b[1];srcerr=errors(source,t);fullerr=errors(full,t)
 active=np.asarray(atomic_active6,bool);assert active.shape==(6,) and np.array_equal(active,full!=source)
 expected=np.tile(source,(6,1));expected[np.arange(6),np.arange(6)]=full
 assert np.array_equal(expected,atomic)
 row={'id':sid,'platform':platform,'scene':scene,'arms':{},'units_coordinate_application':[]}
 for j,name in enumerate(ARMS):
  er=errors(b[j],t);changed=b[j]!=source
  row['arms'][name]={'iou':float(v[j]),'coordinate_errors':er,'actual_changed':bool(changed.any()),'actual_changed_axes':changed.tolist(),'status':str(statuses[j]),'raw_nonpositive_sizes':int((b[j,3:]<=0).sum()),'center_shift_L2_m':float(np.linalg.norm(b[j,:3].astype(float)-source[:3].astype(float))),'size_shift_L2_m':float(np.linalg.norm(b[j,3:].astype(float)-source[3:].astype(float)))}
 for j in range(6):
  row['units_coordinate_application'].append({'case_id':sid,'coordinate':j,'candidate':bool(active[j]),'score':float(coordinate_scores6[j]),'score_defined':bool(coordinate_defined6[j]),'effect':float(v[len(ARMS)+j]-v[0]),'coordinate_error_reduction_m':srcerr['abs_error6_m'][j]-fullerr['abs_error6_m'][j]})
 rgb_changed=bool(np.any(b[1]!=b[2]));row['unit_RGB_term']={'case_id':sid,'candidate':rgb_changed,'score':float(RGB_score),'score_defined':bool(RGB_defined),'effect':float(v[1]-v[2]),'center_error_reduction_m':row['arms']['no_rgb_coordinate']['coordinate_errors']['center_L2_m']-fullerr['center_L2_m'],'size_error_reduction_m':row['arms']['no_rgb_coordinate']['coordinate_errors']['size_L2_m']-fullerr['size_L2_m']}
 row['case_unit']={'case_id':sid,'candidate':bool(active.any()),'score':float(case_score),'score_defined':bool(case_defined),'effect':float(v[1]-v[0])}
 row['gate_selected']={name:bool(gate_selected6[j]) for j,name in enumerate(GATES)}
 row['case_controls']={str(name):{'value':float(control_scores[j]),'defined':bool(control_defined[j])} for j,name in enumerate(control_names)}
 return row,cost

def summarize(rows):
 tables=[];paired=[];discrimination={};risk=[]
 for platform in ('drone','quad'):
  pp=[r for r in rows if r['platform']==platform]
  masks={'all':lambda r:True,'original_G_changed':lambda r:r['case_unit']['candidate'],'weighted_union_changed':lambda r:any(r['arms'][a]['actual_changed'] for a in ARMS[1:2]+ARMS[3:5]),'RGB_term_changed':lambda r:r['unit_RGB_term']['candidate']}
  for name in ARMS:
   for pop,fn in masks.items():
    rr=[r for r in pp if fn(r)];aa=[r['arms'][name] for r in rr];n=len(rr)
    tables.append({'platform':platform,'arm':name,'population':pop,'n':n,'scene_clusters':len({r['scene'] for r in rr}),'mean_iou':mean(a['iou'] for a in aa),'Acc25':sum(a['iou']>.25 for a in aa)/n if n else None,'Acc50':sum(a['iou']>.5 for a in aa)/n if n else None,'center_L2_m_mean':mean(a['coordinate_errors']['center_L2_m'] for a in aa),'size_L2_m_mean':mean(a['coordinate_errors']['size_L2_m'] for a in aa),'actual_changed_n':sum(a['actual_changed'] for a in aa),'actual_changed_axis_counts':[sum(a['actual_changed_axes'][j] for a in aa) for j in range(6)],'status_histogram':dict(collections.Counter(a['status'] for a in aa))})
  comparisons=[(a,'source') for a in ARMS[1:]]+[('unconditional_full','no_rgb_coordinate'),('weighted_constraints','uniform_weighted_constraints')]+[('reliable_50',a) for a in GATES if a!='reliable_50']
  for arm,ref in comparisons:
   for pop,fn in masks.items():
    rr=[r for r in pp if fn(r)];iou=[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in rr];ce=[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in rr];se=[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in rr]
    paired.append({'platform':platform,'arm':arm,'reference':ref,'population':pop,'n':len(rr),'IoU_gain':signed(iou),'center_error_reduction_m':signed(ce,1e-6),'size_error_reduction_m':signed(se,1e-6),'IoU_scene':scene_stats(rr,iou),'center_scene':scene_stats(rr,ce),'size_scene':scene_stats(rr,se)})
  coord=[u for r in pp for u in r['units_coordinate_application']];rgb=[r['unit_RGB_term'] for r in pp];cases=[r['case_unit'] for r in pp]
  discrimination[platform]={'coordinate_primary_IoU':assay(coord),'coordinate_secondary_coordinate_error':assay(coord,'coordinate_error_reduction_m',1e-6),'RGB_primary_matched_IoU':assay(rgb),'RGB_secondary_center_error':assay(rgb,'center_error_reduction_m',1e-6),'case_originalG_IoU':assay(cases)}
  for name in GATES:
   selected=[r for r in pp if r['gate_selected'][name]]
   assert all(r['case_unit']['candidate'] for r in selected)
   d=[r['case_unit']['effect'] for r in selected]
   risk.append({'platform':platform,'policy':name,'candidate_n':sum(r['case_unit']['candidate'] for r in pp),'candidate_defined_n':sum(r['case_unit']['candidate'] and r['case_unit']['score_defined'] for r in pp),'selected_n':len(selected),'selected_undefined_primary_reliability_n':sum(not r['case_unit']['score_defined'] for r in selected),'risk':risk_summary(d,len(pp)),'scene':scene_stats(pp,[r['case_unit']['effect'] if r['gate_selected'][name] else 0. for r in pp])})
 return {'summary':tables,'paired_contrasts':paired,'discrimination':discrimination,'coverage_risk_primary50':risk,'scope':'Exploratory new320 within previously research-exposed targetVAL; 2Drone/3Quad scenes. Coordinate replacement is not face-refit causal attribution; heuristic reliability is not correctness probability. No learned TTA.'}
