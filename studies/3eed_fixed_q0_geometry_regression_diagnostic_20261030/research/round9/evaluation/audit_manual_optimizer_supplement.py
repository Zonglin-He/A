"""Independent postseal aggregate/step-bound review, no newIoU or human relabeling."""
from pathlib import Path
import json,hashlib,math,collections
import numpy as np
R=Path(__file__).resolve().parents[3];E=R/'research/round9/experiment';P=R/'crossplatform/round9'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def near(a,b,tol=1e-12):assert np.allclose(a,b,rtol=0,atol=tol),(a,b)
def mean(v):return math.fsum(v)/len(v)
manual=read(E/'MANUAL_IDENTITY_REVIEW_PRIVATE.json');agg=read(E/'MANUAL_SUBSET_ERROR_SUMMARY.json');opt=read(E/'OPTIMIZATION_FIDELITY_DIAGNOSTIC.json');assert agg['manual_review_sha256']==sha(E/'MANUAL_IDENTITY_REVIEW_PRIVATE.json')
labels=manual['rows_PRIVATE'];assert len(labels)==20 and len({r['id_PRIVATE'] for r in labels})==20;assert manual['all20_actual_images_viewed'] and manual['human_visual_assessment_not_blinded_to_GT_overlay_or_spatial_status'] and manual['normal_or_GT_predictions_or_G_not_changed_from_visuals']
counts=collections.Counter(r['label'] for r in labels);assert dict(counts)==manual['label_counts'];chosen={r['id_PRIVATE'] for r in labels if r['label']=='same_visible_target_plausible'};assert len(chosen)==agg['subset_total']==12
branches={'normal':'scored_normal160_attempt001','GT_controlled_NOT_TTA':'scored_gt160_attempt001','pseudo_bridge_postscore_exploratory':'scored_bridge160_attempt001'};data={}
for branch,directory in branches.items():
 root=E/directory;seal=read(root/'SEALED.json')
 for rel,h in seal['files'].items():assert sha(root/rel)==h
 data[branch]={r['id']:r for r in read(root/'RESULTS_PRIVATE.json')['rows_PRIVATE']}
assert all(chosen<=set(v) for v in data.values());maxmanual=0.;manualcells=0
for s in agg['summary']:
 pp=[r for sid,r in data[s['branch']].items() if sid in chosen and r['platform']==s['platform']];assert s['n']==len(pp) and s['scene_clusters']==len({r['scene'] for r in pp});a=s['arm'];expected={'mean_iou':mean([r['arms'][a]['iou'] for r in pp]),'center_L2_m_mean':mean([r['arms'][a]['coordinate_errors']['center_L2_m'] for r in pp]),'size_L2_m_mean':mean([r['arms'][a]['coordinate_errors']['size_L2_m'] for r in pp])}
 for k,v in expected.items():near(s[k],v);maxmanual=max(maxmanual,abs(s[k]-v));manualcells+=1
# Optimization fidelity derives only from saved normal/bridge arrays and traces.
normal=P/'normal160_attempt002';bridge=P/'pseudo_bridge160_attempt001';nt={r['id']:r for r in read(normal/'trace.json')};gttrace={r['id']:r for r in read(P/'gt_control160_attempt001/trace.json')};samples=read(E/'diagnostic160_gpu_inputs.json');stats={};bound_steps=0;max_physical_bound_excess=0.;max_analytic_bound_excess=0.;max_roundoff_step_error=0.
def verify_steps(t,weights,lr,kcenter,ksize):
 global bound_steps,max_physical_bound_excess,max_analytic_bound_excess,max_roundoff_step_error
 weights=np.asarray(weights,float);sw=weights.sum();Kmax=max(kcenter,ksize);assert lr==.25*.1/Kmax;K=np.r_[np.full(3,kcenter),np.full(3,ksize)]
 for z in t:
  g=np.array(z['weighted_SmoothL1_box_gradient6'],float);near(np.clip(g,-weights/sw,weights/sw),g,tol=5e-8);analytic=-lr*K*g;bound=.25*.1*(K/Kmax)*weights/sw;actual=np.array(z['selected_box_after'])-np.array(z['selected_box_before']);max_analytic_bound_excess=max(max_analytic_bound_excess,float(np.maximum(abs(analytic)-bound,0).max()));max_physical_bound_excess=max(max_physical_bound_excess,float(np.maximum(abs(actual)-bound,0).max()));max_roundoff_step_error=max(max_roundoff_step_error,float(abs(actual-analytic).max()));bound_steps+=1
for sample in samples:
 sid=sample['id'];platform=sample['platform']
 with np.load(normal/(sid+'.npz'),allow_pickle=False) as z:n={k:z[k].copy() for k in z.files}
 with np.load(bridge/(sid+'.npz'),allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
 source=n['arm_boxes_raw6'][0,:3].astype(float);names=n['arm_names'].tolist()
 for mi,mode in enumerate(('rgb_fulltext','pc_only')):
  key=platform+'|'+mode;v=stats.setdefault(key,dict(K=[],proj=[],normal=[],bridge=[],target=[]));t=nt[sid]['G_modes'][mode];cn=t['feature_norm2_plus1']['center'];sn=t['feature_norm2_plus1']['size'];v['K'].append(cn/max(cn,sn));target=n['arm_boxes_raw6'][names.index(mode+'__direct_G'),:3].astype(float);p=n['arm_boxes_raw6'][names.index(mode+'__param3'),:3].astype(float);bf=b['arm_boxes_raw6'][mi+1,:3].astype(float);d=target-source;dn=np.linalg.norm(d)
  if dn>0:v['proj'].append(float((p-source)@d/(d@d)));v['normal'].append(float(np.linalg.norm(p-target)/dn));v['bridge'].append(float(np.linalg.norm(bf-target)/dn));v['target'].append(dn)
  for step in ('1','3'):verify_steps(t['steps'].get(step,{}).get('trace',[]),n['pseudo_reliability6'][mi],t['actual_lr'],cn,sn)
 for group,w in [('C',[1,1,1,0,0,0]),('S',[0,0,0,1,1,1]),('CS',[1]*6)]:
  t=gttrace[sid]['groups'][group]
  for k in ('SGD1','SGD3'):
   norm=t[k]['feature_norms'];cn=norm['center_squared_feature_norm']+1.;sn=norm['size_squared_feature_norm']+1.;verify_steps(t[k]['trace'],w,t['actual_SGD_lr'],cn,sn)
optcells=0;maxopt=0.;maxbridge_ratio_difference=0.
for s in opt['summary']:
 v=stats[s['platform']+'|'+s['mode']];assert s['all_n']==len(v['K']) and s['nonzero_target_n']==len(v['target'])
 for field,key in [('center_self_kernel_over_shared_max_kernel','K'),('projection_fraction','proj'),('normal3_relative_residual','normal'),('bridge_relative_residual','bridge'),('target_delta_L2_m','target')]:
  x=v[key]
  if 'n' in s[field]:assert s[field]['n']==len(x)
  expected={'min':min(x),'median':float(np.median(x)),'max':max(x),'mean':float(np.mean(x))}
  for k,value in expected.items():
   if k not in s[field]:continue
   diff=abs(value-s[field][k]);optcells+=1
   if field=='bridge_relative_residual':near(s[field][k],value,tol=2e-11);maxbridge_ratio_difference=max(maxbridge_ratio_difference,diff)
   else:near(s[field][k],value);maxopt=max(maxopt,diff)
assert max_analytic_bound_excess<1e-9 and max_roundoff_step_error<=5e-5
out={'state':'independent_manual_subset_and_optimizer_fidelity_supplement_pass','auditor_sha256':sha(Path(__file__)),'manual_review_sha256':sha(E/'MANUAL_IDENTITY_REVIEW_PRIVATE.json'),'manual_aggregate_sha256':sha(E/'MANUAL_SUBSET_ERROR_SUMMARY.json'),'optimization_diagnostic_sha256':sha(E/'OPTIMIZATION_FIDELITY_DIAGNOSTIC.json'),'human_labels_not_reassessed_or_changed':True,'human_review_nonblind_GT_overlay_status_explicit':True,'manual_label_counts':dict(counts),'manual_same_target_n':len(chosen),'manual_same_target_platform_counts':dict(collections.Counter(r['platform'] for r in labels if r['id_PRIVATE'] in chosen)),'manual_same_target_but_not_spatial_condition_n':sum(r['id_PRIVATE'] in chosen and not r['annotation_spatial_condition'] for r in labels),'manual_aggregate_numeric_cells':manualcells,'max_manual_aggregate_error':maxmanual,'optimization_numeric_cells':optcells,'max_nonbridge_optimization_stat_error':maxopt,'max_bridge_relative_stat_rounding_difference':maxbridge_ratio_difference,'box_coordinate_step_bound_check':{'b_j_means_selected_box_coordinate_not_network_bias_parameter':True,'formula_real_arithmetic':'abs(delta_box_j)<=alpha*beta*(Khead/Kmax)*w_j/sum(w)','derivation':'SmoothL1 normalized gradient=clip(error/beta,-1,1)*w/sum(w); fixed-feature final-linear SGD induces delta_box=-lr*Khead*g; lr=alpha*beta/Kmax.','steps_checked':bound_steps,'max_analytic_bound_excess_m':max_analytic_bound_excess,'max_observed_FP32_bound_excess_m':max_physical_bound_excess,'max_observed_step_vs_analytic_error_m':max_roundoff_step_error,'floating_point_note':'Observed FP32 step can exceed exact real-arithmetic bound by rounding; actual observed error retained, not silently claimed strict zero.','equal_C3_or_S3_weights_headratio1_upper_mm':1000*.25*.1/3,'equal_CS6_weights_headratio1_upper_mm':1000*.25*.1/6,'quadratic_regime_contraction':'1-alpha*(Khead/Kmax)*w_j/sum(w)','unsupervised_size_kernel_can_reduce_center_step':True},'scope':['Root humanlabels accepted as supplied after sealed predictions, no independent image adjudication; nonblind and stratified20 not random population.','Manual12 same-visible-target plausibility subset, not semantic3D identity groundtruth or a new heldout set.','Independent reductions use sealed score rows; optimization audit uses normal/bridge/GT trace arrays, no new IoU, metadata, GPU or model.','L2+perheadstep exactfit changes two optimization components together; cannot attribute solely to learning rate.','Manualsubset does not replace all160 primary or spatial13/47 conditional diagnostics.']};f=R/'research/round9/evaluation/MANUAL_OPTIMIZER_SUPPLEMENT_REVIEW.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
