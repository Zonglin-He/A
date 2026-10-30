"""Independent sealed mechanism aggregates + resource gate, no freshGT or IoU."""
from pathlib import Path
import json,collections,hashlib,math
import numpy as np
R=Path(__file__).resolve().parents[3];E=R/'research/round10/experiment';G=R/'crossplatform/round10/geometry160_attempt001';M=R/'research/round10/mechanism/actual160_attempt001';P=R/'research/round10/evaluation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
ARMS=('source','full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation')
maxdiff=0.;cells=0

def compare(expected,actual):
 global maxdiff,cells
 if isinstance(expected,dict):
  assert set(expected)<=set(actual),(set(expected)-set(actual))
  for k,v in expected.items():compare(v,actual[k])
 elif isinstance(expected,(list,tuple)):
  assert len(expected)==len(actual)
  for x,y in zip(expected,actual):compare(x,y)
 elif expected is None or isinstance(expected,(str,bool)):
  assert expected==actual,(expected,actual);cells+=1
 else:
  assert np.isclose(expected,actual,rtol=1e-14,atol=1e-12),(expected,actual);maxdiff=max(maxdiff,abs(float(expected)-float(actual)));cells+=1

def desc(x):
 v=[float(z) for z in x if z is not None]
 return {'n':len(v),'min':min(v,default=None),'median':float(np.median(v)) if v else None,'mean':math.fsum(v)/len(v) if v else None,'max':max(v,default=None),'sum':math.fsum(v)}

ms=read(M/'SEALED.json');gs=read(G/'SEALED.json');score=E/'scored160_attempt001';ss=read(score/'SEALED.json')
for root,s in [(M,ms),(G,gs),(score,ss)]:
 for rel,h in s['files'].items():assert sha(root/rel)==h
summary=read(M/'SUMMARY.json');assert summary['binding']['geometry_seal_sha256']==sha(G/'SEALED.json') and summary['binding']['source_seal_sha256']==gs['source_seal_sha256']
assert summary['binding']['program_sha256']==gs['program_sha256'] and summary['binding']['reducer_sha256']==ms['reducer_sha256']==sha(R/'research/round10/mechanism/reduce_mechanism.py')
traces=read(G/'trace.json');records=[];provenance=collections.defaultdict(collections.Counter)
for t in traces:
 with np.load(G/(t['id']+'.npz'),allow_pickle=False) as z:b=z['arm_boxes_raw6'].astype(float);mask=z['acceptance_mask6'].astype(bool);pre=z['pre_constraint_geometry6'];status=z['arm_status'].tolist();prov=z['provenance'].tolist();changed=z['actual_changed_axes']
 d=t['geometry'].get('diagnostics',{});faces=d.get('source_face_evidence_RGB',{});rgb=t['RGB'];sol=d.get('solver',{})
 rec={'platform':t['platform'],'segmentation_valid':rgb.get('state')=='point_seeded_grabcut_component','segmentation_reasons':rgb.get('reason','unknown'),'global_gate_reasons':d.get('global_gate_reason',d.get('reason','unknown')),'accepted':mask[1],'basicpaired':[all(faces.get(str(j)+':'+str(sign),{}).get('accepted_plane',False) for sign in (-1,1)) for j in range(3)],'full_identical':np.array_equal(b[1],b[2]),'sharedpre':np.array_equal(pre[1],pre[2]),'status':status,'changed':changed,'center_delta':np.linalg.norm(b[:,:3]-b[0,:3],axis=1),'size_delta':np.linalg.norm(b[:,3:]-b[0,3:],axis=1),'solver':sol,'model_input_rows':t['model_input_rows'],'unique_raw_indices':t['unique_raw_indices'],'unique_XYZ':t['geometry_unique_XYZ_rows'],'visible_unique_count':d.get('visible_unique_count'),'geometry_support_count':d.get('geometric_support_count'),'RGB_support_count':d.get('RGB_support_count'),'RGB_over_geometric_support':d.get('RGB_support_count')/d['geometric_support_count'] if d.get('geometric_support_count') else None,'RGB_boundary_sample_count':d.get('RGB_boundary_sample_count'),'full_noRGB_maxabs_coordinate_difference_m':float(abs(b[1]-b[2]).max()),'ablation_full_maxabs_coordinate_difference_m':float(abs(b[3]-b[1]).max())};records.append(rec)
 for j,name in enumerate(ARMS):
  for axis,label in enumerate(prov[j]):provenance[t['platform']+'|'+name+'|axis'+str(axis)][label]+=1

def aggregate(rr):
 out={'n':len(rr),'segmentation_valid':sum(r['segmentation_valid'] for r in rr),'segmentation_reasons':dict(collections.Counter(r['segmentation_reasons'] for r in rr)),'global_gate_reasons':dict(collections.Counter(r['global_gate_reasons'] for r in rr)),'any_accepted':sum(r['accepted'].any() for r in rr),'accepted_axis_counts6':np.sum([r['accepted'] for r in rr],axis=0).tolist(),'two_qualified_opposing_plane_axis_counts':np.sum([r['basicpaired'] for r in rr],axis=0).tolist(),'strict_size_pair_axis_counts':np.sum([r['accepted'][3:] for r in rr],axis=0).tolist(),'any_strict_size_pair':sum(r['accepted'][3:].any() for r in rr),'full_noRGB_identical_float32':sum(r['full_identical'] for r in rr),'full_noRGB_different_float32':sum(not r['full_identical'] for r in rr),'full_noRGB_shared_preconstraint_exact_all':all(r['sharedpre'] for r in rr)}
 for key in ('model_input_rows','unique_raw_indices','unique_XYZ','visible_unique_count','geometry_support_count','RGB_support_count','RGB_over_geometric_support','RGB_boundary_sample_count','full_noRGB_maxabs_coordinate_difference_m','ablation_full_maxabs_coordinate_difference_m'):out[key]=desc([r[key] for r in rr])
 out['arms']={}
 for j,name in enumerate(ARMS):
  a={'actually_changed_samples':sum(r['changed'][j].any() for r in rr),'changed_axis_counts6':np.sum([r['changed'][j] for r in rr],axis=0).tolist(),'center_increment_l2_m':desc([r['center_delta'][j] for r in rr]),'size_increment_l2_m':desc([r['size_delta'][j] for r in rr]),'statuses':dict(collections.Counter(r['status'][j] for r in rr))}
  if j:
   sol=[r['solver'][name] for r in rr if name in r['solver']];a.update({'solver_executed':len(sol),'solver_success':sum(s['success'] for s in sol),'numerical_invalid':sum(s['numerical_invalid'] for s in sol),'finite_nonconverged':sum(s['status']=='finite_nonconverged_bounded_result' for s in sol),'iterations':desc([s['iterations'] for s in sol]),'terms':{term:{stage:desc([s[stage+'_terms'][term] for s in sol]) for stage in ('initial','final')} for term in ('prior','observed_surface','RGB_one_sided_containment','total')},'RGB_term_positive_at_source':sum(s['initial_terms']['RGB_one_sided_containment']>0 for s in sol),'RGB_term_positive_after':sum(s['final_terms']['RGB_one_sided_containment']>0 for s in sol),'RGB_term_recorded_as_zero_by_design_when_disabled':name=='no_rgb_coordinate'})
  out['arms'][name]=a
 return out
compare(aggregate(records),summary['all'])
for platform in ('drone','quad'):compare(aggregate([r for r in records if r['platform']==platform]),summary['by_platform'][platform])
# Complete remaining scalar columns/tails beyond frozen main reduction audit.
result=read(score/'RESULTS_PRIVATE.json');rows=result['rows_PRIVATE'];assert result['coordinate_provenance_counts']=={k:dict(v) for k,v in provenance.items()}
for s in result['summary']:
 pp=[r for r in rows if r['platform']==s['platform']]
 from audit_geometry_scored import eligible
 pp=[r for r in pp if eligible(r,s['population'],s['arm'])];v=[r['arms'][s['arm']] for r in pp]
 compare({'own_subset_not_matched_causal_comparison':s['population']=='own_changed_descriptive','flagged_failed_n':sum(x['status'].startswith('failed') for x in v),'finite_nonconverged_n':sum(x['status']=='finite_nonconverged_bounded_result' for x in v),'numerical_invalid_source_fallback_n':sum(x['status']=='numerical_invalid_source_fallback' for x in v),'explicit_source_fallback_n':sum(x['status']=='fallback' or 'source_fallback' in x['status'] or x['status'].startswith('camera_contract_invalid:') or x['status']=='no_accepted_observed_axis' for x in v)},s)
for c in result['paired_contrasts']:
 pp=[r for r in rows if r['platform']==c['platform'] and eligible(r,c['population'])];arm=c['arm'];ref=c['reference']
 for key,vals,tol in [('IoU_gain',[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in pp],1e-8),('center_error_reduction_m',[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in pp],1e-6),('size_error_reduction_m',[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in pp],1e-6)]:compare({'min':min(vals,default=None),'max':max(vals,default=None),'tolerance':tol},c[key])
 for key in ('IoU_scene','center_scene','size_scene'):compare({'bootstrap_seed':1001,'bootstrap_draws':2000,'few_scene_not_significance_or_population_proof':True},c[key])
 compare({'all_positive_direction_means_improvement':True,'pointmask_common_case_does_not_mean_same_support':c['comparison']=='RGB_gated_pointmask_increment'},c)
# Gate mirrors prelocked engineering resource criterion, not adoption/significance.
gate=[]
for platform in ('drone','quad'):
 get=lambda name:next(x for x in result['paired_contrasts'] if x['platform']==platform and x['population']=='all' and x['comparison']==name)
 src=get('full_vs_source');rgb=get('RGB_coordinate_increment');task=src['IoU_gain']['mean']>0 and src['IoU_gain']['gain_sum_exceeds_abs_harm_sum'];coordinate=any(src[k]['mean']>0 and src[k]['gain_sum_exceeds_abs_harm_sum'] for k in ['center_error_reduction_m','size_error_reduction_m']);increment=rgb['IoU_gain']['mean']>0 and rgb['IoU_gain']['gain_sum_exceeds_abs_harm_sum']
 gate.append({'platform':platform,'full_vs_source_task_gain_pass':bool(task),'full_vs_source_center_or_size_error_pass':bool(coordinate),'matched_RGB_coordinate_increment_pass':bool(increment),'platform_gate_pass':bool(task and coordinate and increment),'source_IoU_gain':src['IoU_gain'],'center_error_reduction_m':src['center_error_reduction_m'],'RGB_coordinate_IoU_gain':rgb['IoU_gain']})
passed=all(x['platform_gate_pass'] for x in gate);assert not passed
out={'state':'independent_R10_mechanism_aggregate_and_progression_gate_review_complete','auditor_sha256':sha(Path(__file__)),'main_scoring_audit_sha256':sha(P/'SCORING_INDEPENDENT_AUDIT.json'),'results_sha256':sha(score/'RESULTS_PRIVATE.json'),'scored_seal_sha256':sha(score/'SEALED.json'),'geometry_seal_sha256':sha(G/'SEALED.json'),'mechanism_seal_sha256':sha(M/'SEALED.json'),'mechanism_summary_sha256':sha(M/'SUMMARY.json'),'G_plan_sha256':sha(E/'GEOMETRY_PLAN_LOCK.json'),'n':160,'additional_scalar_fields_checked':cells,'max_scalar_float_difference':maxdiff,'all_mechanism_and_remaining_summary_signedtail_columns_reduced':True,'basic_opposing_plane_axes3_vs_strict_size_axes0_verified':True,'gate_by_platform':gate,'progression_gate_pass':bool(passed),'decision':'Stop current G branch; do not run conditional TTA. Preserve allnegative/zero/finite_nonconverged cases.','interpretation':['RGB coordinate constraint slightly reduces damage relative to removed-coordinate control, but fullG stillworse than source on bothplatforms.','RGB-gated pointmask ablation is slightly better than fullG taskIoU; do not claim RGBpointmask helps from these results.','Observed plane loss/totalobjective decline is not pseudo correctness. Full mean observed-surface loss9.50408→1.24324 while RGBcontainment component1.11413→1.34492 rises; separate components retained.','No strict opposing-plane sizequalification met; unchanged dimensions cannot establish correct sizes or failure of sizeTTA.','Single-plane center correction remains conditional on sourcehalfsize and sourceassignment; failure does notreject regressionadaptation generally.','Only2Drone/3Quad scenes and earlierVALresearch exposure; diagnostic stopgate, no significance/generalizationclaim.'],'no_fresh_GT_rawmetadata_newIoU_GPU_model_or_G_reexecution':True};p=P/'MECHANISM_GATE_INDEPENDENT_REVIEW.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='gate_by_platform'}))
