"""Independent R13 sealed noGT mechanism fidelity/counts plus taskgate readback.
Reads sealed calls/selected boxes/scalars only; no labels, sensors or taskIoU API.
"""
from pathlib import Path
import collections,hashlib,json,math
import numpy as np
B=Path(__file__).resolve().parents[3];R=B/'research/round13';V=R/'evaluation';E=R/'experiment';M=R/'mechanism/actual116_attempt001';A=B/'crossplatform/round13/association116_attempt001';G=B/'crossplatform/round13/geometry116_attempt001';ENG=B/'crossplatform/round13/association_preflight2_attempt001'
AARMS=('source','full_order0_raw','full_two_numbering_consensus','noun_two_numbering_consensus','deranged_two_numbering_consensus');BARMS=('source','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality');CONDS=('full','noun','deranged')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checks=0;maxdiff=0.;failures=[]
def compare(x,y,path):
 global checks,maxdiff
 checks+=1
 if isinstance(y,dict):
  if not isinstance(x,dict) or set(x)!=set(y):failures.append(path+' keys');return
  for k,v in y.items():compare(x[k],v,path+'/'+k)
 elif isinstance(y,list):
  if not isinstance(x,list) or len(x)!=len(y):failures.append(path+' len');return
  for i,v in enumerate(y):compare(x[i],v,path+'/'+str(i))
 elif y is None or isinstance(y,(str,bool,int)):
  if x!=y:failures.append(path+' exact')
 elif not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x-y)>1e-9:failures.append(path+' numeric')
 else:maxdiff=max(maxdiff,abs(x-y))
def stat(xs):
 vals=[float(v) for v in xs if v is not None]
 return {'defined_n':len(vals),'missing_n':len(xs)-len(vals),'sum':math.fsum(vals),'mean':math.fsum(vals)/len(vals) if vals else None,'median':float(np.median(vals)) if vals else None,'min':min(vals,default=None),'max':max(vals,default=None)}
def hist(xs):return dict(collections.Counter(map(str,xs)))
def npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if k in ('arm_boxes_raw6','arm_names','arm_query_index','source_q0','q_source','active_coordinate_mask','actual_changed_axes')}
def fromA(t,z):
 raw=z['arm_boxes_raw6'];q=z['arm_query_index'];q0=int(z['source_q0']);m=t['mapping'];conditions={};calls=[]
 for cond in CONDS:
  cc=t['calls'][cond];c=t['consensus'][cond]
  conditions[cond]={'same_mask_across_numberings':bool(c['two_numberings_same_physical_mask']),'selected_mask_index':c['selected_mask_index'],'selected_mask_index_by_order':c['selected_mask_index_by_order'],'readout_status':c['readout_status'],'q_selected':int(c['q_selected']),'query_changed':bool(c['query_changed']),'same_raw_answer_across_orders':cc[0]['raw_answer_PRIVATE']==cc[1]['raw_answer_PRIVATE']}
  for j,v in enumerate(cc):calls.append({'condition':cond,'numbering':j,'parse_status':v['parsed']['parse_status'],'explicit_UNKNOWN':v['parsed']['parse_status']=='explicit_UNKNOWN','raw_answer_sha256':hashlib.sha256(v['raw_answer_PRIVATE'].encode()).hexdigest(),'input_token_n':int(v['input_token_n']),'generated_token_n':int(v['generated_token_n']),'visual_token_n':sum(v['image_visualtoken_n']),'visual_tokens_per_image':list(map(int,v['image_visualtoken_n'])),'ended_with_eos':bool(v['ended_with_eos']),'hit_max_new_tokens':v['generated_token_n']>=v['max_new_tokens'],'truncation_policy':'no text truncation, cap violations fail process; image native bounded resizing','model_generate_seconds':float(v['seconds_model_generate'])})
 arms={name:{'query_changed':int(q[j])!=q0,'box_changed':bool((raw[j]!=raw[0]).any()),'center_changed':bool((raw[j,:3]!=raw[0,:3]).any()),'size_changed_by_original_query_selection_not_geometry':bool((raw[j,3:]!=raw[0,3:]).any()),'selected_query':int(q[j])} for j,name in enumerate(AARMS)}
 f,n,d=[conditions[k]['selected_mask_index'] for k in CONDS];raw0=t['full_order0_raw'];sem={}
 for key,x,y in [('full_vs_noun',f,n),('full_vs_deranged',f,d)]:sem[key]={'both_defined':x is not None and y is not None,'different_including_NA':x!=y,'different_when_both_defined':x is not None and y is not None and x!=y}
 sem['full_consensus_vs_order0']={'both_defined':f is not None and raw0['selected_mask_index'] is not None,'different_including_NA':f!=raw0['selected_mask_index']}
 return {'group_n':len(m['groups']),'all256_covered_once':sorted(q for g in m['groups'] for q in g['query_indices'])==list(range(256)),'group_size_histogram':hist([len(g['query_indices']) for g in m['groups']]),'source_group_size':len(m['groups'][m['source_group']]['query_indices']),'mask_n':len(m['masks']),'mapped_mask_n':sum(x['mapping_status']=='mapped_unique_geometry_hypothesis' for x in m['masks']),'mask_mapping_status_histogram':hist([x['mapping_status'] for x in m['masks']]),'mask_eligible_group_counts':[x['eligible_group_n'] for x in m['masks']],'conditions':conditions,'calls':calls,'arms':arms,'full_order0_readout_status':raw0['readout_status'],'full_order0_selected_mask_index':raw0['selected_mask_index'],'all_six_raw_answers_identical':len({x['raw_answer_sha256'] for x in calls})==1,'semantic_mask_comparisons':sem,'deranged_same_literal_caption_as_full':bool(t['deranged_same_caption_as_full']),'scope':'Mask/query response differences or identical raw strings are not semantic identity correctness. Size changes here are source-candidate selection, not geometric regression.'}
def fromB(t,z):
 s=t['support'];d=s['diagnostics'];g=t['inequality'];raw=z['arm_boxes_raw6'];initial=g.get('initial_terms');solver=g.get('solver',{});arms={}
 ps=None if initial is None else initial['point_containment']<=0;rs=None if initial is None else initial['RGB_positive_containment']<=1e-16
 vi='NA_gate_or_projection' if initial is None else 'neither_violated' if ps and rs else 'RGB_only_violated' if ps else 'point_only_violated' if rs else 'both_violated'
 for j,name in enumerate(BARMS):
  v=solver.get('direct_full' if j==1 else 'direct_no_rgb' if j==2 else '',{});changed=raw[j]!=raw[0]
  arms[name]={'changed':bool(changed.any()),'center_changed':bool(changed[:3].any()),'size_changed':bool(changed[3:].any()),'changed_axis6':changed.tolist(),'center_shift_L2_m':float(np.linalg.norm(raw[j,:3].astype(float)-raw[0,:3].astype(float))),'size_shift_L2_m':float(np.linalg.norm(raw[j,3:].astype(float)-raw[0,3:].astype(float))),'status':str(z['arm_status'][j]) if 'arm_status' in z else None,'solver_success':v.get('success'),'solver_iterations':v.get('iterations'),'finite_nonconverged':bool(v.get('finite_nonconverged_retained',False)),'half_has_no_solver':j==3,'half_derived_from_nonconverged_full':bool(j==3 and solver.get('direct_full',{}).get('finite_nonconverged_retained',False))}
 return {'unique_mask':bool(s['segmentation_valid']),'support_status':s['status'],'unique_support_n':d.get('selected_unique_model_XYZ',0),'min8_met':d.get('selected_unique_model_XYZ',0)>=8,'candidate_mask_n':len(d.get('candidates',[])),'source_roi_unique_visible':d.get('visible_in_source_ROI'),'shared_gate_passed':bool(g['gate_passed']),'gate_reason':g['gate_reason'],'initial_terms_defined':initial is not None,'point_already_satisfied':ps,'RGB_already_satisfied':rs,'initial_violation_class':vi,'initial_point_violation_fraction':initial.get('point_violation_fraction') if initial else None,'initial_RGB_boundary_violation_fraction':initial.get('RGB_boundary_violation_fraction') if initial else None,'all_full_off_search_domains_equal':bool(np.array_equal(z['active_coordinate_mask'][1],z['active_coordinate_mask'][2])),'full_off_final_boxes_equal':bool(np.array_equal(raw[1],raw[2])),'arms':arms,'solver_terms':{k:{n:v.get(n) for n in ('initial_terms','final_terms','initial_total','final_total','numerical_invalid','status')} for k,v in solver.items()},'scope':'Containment/response mechanism only. Active coordinates are search domain, not observed amodal parameters.'}

def aggregate(rr):
 aa=[r['A'] for r in rr];bb=[r['B'] for r in rr];n=len(rr)
 b={'case_n':n,'unique_independent_mask_n':sum(x['unique_mask'] for x in bb),'min8_support_n':sum(x['min8_met'] for x in bb),'shared_gate_passed_n':sum(x['shared_gate_passed'] for x in bb),'support_status_histogram':hist([x['support_status'] for x in bb]),'gate_reason_histogram':hist([x['gate_reason'] for x in bb]),'unique_support_count':stat([x['unique_support_n'] for x in bb]),'initial_terms_defined_n':sum(x['initial_terms_defined'] for x in bb),'point_already_satisfied_n':sum(x['point_already_satisfied'] is True for x in bb),'RGB_already_satisfied_n':sum(x['RGB_already_satisfied'] is True for x in bb),'initial_violation_class_histogram':hist([x['initial_violation_class'] for x in bb]),'full_off_boxes_identical_n':sum(x['full_off_final_boxes_equal'] for x in bb),'full_off_boxes_different_n':sum(not x['full_off_final_boxes_equal'] for x in bb),'arms':{}}
 for arm in BARMS:
  v=[x['arms'][arm] for x in bb];b['arms'][arm]={'denominator':n,'changed_n':sum(x['changed'] for x in v),'center_changed_n':sum(x['center_changed'] for x in v),'size_changed_n':sum(x['size_changed'] for x in v),'changed_axis6_counts':[sum(x['changed_axis6'][j] for x in v) for j in range(6)],'center_shift_L2_m':stat([x['center_shift_L2_m'] for x in v]),'size_shift_L2_m':stat([x['size_shift_L2_m'] for x in v]),'finite_nonconverged_n':sum(x['finite_nonconverged'] for x in v),'half_derived_from_nonconverged_full_n':sum(x['half_derived_from_nonconverged_full'] for x in v),'status_histogram':hist([x['status'] for x in v])}
 a={'case_n':n,'dedup_group_count':stat([x['group_n'] for x in aa]),'dedup_covers_all256_once_n':sum(x['all256_covered_once'] for x in aa),'source_group_size':stat([x['source_group_size'] for x in aa]),'masks_total':sum(x['mask_n'] for x in aa),'mapped_masks_total':sum(x['mapped_mask_n'] for x in aa),'mask_mapping_status_histogram':dict(sum((collections.Counter(x['mask_mapping_status_histogram']) for x in aa),collections.Counter())),'all_six_raw_answers_identical_n':sum(x['all_six_raw_answers_identical'] for x in aa),'deranged_same_literal_caption_as_full_n':sum(x['deranged_same_literal_caption_as_full'] for x in aa),'conditions':{},'arms':{},'semantic_mask_comparisons':{}}
 for cond in CONDS:
  v=[x['conditions'][cond] for x in aa];calls=[c for x in aa for c in x['calls'] if c['condition']==cond];a['conditions'][cond]={'case_denominator':n,'call_denominator':len(calls),'strict_parse_status_histogram':hist([x['parse_status'] for x in calls]),'explicit_UNKNOWN_n':sum(x['explicit_UNKNOWN'] for x in calls),'same_physical_mask_across_numberings_n':sum(x['same_mask_across_numberings'] for x in v),'same_raw_answer_across_orders_n':sum(x['same_raw_answer_across_orders'] for x in v),'readout_status_histogram':hist([x['readout_status'] for x in v])}
 for arm in AARMS:a['arms'][arm]={'denominator':n,**{k:sum(x['arms'][arm][k] for x in aa) for k in ('query_changed','box_changed','center_changed','size_changed_by_original_query_selection_not_geometry')}}
 for key in ('full_vs_noun','full_vs_deranged','full_consensus_vs_order0'):a['semantic_mask_comparisons'][key]={'case_denominator':n,**{k:sum(x['semantic_mask_comparisons'][key][k] for x in aa) for k in aa[0]['semantic_mask_comparisons'][key]}}
 return b,a

def callcost(aa):
 cc=[c for r in aa for c in r['calls']]
 return {'n_cases':len(aa),'n_calls':len(cc),'model_generate_seconds':stat([c['model_generate_seconds'] for c in cc]),'input_tokens':stat([c['input_token_n'] for c in cc]),'generated_tokens':stat([c['generated_token_n'] for c in cc]),'visual_tokens_both_images':stat([c['visual_token_n'] for c in cc]),'ended_with_eos_n':sum(c['ended_with_eos'] for c in cc),'hit_max_new_tokens_n':sum(c['hit_max_new_tokens'] for c in cc),'hit_cap_without_eos_n':sum(c['hit_max_new_tokens'] and not c['ended_with_eos'] for c in cc),'text_truncation_enabled':False,'image_resizing_is_not_text_truncation':True}

ms=read(M/'SEALED.json');assert sha(M/'SEALED.json')=='220db2eaffef89b5bd88d39f29f737a810a27176996c37b60e766c2306106138' and ms['state']=='sealed_R13_noGT_mechanism_reduction'
assert ms['release_sha256']==sha(E/'MECHANISM_READ_RELEASE.json');release=read(E/'MECHANISM_READ_RELEASE.json');assert release['mechanism_reduction_permitted'] and release['GT_read_permitted'] is False
assert read(E/'PROGRESSION_PREPLAN_LOCK.json')['card_sha256']==sha(E/'EXPERIMENT_CARD.md')
for rel,h in ms['files'].items():assert sha(M/rel)==h
assert ms['association_seal_sha256']==sha(A/'SEALED.json') and ms['geometry_seal_sha256']==sha(G/'SEALED.json') and ms['engineering_association_seal_sha256']==sha(ENG/'SEALED.json')
assert sha(R/'mechanism/reduce_unlabelled.py')==ms['reducer_sha256'];summary=read(M/'SUMMARY.json');private=read(M/'CASE_MECHANISM_PRIVATE.json');bt={t['id']:t for t in read(G/'TRACE_PRIVATE.json')};derived=[]
for row in private:
 at=read(A/'cases'/(row['case_alias']+'.json'));ap=npz(A/'cases'/(row['case_alias']+'.npz'));bp=npz(G/(row['id']+'.npz'))
 with np.load(G/(row['id']+'.npz'),allow_pickle=False) as z:bp['arm_status']=z['arm_status'].copy()
 assert at['id_PRIVATE']==row['id'] and bt[row['id']]['case_alias']==row['case_alias']
 freshA=fromA(at,ap);freshB=fromB(bt[row['id']],bp);compare(row['A'],freshA,'Araw');compare(row['B'],freshB,'Braw');derived.append(dict(row,A=freshA,B=freshB))
assert len(derived)==len({r['id'] for r in derived})==116
compare(summary['case_role_counts'],dict(collections.Counter(r['case_role'] for r in derived)),'rolecounts')
assert {(s['case_role'],s['platform']) for s in summary['stratified']}=={(r,p) for r in ('new_content108','regression8') for p in ('drone','quad','all_platforms')}
for s in summary['stratified']:
 rr=[r for r in derived if r['case_role']==s['case_role'] and (s['platform']=='all_platforms' or r['platform']==s['platform'])];bb,aa=aggregate(rr);compare(s,{'case_role':s['case_role'],'platform':s['platform'],'n':len(rr),'B_geometry':bb,'A_association':aa},'mechanism_summary')
compare(summary['formal_call_cost'],callcost([r['A'] for r in derived]),'formal_cost');eng=[]
for path in sorted((ENG/'cases').glob('*.json')):eng.append(fromA(read(path),npz(path.with_suffix('.npz'))))
compare(summary['engineering_call_cost_separate'],callcost(eng),'engineering_cost');compare(summary['formal_actual_calls'],696,'formal696');compare(summary['engineering_actual_calls'],12,'eng12');compare(summary['total_actual_calls_both_scopes'],708,'total708')
for key,value in [('formal_process_seconds',read(A/'SEALED.json')['seconds_this_process']),('engineering_process_seconds',read(ENG/'SEALED.json')['seconds_this_process']),('B_process_seconds',read(G/'SEALED.json')['seconds']),('formal_peak_allocated_bytes',read(A/'SEALED.json')['peak_allocated_bytes'])]:compare(summary[key],value,key)
sc=read(E/'scored116_attempt001/SCALAR_SUMMARY.json');idx={(x['case_role'],x['platform'],x['arm']):x for x in sc['summary']};ci={(x['case_role'],x['platform'],x['arm'],x['reference']):x for x in sc['paired_contrasts']};gates={};numbers={}
for platform in ('drone','quad'):
 get=lambda arm:idx['new_content108',platform,arm]
 gain=lambda arm,ref,key: get(arm)[key]-get(ref)[key]
 af=get('A_full_consensus');bf=get('B_inequality_full');src=get('source');noun=get('A_noun_consensus');half=get('B_inequality_half')
 ac={'nonzero_box_changes':af['actual_changed_n']>0,'IoU_over_source_micro':af['mean_iou']>src['mean_iou'],'IoU_over_source_scene_equal':af['equal_scene_mean_iou']>src['equal_scene_mean_iou'],'IoU_over_noun_micro':af['mean_iou']>noun['mean_iou'],'IoU_over_noun_scene_equal':af['equal_scene_mean_iou']>noun['equal_scene_mean_iou']};ac['all_prerequisites_pass']=all(ac.values())
 bc={'nonzero_box_changes':bf['actual_changed_n']>0,'IoU_over_source_micro':bf['mean_iou']>src['mean_iou'],'IoU_over_source_scene_equal':bf['equal_scene_mean_iou']>src['equal_scene_mean_iou'],'center_error_reduction_micro':bf['center_L2_m_mean']<src['center_L2_m_mean'],'center_error_reduction_scene_equal':bf['equal_scene_center_L2_m_mean']<src['equal_scene_center_L2_m_mean'],'size_error_reduction_micro':bf['size_L2_m_mean']<src['size_L2_m_mean'],'size_error_reduction_scene_equal':bf['equal_scene_size_L2_m_mean']<src['equal_scene_size_L2_m_mean'],'full_over_half_micro':bf['mean_iou']>half['mean_iou'],'full_over_half_scene_equal':bf['equal_scene_mean_iou']>half['equal_scene_mean_iou'],'positive_net_task_signed_sum':ci['new_content108',platform,'B_inequality_full','source']['IoU_gain']['sum']>0};bc['all_prerequisites_pass']=all(bc.values());gates[platform]={'A':ac,'B':bc}
 numbers[platform]={arm:{'n':get(arm)['n'],'mIoU_percent':get(arm)['mean_iou']*100,'equal_scene_mIoU_percent':get(arm)['equal_scene_mean_iou']*100,'center_L2_m':get(arm)['center_L2_m_mean'],'size_L2_m':get(arm)['size_L2_m_mean'],'changed':get(arm)['actual_changed_n']} for arm in ('source','A_full_consensus','A_noun_consensus','A_full_order0','B_inequality_full','B_inequality_half','B_inequality_noRGB')}
Apass=all(gates[p]['A']['all_prerequisites_pass'] for p in gates);Bpass=all(gates[p]['B']['all_prerequisites_pass'] for p in gates)
result={'state':'PASS_R13_INDEPENDENT_MECHANISM_AND_PRELOCKED_GATE_REVIEW' if not failures else 'FAIL_R13_MECHANISM_AUDIT','auditor_sha256':sha(Path(__file__)),'mechanism_seal_sha256':sha(M/'SEALED.json'),'mechanism_summary_sha256':sha(M/'SUMMARY.json'),'association_seal_sha256':sha(A/'SEALED.json'),'geometry_seal_sha256':sha(G/'SEALED.json'),'scored_seal_sha256':sha(E/'scored116_attempt001/SEALED.json'),'scoring_audit_sha256':sha(V/'SCORING_INDEPENDENT_AUDIT.json'),'preplan_sha256':sha(E/'PROGRESSION_PREPLAN_LOCK.json'),'card_sha256':sha(E/'EXPERIMENT_CARD.md'),'checks':checks,'max_numeric_difference':maxdiff,'by_platform_prelocked_conditions':gates,'A_both_platforms_pass':Apass,'B_both_platforms_pass':Bpass,'combination_or_neural_TTA_progression_pass':Apass and Bpass,'key_numbers':numbers,'interpretation':['A improves samplemicroIoU inbothplatforms but Quadsceneequaldecreases; no passingcrossplatform A prereq underconsistentweights.','B Quad taskmicroandsceneequal improve, Drone taskandcoordinateerror worsen; B sourceprior expansion sizeerror increases bothplatforms.','No A+B combination or neuralTTA oncurrentfeedback; not a routewide impossibility.','All696 validIDs/EOS and two-number consensus are response/format evidence, not literalidentity truth; semantic_mask_comparisons include choices withoutvalid3Dmapping.','A sizechanges come fromoriginalproposalchoice, B sizechanges are inequality/prior results; sixactivecoordinates are searchdomain not sixidentifiedparameters.','R7/R8 full5912historical exposure; new108 onlyrelative recent914groups exclusion; few1/3scenes and old8outputselection remain exploratory.'],'no_new_GT_index_payload_IoU_sensor_media_model_GPU_optimization_or_prediction_changes':True,'failures':failures}
out=V/'MECHANISM_GATE_INDEPENDENT_REVIEW.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':checks,'maxdiff':maxdiff,'Apass':Apass,'Bpass':Bpass,'failures':failures[:10],'sha256':sha(out)}))
if failures:raise SystemExit(1)
