"""R13 finite final narrative/scalar/binding review; no GT or new geometry."""
from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parents[1];E=R/'experiment';V=R/'evaluation';D=R/'association_diagnostic'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
text=(E/'REPORT.md').read_text();checks=[]
def ck(name,value):
 checks.append({'check':name,'pass':bool(value)})
 assert value,name
ck('final_report_expected_sha',sha(E/'REPORT.md')=='b93b4b0ee75794b873c9ec7c2dd6f0fb3c100c36310619009c1c5489ec2e4aba')
ck('decision_expected_sha',sha(E/'DECISION.json')=='88213637d56aa44e332cfcaef0f400a0aa7ef324604e5f1f2abe0cfc020f416a')
pre=read(V/'REPORT_BODY_PRELIMINARY_REVIEW.json');body=re.sub(r'### 5\.1 .*?(?=## 6\.)','',text,flags=re.S);ck('all_main_body_outside_postscore_section_byteexact_to_preliminary',hashlib.sha256(body.encode()).hexdigest()==pre['observed_report_sha256'])
sc=read(E/'scored116_attempt001/SCALAR_SUMMARY.json');idx={(x['case_role'],x['platform'],x['arm']):x for x in sc['summary']}
labels={'冻结源模型':'source','A完整文字，两编号共识':'A_full_consensus','A完整文字，仅order0':'A_full_order0','A名词，两编号共识':'A_noun_consensus','A同场景错配描述，两编号共识':'A_deranged_consensus','B单侧不等式full，直接输出':'B_inequality_full','B同门控noRGB坐标项':'B_inequality_noRGB','B固定half，直接输出':'B_inequality_half','B旧可见面等式':'B_old_surface_equality'}
for label,arm in labels.items():
 line=next(x for x in text.splitlines() if x.startswith('| '+label+' |'));vals=[float(x.strip()) for x in line.split('|')[2:4]]
 for platform,v in zip(('drone','quad'),vals):ck('primary_table_'+arm+'_'+platform,abs(v-100*idx['new_content108',platform,arm]['mean_iou'])<=5.01e-7)
diag=read(D/'actual116_attempt001/SUMMARY.json');ds={(s['case_role'],s['platform'],s['stratum']):s for s in diag['summary']}
for label,cat in [('目标标注几何占优','target_spatially_dominant'),('其他已标注几何占优','annotated_other_spatially_dominant'),('未确认','unconfirmed_spatial_relation')]:
 line=next(x for x in text.splitlines() if x.startswith('| '+label+' |'));values=[x.strip() for x in line.split('|')[2:4]]
 for p,v in zip(('drone','quad'),values):
  counts=[ds['new_content108',p,'all']['readouts'][k]['spatial_category_counts'].get(cat,0) for k in ('q0','fullA')];ck('spatial_table_'+cat+'_'+p,v=='→'.join(map(str,counts)))
qc=ds['new_content108','quad','all']['predicted_COCO_category_not_GT']['A_full'];ck('COCO44_and5_predicted_notGT',qc['defined_n']==44 and qc['incompatible_n']==5)
ck('explicit_signed_target_rule','targetIoU>.25且target−other>.10' in text);ck('explicit_signed_other_rule','othermaxIoU>.25且other−target>.10' in text);ck('hypotheses_not_identity_verdict','这些结果提示至少两类待核解释' in text)
sr=read(V/'SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW.json');ck('spatial_review_PASS',sr['state']=='PASS_R13_POSTSCORE_SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW' and not sr['failures']);ck('348_target_bridge0',sr['target_primary_scalar_bridge_n']==348 and sr['target_primary_max_abs_difference']==0.)
rr=read(E/'ROOT_POSTSCORE_DIAGNOSTIC_REVIEW.json');ck('root_diag_complete_bound',rr['state']=='root_descriptive_association_mapping_diagnostic_review_complete_independently_audited' and rr['independent_review_sha256']==sha(V/'SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW.json'))
ck('root_diag_scientific_outputs_unchanged',all(rr[k] for k in ('primary_prediction_seal_unchanged','primary_scored_seal_unchanged','no_G_filter_prompt_threshold_metric_or_adoption_rule_changes','no_literal_identity_truth_claim')))
decision=read(E/'DECISION.json');gate=read(E/'PROGRESSION_GATE.json');indgate=read(V/'MECHANISM_GATE_INDEPENDENT_REVIEW.json');ck('decision_gate_sha',decision['progression_gate_sha256']==sha(E/'PROGRESSION_GATE.json'));ck('root_independent_gate_exact',gate['by_platform_prelocked_conditions']==indgate['by_platform_prelocked_conditions'] and gate['independent_gate_review_sha256']==sha(V/'MECHANISM_GATE_INDEPENDENT_REVIEW.json'))
ck('no_adoption_no_combination_noTTA',all(decision[k] is False for k in ('current_full_pipeline_adopted','current_G_crossplatform_adopted','new_TTA_adopted','neural_TTA_executed','A_B_composition_executed','regression_route_rejected','partial_observation_route_rejected')))
ck('A_B_gate_fail',not gate['A_both_platforms_pass'] and not gate['B_both_platforms_pass'] and not gate['combination_or_neural_TTA_progression_pass'])
ck('historical_full5912_exposure_explicit','R7/R8已经对全部5912' in text and '不是从未见过的输入、标签或场景' in text and decision['all5912_historical_R7_R8_model_taskscore_exposure_explicit'])
ck('positive_Quad_signal_preserved','39正/2负' in text and '三个场景均正' in text and decision['Quad_inequality_geometry_positive_exploratory_signal_retained'])
ck('Q_equal_scene_negative_not_hidden','−.247880pp' in text)
ck('raw_target9_other7_no_fakepadding_scope','425条raw7D' in text and '不补造尾字段' in text and '没有显式实例ID' in text)
ck('postscore_diagnostic_not_method','以下为看完封存正负例后新增的描述性分析' in text and '没有据此新增过滤器或重算主成绩' in text)
ck('no_all_A_harms_language_attribution','不能把全部A负增量统一归因于文字理解' in text)
gallery=read(E/'ROOT_GALLERY_REVIEW.json');ck('root_view_receipt8_plus2_and_awareness',len(gallery['viewed_files'])==8 and gallery['root_taskscore_aware'] and gallery['not_blind_or_identity_ground_truth_or_causal_proof'])
scientific_scope=[
 'Main9arm values, signedcounts, scenegains, candidatecoverage, regressioncases and timing reviewed in preliminary+independent scalar/mechanism audits; currentnon-diagnostic body byteexact.',
 'Afull micropositive bothplatforms but Quadsceneequalnegative. BQuad39help/2harm andall3scenespositive plus noRGB/half gains preserved; Droneharm and sizeerrorincrease prevent uniformprereq pass.',
 'A selects unchangedoriginalproposal, B fixedq0 inequality/sizeprior expansion; oldsurface comparator changesactivation/domain and isnot strict singleloss causalproof.',
 'All targetVAL inputs/tasklabels alreadyR7/R8exposed. Recent914content exclusion isrelativeonly; 108newrole and8regression areexploratory, not pristineconfirmation or newdomains.',
 'Frozen2Dmask/VLM helpers additional observations; two-number stablemask, validformat and COCOtypecompatibility do not certify literalidentity.',
 'Postscoretarget9/other7 explicitadapter, exactf32geometrydedup, strictsignedmargins/NA and zero primarybridge correct. Otherannotation inventory incomplete/noIDs; dominance is annotationrelative spatialproxy.',
 'new67 same2Dmask,targetIoUdecline andzerootheroverlap iscompatible withproposal/mapping/extent problem but not causaloridentityproof; reg006COCO41vs1 ispredictionplusrootqualitativeobservation, not GTcategory adjudication.',
 'Rootactualgallery2inputpreviews+8scoreawarepostscorepages reviewedbyreceipt here; thisreview didnot reopenimages or claim independentvisualjudgment.',
 'No method/score/threshold/prompt/filter change fromdiag. No A+B orneuralTTA. Potentialfuture transferpilot remainsconditional andrequires unusedobservation plusmatched directcontrol.'
]
files=[E/'REPORT.md',E/'DECISION.json',E/'PROGRESSION_GATE.json',E/'ROOT_POSTSCORE_DIAGNOSTIC_REVIEW.json',E/'ROOT_GALLERY_REVIEW.json',E/'ROOT_MECHANISM_REVIEW.json',E/'HISTORICAL_EXPOSURE_CLARIFICATION.json',V/'SCORING_INDEPENDENT_AUDIT.json',V/'MECHANISM_GATE_INDEPENDENT_REVIEW.json',V/'SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW.json',V/'REPORT_BODY_PRELIMINARY_REVIEW.json']
out={'state':'PASS_R13_FINAL_REPORT_INDEPENDENT_REVIEW','report_sha256':sha(E/'REPORT.md'),'decision_sha256':sha(E/'DECISION.json'),'progression_gate_sha256':sha(E/'PROGRESSION_GATE.json'),'root_postscore_diagnostic_review_sha256':sha(E/'ROOT_POSTSCORE_DIAGNOSTIC_REVIEW.json'),'scoring_audit_sha256':sha(V/'SCORING_INDEPENDENT_AUDIT.json'),'mechanism_gate_review_sha256':sha(V/'MECHANISM_GATE_INDEPENDENT_REVIEW.json'),'spatial_diagnostic_review_sha256':sha(V/'SPATIAL_DIAGNOSTIC_INDEPENDENT_REVIEW.json'),'auditor_sha256':sha(Path(__file__)),'finite_checks':len(checks),'checks':checks,'scientific_scope_review':scientific_scope,'evidence_sha256':{str(p.relative_to(R)):sha(p) for p in files},'issues':[],'no_new_GT_index_payload_task_IoU_rawsensors_images_model_GPU_or_scientific_edits':True}
f=V/'FINAL_REPORT_INDEPENDENT_REVIEW.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'checks':len(checks),'report_sha256':out['report_sha256'],'decision_sha256':out['decision_sha256'],'review_sha256':sha(f)}))
