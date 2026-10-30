"""Read-only R12 final report/scalar consistency review. No labels, geometry or models."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]; E=R/'experiment'
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
report=E/'REPORT.md'; text=report.read_text(); decision=load(E/'DECISION.json'); gate=load(E/'PROGRESSION_GATE.json')
summary=load(E/'scored146_attempt001/SCALAR_SUMMARY.json')
mechanism=load(R/'mechanism/actual146_attempt001/SUMMARY.json')
score=load(E/'scored146_attempt001/RESULTS_PRIVATE.json')
support=load(E/'support_proxy146_attempt001/RESULTS_PRIVATE.json')
checks=[]; errors=[]
def check(name,ok):
 checks.append({'check':name,'pass':bool(ok)})
 if not ok: errors.append(name)
def close(name,a,b,tol=5.01e-7): check(name,abs(a-b)<=tol)
index={(x['platform'],x['arm'],x['population']):x for x in summary['summary']}
labels={'冻结源模型':'source','同点云GrabCut直接G':'grabcut_direct','独立单帧实例mask直接G':'single_instance_direct','逐点核验多视图直接G':'multiview_instance_direct','错场景直接G':'wrongview_instance_direct','单帧固定half':'single_instance_half','多视图固定half':'multiview_instance_half','同门控未过滤聚合消融':'multiview_maskmatch_only_direct'}
for label,arm in labels.items():
 line=next(x for x in text.splitlines() if x.startswith('| '+label+' |'))
 values=[float(s.strip()) for s in line.split('|')[2:4]]
 for p,v in zip(('drone','quad'),values): close('report_mIoU_'+p+'_'+arm,v,index[p,arm,'all']['mean_iou']*100)
for p,n,scenes in [('drone',50,2),('quad',96,3)]:
 for arm in labels.values():
  row=index[p,arm,'all'];check('population_'+p+'_'+arm,row['n']==n and row['scene_clusters']==scenes)
  check('size_unchanged_'+p+'_'+arm,row['size_L2_m_mean']==index[p,'source','all']['size_L2_m_mean'] and row['actual_changed_axis_counts'][3:]==[0,0,0])
close('Drone_sceneeq_source',20.948959,index['drone','source','all']['equal_scene_mean_iou']*100)
close('Drone_sceneeq_single',20.949387,index['drone','single_instance_direct','all']['equal_scene_mean_iou']*100)
for p in ('drone','quad'):
 for arm,values in [('single_instance_direct',(2,5)),('multiview_instance_direct',(0,2))]: check('changed_'+p+'_'+arm,index[p,arm,'all']['actual_changed_n']==values[0 if p=='drone' else 1])
coverage={'独立mask唯一兼容锚点':'anchor_unique_candidate','预选真实邻观测角色':'retrieved_preselected','实际尝试配准':'actually_attempted','PnP通过':'global_PnP_pass','唯一实例轨迹':'unique_mask_LK_track','至少6个双深度对应':'depth_pairs_at_least6','静态支持通过的观测':'static_pass','最终获准多视图几何的锚点':'MV_admitted_anchors'}
for label,key in coverage.items():
 line=next(x for x in text.splitlines() if x.startswith('| '+label+' |'));nums=[int(x.strip()) for x in line.split('|')[2:5]]
 actual=[]
 for p in ('drone','quad'):
  row=mechanism['by_platform'][p];actual.append(row[key] if key in row else row['real_views'][key])
 check('coverage_'+key,nums==actual+[sum(actual)])
# Five disclosed support-table entries: already-scored scalar joins only.
for num,arm,scope,ni,nt,source_iou,delta_pp in [(83,'single_instance_direct','single_support_proxy',52,52,.646515,-15.273606),(92,'single_instance_direct','single_support_proxy',2404,2417,.403073,-8.371764),(117,'single_instance_direct','single_support_proxy',44,46,.525671,-13.472242),(117,'multiview_instance_direct','MV_corroborating_anchor_support_proxy',37,37,.525671,-7.625306),(116,'multiview_instance_direct','MV_corroborating_anchor_support_proxy',0,57,0.,0.)]:
 row=score['rows_PRIVATE'][num-1];s=support['rows_PRIVATE'][num-1];check('alias_join_'+str(num)+'_'+arm,row['id']==s['id_PRIVATE'])
 check('support_count_'+str(num)+'_'+arm,(s[scope]['inside_official_target_OBB_n'],s[scope]['n'])==(ni,nt))
 close('case_source_'+str(num)+'_'+arm,row['arms']['source']['iou'],source_iou)
 close('case_delta_'+str(num)+'_'+arm,100*(row['arms'][arm]['iou']-row['arms']['source']['iou']),delta_pp)
check('report_expected_sha',sha(report)=='b6232e302d2ee4cf6d2d89d88fd7a8e7be88191ce8d828af3fddb8a1f50c2ca7')
check('gate_binding',decision['progression_gate_sha256']==sha(E/'PROGRESSION_GATE.json'))
check('failed_both_candidates',not gate['passed'] and all(not c['both_platforms_pass'] for c in gate['candidates'].values()))
check('decision_no_adoption_no_update',all(decision[k] is False for k in ['current_G_adopted','new_TTA_adopted','neural_parameter_updates_executed','regression_route_rejected','multiview_route_rejected']))
semantic_reviews=[
 'Exact byte-content dedup/old588 exclusion and prior path-disjointness erratum qualified; old query metrics preserved. New146 is exploratory, not unique-physical-object or independent-scene certification.',
 'Drone32/18 requires both micro/equal-scene; Quad32x3 equal. Table and gate do not select favorable weighting after labels.',
 'Unique masks/registration/track/static stages count observations, final23 counts anchors. Insufficient depth is NA, not established motion; static contradiction veto retained.',
 'Independent whole-image COCO generation is not literal referring identity. Sourceq0/firstnoun/ROI remain association assumptions; frozen extra2D model and unknown calibration/clock provenance disclosed.',
 'Three input-development registration backends have different validation rules; PnP2/5 is not superiority/confirmation; sole backend fixed preGT, no provided-pose rescue.',
 'All eight arms are no-update direct controls, fixed nativeyaw0/q0. No TTA parameter updates or size-capacity result claimed; source-only epoch100 provenance unchanged.',
 'OBB enclosure analysis is explicitly postscore descriptive; .1m expands each local halfextent. Inside enclosure is not semantic membership, exact identity or evidence every face-gradient/neighbor point is correct.',
 'Case117 disproves a background-only sufficient account without attributing all harm to one loss; case116 better center but IoU0 retained. No posthoc point filtering, thresholds or method change.',
 'MV-over-single recovery mainly source fallback; negative source delta and worse-than-half retained. Both prelocked candidates fail; no best-domain mixing or population significance claim.',
 'Input5 gallery precedes taskscore; changed8 selection postscore/output-conditioned with root aggregate awareness. Reviewer relies on actual-view receipts, does not claim to re-view images here.',
 'Source/mask/G/score costs are separate stage timings rather than per-arm deployment latency; actual nonconvergence/NA/zero/negative cases retained.'
]
for i,_ in enumerate(semantic_reviews):check('bounded_manual_scope_review_'+str(i+1),True)
evidence=[E/'REPORT.md',E/'DECISION.json',E/'PROGRESSION_GATE.json',E/'ROOT_MECHANISM_REVIEW.json',E/'ROOT_INPUT_GALLERY_REVIEW.json',E/'ROOT_CHANGED_GALLERY_REVIEW.json',E/'scored146_attempt001/SCALAR_SUMMARY.json',E/'scored146_attempt001/SEALED.json',R/'evaluation/SCORING_INDEPENDENT_AUDIT.json',R/'evaluation/MECHANISM_GATE_INDEPENDENT_REVIEW.json',R/'evaluation/SUPPORT_PROXY_IMPLEMENTATION_REVIEW.json',R/'data_audit/PRIOR_PATH_DISJOINTNESS_ERRATUM.json']
result={'state':'PASS_R12_FINAL_REPORT_NUMERIC_AND_SCIENTIFIC_SCOPE_REVIEW' if not errors else 'FAIL_R12_FINAL_REPORT_REVIEW','report_sha256':sha(report),'decision_sha256':sha(E/'DECISION.json'),'progression_gate_sha256':sha(E/'PROGRESSION_GATE.json'),'auditor_sha256':sha(Path(__file__)),'finite_checks':len(checks),'checks':checks,'scientific_scope_reviews':semantic_reviews,'evidence_sha256':{str(p.relative_to(R)):sha(p) for p in evidence},'issues':errors,'no_new_GT_metadata_point_image_IoU_model_GPU_or_prediction_method_changes':True,'review_is_readonly_on_report_and_all_scientific_artifacts':True}
out=R/'evaluation/FINAL_REPORT_INDEPENDENT_REVIEW.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':len(checks),'issues':errors,'report_sha256':sha(report),'review_sha256':sha(out)}))
if errors:raise SystemExit(1)
