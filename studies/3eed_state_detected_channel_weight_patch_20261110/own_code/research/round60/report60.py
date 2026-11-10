"""Build bounded decision/report from validated saved evidence only."""
from common60 import *
H=ROOT/'research/round60';p=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json');a=read(H/'ANALYSIS_SUMMARY.json');cost=read(ROOT/p['prediction_root']/'COSTS.json');det=read(ROOT/p['detection_root']/'DETECTION_SUMMARY.json');audit=read(H/'prediction_audit_attempt002/AUDIT_RESULT.json');scalar=read(H/'saved_scalar_audit_attempt001/AUDIT_RESULT.json');process=read(H/'independent_process_audit_attempt001/AUDIT_RESULT.json');assert all(q['state'].startswith('PASS_') for q in [audit,scalar,process]);metric={r['arm']:r for r in a['metrics']};pr={r['arm']:r for r in a['process']}
checks={}
for z in MECHANISMS:
 refs=['frozen_ABC',z+'_fixed',z+'_shuffled',z+'_random'];checks[z]=all(metric[z+'_dynamic']['balanced_mIoU_percent']>metric[x]['balanced_mIoU_percent']+1e-10 for x in refs)
decision={'state':'R60_COMPLETED_AUDITED_NO_INPUT_STATE_SPECIFIC_PATCH_ADVANTAGE','noise_dynamic_adopted':checks['noise'],'common_dynamic_adopted':checks['common'],'dynamic_weight_edit_adopted':any(checks.values()),'noise_delta_vs_ABC_pp':metric['noise_dynamic']['gain_vs_frozen_ABC_pp'],'common_delta_vs_ABC_pp':metric['common_dynamic']['gain_vs_frozen_ABC_pp'],'common_matches_fixed_and_shuffled_task_decisions':True,'noise_matches_ABC_task_decisions':True,'new_reliable_fault_component_identified':False,'no_all_channel_editing_impossibility_claim':True,'revisions_used':0,'source_retraining_started':False,'protected_holdout_accessed':False,'full_target_evaluated':False,'no_further_experiment_scheduled':True};dump(H/'DECISION.json',decision)
config={k:v for k,v in p.items() if k not in ['manifest','source_parameters','feature_root','association_root','detection_root','prediction_root','programs_sha256','inputs_sha256']};config['cohort']='exposed_DEV336_168_per_platform_2Drone_3Quad_scene_clusters';config['mechanism_definitions']='see_mechanism60_implementation_and_ITERATION_RECORD';dump(H/'PUBLIC_FIXED_CONFIGURATION.json',config)
dump(H/'NORMALIZED_COSTS.json',{'counts':{'inputs':336,'mechanisms':2,'arms':16,'new_head_replays':cost['new_head_replays'],'new_encoder_forwards':0,'new_gradient_steps':0,'solver_calls':0,'pseudo_label_fits':0,'external_model_calls':0,'saved_candidate_IoU_lookups':5376,'new_rawGT_or_geometry_calls':0,'ordered_negative_comparisons':1424,'independent_readouts':3360,'independent_prefix_feature_checks':1008,'independent_process_rows':3360},'timing_seconds':{'detect_all336':det['seconds'],'patch_controls_all336':cost['seconds'],'independent_prediction_audit':audit['seconds'],'independent_scalar_audit':scalar['seconds'],'independent_process_audit':process['seconds']},'peak_bytes':{'patch_controls_RSS':cost['max_RSS_bytes']},'flags':{'GPU_used':False,'feature_cache_reuse':True,'fresh_encoder_latency_measured':False,'target_training':False,'FINAL_access':False,'all_negative_rows_retained':True,'fixed_control_target_transductive_calibration':True}})
dump(H/'NORMALIZED_HISTORY.json',{'events':[{'stage':'prediction_audit','attempt':1,'status':'failed','failure_type':'local_seal_variable_shadowed_abs_builtin','repair_type':'rename_auditor_local_only','valid_outputs':0,'before_GT':True,'producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True},{'stage':'prediction_audit','attempt':2,'status':'passed','valid_outputs':3360,'before_GT':True,'producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True}]})
table='\n'.join('| '+r['arm']+' | '+' | '.join(f"{r[k]:.6f}" for k in ['Drone_mIoU_percent','Quad_mIoU_percent','balanced_mIoU_percent','gain_vs_frozen_ABC_pp','balanced_Acc25_percent','balanced_Acc50_percent'])+' |' for r in a['metrics'])
proc='\n'.join(f"| {r['arm']} | {r['actual_edit_n']} | {r['median_relative_weight_edit_norm']*100:.4f} | {r['median_matched_logodds_disagreement_change_percent']:+.6f} | {r['median_candidate_separation_change_percent']:+.6f} |" for r in a['process'])
pairs='\n'.join(f"| {r['arm']} | {r['reference']} | {r['balanced_delta_pp']:+.6f} | {r['improved']} | {r['worsened']} | {r['query_changes']} |" for r in a['pairwise_comparisons'])
scenes=read(H/'ALL_SCENE_METRICS_PUBLIC_SAFE.json');sct='\n'.join(f"| {r['platform']} | {r['scene_cluster']} | {r['n']} | {r['arm']} | {r['mIoU_percent']:.6f} |" for r in scenes if r['arm'] in ['frozen_ABC','r58_online_ema_teacher','r59_direct_all','noise_dynamic','common_dynamic'])
text=f'''# R60 — internal-state detector, actual channel weight patch, native replay

**Decision: neither dynamic patch is adopted.** Noise-dominance editing leaves every final selection unchanged versus frozenABC. Candidate-common-response editing yields only +0.002229pp; its fixed and wrong-input controls give exactly the same final selections. No input-specific editing advantage is established. Do not enlarge attenuation merely to force a result.

## Scope and actual mechanism

Fixed Waymo-only epoch100, historically exposedDEV336(168per platform,five scene clusters), originalA256boxes and R57ABC sampling/soft transport/source-neutral UNKNOWN. Source resets for each input. Use audited source hidden states after the second ReLU and original text masks. No new external model, source training, rawRGB semantic encoder, target GT in detector, gradient, least-squares, pseudo-label fitting or per-input patch search. Development is AI-assisted; no separate Astra process was invoked, and no inference-stage external AI is used.

Only final64x288weight columns change; bias and earlier parameters stay fixed. At most8columns attenuated, alpha=.25 fixed before task scores, no hyperparameter grid. Formula Wx=W0 diag(1-a(x)),0<=a<=.25. Source restored after native final normalization/token-logsumexp matching/temperature/.07 and originalABC readout. There are two distinct diagnostic hypotheses, not assertions of discovered harmful channels:

1. Noise: normalize additive hidden-column contributions by source output norm, project to a unit subject-versus-other-token mean direction. Use fixed soft correspondence to compare within-sampling fluctuation with between-candidate variation. Require B/C support, select at most8positive excess channels, attenuation proportional to excess/(noise+signal). No hard identity claim is made from correspondence or token direction. This statistic is not raw activation variance and is not the final score.
2. Common response: candidate-common activation fraction above.5, normalized projection energy and low entity-direction alignment identify possible angle-compressing channels. Select at most8. These channels can still encode useful category semantics; the intervention must establish task benefit.

Both detectors triggered on336/336inputs,median8columns,median233supported anchors. Absence of evidence would give a zero patch, but no such input occurred in this cohort. No sample IDs, scene names, GT or caption-specific exceptions enter the detector. Hashes are only for reproducible negative-control randomization, not deployment logic.

## Controls and results

Shared fixed patch: top8mean dynamic attenuation over all336unlabeled internal states, shared across platforms. It is a target-transductively calibrated repair control, not source-only; it does not have identical norm to every dynamic input. Wrong-input control permutes patches within platform, preserving the exact patch multiset and no fixed points. Random control excludes selected channels, matching channel count, individual column edit magnitudes and total Frobenius norm; its safety attenuation cap.5 is registered. Exact activation-gating counterparts verify the weight-scaling equivalence. R58online is an inherited system reference with different history/supervision; R59directAll is the stronger same-internal-information no-edit reference.

| Arm | Drone mIoU% | Quad mIoU% | Balanced mIoU% | Delta vs ABC pp | Acc25% | Acc50% |
|---|---:|---:|---:|---:|---:|---:|
{table}

## All primary pairwise comparisons

| Dynamic arm | Reference | Delta pp | Improved | Worsened | Query changes |
|---|---|---:|---:|---:|---:|
{pairs}

Noise dynamic changes no final query. Common dynamic changes2queries, one tinyIoUimprovement and one unchangedIoU; fixed and shuffled pick identical queries on all336. No accuracy-threshold gain from either dynamic arm. Wrong-input noise patch has +0.083196pp from one improved observation, not evidence to adopt the wrong-input control. All16arms and1424directed negative comparisons retained; these are comparison rows, not1424unique failed observations. Zero unknown/fallback/no-update cases are removed from denominators.

## Process changes do not establish task repair

| Arm | Actual edits n | Median edit norm/source % | Median disagreement change % | Median separation change % |
|---|---:|---:|---:|---:|
{proc}

Noise dynamic changes native probabilities on335inputs but the median true log-odds disagreement decreases only0.015964%,with zero query correction. The diagnostic proxy thus finds some small score fluctuations, not an established cause of grounding failures. Common dynamic increases candidate separation about5.9316%but also disagreement about5.0210%; greater score separation does not mean better instance discrimination. Its selected channel-set overlap with fixed control has median Jaccard.77778, compared with.06667for noise. This aligns with its near-static behavior, but is not an independent causal proof. Inputwise dynamic patches have not outperformed fixed/wrong-input controls. Do not interpret harmlessness or nonzero parameters as a method success.

Weight scaling versus equivalent activation gating has max probability discrepancy {cost['activation_weight_control_max_score_difference']:.3g}, from FP32 operation ordering, and every final query matches. Storage as weights is not an additional information source.

## Five exposed scene clusters

| Platform | Anonymous cluster | n | Arm | mIoU% |
|---|---:|---:|---|---:|
{sct}

## Independent checks and failed attempt

Reconstructed1008prefix states, both detector formulas, fixed patch, donor derangements, randomized count/norm controls and3360noninherited native readouts. Max detector error {audit['max_detector_error']:.3g},native score error0. Every actual edit restores source, all boxes/reference outputs fixed. Independent task aggregate max error {scalar['independent_summary_max_abs_error_pp']:.3g}pp. Independently checked3360process rows and original final log-odds stability/separation. First auditor attempt failed because a local seal dictionary shadowed builtin abs; preserved failed code/plan/receipt, renamed that auditor variable, changed no predictions/tolerance/method, and passed attempt2before scoring.

One saved-candidate-IoU evaluation after sealed predictions and independent audit:5376values,zero new rawGT/geometry calls. DEV336was already exposed, so no independent-confirmation or frame-independent-CI claim. FINAL384and ALL5912 untouched. Main replay takes{cost['seconds']:.2f}s plus detector{det['seconds']:.2f}s on CPU from frozen caches; no GPU. This is cached-head timing, not full deployment latency.

## Paper and claim boundaries

DICE uses contribution-directed sparsity for OOD detection, and ASH post-hoc activation shaping. We checked their original abstract/official implementation. Neither guarantees 3Dgrounding accuracy, neither is reproduced here. [DICE official code](https://github.com/deeplearning-wisc/dice/blob/master/models/route.py), [DICE paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4405_ECCV_2022_paper.php), [ASH official code](https://github.com/andrijazz/ash/blob/main/ash.py), [ASH paper](https://arxiv.org/abs/2209.09858).

## Adoption and next boundary

Do not adopt either detector/column-edit recipe, and do not use the wrong-input control as a method. Two of maximum3mechanisms tested,zero revisions; no evidence-based revision is compelled by these near-zero overall effects. This stops this bounded column-damping recipe, not all internal editing or TTA. Small, nuisance-ranked columns were not shown to cause the task failures; distributed effects, stronger justified interventions and correspondence quality remain unresolved. Preserve R58and R59direct internal fusion as exposed-development references. No new model/loss/EMA combination/source preparation/full-target/holdout run is scheduled. Deployment interface detect_patch_replay is supplied for reproducibility, not production adoption.
'''
(H/'REPORT.md').write_text(text);print(decision)
