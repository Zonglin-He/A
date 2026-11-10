"""Render saved audited scalars; never infer, fit, score or select a method."""
from pathlib import Path
import json, datetime
H=Path(__file__).resolve().parent;R=H.parents[1]
def rd(p):return json.loads(Path(p).read_text())
def wr(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def table(rows,keys):return '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(f'{r[k]:.6f}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |' for r in rows)+'\n'
v1=rd(H/'ALL_METHOD_ROWS.json');v2=rd(H/'ALL_METHOD_ROWS_QUERY.json');d1=rd(H/'DECISION.json');d2=rd(H/'DECISION_QUERY.json');di1=rd(H/'POSTSEAL_TASK_FEEDBACK_DIAGNOSTICS.json');di2=rd(H/'POSTSEAL_TASK_FEEDBACK_DIAGNOSTICS_QUERY.json');strata=rd(H/'PAIRED_ELIGIBILITY_STRATA.json')
report='''# R66 — Pre-fusion regional evidence and conditional-set matching adaptation

Completed on historically exposed DEV336: 168 Drone and 168 Quad observations in two Drone and three Quad scene clusters. All results are development measurements, not independent confirmation. No FINAL384 access, full5912 evaluation, source training, external predictor or external-model feedback cache. Only the primary regional-contrast path was executed; RobOT correspondence and 3DSSD sampling changes were not combined.

**Result:** after one disclosed control-validity repair, a real one-proposal matching-head update plus the unchanged R59 fusion reaches **22.818641%**, **+0.207525 pp** above R59. Both platform mIoUs improve. Its direct regional-evidence readout remains below R59, ordinary ABC updating also declines, and heldout-sampling C declines **0.117246 pp** from its own frozen source. Thus the R59-specific development gain is retained, but the predeclared joint supervision/transfer criterion is not met. Do not adopt this recipe as robust TTA, and do not describe all arms as failures.

## Fixed intervention and supervision

Waymo-only epoch100, original 256 final boxes, source backbone and candidate geometry fixed. Capture source A/B before point–language fusion. A region is a hypothesis group's seed support. Remove its visual features before fusion, re-zero masked residuals after each cross-encoder layer, and block visual self-attention and language-to-visual keys. Re-gather decoder queries from the intervened fused features. Persistently mask decoder visual keys and keys of queries anchored on removed seeds. Preserve original spatial anchors and six decoder geometry inputs; do not silently transplant an unrelated forward's query identities.

This is **pre-fusion seed-feature ablation**, not physical-object deletion: PointNet exterior receptive fields and fixed geometry priors can retain information. Engineering checks on real samples and full independent replays show zero-intervention equality, zero blocked attention and no influence from replacing blocked features with a large constant. These establish implementation validity, not object identity or supervision quality.

Computational duplicates require box IoU >=0.5 AND point-support Jaccard >=0.5 with at least four shared points. Use four highest A/B source group-mass hypotheses, not GT-selected recoverable candidates. These groups are not instance truth. A/B correspondence uses the existing transport formula on A/B point support only. Category uses the existing native first-lexicon-noun proxy from the actual description, without external parsing; it is not independently verified subject-class annotation. Category fusion is recomputed at the same full-description spatial hypotheses and positions. It is a conditional spatial probe, not unconstrained category grounding.

For group g, let D(t,g,r)=log p(g|t)-log p(g|t,ablate r), using native candidate matching group mass. Evidence is E_g=[D(full,g,region)-D(full,g,control)]-[D(category,g,region)-D(category,g,control)]. Stable positive signs in both A/B at epsilon 1e-6 define partial positive set S. Stable positive/negative groups define tested set Cset; other candidates stay UNKNOWN. Both signs are required for informative exclusion. UNKNOWN is never promoted to negative. No single local preference becomes a full-candidate positive label.

Direct use reweights only tested-pool mass and leaves every untested candidate's probability unchanged, invariant to common evidence offsets. A second no-update control assigns the tested-pool mass to S. Full-only, category-only, control-only and unchanged R59 are retained.

In parallel, run a real episodic update of the existing 184,960-parameter matching MLP: one Adam proposal, lr 0.0005, gradient clip 1, conditional set-mass loss plus source full-output KL with weight 1. Use the previously fixed nine half-scales for loss-only acceptance; no GT step selection. No layer/strength/loss grid. A/B evidence and all fits are sealed before C features are opened. C is a heldout sampling of the same original observation, not a new physical viewpoint or disjoint measurement. The A256 geometry and the existing C transport/UNKNOWN conventions remain fixed.

## Initial v1 and the one permitted repair

V1 matched the number of removed seeds but did not match removed decoder query keys. Only **14/2174** paired valid view-groups matched query-key removal; aggregate counts were **36231 vs 3983**. Thus regional and control ablations had unequal computational strength. V1 task scores remain real, but do not cleanly identify a semantic regional effect.

After v1 task/control diagnostics were exposed, lock exactly one validity revision before any v2 task score: choose pool-external control seeds in query-anchor/nonanchor strata to match both counts. Reuse all sealed source/region responses and regenerate only controls. No E formula, sign epsilon, optimizer, loss, source, pool, update site or step-budget changes. No third revision or positive-support threshold run.

V2 matches both counts for **2174/2174 valid paired view-groups**, aggregate **36231 vs 36231**. Independent audit also verifies all 2240 supported individual view-groups. Both-view valid groups remain 1087, so this is not simply a reduction in structural coverage. The control is outside compared regions, not certified semantically irrelevant; it may still contain useful contextual information. Same-v2-tested-pool v1-evidence direct controls separate coverage from changed evidence. This repair is disclosed development iteration, not a fresh preregistered confirmation.

## Complete v1 task table

Units: percent or percentage points. Improvement/worsening counts compare IoU against R59, not instance identity.

'''
keys=['method','Drone_mIoU','Quad_mIoU','balanced_mIoU','Acc25','Acc50','improved_vs_R59','worsened_vs_R59']
report+=table(v1,keys)+'\n## Complete matched-control v2 task table\n\n'+table(v2,keys)
report+='''
The primary v2 R59-updated readout improves five inputs and worsens two versus R59, with 12 query changes. Acc@25 changes from 43.154762% to 42.857143%; Acc@50 changes from 21.726190% to 22.023810%. Hence even the positive mIoU result does not improve every task measure. Ordinary updated ABC is 22.213673% versus frozen ABC 22.302460%. Updated C is 22.596512% versus C source 22.713758%. R59 uses the same unchanged representation-fusion formula, recomputed with the updated matching head. The positive result is readout-dependent; it is not evidence that arbitrary matching updates or future observations improve.

V2 R59 E-direct is 22.189728%, above v1 evidence on the identical v2 tested pool (21.676172%), yet still below frozen R59. This demonstrates that matching computational intervention strength changes evidence use, without establishing that the new evidence correctly identifies the referent.

## Coverage, reliability and accepted updates

'''
report+=table([{'diagnostic':k,'v1':di1[k],'v2':di2[k]} for k in di1],['diagnostic','v1','v2'])
report+='''
All task-proxy diagnostics above use saved candidate IoUs only after predictions are sealed. A group's best IoU >0.25 is a localization proxy, **not independent instance-identity truth**. They never enter candidate generation, evidence signs, acceptance or optimization.

A256 contains a potentially good box for 123 R59 failures, but the four-group pool covers only 51. V2 tests 35 and retains a good box in its positive set for 28. Thus 72/123 recoverable errors are already outside the compared pool; later evidence cannot address them. Among the 47 eligible v2 inputs, positive-set best IoU exceeds negative-set best IoU in 14, is lower in 13 and equal in 20. Seven eligible inputs exclude every tested good candidate from S; seven place an originally good R59 selection in the negative set. This is substantive erroneous supervision alongside limited coverage, not just an optimizer issue.

A second defect is interpretive: **479/545 positive-E groups have nonpositive mean control-corrected full-description support**. Relative full-minus-category evidence can become positive because the category term is even more negative; it need not indicate that the full sentence actually relies on the region. Only 39 positive-E groups have positive control-corrected full support in both views. Positive sign stability therefore does not establish positive referent evidence. A positive-support filter was not tested and must not be advertised as a solution.

V1: 85 eligible, 84 accepted nonzero updates, one loss-only rollback, 251 with no feedback. V2: 47 eligible, 47 accepted, 289 without feedback. No-feedback, UNKNOWN, rejection and fallback observations remain in the full336 denominator. Independent functional gradients, accepted parameters and all direct/C readouts match exactly; scores are not explained by zero gradients or a missing parameter update.

## Paired influence of the control repair

The following is post-seal saved-scalar stratification, not an evaluated deployment filter or a GT-selected method. Contributions use denominator336, not only the active subset.

'''
report+=table(strata,['v1_update_eligible','v2_update_eligible','n','v1_R59_full336_contribution_pp','v2_R59_full336_contribution_pp','v1_C_full336_contribution_pp','v2_C_full336_contribution_pp'])
report+='''
The 46 inputs eligible only in v1 contributed -0.173189 pp to R59 and are inactive in v2. The 39 shared-eligible inputs contribute +0.125796 pp with v1 versus +0.207525 pp with v2. Eight newly eligible inputs produce no task change. Thus v2 is helped both by avoiding harmful v1 activity and by changing updates on shared inputs. This does not identify the exact causal role of correct semantic evidence: no shuffled-positive-set update control was run, and both versions' C contributions on the shared inputs remain negative.

## Five scene clusters, all retained

No frame-independent significance claim or pooled CI is made from the five exposed scene clusters. All arms for both versions are exported; the main v2 comparisons are shown below.

'''
sc=rd(H/'ALL_SCENE_ROWS_QUERY.json');report+=table([s for s in sc if s['method'] in ['R59','R59_E_direct','R59_tta1','C_source','C_tta1']],['method','platform','scene_alias','n','mIoU','gain_vs_R59_pp'])
report+='''
The scene table's delta column always references R59. For heldout-sampling transfer, compare C_tta1 against C_source within the same scene instead; these are different baselines. All original negative comparisons (8254 within-run ordered losses and 71 v2-minus-v1 losses) are preserved as anonymous scalar rows. No adverse input is deleted from scoring.

## Original-method audit and limits

[CRG paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10172.pdf) and [official code](https://github.com/meetdavidwan/crg) combine original and blacked-out generation logits as (1+alpha)original-alpha*blackout. Here we test native grounding probabilities and a fixed spatial feature-ablation interface, not sentence generation likelihood or a CRG reproduction.

[DeYO official implementation](https://github.com/Jhyun17/DeYO/blob/main/methods/deyo.py) checks prediction confidence change under perturbation, with positive PLPD selection and entropy filtering/reweighting. Our A/B stable relative contrast does not reproduce its full reliability recipe. In particular positive E does not imply positive full-description support. These results cannot be generalized to complete DeYO.

[PASLE official implementation](https://github.com/palm-ml/PASLE/blob/main/code/adapt_algorithm.py) includes negative log probability mass over a partial candidate set, alongside its own selection/buffer/prototype and temperature machinery. Our denominator conditions on tested groups and adds source-output KL to protect untested predictions. It is PASLE-inspired partial-set supervision, not original PASLE. Different text inputs have different candidate identities; no universal query-class prototypes are assumed.

Readout, correspondence, grouping, native subject proxy and intervention validity remain competing explanations. Same-category neighbors, bad source supports, contextual control regions and unchanged geometry priors can all affect the contrast. The experiment does not prove that internal regional evidence, all TTA, or the 3EED direction is intrinsically impossible.

## Decision and reproducibility

The locked G criterion fails in both versions. V2 R59+TTA exceeds R59 and every registered same-information direct control, so its R59-specific gate passes. The independently heldout C gate fails, hence the joint TTA gate fails. **Stop this fixed regional-evidence/set-update recipe; retain v2 as a positive R59-specific development observation, not a new selected method.** No further threshold tuning, source training, transport/sampling mixing or protected evaluation was used to rescue it.

Engineering/provenance, independent encoder/decoder replay, independent gradient/readout, saved-scalar and derived audits are exported. All source/backbone inference and auxiliary replay counts are separately recorded in costs/COLLECTION_COUNTS_TIMES.json. The one validity repair and two saved-analysis implementation failures are retained in history; neither changed sealed predictions retroactively. Source artifacts and exact local provenance remain private. Public own_code includes only our programs, not upstream snapshots, raw media/descriptions/coordinates/IDs, features, weights or secrets. Remote publication uses immutable-byte verification and preservation of all prior study/root entries before moving main.
'''
(H/'REPORT.md').write_text(report)
wr(H/'DECISION_FINAL.json',{'state':'R66_COMPLETED_R59_SPECIFIC_POSITIVE_DEV_SIGNAL_JOINT_TTA_NOT_ADOPTED','new_G_adopted':False,'new_TTA_adopted':False,'v1':d1,'v2':d2,'v2_positive_development_result_retained':True,'v2_G_reliable_instance_supervision_established':False,'v2_heldout_sampling_transfer_established':False,'further_revision_run':False,'FINAL384_accessed':False})
plan=rd(H/'experiment/MATCHED_QUERY_DESIGN_LOCK_v2.json');config={k:v for k,v in plan.items() if k in ['source','candidate_pool_groups','group_box_IoU','group_point_Jaccard','group_shared_points_min','mask_seed_min','control_same_seed_count','pre_and_post_layer_zero_masked_visual','persistent_encoder_and_decoder_KV_masks','fixed_full_query_seed_indices','fixed_full_six_geometry_positions','category_phrase','evidence','set','sign_epsilon','optimizer','prior_KL_weight','C_not_used_in_supervision','AB_transport_support','source_training','new_external_models','n']};config.update({'candidate_boxes':256,'matching_head_trainable_parameters':184960,'v1_control':'equal_seed_count_pool_external','v2_only_change':'match_seed_count_and_original_query_anchor_count','versions_after_v1_task_exposure':True,'max_validity_revisions':1,'AB_used_for_supervision_and_fit':True,'C_only_after_all_fits_sealed':True,'same_information_direct_controls':True,'heldout_C_is_same_cloud_sampling_not_new_view':True,'protected_evaluation_accessed':False,'full_target_evaluation':False,'original_method_reproduction':False});wr(H/'PUBLIC_FIXED_CONFIGURATION.json',config)
counts={'inputs_per_run':336,'drone_inputs':168,'quad_inputs':168,'scene_clusters':5,'v1_task_arms':15,'v2_task_arms':17,'task_values':10752,'new_raw_GT_scoring_calls':0,'within_run_ordered_negatives':8254,'paired_v2_v1_negatives':71,'primary_full_source_forwards':672,'independent_full_source_forwards':672,'engineering_full_source_forwards':4,'v1_primary_encoder_decoder_replays':10304,'v1_independent_encoder_decoder_replays':10976,'v2_primary_control_replays':4480,'v2_independent_control_replays':4480,'v1_updates_eligible':85,'v1_updates_accepted':84,'v2_updates_eligible':47,'v2_updates_accepted':47};timing={};peak={}
for key,path in [('engineering',H/'engineering_capture_attempt001/COSTS.json'),('capture',R/'crossplatform/round66/prefusion336_attempt001/COSTS.json'),('v1_evidence',R/'crossplatform/round66/evidence336_attempt001/COSTS.json'),('v1_fit',R/'crossplatform/round66/fitted336_attempt001/COSTS.json'),('v1_readout',R/'crossplatform/round66/predicted336_attempt001/COSTS.json'),('v2_evidence',R/'crossplatform/round66/evidence336_matched_query_attempt001/COSTS.json'),('v2_fit',R/'crossplatform/round66/fitted336_matched_query_attempt001/COSTS.json'),('v2_readout',R/'crossplatform/round66/predicted336_matched_query_attempt001/COSTS.json'),('independent_v1_evidence',H/'independent_evidence_audit_attempt001/AUDIT_RESULT.json'),('independent_v2_evidence',H/'independent_query_evidence_audit_attempt001/AUDIT_RESULT.json'),('independent_v1_readout',H/'independent_audit_attempt001/AUDIT_RESULT.json'),('independent_v2_readout',H/'independent_matched_audit_attempt001/AUDIT_RESULT.json')]:
 c=rd(path);timing[key]=c.get('seconds');peak[key]=c.get('peak_GPU_bytes',c.get('peak_gpu_bytes'))
wr(H/'NORMALIZED_COSTS.json',{'counts':counts,'timing_seconds':timing,'peak_bytes':peak,'flags':{'raw_media_or_weights_public':False,'external_runtime_model':False,'source_training':False,'FINAL_access':False,'CPU_saved_task_only':True,'engineering_zero_leak_replays_separate_from_formal_counts':True}})
wr(H/'NORMALIZED_HISTORY.json',{'events':[{'stage':'v1','attempt':1,'status':'completed_with_control_confound','failure_type':'query_key_ablation_strength_mismatch','valid_outputs':336,'prior_attempt_preserved':True},{'stage':'v2','attempt':1,'status':'completed','repair_type':'match_seed_and_query_key_counts_only','valid_outputs':336,'before_GT':True,'prior_attempt_preserved':True},{'stage':'query_analysis','attempt':1,'status':'failed_then_repaired','failure_type':'saved_label_path_substitution','repair_type':'restore_existing_label_path','producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True},{'stage':'derived_audit','attempt':1,'status':'failed_then_repaired','failure_type':'numpy_scalar_numeric_dispatch','repair_type':'recognize_numbers_Real','producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True}]})
state=rd(H/'STATE.json');state.update(state='R66_SCIENTIFICALLY_COMPLETE_PUBLICATION_PENDING',GPU_stage_authorized=False,new_loss_authorized=False,scientific_complete=True,v2_R59_gain_pp=d2['TTA_gain_vs_R59_pp'],v2_C_gain_pp=d2['C_TTA_gain_vs_C_source_pp'],new_TTA_adopted=False);wr(H/'STATE.json',state)
(H/'HANDOFF.md').write_text('# R66 completed scientific work\n\nBoth versions sealed/scored/audited. One control-strength repair only. R59+TTA v2 +0.207525pp; directG and heldout C fail. No new model/update/score experiments. Finish privacy export, remote immutable bytes, CAS, receipt/archive and close state. Report and DECISION_FINAL.json are current. All negative alternatives retained. FINAL384 untouched.\n')
print({'report_bytes':(H/'REPORT.md').stat().st_size,'scientific_complete':True})
