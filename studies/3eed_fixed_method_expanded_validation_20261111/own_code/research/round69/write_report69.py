"""Report the prelocked expanded validation; never selects a new method."""
from common69 import *
H=ROOT/'research/round69';p=read(H/'experiment/DESIGN_LOCK.json');m=read(H/'ALL_METHOD_ROWS.json');sc=read(H/'ALL_SCENE_ROWS.json');di=read(H/'COHORT_DIAGNOSTICS.json');re=read(H/'RANDOM_REPEAT_SUMMARY.json');rest=read(H/'SOURCE_RESTORATION_BREAKDOWN.json');de=read(H/'DECISION.json');frames=read(H/'ALL_FRAME_CLUSTER_ROWS.json');stats=read(H/'score_all2304_attempt001/TASK_SUMMARY.json')
assert read(H/'independent_derived_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_')
def table(rows,keys):
 return '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(f'{r[k]:.6f}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |' for r in rows)+'\n'
out='''# R69 — fixed-method expanded validation, without new algorithms or tuning

This is the user-authorized validation of the original R68-selected configurations on more target observations and additional target recording sequences. It is not R69 method development: gamma=1, eta=1, update module, loss, acceptance rule, candidate transport, UNKNOWN handling, seed prefixes and sampling quota are unchanged. No source retraining, external visual model or new RGB loss. VAL1280 and TRAIN1024 are reported separately, with no favorable-cohort selection and no method changes after new task labels.

## Data and exposure boundary

- VAL1280: 640 expressions/platform, outside DEV336 and FINAL384 by identifier, point-media hash and image-media hash; 1,033 distinct media pairs (428 Drone,605 Quad), at most two distinct descriptions/media. These are the same two Drone and three Quad official recording sequences as existing VAL, not five new scenes. Full VAL historically received source/other-method evaluation: this is expanded fixed-method evidence outside the main development/protected-media subsets, not never-exposed target data.
- TRAIN1024: 512 expressions/platform,1,024 unique media pairs, one expression/media; two additional official recording sequences/platform and distinct nominal place-name families. Three TRAIN sequence families resembling VAL place names were excluded before model execution. Every selected TRAIN point/image hash is disjoint from all target VAL media across both platforms. The source checkpoint remains Waymo-only; target TRAIN labels are used solely for this postseal evaluation, never source training, G construction, fit acceptance or method selection.
- Origin headers contain empty sequence_name and local integer timestamps. Equal local timestamps from different sequences do not prove repeated frames. Byte hashes and official sequence identity establish the stated exclusions; global geographical non-revisit and absolute-wallclock separation cannot be certified from available metadata. Do not call TRAIN a certified geographic holdout.
- FINAL384 was historically scored in R46. This round reads its manifest identities and hashes its media bytes only to exclude overlaps; it does not decode those media for model input, generate predictions, read numeric GT, reuse historical FINAL predictions, or score FINAL again. This is not a claim of zero FINAL byte access or a pristine historical holdout.
- Selection is stratified across sequences and eight local-time bins, deterministic hash ordering, media deduplication and a fixed per-media description cap. Target TRAIN indexes all caption records with the source text-length interface constraint, without GT class/box/correctness filtering. All selected captions fit the fixed interface. Paths, captions, hashes and raw sequence names remain private.

An initial metadata-only selection incorrectly treated empty-sequence/local-timestamp pairs as global identity. It was repaired before formal model execution or task scoring; the failed selection is preserved. It did not change an algorithm, use task outcomes, or remove negative predictions. Independent data audit verifies all2,304 joins, exclusion hashes and cohort counts.

## Fixed algorithms and execution

Source: strict Waymo-only epoch100 with pinned weights and fingerprint. All11 task arms select from the same input's original256 final A boxes; no new box proposals, geometry refinement or regression update. A,B,C are different samplings of the same physical point cloud, not new camera views or independent scenes. All groups/fits use A/B only; every fitted matching-head snapshot is sealed before any C inference. Source model weights remain unchanged.

R59 uses original representation fusion gamma1 with the original R57 correspondence, fixed transport mass/UNKNOWN, temperature and readout. R66 is exact v2 region-vs-matched-control full-description-minus-category contrast, matching both seed and masked query-key counts. Initial group/pool/stable-sign/partial-set rules remain fixed. An eligible example must have both positive and negative groups; the existing matching MLP performs one original Adam .0005 step with clip1 and nine label-free loss-backtracking scales. Accepted eta1 snapshots are loaded exactly. No GT-dependent trigger, early stop, sample exclusion, update-layer change or strength search. Noneligible/rejected examples remain in every denominator.

Random SA3 retains256 native D-FPS seeds plus256 random remaining seeds at512 total; original source SA2, query anchors and six decoder geometry inputs remain fixed. Three seed prefixes are the original R67/R68 rules; they are three repeated policies, not an output ensemble, bestseed selection, or extra information for R66. Each policy receives the same three A/B/C point samplings. Source/ABC/R59 use no regional perturbation forwards; R66 has additional evidence-construction cost. E-direct and set-direct are retained same-evidence/no-update diagnostics, not new optimized alternatives. C distributions are transported to the same canonical A256 candidates using the fixed correspondence; they do not score a new physical scene or new box set. C source versus C adapted is a transfer diagnostic and not a posthoc veto of the deployment-readout comparison.

Predictions and statuses were sealed before numeric task GT was released. One official-convention GT pass computes all original256 candidate IoUs/input, then looks up every11 selected index. The two cohort tables derive from this one pass, not two extra scores. Independent geometry uses rectangle-polygon intersection and vertical overlap rather than the producer convex-polyhedron implementation. All UNKNOWN, fallback, no-update and negative comparisons are exported.

## Main fixed-method results

Units: percent for IoU/accuracy, percentage points for differences. Both cohorts have equal platform counts, so expression mean equals the predeclared platform-equal mIoU. Random mean is over three policy scores and has no single associated selected-query output.

'''
keys=['method','Drone_mIoU','Quad_mIoU','balanced_mIoU','Acc25','Acc50','gain_vs_R59_pp','improved_vs_R59','worsened_vs_R59']
for co in ['VAL1280','TRAIN1024']:
 out+='\n### '+co+'\n\n'+table([z for z in m if z['cohort']==co],keys)
out+='\n## Three prespecified random policies\n\n'+table(re,list(re[0]))
out+='''
The repeat mean/range/std measure seed-policy variability, not an independent-scene confidence interval. All three outputs and ordered negatives are preserved. The best repeat is not selected for deployment or substituted for the policy mean.

## Update coverage, C transfer and source-restoration attribution

'''+table(di,list(di[0]))+'\n'+table(rest,list(rest[0]))
out+='''
Restoration bins describe R66 improvements relative to R59. Exact source query restoration and equal-source-IoU alternative queries are separate from improvements beyond source; IoU equality alone is not independent instance-identity certification. These are postseal diagnostics, not deployment gates. Negative contributions are not hidden by the positive-only attribution bins; the full net and all harmed rows remain in the next table and exported scalars. Largest single-case contributions and the remainder retain the full original cohort denominator.

'''+table([z for z in m if z['method'] in ['R66_R59','random_s0','random_s1','random_s2']],['cohort','method','gain_vs_R59_pp','improved_vs_R59','worsened_vs_R59','query_changes_vs_R59','largest_positive_full_denominator_pp','rest_after_largest_full_denominator_pp'])
out+='\n## Every recording-sequence cluster\n\n'+table(sc,['cohort','method','platform','scene_alias','n','mIoU','gain_vs_R59_pp'])
frame_summary=read(H/'FRAME_BALANCED_SUMMARY.json')
out+='''
Expression-level observations within a recording, and repeated descriptions of the same media, are not independent scenes. The complete per-frame-cluster table and this equal-frame/platform sensitivity readout are supplied; primary metrics remain expression-level. With only2/3 VAL sequences and2/2 TRAIN sequences, no frame-iid confidence interval or broad cross-city generalization claim is made. A single harmed recording is not an automatic rejection gate, but platform/sequence asymmetry remains a material limit.

'''+table(frame_summary,list(frame_summary[0]))
out+='\n## Fixed-validation decision\n\n'+ '\n'.join('- '+k+': '+str(v) for k,v in de.items())+'\n\n'
for d,r in zip(di,re):
 out+=f"{d['cohort']}: R66 minus R59 **{d['R66_gain_vs_R59_pp']:+.6f}pp**, R66 minus three-repeat random policy mean **{r['R66_minus_random_policy_mean_pp']:+.6f}pp**, C adapted minus C source **{d['R66_C_gain_vs_C_source_pp']:+.6f}pp**. Update eligibility is {d['update_eligible']}/{d['n']}; accepted nonzero snapshots {d['accepted_nonzero']}.\n\n"
out+='''
No algorithm is promoted merely because loss falls, parameters change, a single expression is restored, or source is exceeded. Evidence for R66's incremental task value requires comparison with R59 and the cheaper fixed random policy under the reported information/cost differences. Consistent positive direction on these cohorts would support expanded fixed-method validation, not automatically establish general semantic supervision; absent/reversed gains do not justify another eta/gamma/layer/threshold sweep on these labels. This round terminates after reporting the fixed comparisons. No new method is developed using these outcomes and no follow-on run is queued.

## Independent verification, resources and reproducibility

All2,304 examples receive independent CPU reconstruction of AB groups/control counts/evidence algebra, gradient/Adam/backtracking, every saved parameter, ABC candidate transport/UNKNOWN and all11 readouts. A pre-model locked72-input subset (eight per cohort/platform/recording) receives independent full source, regional encoder/decoder and random-suffix GPU replay. It is a stratified mechanism replay subset, not a claim of repeating every full GPU computation. Runtime zero-mask h/query/token equality, unchanged source fingerprints and sampler/anchor checks additionally run throughout the full producer. All589,824 candidate IoUs, task/status, scene, negative, repeat, source-restoration and frame-cluster summaries are independently verified. Public package has an independent per-file private-token/secret/schema scan and immutable remote-byte verification before main compare-and-swap; no original media/descriptions/IDs/coordinates/features/weights/secrets or third-party source snapshots are published.

Collection timing includes input decoding, cache I/O, fits and audits. Regional replays/suffixes are not full-backbone forwards. Random repeat means are not three-repeat deployment ensembles. Per-policy deployment latency is not inferred by dividing total time across heterogeneous computation. Engineering four-input checks are disclosed separately and not counted as extra evaluation observations. Source loader's unused RoBERTa pooler-initialization warning does not change the strict loaded grounding checkpoint or matching-state equality.
'''
(H/'REPORT.md').write_text(out)
cfg={k:v for k,v in p.items() if k not in ['programs_sha256','inputs_sha256'] and not k.endswith('_root') and k not in ['manifest','cohort_map','scoring_index','media_hashes','source_plan','initial_head']};cfg.update(data_selection=read(H/'DATA_SELECTION_AUDIT.json'));dump(H/'PUBLIC_FIXED_CONFIGURATION.json',cfg)
ab=read(ROOT/p['AB_capture_root']/'COSTS.json');ev=read(ROOT/p['evidence_root']/'COSTS.json');cr=read(ROOT/p['C_random_root']/'COSTS.json');pr=read(ROOT/p['prediction_root']/'COSTS.json');ga=read(H/'independent_gpu_audit_attempt001/AUDIT_RESULT.json');sr=read(H/'score_all2304_attempt001/SCORING_RECEIPT.json')
dump(H/'NORMALIZED_COSTS.json',{'counts':{'evaluation_inputs':2304,'VAL_expressions':1280,'TRAIN_expressions':1024,'VAL_unique_media':1033,'TRAIN_unique_media':1024,'formal_AB_source_forwards':ab['full_source_forwards'],'formal_C_source_forwards':cr['C_source_forwards'],'formal_regional_encoder_decoder_replays':ev['encoder_decoder_replays'],'formal_random_suffix_replays':cr['random_suffix_replays'],'update_eligible':ev['eligible'],'accepted_nonzero':ev['accepted'],'task_arms':len(p['arms']),'task_values':sr['task_values'],'candidate_IoUs':2304*256,'ordered_negatives':sr['ordered_negative_comparisons'],'engineering_AB_forwards':8,'engineering_regional_replays':128,'independent_GPU_inputs':72},'timing_seconds':{'AB_capture':ab['seconds'],'AB_evidence_and_fit':ev['seconds'],'C_and_random_capture':cr['seconds'],'readout':pr['seconds'],'task_scoring':sr['seconds']},'peak_bytes':{'AB':ab['peak_GPU_bytes'],'evidence':ev['peak_GPU_bytes'],'C_random':cr['peak_GPU_bytes']},'flags':{'source_training':False,'no_parameter_search':True,'all_negatives_retained':True,'FINAL_identity_hash_exclusion_only':True,'FINAL_model_prediction_or_scoring':False,'new_geographic_independence_claim':False}})
dump(H/'NORMALIZED_HISTORY.json',{'events':[{'stage':'cohort_metadata_selection','attempt':1,'status':'repaired_before_models','failure_type':'local_timestamp_treated_as_global_identity','repair_type':'media_byte_identity_and_sequence_local_strata','before_GT':True,'producer_predictions_changed':False,'prior_attempt_preserved':True},{'stage':'engineering_source','attempt':1,'status':'passed','completed_calls':8,'before_GT':True},{'stage':'engineering_evidence_fit','attempt':1,'status':'passed','completed_calls':128,'before_GT':True},{'stage':'formal_source_AB','attempt':1,'status':'passed','completed_calls':ab['full_source_forwards'],'before_GT':True},{'stage':'formal_evidence_fit','attempt':1,'status':'passed','completed_calls':ev['encoder_decoder_replays'],'before_GT':True},{'stage':'formal_C_random','attempt':1,'status':'passed','completed_calls':cr['C_source_forwards']+cr['random_suffix_replays'],'before_GT':True}]})
(H/'ITERATION_RECORD.md').write_text('''# R69 research iteration record

Problem: R68 gains were small on repeatedly exposed DEV336 and R66 was dominated by restoration of an R59 error. Competing explanations are true incremental task correction, single-example concentration, sampling-policy variance and recording/domain dependence. This round changes evaluation coverage only, with unchanged algorithms and no new papers/algorithm grafts needed.

Locked intervention: VAL1280 outside DEV/FINAL media and TRAIN1024 in additional official recording sequences. Source, frozenABC, R59gamma1, R66v2eta1, three original random policies, same-evidence E/set direct and heldout-sampling C diagnostics. Fixed selection, no GT in feedback/fits/acceptance, all fits before C, all predictions before task scores. Main decisions use total/platform/recording results rather than case-specific repair or per-scene nonnegative gates.

Validity: byte identity supersedes misleading local timestamps. TRAIN official split is evaluated only; no target-label training. Geographic/time independence not certified. Expanded VAL is outside current DEV/FINAL media but historically fullVAL-exposed. FINAL identities/media hashes used only for exclusions; no new FINAL inference or GT scoring. All failed selection artifacts/engineering attempts and all later negative comparisons retained. Independent numeric and scalar audits precede publication.

Stop: report these fixed methods, seed variability, source-restoration contributions and C behavior; do not tune the methods on newly seen outcomes or extend the grid. No automated continuation is registered.
''')
print({'report_bytes':len(out.encode()),'decisions':de})
