# R11 implemented bounded reliability contract

This file supersedes proposal-only choices in RELIABILITY_PROPOSAL.md. No real data, GPU, labels or scoring are used by implementation preparation. Old R6–R10 files are unchanged.

## Array-only interfaces

`reliability.compute_reliability(*, sample_id, points, raw_indices, rgb, source_box6, camera_spec, base_mask2d, base_result, config=ReliabilityConfig())` accepts explicit arrays plus validated camera and the original same-sample R10-style result. It does not read files. Return:

- `face_weights`: six `axis:sign` entries; undefined gives0 and an explicit reason.
- `RGB_weight`, `RGB_weight_defined`: absolute RGB objective strength, not calibrated probability.
- `axis_reliability6`, `axis_reliability_defined6`: minimum corresponding face weight for paired-size or single nearface; inactive coordinates0/undefined.
- `active_face_keys`: actual accepted coordinate's source-facing observation keys, used for separately bound uniform mean/control reductions.
- `case_reliability`: minimum **actual float32-changed** accepted-axis score, or `None` when any such axis is undefined. Accepted-but-unchanged axes never enter the case minimum. Not multiplied by a global source confidence.
- `association_status`: association_supported/conflicted/undefined means a source-conditioned geometric correspondence hypothesis only. Never identity_verified or literal fulltext identity evidence.
- `raw_face_weights`, `raw_face_weight_defined`, `raw_RGB_weight`, `raw_RGB_weight_defined`, `raw_axis_reliability6`, `raw_axis_reliability_defined6`, `raw_case_reliability` preserve pre-correspondence diagnostics. Effective weights and effective defined flags are separate.
- `partition_hashes`, `features`, `undefined_reasons`, `control_scores`. Features retain mask stability, cross-held-out foreground support, graph fragmentation, actual prior-incompatible components, per-face partition counts, offset/normal/coverage factors, source-normalized displacement, and final plane residual. Controls are source_agreement, low_plane_residual, point_count, stability_only, reliability. Missing values are `None` and must remain NA-last.

`weighted_geometry.generate_weighted_geometry_targets(*, points_fit, raw_indices, source_box6, q0, camera_spec, rgb_mask2d, segmentation_valid, face_weights, RGB_weight, config=GeometryConfig())` is a new explicitly derived R10 solver. It returns the original result schema plus weighted diagnostics. Root supplies uniform weights; this module cannot inspect the cohort or target labels.

`select_exact_half(units, score_key)` accepts only dictionaries `{id,platform,eligible,scores}`. Per-platform eligible count N is original full-G actual-change cases, K=ceil(N/2), higher score first, NA last, same fixed input-hash tie order. Random ranking uses only that hash. It preserves K even if selected scores are undefined; that limitation is reported rather than dropping rows. No alternative coverage is searched.

## Frozen features and dependency

Physical raw returns are deduplicated and coincident float32XYZ grouped. Metric0.10m voxels are deterministically assigned by sample-salted SHA parity; same physical observations cannot cross halves. Each A/B subset independently computes visibility, k16 normals/planarity on its own whole cloud and GrabCut seeds/mask. Shared RGB, source q0, projected source ROI, and source size prior are retained dependencies. These are disjoint point subsets, not independent views/scenes/identity labels. No additional language/image semantic model is introduced.

R10 observed normal alignment/planarity and sourceface windows define candidate face points, but each partition only needs3 distinct points to define a reliability estimate; the old8-point plane-acceptance gate is not imposed twice. Face reliability:

`exp(-abs(medianA-medianB)/max(0.03m,1.4826*(MAD_A+MAD_B)/2)) * abs(normalA·normalB) * clip(min(spanA/spanBase,spanB/spanBase across both tangential axes),0,1)`.

Normals are sign-aligned to the axis before averaging; the cross-partition dot is unsigned. This is disagreement between observations, not preference for an unchanged source residual. Single-face center inference remains conditional on source size. Jointly wrong source/background planes can score1; the synthetic fixture explicitly demonstrates that counterexample.

RGB reliability:

`maskIoU(A,B) * exp(-symmetric_boundary_distance(A,B)/R10_pixel_sigma) * sqrt(fraction(maskA covers heldout visible crop B) * fraction(maskB covers heldout visible crop A)) * ambiguity_factor`.

Mask boundary sets are deterministic row-major samples, maximum128 points each; boundary distance is symmetric nearest-boundary mean. Pixel sigma is unchanged R10 `max(3px,0.05*min(projected source bbox sides))`. Undefined partition mask/crossmodal support gives RGB weight0. Mask agreement alone is not identity evidence. Before any fusion, association_undefined forces all effective face/RGB weights0, effective axis defined flags false and case scoreNA; association_conflicted forces effective weights0 with a defined case score0 on actual-change cases. Only association_supported permits the raw weights to be applied. Raw weights/features/defined flags are always retained for the discriminative analysis. Supported remains geometric compatibility under the source prior, never literal text identity confirmation.

3D component graph radius is `clip(0.1*min(source dimensions),0.05m,0.25m)`. A substantial component has at least `max(3,ceil(0.15*N))` unique physical points. Two substantial components are conflicting only when their centroid distance exceeds `1.25*norm(source dimensions)`, an explicit source-size compatibility prior. `ambiguity_factor=1−fraction of foreground support belonging to any conflicting component pair`. Singleton/small disconnected fragments, depth layering or ordinary disconnected car surfaces do not automatically become wrong-object conflicts. With fewer than2 substantial components no conflict is observed and factor1; if none is substantial association status is undefined. Fragmentation purity/counts and depth separation are diagnostics/controls, not the ambiguity score. Source-size error can defeat this test. No physical target identity is certified.

## Actual loss weighting and fixed targets

Original source/full/no-RGB outputs and original G targets remain byte-preserved. Weighted and uniform-weighted outputs are separate arms. Original q0, source prior, crop/mask, support, frozen face assignments, acceptance and bounds stay the same. Each face mean loss is multiplied by its reliability and averaged with the original number of eligible face terms; there is no division by sum(weights). The RGB objective receives its absolute scalar weight. Prior is unchanged. Thus global confidence is not canceled by normalization, and weighted optimization is not output shrinkage relabelled as a new mechanism.

All-one weights reproduce R10 boxes, preconstraint geometry, support and acceptance bitexact in synthetic tests. All observational weights zero have source as the exact prior minimizer; return source without finite-difference drift. A single zero face does not imply source fallback while another face/RGB constraint remains active. RGB_weight0 removes both the RGB loss calculation and RGB-invalid-hull veto; a disabled term cannot retain a hidden geometry gate. Finite nonconverged results and numerical source fallback keep the original explicit policies. No extra yaw, source retraining, EMA or neural adaptation is introduced.

Root's11 delivered arms: source; unconditional_full; no_rgb_coordinate; weighted_constraints; uniform_weighted_constraints; reliable_50; hash_random_50; source_agreement_50; low_plane_residual_50; point_count_50; stability_only_50. Reliability ranking applies original full-G targets. Uniform face/RGB means must be computed separately from actual contributing/defined constraints with explicit counts and provenance before GT; inactive/undefined entries are not silently inserted into a mean as zeros. Undefined operational per-constraint weights remain zero when applied.

## Execution bounds and claims

Extra two-mask/normal work is restricted to original actual-change cases; all other samples keep identity output and explicit undefined diagnostics. Fresh320 source/G predictions, weights/features, cohort means and selections must seal before labels or scoring. Budget<=20CPUmin, one configuration, no sweeps. Positive/negative/zero effects remain in fixed denominators. This is an exploratory reliability-conditioned geometric-constraint experiment. Scores are heuristic strengths, not probabilities or evidence of verified identity. Matched controls test whether reliability is merely density, original-source agreement, low fitting residual or segmentation stability.

One synthetic preparation error called the existing visibility helper with an extra config argument; it failed before any real input, was corrected to the helper's three-argument API, and the complete fixture was rerun. This was an engineering interface correction, not a scientific configuration revision.

## Preflight correction lineage

Before any actual reliability/G run, independent review found case min included accepted-but-unchanged axes, and a zero RGB weight retained its hull-invalid veto. Root also required correspondence-before-fusion to be an explicit operational gate. All previous code, fixture, contract and lock files are preserved under `history/preflight_v1_before_correspondence_gate/`. The single pre-run correction batch changes case min to actual float32-changed axes, separates defined zero from NA, removes the disabled RGB veto, and preserves raw features while unsupported correspondence zeros effective weights. No old R10 code/configuration or target labels are involved. Synthetic controls directly exercise all four corrected behaviors.
