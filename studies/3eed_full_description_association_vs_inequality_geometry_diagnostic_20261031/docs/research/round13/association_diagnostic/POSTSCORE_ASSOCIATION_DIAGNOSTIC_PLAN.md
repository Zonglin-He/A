# R13 post-score annotated spatial association diagnostic — plan only

State: PREPARED, not executed. This plan grants no label access. The original task-score release does not authorize all-object annotation access; a new explicit root diagnostic release is required before even the schema-only read.

## Motivation and fixed scope

Root's post-score inspection of new_00067 suggests A's chosen independent 2D mask may coincide with B's support mask while A maps it to a worse original 3D proposal. This observation is a competing explanation, not proof of either correct text identity or wrong mapping. A negative target IoU must not automatically be called a semantic identity error.

Bounded to the same sealed116 cases, separately new_content108/regression8 and platform. No new query selection, images, point clouds, model, G, optimization, mask loss, registration or new task prediction. Main task scores, thresholds and progression decision remain unchanged.

## New release and evidence binding

Require a separate root release `stage=R13_postscore_annotated_spatial_association_diagnostic`, `POSTSCORE_DIAGNOSTIC_RELEASE=true`, `all_task_predictions_and_main_scores_sealed=true`, `method_or_primary_metric_modification_permitted=false`.

It must bind this plan, eventual diagnostic program/dependencies, original R5 scoring-index SHA, all source/A/B/composite/task-score seals, exact case116/strict9 manifest and actual source checkpoint. Every seal/member and input/program binding is checked before the scoring-index JSON payload or any original metadata is opened. This plan is the pre-label measurement rule; no tuning from diagnostic results.

## Stage 1: schema-only qualification after new release

Read only the structural schema of the already hash-bound original metadata containers. Record aggregate key names, value types, list/object counts and bbox field presence/shape, without printing or exporting captions, category values, object-ID values or numeric box values. The metadata container itself can physically include labels; this is disclosed label access, not a target-label-free preparation claim.

Expected candidate source is the actual same-frame `ground_info` annotation list. Do not infer an unobserved all-scene object list from a generic field. If another explicit list is discovered, record its structural presence but do not silently expand to it: this plan uses `ground_info` only. Do not infer instance IDs from array position, class or caption. Optional explicit ID key presence/types are reported only; geometry grouping below does not require IDs and never declares true physical identity.

If bbox_3d is absent, not an array, platform dimensions differ from official expected shape, or no non-target annotated geometry is available, the affected diagnostic is NA. Malformed/missing-other counts remain visible. An invalid/missing target or inconsistent target join is an explicit failure rather than silently changing target/sample.

## Stage 2: fixed annotation geometry grouping

For each actual ground_info entry with a valid bbox_3d, use the frozen R5 `independent_core_v2.transform_gt`: Drone/Quad raw9D truncates to first7 official yaw-only coordinates; Drone +1.8 shift in original convention; native float32 transformed target. No tail-field rotation interpretation and no predicted yaw update.

Deduplication is **exact elementwise equality of the transformed float32 seven-vector** (`numpy.array_equal` semantics; signed zero numerically equal), before any IoU computation. No IoU threshold, GT-directed radius, semantic text rule, nearest center or task-driven clustering is used. Retain group membership/counts privately. The primary target is the immutable original scoring-index object_index; its exact-geometry group is the target group. Other groups are distinct annotated geometry groups, not certified physical identities. Geometrically equivalent boxes with different parameterization (e.g. yaw+pi) are not merged; disclose this limitation rather than introduce an unregistered IoU merge.

This removes repeated identical boxes attached to multiple descriptions, but does not establish all objects in the RGB/point cloud have annotations. If only the target geometry group is available, other-max/margin/classification are NA. Optional IDs, if structurally present, are descriptive provenance only and do not override this exact grouping.

## Stage 3: three immutable original proposals

Evaluate only sealed original-source candidates:

- q0: original pure-text source argmax;
- fullA: sealed A_full_consensus query;
- nounA: sealed A_noun_consensus query.

Each box must exactly match source_pool[stored query] in its own immutable forward cache. Preserve repeated queries/boxes; do not select an oracle query, rerank candidates or replace a fallback. Use existing official size clamp for evaluation and native prediction yaw0. Compute general3D float64 IoU against the one target7D and each deduplicated annotated-other7D using the frozen R5 core/OBB implementation. No 2D mask projection IoU or new image/point measurements.

Per candidate report continuous:

- target_IoU;
- annotated_other_max_IoU, or NA when no distinct valid other group exists;
- signed margin target_IoU − annotated_other_max_IoU;
- annotated group counts and malformed/missing annotation counts.

These are offline annotation-relative spatial compatibility measurements, not literal referring-identity ground truth.

## Fixed high-confidence spatial categories

Apply only when target and other-max are both defined:

- `target_spatially_dominant`: target_IoU > .25 AND signed margin > .10;
- `annotated_other_spatially_dominant`: other-max_IoU > .25 AND −signed margin > .10;
- `unconfirmed_spatial_relation`: all remaining defined cases;
- `NA_no_distinct_annotated_other_geometry`: no valid distinct other group;
- `NA_schema_unavailable`: required observation schema unavailable.

Strict inequalities at .25/.10. Unconfirmed/NA is not "wrong". Annotated-other dominance is not a definitive wrong-language-target label; boxes can overlap, annotations can be sparse and the source output is yaw0. No category threshold scan or gate adoption follows.

## Mask/query strata registered before diagnostic labels

Use already sealed A/B outputs only:

- A_full_consensus selected_mask_index versus B selected_mask_index: both-defined and equal / both-defined and different / A undefined / B undefined / both undefined. Equality is same frozen detector mask, not semantic truth.
- fullA query versus q0: equal / different; fullA raw box versus q0 box: equal / different. Query changes can leave geometry identical.
- Primary mechanism stratum: same defined A/B mask AND fullA query changed. Report its fullA−q0 target-IoU/margin changes and spatial-category transitions.
- Other strata retain the same complete116 and within-cohort/platform denominators. The motivating new_00067 is identified privately but receives no different rule.
- Optionally compare fullA versus nounA under the same three immutable boxes; no new semantic prediction.

If A maps a mask shared with B to a lower-IoU candidate, report this as "same selected 2D observation, different original 3D proposal; target-overlap loss at mapping/readout stage is compatible with the evidence". It does not establish the 2D mask was the literal target, nor causally prove a text model/mapping algorithm is solely responsible.

## Outputs and non-adoption boundary

Write anonymous aggregate SUMMARY.json and CASES_PRIVATE.json, SCHEMA_KEYS_SUMMARY.json, SCHEMA_QUALIFICATION_PRIVATE.json and SEALED.json. Public aggregates contain no captions, physical scene/frame IDs, ID values, GT coordinates or raw predictions. Per-case/private rows can retain internal IDs and spatial scalar values for audit; GT box arrays need not be re-exported.

Separate new108 from exposed regression8, platform and scene counts; report micro counts and NA/missingness, continuous target/other/margin summaries and fixed category transitions. Do not compute an AUC, claim correctness from response changes, or revise method/main task metrics. This is a bounded post-score descriptive diagnosis, not a new method-selection validation or proof of literal identity.
