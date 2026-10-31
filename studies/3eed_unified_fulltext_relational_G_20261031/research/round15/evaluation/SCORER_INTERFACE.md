# R15 post-seal eight-arm CPU scoring

Prepared only. No target index payload, annotations, actual predictions or task labels have been read by this scorer preparation. Frozen source remains Waymo-only epoch100; no retraining, PCA, GTyaw or identity oracle enters a method arm.

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/absolute/workspace/crossplatform/env/bin/python \
/absolute/workspace/research/round15/evaluation/score_direct.py \
--predictions /absolute/sealed/formal204/direct_directory \
--manifest /absolute/workspace/research/round15/input/FORMAL204_GPU_INPUTS.json \
--cases /absolute/workspace/research/round15/input/FORMAL_CASE204_PRIVATE.json \
--scoring-index /absolute/R5/SCORING_INDEX_PRIVATE.json \
--release /absolute/root/CPU_SCORE_RELEASE.json --release-sha256 ROOT_RELEASE_SHA \
--output /absolute/new/scored204_attempt_directory
```

The root release is required and must bind `CPU_SCORE_RELEASE=true`, stage `R15_unified_full_description_direct8`, and `all_direct_predictions_and_helpers_sealed_before_GT=true`. It supplies exact hashes for prediction seal, manifest, cases, scoring-index bytes, scorer, source checkpoint path/hash and the literal `geometry_dependency_sha256` map in `score_direct.py`. Nonempty `program_bindings`, `input_bindings`, and `prompt_bindings` maps are byte-verified. `sealed_artifacts` is an explicit list of `{role,directory,seal_sha256}` entries, with the same unique role set in `required_seal_roles`: exactly eight source/mask capture directories, the formal204/612-call semantic directory, and the composite directory. A mask producer's single string hash must occur in the explicit verified program bindings. Engineering output and historical G endpoints are excluded.

All artifact members and producer code bindings are checked before annotation access. Per-case immutable source cache pool/scores/q0 are compared bitexact with composite fields. Selected original proposal arms, main graph association/geometric q1, source anchors, common support/mask/gate, raw/evaluation clamp and changed-axis fields are validated. The scorer writes `SCORER_LOCK.json` before parsing the R5 index payload. Original metadata are then hash-checked and only the target bbox entry is transformed by the frozen R5 helper. Every original expression is scored; missing/corrupt cases abort and remain explicit failures rather than filtered rows.

The prediction contract is the actual assembler's eight roles:

1. source
2. proposal_assoc
3. proposal_geo
4. graph_assoc
5. graph_geo
6. graph_geo_noRGB
7. graph_geo_relation_disabled
8. graph_geo_relation_flipped_diagnostic

The native prediction yaw is zero. Task IoU uses official size clamp `max(size,1e-6)` only in evaluation, float64 general 3D intersection and strict `IoU>.25` / `IoU>.5`. Raw center/size L2 and per-axis absolute errors are separately retained in meters; GT dimensions at GTyaw are not an amodal yaw0 enclosure or a strict achievable IoU upper bound. Pool coverage, if reported, is a post-seal GT-assisted diagnostic; its CPU score/index tie ordering is explicitly not native GPU tie ordering and it is not instance identity ground truth.

Fresh64 and regression140 have separate summaries and contrasts. Each reports Drone/Quad and a supplementary pooled row, including micro and equal-scene IoU/accuracy/coordinate errors, status histograms, query changes and actual coordinate changes. Every paired contrast retains signed sums, gains, harms, ties and tails. Scene cluster descriptive intervals reuse the frozen R12 resampling convention and are not significance or population evidence. Fresh Quad has two scenes and Drone one; the two averaging schemes for one scene are identical evidence.

The fixed progression gate uses **all fresh32 per platform**, no accepted/changed selection. `graph_geo` must exceed **both source and graph_assoc** by more than `1e-8` IoU in micro and equal-scene means, with Acc25/50 nondecreasing in both averaging schemes against both references. Regression cannot rescue it. A pass only enables the separately authorized conditional A→unusedB stage; it does not establish same-instance correspondence or parameter-specific value. Relation-disabled/flipped paths may change association and support, so their scores are whole-G mechanism counterfactuals rather than fixed-identity geometry loss ablations.

All current target VAL contents were historically scored in R7/R8. Recent-content exclusion is not pristine validation. The scorer does not read labels to choose G, thresholds, clauses, samples, accepted subsets or prediction arms.

Outputs `RESULTS_PRIVATE.json` (all204×8 rows, query indices, offline pool diagnostics and private tails), explicit aggregate `SCALAR_SUMMARY.json`, `SCORER_LOCK.json` and `SEALED.json` remain local. Public export requires a later independent explicit scalar projection; the PRIVATE originals, targets, indices, raw source arrays, captions and sensor data are not exportable as-is.
