# R13 spatial diagnosis v2: explicit released annotated-other adapter

This is a **pre-computation** schema adapter amendment approved by root after schema-only inspection. v1 plan SHA34cbc9157cb57fabfdd3fc67155c1753fd597848bd40b887208fcd3144db7f52 and its NA branch remain intact. No actual diagnostic IoU/category results have been computed. Only schema keys/types/counts/shapes were inspected under separate written schema releases.

## Qualified actual schema and source

116 metadata containers, each ground_info length1. Target: ground_info[immutable scoring-index object_index].bbox_3d is raw9D. Other annotations: **that same entry.others[].bbox_3d_other**, each raw7D, with class_other strings unused by spatial geometry. All116 others_num equal actual list length; 425 other entries have raw7D structure. There are no explicit object-ID keys. Official released statistics/analyze_dataset_statistics.py processes target bbox_3d[:7] and others[].bbox_3d_other[:7] in the same raw-point coordinate system. This is a list of annotated other geometry, not a complete scene inventory or object-identity truth.

Scope is only current Drone/Quad. `transform_other7` validates finite positive-size raw7D, copies to float64, adds+1.8 once to raw z for Drone, leaves Quad unchanged, then casts official7D to float32. It never fabricates a9D observation or pitch/roll/tail fields. Pure synthetic tests compare128 float64-JSON-like raw7D cases against frozen R5 transform_gt with arbitrary unused tails and require exact output equality. Other platforms are rejected. Target raw9D continues to use unmodified R5 transform_gt.

All other v1 rules remain: exact transformed float32 elementwise7D dedup before IoU; target exact group removed from other groups; no IoU-based geometry merge, no inferred IDs, no GT-reselected query. Use q0/fullA/nounA from the exact original sealed source pool. All primary outputs/metrics/G/thresholds remain immutable. As an engineering bridge, recomputed target IoU for these existing selected boxes must match the already sealed primary target IoU within1e-12; this never substitutes or changes primary metrics.

Continuous target/annotated-other maxIoU/margin and strict IoU>.25+margin>.10 categories remain offline descriptive. Missing distinct other geometry is NA; malformed other schema is separately retained and cannot certify target dominance. No unresolved/NA observation is counted as wrong semantic identity.

## Independent predicted-category addendum

For fullA mask, nounA mask and B support mask, load only frozen mask-cache class_ids (no new detector). Compare COCO predicted class ID to the family implied by the original input caption's existing first lexicon noun via unchanged R13 support parser. Retain undefined/unsupported family separately. This is **predicted-category compatibility**, not GT category or literal identity truth. Do not filter any arm/sample, change mask/query, recompute main metrics or reinterpret an incompatible prediction as definitive error.

Same A/B mask plus q-change is a predeclared stratum. A different original proposal for the same mask can yield a worse target overlap; neither that overlap loss nor a predicted category mismatch proves the literal text target was selected incorrectly. The already inspected new_00067/regression_00006 remain ordinary cases under this uniform rule.

The actual diagnostic program and all schema/source/scoring seals are bound in PLAN_V2_LOCK.json and the eventual new root compute release. Preparing code/fixtures does not authorize actual numeric annotation processing. A separate root release is still required.
