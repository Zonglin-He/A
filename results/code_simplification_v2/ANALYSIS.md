# Equivalent computation checks (not algorithm component removal)

The latest user clarified the actual goal is simplifying the algorithm, not writing a Method section or merely refactoring code. These completed equivalence checks remain useful implementation evidence; the separate algorithm_pruning_v2 study tests actual step removal.

- Eight-singleton robust initialization: same source-calibrated truncation/scale, same subset enumeration, objective/tie order and gamma writeback. All2030(HC1902+Vid128) deltas, admitted observations, objectives and E boxes bitwise match. On67paired warm/3repeat cases, mean per-query median CPU cost10.3104ms ->2.9994ms. General multiple-candidate/scoring machinery is unnecessary for the current single selected expert track.
- Known native H/W grid adapter: every feature and valid mask exact on67; earlier paired CPU extraction42.9137ms ->6.2757ms. Actual encoder shape is the adapter contract, not inferred from GT.
- Exact directed-affinity caching:2077geometry tensor fields, every requested affinity and all67aux/final boxes/actions/scores match bitwise. Original spatial core87.7640ms ->cache90.1513ms. It does not improve speed here, so it is NOT recommended. Cache hits and memory are fully reported.
- The first attempt shared an unordered-pair cosine matrix via transpose; actual FP32 directional affinity differs by1.1093e-6 on the first case. It failed the strict test and was stopped. Its separate plan/source/failure remain in code_simplification_v1; no tolerance relaxation or production fallback was used.
- Positive cumulative hull and vector prototype computation have their2030-cache audit in algorithm_simplification_v1/EXACT_MATH.json. They are mathematical rewrites, distinct from deleting algorithm steps.

No model/RGB/teacher/GPU/labels/training/parameter update, no production changes. The running full remains bound to its original source and fingerprint. Original robust propagation, cycle screening/weighting, mask links, guard, full-box extent, two anchors, Native2D feedback and DP are all unchanged in this exact study. [All measurements](report.json), [fixed protocol](protocol.json).
