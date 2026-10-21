# Sparse trajectory feedback engineering

The old 1,321-evaluation experiment was stopped at a complete 12-query boundary by user instruction; those outputs remain private, preserved and unscored. Its later 256-budget full HC64 run was cancelled before execution. The output-Gaussian six-forward variant also remained prepared only. None has a new complete-panel task-performance result.

Fixed top-20 correspondences now persist as indices and weights. Candidate rewards batch the same occupancy, projection, link and guard calculations; event groups, confidence products, grid boundaries and fixed reference statistics are cached. All 64 existing caches passed CPU reward/component/gradient comparisons. In the two-existing-query GPU regression at the same 256-evaluation allowance, all four optimizers preserved final states and boxes bit for bit, all acceptance decisions, final modes and actual decoder/backward counts. New RGB/encoder/expert calls were zero.

The GPU microbenchmark compares 32 reward evaluations after warmup, not deployment inference. Sparse batching sharply reduced repeated scoring, but decoder and backward costs remained: one DirectMC case became slower overall. Report the signed per-arm times in the JSON rather than claiming a uniform speedup.

Public evidence: results/trajectory_engineering/cpu64.json and gpu_two_query.json. Raw tensors, private identities, masks, traces and features stay local. The new propagation-policy study is a method change, not covered by this numerical-equivalence result.
