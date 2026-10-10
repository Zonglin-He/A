# Fixed OPD P5: complete budget, cost and feedback-chain review

On the fixed historical cross-domain cohort, VidSTG has a positive paired parent interval against Frozen at every K, while all four HC2 intervals contain zero. VidSTG K4−K2 and K8−K4 intervals contain zero, so the results do not establish a uniformly increasing return to more observations. Every paired contrast and harmful-source tail is retained, and the original main Uniform4 is unchanged.

All ten deployment streams globally sealed before GT: two targets × four Uniform K budgets plus two pre-GT unified-configuration appendix streams. Each target has 128 historical parent sources, one query per parent, two locked source-blocked orders. The 2,560 logical arrivals contain 2,048 new formal fits and 512 exact complete original K4 stream aliases. Every budget/order resets to its original source; query residual and Adam reset per query, LN inheritance/writeback and configured last output stay fixed. Qualifications are never spliced into formal results. Original main Uniform4 remains selected without P5-result tuning.

Source checkpoints: VidSTG `5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83` → HC2 validation, HC-STVG2 `47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036` → VidSTG test. HC main lr .01, sigma .025, tau .05, 40 rounds, LN writeback1/16; Vid main .03/.1/.25, 10 rounds, writeback1/8. All use M32 antithetic on-policy Gaussian likelihood, detached admitted Top1 frozen-DINO IoU, joint1792 parameters and Native WHEN.

Scores and intervals below are independently read back from all anonymous rows. Query macro equals parent macro here. The 10,000 paired parent bootstrap uses seed20261006 and averages the two fixed orders within each parent before resampling. Intervals are conditional on the historically exposed roster, fixed source checkpoints and order histories; they are not fresh-source stability guarantees. The full analysis retains both individual orders and all six pairwise budget contrasts per target; intervals are not multiplicity-adjusted.

| Target | K | After−Frozen vIoU, pp [95% CI] | After−Before, pp | Before−Frozen, pp | Harm >5/>20pp parents |
|---|---:|---|---|---|---:|
| hc2 | 1 | +0.655 [-0.791, +1.928] | -0.862 [-2.279, +0.360] | +1.517 [+1.197, +1.844] | 12/3 |
| hc2 | 2 | +0.664 [-0.739, +1.859] | -0.810 [-2.196, +0.398] | +1.473 [+1.091, +1.870] | 10/3 |
| hc2 | 4 | +1.346 [-0.046, +2.526] | -0.430 [-1.789, +0.669] | +1.776 [+1.417, +2.157] | 11/2 |
| hc2 | 8 | +0.342 [-1.393, +1.884] | -1.384 [-3.151, +0.134] | +1.725 [+1.345, +2.125] | 15/5 |
| vidstg | 1 | +2.249 [+0.985, +3.483] | +0.684 [-0.432, +1.789] | +1.565 [+1.068, +2.076] | 8/1 |
| vidstg | 2 | +3.649 [+2.205, +5.081] | +2.097 [+0.767, +3.409] | +1.552 [+1.087, +2.039] | 5/1 |
| vidstg | 4 | +3.871 [+2.430, +5.290] | +2.037 [+0.753, +3.317] | +1.834 [+1.327, +2.367] | 6/1 |
| vidstg | 8 | +4.458 [+2.847, +6.078] | +2.738 [+1.232, +4.225] | +1.720 [+1.245, +2.213] | 7/2 |

The inherited/current components describe each budget’s own state trajectory. Before−Frozen is not a Full−alpha0 causal contrast. K changes the observation support and ensuing state history. Admitted-observed versus other GT frames are different frame populations, so their mean differences are descriptive; they are not an independent causal transfer experiment.

| Target | Matched After contrast | vIoU, pp [95% CI] |
|---|---|---|
| hc2 | K2 minus K1 | +0.008 [-1.315, +1.402] |
| hc2 | K4 minus K1 | +0.691 [-0.411, +1.849] |
| hc2 | K8 minus K1 | -0.314 [-1.930, +1.232] |
| hc2 | K4 minus K2 | +0.682 [-0.236, +1.656] |
| hc2 | K8 minus K2 | -0.322 [-1.778, +0.920] |
| hc2 | K8 minus K4 | -1.005 [-2.264, +0.033] |
| vidstg | K2 minus K1 | +1.400 [+0.435, +2.404] |
| vidstg | K4 minus K1 | +1.622 [+0.910, +2.366] |
| vidstg | K8 minus K1 | +2.209 [+1.243, +3.200] |
| vidstg | K4 minus K2 | +0.222 [-0.455, +0.843] |
| vidstg | K8 minus K2 | +0.809 [+0.152, +1.444] |
| vidstg | K8 minus K4 | +0.587 [-0.083, +1.205] |

| Target | K | Recorded fit s/arrival | Recorded shared capture s/arrival | Recorded CPU math s/arrival | Original new DINO calls | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| hc2 | 1 | 1.3261 | 1.3721 | 0.0576 | 125 | 3.641 |
| hc2 | 2 | 1.3505 | 1.4030 | 0.0769 | 250 | 3.606 |
| hc2 | 4 | 1.3562 | 1.2502 | 0.0377 | 0 | 3.617 |
| hc2 | 8 | 1.4125 | 1.6301 | 0.1802 | 962 | 3.568 |
| vidstg | 1 | 0.3222 | 0.8269 | 0.0130 | 105 | 15.935 |
| vidstg | 2 | 0.3339 | 0.8577 | 0.0174 | 210 | 15.967 |
| vidstg | 4 | 0.3471 | 0.7966 | 0.0078 | 0 | 15.955 |
| vidstg | 8 | 0.3430 | 1.1368 | 0.0382 | 840 | 15.967 |

These are actual recorded wall measurements, including synchronous audit recording and original cache conditions. K4 aliases retain their original measurements; they do not incur new fitting time in this P5 execution. Shared capture is not repeatedly added once per variant. Expert forward receipts can describe reused cached calls; they are not cold deployment latency. Decode/corruption, frozen STVG forward and recorded expert-forward components, available-row counts, actual backward rounds and observed/unobserved denominators remain in COST.json and ACTUAL_ROOT_BUDGET_CONTRASTS.json.

| Unified appendix target | After−Frozen vIoU, pp [95% CI] | Unified−main K4, pp [95% CI] |
|---|---|---|
| hc2 | +0.898 [-0.335, +2.018] | -0.448 [-0.970, +0.083] |
| vidstg | +3.920 [+2.429, +5.364] | +0.050 [-0.149, +0.296] |

The unified appendix was fixed before P0 GT: lr .03, sigma .1, tau .25, 10 rounds, writeback1/16, M32, K4. It neither replaces main configurations nor constitutes a single-factor K contrast. No parameters or budget were selected from these results.

Six distinct post-hoc main-budget cases retain successes, current-query harms and cases in which expert reward rises while task score worsens. The case identity and support are verified against preserved payload/input hashes. Saved original GPU actions are evaluated against official GT only after global seal; offline best-sample curves never select deployment outputs. Private RGB redecoding must match the original pixel SHA. The full case signal chains and all negative rows/tails/strata stay in the result package. Cases illustrate mechanisms and are not efficacy estimates.

Actual root counts: `{"actual_qualified_formal_pairs_bitwise": 20, "complete_math_dictionary_exact": 2560, "dense_metric_scalars": 23040, "exact_stream_reuse_rows": 512, "input_bindings_checked": 2560, "logical_arrivals": 2560, "new_formal_fits": 2048, "observed_unobserved_checks": 5120, "payload_bytes": 1989956682, "qualified_alias_complete_input_comparisons": 4, "qualified_new_formal_pairs_bitwise": 16, "qualified_original_alias_pairs_bitwise": 4, "rounds": 56320, "second_opaque_prediction_input_receipt_checks": 10240, "state_coordinates": 4587520, "strict_complete_input_source_checks": 2560, "strict_original_P0_input_receipt_readbacks": 512, "strict_original_pre_revision_runtime_receipt_readbacks": 256}`. Every complete saved math dictionary is dispatched by its original pinned receipt, full1792 state/query-and-Adam reset/LN inheritance/writeback/input/Native interval is read back, dense metrics and observed/unobserved means are independently recomputed, and prediction/input/receipt SHA checks repeat after the full scan. The original unsealed CPU helper failures (legacy receipt without bytes, duplicate immutable statistics write, equality of differently serialized K4 input containers, and the original pre-revision Vid K4 runtime binding) remain preserved; separately pinned readback revisions preserve the original predictions/scientific runtime and compare existing statistics exactly. K4 alias/control input pairs are checked against the strict original hash-bound NPZ bridge across source-state hash, frame/pixel identity, Native interval/boxes and full packed experts; new formal input SHA checks remain exact.

Public arithmetic reproduces all anonymous source statistics, paired intervals, negative tails, strata, recorded costs and evidence bindings. It does not recreate private inference/GT/sample checks or independently implement the entire decoder Jacobian. The root makes no new model/optimizer calls and does not claim CUDA transcendental proof or future numerical/OOM safety. Private RGB/query/caption/GT geometry/boxes/actions/weights/fit/gradient/Adam/cache payloads are excluded from publication.

Decision: close only original P5 after actual remote content verification and archive maintenance, then implement and qualify the original finite P6 existing temporal-signal/head and post-deployment-seal GT-head-oracle/failure diagnostic. No retuning, new expert/algorithm/scorer/gate/memory or method promotion. EATA and all historical paused queues remain paused; the entire paper suite is still incomplete.
