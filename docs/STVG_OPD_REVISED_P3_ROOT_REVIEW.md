# Original fixed OPD P3: parameter scope and persistent state

Full exceeds Query-only in HC2 same-domain corruption and VidSTG cross-domain clean; the added query residual exceeds LN-only clearly only in VidSTG cross-domain clean. Full minus joint alpha0 is positive in HC2 cross-domain clean, with intervals containing zero in the other three setting means. The complete joint design is not uniformly better than every ablation.

Both targets use the original exposed 128-parent confirmation cohorts, one query per parent. Cross-domain clean has both original orders; same-domain has five physical burst families at 5%, one order. All four arms/conditions/orders/directions globally sealed before GT. There are 7,168 logical fits: 5,376 new formal fits and 1,792 exact complete-stream Full aliases. Qualification predictions are excluded.

TA-STVG sources remain `TASTVG_VidSTG.pth` (5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83) and `TASTVG_HCSTVG2.pth` (47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036). Cross uses the other dataset source; same-domain uses its own source. HC2 lr/sigma/tau/rounds/writeback are .01/.025/.05/40/1/16; VidSTG .03/.1/.25/10/1/8. M=32 antithetic Gaussian actions, original Uniform4, admitted Top1 frozen DINO, per-query Adam/residual reset and final-round output remain fixed. Query-only updates 256 query parameters; LN-only updates 1,536 LN parameters; joint alpha0 and Full update 1,792. Joint alpha0 discards the LN writeback; all streams start independently from source per condition/order/arm. Native WHEN is fixed and every temporal delta is zero.

All numbers below are vIoU percentage points with 10,000 paired parent bootstrap draws (seed 20261006), conditional on these saved histories. Conditions/orders are averaged within query before parent aggregation; query-macro equals parent-macro because this cohort has one query per parent. Intervals are not multiplicity adjusted.

| Setting | Full − Query-only | Full − LN-only | Full − joint alpha0 |
|---|---:|---:|---:|
| P3_hc2_cross_clean | +1.182 [-0.070, +2.247] | -0.170 [-0.734, +0.296] | +0.238 [+0.011, +0.488] |
| P3_hc2_same_5percent | +3.942 [+2.686, +5.152] | +0.009 [-0.490, +0.468] | +0.217 [-0.279, +0.673] |
| P3_vidstg_cross_clean | +3.322 [+2.048, +4.608] | +0.464 [+0.090, +0.909] | +0.258 [-0.182, +0.691] |
| P3_vidstg_same_5percent | +1.438 [-0.676, +3.415] | +0.420 [-0.034, +0.936] | +0.480 [-0.257, +1.211] |

| Setting / arm | After vIoU (%) | Total vs Frozen | Current vs Before | Inherited Before vs Frozen | >5 / >20 pp harmful parents |
|---|---:|---:|---:|---:|---:|
| P3_hc2_cross_clean / query_only | 22.320 | +0.164 [-0.577, +0.829] | +0.164 [-0.577, +0.829] | +0.000 [+0.000, +0.000] | 5 / 2 |
| P3_hc2_cross_clean / LN_only | 23.672 | +1.517 [+0.239, +2.524] | -0.262 [-1.496, +0.654] | +1.779 [+1.396, +2.175] | 8 / 1 |
| P3_hc2_cross_clean / joint_alpha0 | 23.264 | +1.108 [-0.303, +2.297] | +1.108 [-0.303, +2.297] | +0.000 [+0.000, +0.000] | 11 / 2 |
| P3_hc2_cross_clean / on_policy | 23.502 | +1.346 [-0.046, +2.526] | -0.430 [-1.789, +0.669] | +1.776 [+1.417, +2.157] | 11 / 2 |
| P3_hc2_same_5percent / query_only | 29.082 | -0.011 [-0.726, +0.604] | -0.011 [-0.726, +0.604] | +0.000 [+0.000, +0.000] | 3 / 1 |
| P3_hc2_same_5percent / LN_only | 33.015 | +3.921 [+2.628, +5.178] | +0.137 [-1.187, +1.360] | +3.784 [+3.173, +4.429] | 8 / 2 |
| P3_hc2_same_5percent / joint_alpha0 | 32.807 | +3.714 [+2.265, +5.126] | +3.714 [+2.265, +5.126] | +0.000 [+0.000, +0.000] | 8 / 3 |
| P3_hc2_same_5percent / on_policy | 33.024 | +3.931 [+2.497, +5.304] | +0.045 [-1.428, +1.405] | +3.885 [+3.263, +4.550] | 10 / 3 |
| P3_vidstg_cross_clean / query_only | 15.015 | +0.549 [+0.114, +1.025] | +0.549 [+0.114, +1.025] | +0.000 [+0.000, +0.000] | 3 / 0 |
| P3_vidstg_cross_clean / LN_only | 17.873 | +3.407 [+2.043, +4.750] | +1.616 [+0.422, +2.796] | +1.791 [+1.286, +2.328] | 7 / 1 |
| P3_vidstg_cross_clean / joint_alpha0 | 18.079 | +3.613 [+2.219, +4.969] | +3.613 [+2.219, +4.969] | +0.000 [+0.000, +0.000] | 5 / 1 |
| P3_vidstg_cross_clean / on_policy | 18.337 | +3.871 [+2.430, +5.290] | +2.037 [+0.753, +3.317] | +1.834 [+1.327, +2.367] | 6 / 1 |
| P3_vidstg_same_5percent / query_only | 24.331 | +0.737 [+0.072, +1.530] | +0.737 [+0.072, +1.530] | +0.000 [+0.000, +0.000] | 2 / 0 |
| P3_vidstg_same_5percent / LN_only | 25.349 | +1.755 [-0.468, +3.853] | +0.338 [-1.762, +2.283] | +1.416 [+0.674, +2.169] | 14 / 5 |
| P3_vidstg_same_5percent / joint_alpha0 | 25.289 | +1.695 [-0.485, +3.775] | +1.695 [-0.485, +3.775] | +0.000 [+0.000, +0.000] | 12 / 6 |
| P3_vidstg_same_5percent / on_policy | 25.769 | +2.175 [-0.114, +4.383] | +0.410 [-1.782, +2.472] | +1.765 [+0.988, +2.566] | 15 / 6 |

LN-only retains much of the Full gain in HC2; this does not establish an incremental query benefit there. Query-only has exactly zero inherited delta because the query residual resets and LN stays at source. Joint alpha0 also has zero inherited delta, yet its current adaptation can be useful. Full Before−Frozen is a decomposition of its own trajectory, not the causal Full−alpha0 contrast: turning off carry changes subsequent adaptation states and current effects. A small paired Full−alpha0 difference must not be relabelled as the larger inherited component.

Severe failures, condition-specific results, per-order results, observed/unobserved effects, expert quality, motion/duration strata and query types remain in ROOT_STATISTICS, ALL_PARENT_EFFECTS, FAILURE_STRATA, COST and complete anonymous stage rows. No failing source is dropped or used to retune.

| Setting / arm | Stored fit wall seconds / arrival | CPU math seconds / arrival |
|---|---:|---:|
| P3_hc2_cross_clean / query_only | 1.468057 | 0.114888 |
| P3_hc2_cross_clean / LN_only | 1.262824 | 0.115838 |
| P3_hc2_cross_clean / joint_alpha0 | 1.484230 | 0.116350 |
| P3_hc2_cross_clean / on_policy | 1.356214 | 0.037666 |
| P3_hc2_same_5percent / query_only | 1.399904 | 0.112120 |
| P3_hc2_same_5percent / LN_only | 1.201765 | 0.113431 |
| P3_hc2_same_5percent / joint_alpha0 | 1.412621 | 0.113790 |
| P3_hc2_same_5percent / on_policy | 1.440677 | 0.108448 |
| P3_vidstg_cross_clean / query_only | 0.341442 | 0.023827 |
| P3_vidstg_cross_clean / LN_only | 0.298703 | 0.024207 |
| P3_vidstg_cross_clean / joint_alpha0 | 0.342832 | 0.024146 |
| P3_vidstg_cross_clean / on_policy | 0.347119 | 0.007815 |
| P3_vidstg_same_5percent / query_only | 0.338269 | 0.024534 |
| P3_vidstg_same_5percent / LN_only | 0.297805 | 0.024851 |
| P3_vidstg_same_5percent / joint_alpha0 | 0.339874 | 0.024807 |
| P3_vidstg_same_5percent / on_policy | 0.342704 | 0.024557 |

Stored fit wall time includes synchronous numerical recording. Shared capture is not multiplied by four; alias timing is historical cost, not new GPU compute or cold deployment latency. Full and ablations were run in serial historical streams; these cost comparisons are descriptive rather than matched hardware microbenchmarks.

The actual root checked 7168 complete saved mathematical dictionaries, 179200 rounds, 12845056 full-state coordinates, 5376 matched input comparisons, 64512 independent dense scalars and 14336 observed/unobserved checks. The largest independent dense discrepancy is 2.44249065e-15. Receipt-aware dispatch preserves original P0/P2/006 and new scoped audit dictionaries exactly. All inactive parameters, active Adam scopes, query reset, source reset, LN inheritance/writeback and Native intervals are checked.

Six distinct success/harm/expert-mismatch cases were selected post hoc after all parent rows were read. Each saved Full signal is accompanied by three complete scope-control signal chains on the same query/input/order. Sample-vs-GT arithmetic is checked independently; recorded GPU actions are used when saved, while historical P0 sample logits use an explicitly labelled offline float64 sigmoid. Offline best samples never select a deployment output. Four report plot pairs and six private real RGB case sheets were actually viewed. These cases describe mechanisms and are not independent efficacy estimates.

Public code/configuration/anonymous rows/scalars/plots include every negative result. No RGB/query/caption/GT geometry/weights/raw logits/boxes/actions/gradient/Adam/cache payload is exported. The root performs no model/optimizer calls, proves saved parameter arithmetic and state/dense integrity, and does not independently reimplement the entire decoder Jacobian or prove CUDA transcendental kernels. Historical numerical assertion failures remain preserved; bounded audit bridges do not change the science.

Decision: close only P3 after actual remote publication and archive verification, then continue the original fixed P4 robustness phase. No method promotion, retuning, new algorithm/expert or restoration of paused EATA/historical queues. P4–P6 retain their own GPU qualification/global seal/CPU/root/view/public/archive obligations; paper suite remains incomplete.
