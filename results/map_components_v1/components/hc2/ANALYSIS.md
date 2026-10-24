# TA to HC: complete current-MAP components and final-DP control

All 1,902 queries / 227 parents are sealed, independently officially scored, and cold-replay audited. This is the complete fixed cohort, not a prefix result. All FullMAP arrays reproduce the existing main MAP result exactly. No new model, Source, prefix, expert or GPU execution.

## Five-row progression

| Configuration | vIoU (%) | R.3 (%) | R.5 (%) | vIoU delta vs original Source (pp) |
|---|---:|---:|---:|---:|
| Source | 20.704678 | 29.022082 | 8.937960 | +0.000000 |
| EightReplacement | 24.709592 | 36.698212 | 12.355415 | +4.004913 |
| SharedCorrection | 24.868872 | 37.013670 | 13.827550 | +4.164194 |
| SpatialInference | 25.847764 | 39.011567 | 15.036803 | +5.143086 |
| FullMAP | 26.731488 | 40.378549 | 16.929548 | +6.026810 |

Original Source retains its native per-offset MAP followed by physical hull. The three intermediate adapted rows use the same q0 joint physical MAP; FullMAP uses q1 joint MAP and actually re-solves spatial DP on that interval. Thus EightReplacement minus original Source includes a temporal-readout change. It is not a pure estimate of the eight expert boxes. The explicit SourceJointMAP control is 24.039097 vIoU / 36.225026 R.3 / 11.934805 R.5. EightReplacement improves its mean vIoU by 0.670495pp; no bootstrap CI for this extra pair is claimed.

## Paired effects and negative outcomes

| Contrast | vIoU delta (pp) | 95% parent-cluster CI | Improved / harmed / tied | Severe v loss >5pp |
|---|---:|---|---|---:|
| EightReplacement-Source | +4.004913 | [+3.573244, +4.454615] | 1265 / 447 / 190 | 125 |
| SharedCorrection-EightReplacement | +0.159280 | [-0.056526, +0.382438] | 1139 / 575 / 188 | 148 |
| SpatialInference-SharedCorrection | +0.978892 | [+0.637944, +1.331710] | 841 / 728 / 333 | 180 |
| FullMAP-SpatialInference | +0.883724 | [+0.578536, +1.193539] | 738 / 530 / 634 | 170 |
| FullMAP-Source | +6.026810 | [+5.348843, +6.740112] | 1235 / 484 / 183 | 264 |
| FullMAP-SourceJointMAP | +2.692391 | [+2.133693, +3.275670] | 1126 / 606 / 170 | 291 |
| FullMAP-FinalIndependent | +0.954760 | [+0.851435, +1.061092] | 1127 / 424 / 351 | 10 |

Shared correction versus eight-frame replacement has a positive mean but its vIoU CI crosses zero. Its observed-frame sIoU decreases by 7.088303pp (CI [-8.119354, -6.051199]) while unobserved-frame sIoU improves by 0.916466pp (CI [0.382726, 1.434468]). This tradeoff must not be hidden by the mean. Spatial inference improves mean vIoU by 0.978892pp; temporal evidence plus final interval/path update adds 0.883724pp. The latter observed-frame sIoU changes by -0.152755pp (CI [-0.317949, -0.004313]); these are conditional progression contrasts, not strict deletion effects.

FullMAP versus original Source has 267 severe temporal losses, 264 severe vIoU losses and 396 queries in their union (counted once). Its mean gain is +6.026810 vIoU pp; against the matched SourceJointMAP the gain is +2.692391pp. Both mean gains and tails remain reported. All signed per-query outcomes are retained in the TSVs.

## Final DP with fixed upstream evidence

FinalIndependent uses exactly the same candidates, validity, auxiliary trajectory, pooled features, q1 and final MAP interval as FullMAP, and selects the maximum valid unary independently at each frame. FullMAP retains the full path score and final exact DP. This targets the final trajectory solver; it does not establish the independent contribution of the first DP.

| Solver | vIoU (%) | R.3 (%) | R.5 (%) |
|---|---:|---:|---:|
| FinalIndependent | 25.776728 | 38.538381 | 15.194532 |
| FullMAP | 26.731488 | 40.378549 | 16.929548 |

Final DP improves vIoU by +0.954760pp, CI [0.851435, 1.061092]; 1,127 queries improve, 424 decline, 351 tie, and 10 decline by more than 5pp. Temporal metrics are identical by construction. Unobserved-frame sIoU improves by +1.938439pp, CI [1.760514, 2.120827]; observed-frame sIoU changes by -0.068456pp, CI [-0.186449, 0.048409].

## Duration groups

| Contrast | Short <=5s (544 queries): v delta / CI | Long >5s (1358 queries): v delta / CI |
|---|---|---|
| EightReplacement-Source | +2.687126 [+1.817025, +3.545493] | +4.532805 [+4.015531, +5.063404] |
| SharedCorrection-EightReplacement | -0.278307 [-0.641356, +0.061186] | +0.334573 [+0.055359, +0.604667] |
| SpatialInference-SharedCorrection | +1.293586 [+0.676245, +1.908987] | +0.852829 [+0.479383, +1.259232] |
| FullMAP-SpatialInference | +0.372500 [-0.138711, +0.928671] | +1.088515 [+0.718546, +1.445169] |
| FullMAP-Source | +4.074905 [+2.711878, +5.411140] | +6.808721 [+5.963675, +7.675275] |
| FullMAP-SourceJointMAP | +2.179661 [+1.167613, +3.207907] | +2.897785 [+2.198419, +3.610300] |
| FullMAP-FinalIndependent | +0.791647 [+0.600365, +0.972839] | +1.020101 [+0.895455, +1.146018] |

All CIs are pointwise parent bootstrap intervals, not simultaneous familywise guarantees. The datasets have historical exposure; no unseen-test or fresh confirmation claim, outcome tuning or parameter re-selection is made.

## Verification and costs

The independent official scorer performed 13,314 dense checks (maximum numerical difference 1.3322676295501878e-15). After scoring, the reporter verified 3,804 prediction/receipt SHA bindings, cold-replayed all arrays exactly, and recomputed all means. The separate post-audit scalar exporter produced table3.tsv and final_DP_control.json with no additional GT scoring.

Worker 341748/start240742330: 1,902 cached query computations sum to 88.396076 conditional CPU wall seconds. This excludes parent-cache acquisition, outer setup and other stage overhead; shared-host scheduling and pauses may be included. It is not study elapsed time or uncached Source-plus-expert deployment latency.

The original CG-Vid full and its MAP consumer continue independently. Vid component results and K2/4/8/16 budget results are not yet claimed complete. All old scientific sources, plans, outputs and failures are preserved.
