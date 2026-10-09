# Fixed OPD full-query P1: actual root completion evidence

All 41,355 deployment predictions were globally sealed before target GT scoring: HC-STVG2 validation 3,482 clip/queries from 237 parent movies in three orders, and VidSTG 10,303 queries from 732 videos in three orders. TA-STVG source checkpoints remain fixed in the opposite dataset direction. Existing complete baselines are exact matched reuse.

HC2 uses lr .01, sigma .025, tau .05, 40 rounds and LN writeback 1/16; VidSTG uses .03, .1, .25, 10 rounds and 1/8. Both use the original Uniform4 admission/Top1, one frozen DINO, 32 antithetic actions and 1,792 joint parameters. Native WHEN, query-residual/Adam reset, original histories and final-round readout remain fixed.

## Official full-query table

Values are percentages. Each query is averaged over all three original orders. Parent means separately give each movie/video equal weight; 10,000 paired parent bootstrap resamples use seed 20261008. The 32 development parent sources in each dataset are excluded in the separate primary analysis below. These datasets have historical exposure; no fresh unseen or globally optimal configuration claim is made.

| Target | Method | Query m_vIoU | Query m_tIoU | Query m_sIoU | vIoU > .3 | vIoU > .5 | Parent m_vIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| hc2 | spatial opd | 22.583 | 40.311 | 53.625 | 32.970 | 11.784 | 23.510 |
| hc2 | source only | 20.017 | 40.311 | 48.550 | 27.570 | 7.639 | 21.072 |
| hc2 | tent stvg | 16.598 | 33.333 | 48.536 | 20.132 | 5.035 | 17.437 |
| hc2 | sar stvg | 19.939 | 40.152 | 48.556 | 27.465 | 7.486 | 20.994 |
| hc2 | dino refine | 20.673 | 40.311 | 43.631 | 28.920 | 9.075 | 21.888 |
| hc2 | Supervised in-domain reference | 29.056 | 57.280 | 49.332 | 45.951 | 14.302 | 29.895 |
| hc2 | eata stvg (HC-only supplement) | 20.017 | 40.312 | 48.550 | 27.570 | 7.639 | 21.072 |
| vidstg | spatial opd | 14.097 | 31.703 | 39.855 | 17.904 | 7.037 | 15.376 |
| vidstg | source only | 11.129 | 31.703 | 32.758 | 12.831 | 3.300 | 12.065 |
| vidstg | tent stvg | 8.721 | 24.973 | 32.767 | 9.774 | 2.271 | 9.512 |
| vidstg | sar stvg | 11.129 | 31.703 | 32.758 | 12.831 | 3.300 | 12.065 |
| vidstg | dino refine | 13.985 | 31.703 | 35.710 | 18.024 | 6.920 | 15.372 |
| vidstg | Supervised in-domain reference | 19.296 | 42.750 | 42.325 | 28.137 | 13.103 | 21.449 |

## Primary analysis excluding tuning parents

The exclusion is at the parent-source level: HC2 205 parents/2,997 queries; VidSTG 700 parents/9,918 queries. This does not erase prior historical exposure.

| Target | Parent m_vIoU | OPD − Source Only (pp), 95% paired CI | Current query component (pp), CI | Inherited component (pp), CI |
|---|---:|---:|---:|---:|
| hc2 | 23.152 | +2.409 [+1.891, +2.907] | +0.355 [-0.133, +0.826] | +2.054 [+1.894, +2.218] |
| vidstg | 15.464 | +3.329 [+3.017, +3.646] | +1.629 [+1.387, +1.886] | +1.699 [+1.527, +1.874] |

The full-population paired contrast against DINO Refine is positive in HC2 (+1.622 pp parent m_vIoU, CI [1.200, 2.038]) and inconclusive in VidSTG (+0.004 pp, CI [−0.212, 0.215]). OPD improves relative to Source Only in both directions, but P1 does not establish improvement over DINO Refine in both datasets. The supervised in-domain reference is separately labeled; it is not a source-only TTA baseline. EATA has only the already completed HC2 supplement; its missing direction and media/Fisher remain user paused.

## Feedback, failures and real cases

All positive, negative and no-update arrivals, full parent tails and descriptive expert-quality/duration/motion/query-type strata are retained. Native timing is exactly unchanged. Current and inherited components are a measured Before/Frozen/After decomposition, not an isolated causal mechanism experiment.

The actual root reviewed three public plot pairs and six distinct private query cases (36 Frozen/Before/After frame panels). The deterministic post-hoc extremes show successful correction with accurate experts and harmful identity switches when an admitted expert covers a distractor. In those failures, agreement with the expert rises while GT overlap falls. Complete per-round Gaussian sample/teacher/central-output scalar chains are public; RGB, query text, GT coordinates and raw actions are not. Cases are descriptive, not prevalence or independent efficacy estimates. The best-sampled GT box is an offline diagnostic, never a deployed oracle step.

## Actual cost

Costs retain measured fit, shared capture and CPU audit/score components. Inputs reuse old matched DINO evidence; zero new DINO calls is not zero expert cost, and cached measurements are not cold end-to-end latency.

| Target | Synchronized fit (s) | Shared capture (s) | Saved independent CPU math (s) | Backward calls | Peak allocated GPU bytes |
|---|---:|---:|---:|---:|---:|
| hc2 | 14461.627 | 11783.018 | 397.878 | 397158 | 4072293376 |
| vidstg | 10546.433 | 21313.929 | 269.950 | 219950 | 21685415424 |

## Actual validation and next authorized phase

The root read all 258,576 anonymous baseline/OPD logical rows, all parent sources and all 41,355 saved fits. Saved mathematical dictionaries recompute exactly under their original registered audit revisions. The root checked 148,216,320 state coordinates, 372,195 dense scalars and 165,420 observed/unobserved checks; maximum independent dense error is 4.44e−15. The original GT parser provenance is reused; a complete decoder Jacobian was not independently reimplemented. Earlier failed cross-precision assertions and numerical/allocator recovery receipts remain preserved. No predictions, fitted parameters, original locks or rosters were rewritten.

P1 closure requires actual public byte/ref verification and archive receipts, separately from CPU completion. Once those receipts exist, the next authorized phase is the original P2 Direct L1/GIoU, Shuffled feedback, Fixed Rollout and Full comparison. P3–P6 remain unexecuted until their preceding actual root/public gates. There is no P1-result tuning or additional expert/gate/memory algorithm.
