# TA to Vid: complete current-MAP component progression and final DP

All 10,303 queries / 732 parents are sealed, independently officially scored and cold-replay audited. All FullMAP outputs reproduce the already completed main MAP arrays exactly. No new model, Source, prefix, expert or GPU acquisition.

| Configuration | vIoU (%) | R.3 (%) | R.5 (%) | vIoU delta vs original Source (pp) |
|---|---:|---:|---:|---:|
| Source | 11.138484 | 12.831214 | 3.348539 | +0.000000 |
| EightReplacement | 12.227933 | 14.025041 | 4.212365 | +1.089449 |
| SharedCorrection | 13.243519 | 16.995050 | 7.366786 | +2.105036 |
| SpatialInference | 13.871399 | 17.790935 | 7.590022 | +2.732916 |
| FullMAP | 14.433693 | 18.926526 | 8.317966 | +3.295210 |

Original Source retains its per-offset MAP physical hull. EightReplacement, SharedCorrection and SpatialInference use the same q0 joint physical MAP; FullMAP uses q1 joint MAP with the spatial DP actually re-solved on that interval. The first adaptive row changes both boxes and readout, so its total difference is not a pure effect of the eight boxes. SourceJointMAP (unchanged Source boxes, q0 jointMAP) is 11.641627 vIoU / 13.345627 R.3 / 3.630011 R.5. EightReplacement minus this matched readout is +0.586306 vIoU pp by means; no CI for this extra pair is claimed.

## Paired effects, including harms

| Contrast | v delta (pp) | 95% parent-cluster CI | Improved / harmed / tied | Severe v loss >5pp |
|---|---:|---|---|---:|
| EightReplacement-Source | +1.089449 | [+0.946177, +1.246573] | 5195 / 2384 / 2724 | 356 |
| SharedCorrection-EightReplacement | +1.015587 | [+0.758144, +1.280449] | 4451 / 3280 / 2572 | 1187 |
| SpatialInference-SharedCorrection | +0.627880 | [+0.493177, +0.767989] | 3664 / 3046 / 3593 | 341 |
| FullMAP-SpatialInference | +0.562294 | [+0.426163, +0.707613] | 2851 / 2020 / 5432 | 307 |
| FullMAP-Source | +3.295210 | [+2.893241, +3.708519] | 5053 / 2734 / 2516 | 1154 |
| FullMAP-SourceJointMAP | +2.792066 | [+2.420883, +3.168084] | 4980 / 2891 / 2432 | 1159 |
| FullMAP-FinalIndependent | +0.551463 | [+0.503492, +0.603491] | 4162 / 1523 / 4618 | 16 |

The adjacent mean vIoU gains from shared correction, spatial inference, and temporal revision are positive with pointwise parent CIs above zero on this full direction. These are conditional progression effects, not strict component deletions. Shared correction changes observed-frame sIoU by -8.789776pp while increasing unobserved-frame sIoU by +1.975435pp. The observed/unobserved tradeoff is retained.

FullMAP has 1,139 severe temporal losses, 1,154 severe vIoU losses and 1,934 queries in their union against original Source; the mean improvement does not remove these tails. All signed results and the full severity counts for every arm are preserved.

## Final solver with fixed upstream evidence

| Solver | vIoU (%) | R.3 (%) | R.5 (%) |
|---|---:|---:|---:|
| FinalIndependent | 13.882230 | 18.295642 | 7.434728 |
| FullMAP | 14.433693 | 18.926526 | 8.317966 |

Both solvers use exactly the same candidates, validity, auxiliary trajectory, pooled features, q1 and final MAP interval. Independent choice takes each valid unary maximum without path/link optimization. The final full-path DP improves mean vIoU by +0.551463pp [0.503492, 0.603491], with 4,162 improvements, 1,523 harms, 4,618 ties and 16 vIoU declines above5pp. Time is identical by construction. This supports the targeted final solver contrast, not an independent claim for the first DP.

## Duration groups

| Contrast | Short <=5s (6060 queries): v delta / CI | Long >5s (4243 queries): v delta / CI |
|---|---|---|
| EightReplacement-Source | +0.256555 [+0.117451, +0.405348] | +2.279018 [+2.053177, +2.527708] |
| SharedCorrection-EightReplacement | +0.004080 [-0.216243, +0.228558] | +2.460255 [+2.015357, +2.923026] |
| SpatialInference-SharedCorrection | +0.701956 [+0.545885, +0.860638] | +0.522082 [+0.319007, +0.734358] |
| FullMAP-SpatialInference | +0.043495 [-0.022783, +0.111670] | +1.303261 [+1.001879, +1.622731] |
| FullMAP-Source | +1.006086 [+0.677979, +1.362015] | +6.564616 [+5.889406, +7.278668] |
| FullMAP-SourceJointMAP | +1.277337 [+0.955621, +1.625285] | +4.955456 [+4.320505, +5.608105] |
| FullMAP-FinalIndependent | +0.321099 [+0.278010, +0.365366] | +0.880478 [+0.793566, +0.973179] |

Shared correction and the final temporal revision have short-event vIoU CIs crossing zero, even though their full-domain effects are positive. Short-event FullMAP temporal IoU relative to original Source changes by -0.327366pp with a negative CI; this previously reported harm is retained. The final-DP vIoU effect is positive in both duration groups.

## Verification, costs and scope

72,121 independent official dense checks, maximum difference3.774758283725532e-15. After scoring:20,606 prediction/receipt SHA checks, every array cold-replayed exactly and all means recomputed. The root actually ran the separate scalar exporter after the reporting receipt; no additional GT scoring/model inference.

Worker341761/start240742341:10,303 cached computations sum896.661483 conditional CPU wall seconds, excluding outer setup/GC and original source/expert acquisition and including possible shared-host scheduling. This is not study elapsed time or uncached deployment latency.

All CIs are pointwise parent bootstrap, not familywise guarantees. The fullsets have historical exposure. No outcome-based re-selection, unseen-test claim or general all-backbone module conclusion. The full MAP vs MBR comparison remains separately preserved: MAP-minus-MBR +0.059689t/+0.050379v pp, both CIs cross zero. This component study does not establish MAP superiority to MBR.
