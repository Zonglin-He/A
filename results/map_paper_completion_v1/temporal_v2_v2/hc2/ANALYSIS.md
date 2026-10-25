# Complete current-MAP temporal controls: hc2

**Fully sealed, officially scored and exact-replay audited:** 1902 queries / 227 parents. Native Source/current FullMAP and all numerical settings remain frozen. Changed posteriors each drive their own joint-MAP interval and a new interval-conditioned exact spatial DP. These are complete inference controls, not fixed boxes with a changed interval. Historical full targets are exposed development/evaluation evidence; no configuration is selected from this result.

| Configuration | tIoU % | vIoU % | Severe t / v / union vs original native Source |
|---|---:|---:|---|
| Source | 40.141475 | 20.704678 | 0 / 0 / 0 |
| SourceJointMAP | 46.693293 | 24.039097 | 220 / 118 / 221 |
| FullMAP | 48.586628 | 26.731488 | 267 / 264 / 396 |
| NoPosteriorUpdate | 46.693293 | 25.847764 | 220 / 241 / 364 |
| HardMembership | 48.181471 | 26.542318 | 290 / 274 / 416 |
| EventOnly | 19.076367 | 10.869245 | 1396 / 1051 / 1417 |
| EventCentered | 48.095619 | 26.742967 | 304 / 281 / 427 |
| NoShrink | 49.514796 | 27.214726 | 364 / 305 / 467 |
| UniformPrior | 48.690798 | 26.679948 | 511 / 398 / 597 |

All native-Source severe counts use >5pp losses, union counted once. SourceJointMAP is separate because native Source uses per-offset MAP followed by a physical hull.

| Signed comparison | Δt pp [95% CI] | Δv pp [95% CI] | Severe t/v/union | Rescue union |
|---|---:|---:|---|---:|
| NoPosteriorUpdate-FullMAP | -1.893334 [-2.383192, -1.411589] | -0.883724 [-1.193539, -0.578536] | 527 / 289 / 527 | 299 |
| HardMembership-FullMAP | -0.405156 [-0.688469, -0.116789] | -0.189170 [-0.380785, 0.006301] | 161 / 82 / 162 | 115 |
| EventOnly-FullMAP | -29.510260 [-30.599296, -28.415244] | -15.862243 [-16.570761, -15.162802] | 1574 / 1309 / 1582 | 71 |
| EventCentered-FullMAP | -0.491008 [-0.920104, -0.079079] | +0.011479 [-0.287850, 0.302480] | 336 / 175 / 336 | 262 |
| NoShrink-FullMAP | +0.928168 [0.478495, 1.382096] | +0.483238 [0.224228, 0.739801] | 315 / 168 / 316 | 431 |
| UniformPrior-FullMAP | +0.104170 [-0.665773, 0.895473] | -0.051540 [-0.492543, 0.399650] | 615 / 409 / 621 | 598 |
| FullMAP-Source | +8.445152 [7.590183, 9.310399] | +6.026810 [5.348843, 6.740112] | 267 / 264 / 396 | 1100 |
| FullMAP-SourceJointMAP | +1.893334 [1.411589, 2.383192] | +2.692391 [2.133693, 3.275670] | 296 / 291 / 460 | 777 |

## Duration groups

| Group / signed comparison | Queries | Δt pp [95% CI] | Δv pp [95% CI] |
|---|---:|---:|---:|
| short_le5s / NoPosteriorUpdate-FullMAP | 544 | -0.647884 [-1.468806, 0.142748] | -0.372500 [-0.928671, 0.138711] |
| short_le5s / HardMembership-FullMAP | 544 | -0.325169 [-0.775632, 0.099090] | -0.210683 [-0.526852, 0.064943] |
| short_le5s / EventOnly-FullMAP | 544 | -18.472137 [-20.293451, -16.640246] | -10.006905 [-11.266880, -8.759160] |
| short_le5s / EventCentered-FullMAP | 544 | -1.514096 [-2.380186, -0.708433] | -0.756034 [-1.305831, -0.265927] |
| short_le5s / NoShrink-FullMAP | 544 | -1.159478 [-1.997215, -0.332317] | -0.656972 [-1.157207, -0.152927] |
| short_le5s / UniformPrior-FullMAP | 544 | -3.895632 [-5.417784, -2.386771] | -2.226617 [-3.097991, -1.380588] |
| short_le5s / FullMAP-Source | 544 | +4.745209 [3.100079, 6.431985] | +4.074905 [2.711878, 5.411140] |
| short_le5s / FullMAP-SourceJointMAP | 544 | +0.647884 [-0.142748, 1.468806] | +2.179661 [1.167613, 3.207907] |
| long_gt5s / NoPosteriorUpdate-FullMAP | 1358 | -2.392248 [-2.961033, -1.817096] | -1.088515 [-1.445169, -0.718546] |
| long_gt5s / HardMembership-FullMAP | 1358 | -0.437198 [-0.777810, -0.080059] | -0.180552 [-0.407014, 0.054116] |
| long_gt5s / EventOnly-FullMAP | 1358 | -33.932012 [-35.055287, -32.766885] | -18.207828 [-19.053326, -17.371250] |
| long_gt5s / EventCentered-FullMAP | 1358 | -0.081170 [-0.567212, 0.397313] | +0.318936 [-0.031532, 0.670038] |
| long_gt5s / NoShrink-FullMAP | 1358 | +1.764456 [1.219768, 2.325750] | +0.939994 [0.620933, 1.270070] |
| long_gt5s / UniformPrior-FullMAP | 1358 | +1.706447 [0.800945, 2.614254] | +0.819772 [0.298658, 1.346371] |
| long_gt5s / FullMAP-Source | 1358 | +9.927309 [8.979625, 10.877136] | +6.808721 [5.963675, 7.675275] |
| long_gt5s / FullMAP-SourceJointMAP | 1358 | +2.392248 [1.817096, 2.961033] | +2.897785 [2.198419, 3.610300] |

## Interpretation and costs

NoPosteriorUpdate isolates temporal evidence update with the same auxiliary features and q0 joint-MAP readout. HardMembership replaces the soft event membership by the hard q0-MAP seed. EventOnly omits the background denominator without centering; EventCentered is the separately declared video-centered event-only control. NoShrink uses kappa0; UniformPrior retains the original q0 membership/evidence while changing only the final positive-support prior. FullMAP retains kappa64/beta1/context1.5/one round and the same eight expert observations.

The uncentered EventOnly control loses substantial temporal and video accuracy. Centering largely recovers its vIoU, with a pointwise interval crossing zero against FullMAP. Removing shrinkage **improves** mean tIoU and vIoU here, while increasing the native-Source severe union from396 (FullMAP) to467. Thus this result does not support an unconditional mean-accuracy advantage for shrinkage64. UniformPrior has a small mean difference with intervals crossing zero and a larger native-Source severe union597. The fixed method is not retuned from these exposed full-target comparisons. Soft versus hard membership has a temporal difference, while the vIoU interval crosses zero.

Intervals are pointwise parent-cluster bootstrap intervals, not familywise or independent-unseen confirmation. All scalar signed outcomes, observed/unobserved effects, short/long intervals, severe losses and rescues remain in report.json, all_signed.tsv, all_signed_severe_timegroups.tsv and all_paired_tail_summary.json. The audited cached-query wall sums per worker are {"1083188:245060214": 85.29723130585626}; these exclude setup/outer-loop overhead and may reflect shared-host execution. They are not end-to-end deployment latency. Inherited Source/expert acquisition is nonzero provenance. New Source/expert/GPU/GT scoring during this read-only export: zero.

Audit: all arrays rederived exactly, all reported means recomputed, 9510 parent/prediction/receipt SHA checks. The independent official score receipt and frozen protocol remain authoritative.