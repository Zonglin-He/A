# Complete current-MAP temporal controls: vidstg

**Fully sealed, officially scored and exact-replay audited:** 10303 queries / 732 parents. Native Source/current FullMAP and all numerical settings remain frozen. Changed posteriors each drive their own joint-MAP interval and a new interval-conditioned exact spatial DP. These are complete inference controls, not fixed boxes with a changed interval. Historical full targets are exposed development/evaluation evidence; no configuration is selected from this result.

| Configuration | tIoU % | vIoU % | Severe t / v / union vs original native Source |
|---|---:|---:|---|
| Source | 31.728454 | 11.138484 | 0 / 0 / 0 |
| SourceJointMAP | 33.450806 | 11.641627 | 1015 / 272 / 1015 |
| FullMAP | 34.664067 | 14.433693 | 1139 / 1154 / 1934 |
| NoPosteriorUpdate | 33.450806 | 13.871399 | 1015 / 1109 / 1837 |
| HardMembership | 34.487493 | 14.331920 | 1193 / 1151 / 1979 |
| EventOnly | 14.679894 | 6.162265 | 6610 / 3586 / 6733 |
| EventCentered | 34.724051 | 14.518460 | 1187 / 1161 / 1971 |
| NoShrink | 34.927982 | 14.548371 | 1487 / 1221 / 2238 |
| UniformPrior | 34.021032 | 14.277113 | 2776 / 1585 / 3307 |

All native-Source severe counts use >5pp losses, union counted once. SourceJointMAP is separate because native Source uses per-offset MAP followed by a physical hull.

| Signed comparison | Δt pp [95% CI] | Δv pp [95% CI] | Severe t/v/union | Rescue union |
|---|---:|---:|---|---:|
| NoPosteriorUpdate-FullMAP | -1.213262 [-1.495964, -0.946621] | -0.562294 [-0.707613, -0.426163] | 1493 / 598 / 1498 | 971 |
| HardMembership-FullMAP | -0.176575 [-0.281378, -0.070337] | -0.101773 [-0.160447, -0.046844] | 431 / 158 / 432 | 398 |
| EventOnly-FullMAP | -19.984173 [-21.177270, -18.813384] | -8.271428 [-8.924296, -7.645077] | 6820 / 4060 / 6830 | 846 |
| EventCentered-FullMAP | +0.059984 [-0.110511, 0.232324] | +0.084766 [-0.013476, 0.184133] | 879 / 280 / 882 | 865 |
| NoShrink-FullMAP | +0.263915 [0.044013, 0.485967] | +0.114677 [0.003874, 0.223256] | 1178 / 436 / 1184 | 1313 |
| UniformPrior-FullMAP | -0.643035 [-1.201860, -0.103044] | -0.156580 [-0.421044, 0.097812] | 2706 / 1223 / 2715 | 2532 |
| FullMAP-Source | +2.935613 [2.547106, 3.355527] | +3.295210 [2.893241, 3.708519] | 1139 / 1154 / 1934 | 4275 |
| FullMAP-SourceJointMAP | +1.213262 [0.946621, 1.495964] | +2.792066 [2.420883, 3.168084] | 970 / 1159 / 1880 | 3441 |

## Duration groups

| Group / signed comparison | Queries | Δt pp [95% CI] | Δv pp [95% CI] |
|---|---:|---:|---:|
| short_le5s / NoPosteriorUpdate-FullMAP | 6060 | -0.148318 [-0.295648, -0.004023] | -0.043495 [-0.111670, 0.022783] |
| short_le5s / HardMembership-FullMAP | 6060 | -0.070990 [-0.143564, 0.003033] | -0.026141 [-0.060628, 0.008120] |
| short_le5s / EventOnly-FullMAP | 6060 | -8.678974 [-9.796943, -7.614656] | -3.120653 [-3.608913, -2.652640] |
| short_le5s / EventCentered-FullMAP | 6060 | -0.154496 [-0.317886, 0.008950] | -0.017968 [-0.104457, 0.069229] |
| short_le5s / NoShrink-FullMAP | 6060 | +0.249413 [0.004986, 0.500581] | +0.122483 [-0.001145, 0.252638] |
| short_le5s / UniformPrior-FullMAP | 6060 | -0.802841 [-1.418583, -0.203903] | -0.212497 [-0.486924, 0.064283] |
| short_le5s / FullMAP-Source | 6060 | -0.327366 [-0.591641, -0.052901] | +1.006086 [0.677979, 1.362015] |
| short_le5s / FullMAP-SourceJointMAP | 6060 | +0.148318 [0.004023, 0.295648] | +1.277337 [0.955621, 1.625285] |
| long_gt5s / NoPosteriorUpdate-FullMAP | 4243 | -2.734251 [-3.355731, -2.143682] | -1.303261 [-1.622731, -1.001879] |
| long_gt5s / HardMembership-FullMAP | 4243 | -0.327375 [-0.569042, -0.092088] | -0.209793 [-0.338579, -0.084630] |
| long_gt5s / EventOnly-FullMAP | 4243 | -36.130652 [-37.604226, -34.662254] | -15.627944 [-16.648845, -14.654588] |
| long_gt5s / EventCentered-FullMAP | 4243 | +0.366311 [0.015765, 0.722927] | +0.231495 [0.027575, 0.444873] |
| long_gt5s / NoShrink-FullMAP | 4243 | +0.284626 [-0.118052, 0.674679] | +0.103529 [-0.089427, 0.287221] |
| long_gt5s / UniformPrior-FullMAP | 4243 | -0.414795 [-1.370623, 0.505249] | -0.076718 [-0.532383, 0.358918] |
| long_gt5s / FullMAP-Source | 4243 | +7.595912 [6.906868, 8.334642] | +6.564616 [5.889406, 7.278668] |
| long_gt5s / FullMAP-SourceJointMAP | 4243 | +2.734251 [2.143682, 3.355731] | +4.955456 [4.320505, 5.608105] |

## Interpretation and costs

NoPosteriorUpdate isolates temporal evidence update with the same auxiliary features and q0 joint-MAP readout. HardMembership replaces the soft event membership by the hard q0-MAP seed. EventOnly omits the background denominator without centering; EventCentered is the separately declared video-centered event-only control. NoShrink uses kappa0; UniformPrior retains the original q0 membership/evidence while changing only the final positive-support prior. FullMAP retains kappa64/beta1/context1.5/one round and the same eight expert observations.

Intervals are pointwise parent-cluster bootstrap intervals, not familywise or independent-unseen confirmation. All scalar signed outcomes, observed/unobserved effects, short/long intervals, severe losses and rescues remain in report.json, all_signed.tsv, all_signed_severe_timegroups.tsv and all_paired_tail_summary.json. The audited cached-query wall sums per worker are {"1083189:245060214": 715.2488484489731}; these exclude setup/outer-loop overhead and may reflect shared-host execution. They are not end-to-end deployment latency. Inherited Source/expert acquisition is nonzero provenance. New Source/expert/GPU/GT scoring during this read-only export: zero.

Audit: all arrays rederived exactly, all reported means recomputed, 51515 parent/prediction/receipt SHA checks. The independent official score receipt and frozen protocol remain authoritative.