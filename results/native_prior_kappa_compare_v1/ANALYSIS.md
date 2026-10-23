# Fixed native-prior kappa16 versus kappa64 cached comparison

This comparison changes only shrinkage kappa. Both arms use the exact native final-layer two-offset prior for both prototype seeds and final posterior, beta1, ONEround, context1.5, identical Native2D pools, eight observations, complete geometry, MBR and spatial DP. It is separate from the active full queue; no parameter or method selection. All256 predictions sealed before independent CPU official-label scoring.

All dev96/confirm32 panels are historically exposed. The old confirm32 name is provenance, not a claim of new heldout validation. Official HC1902 K64 results were already read before this study.

## Accuracy and paired uncertainty

All numbers below are percentage points or percentages as labeled.

| Direction / panel | K16 tIoU% | K64 tIoU% | K16 vIoU% | K64 vIoU% | K16-K64 t pp [95% parent-cluster CI] | K16-K64 v pp [95% CI] |
|---|---:|---:|---:|---:|---|---|
| hc2 / all128 | 50.2927 | 49.1896 | 28.8533 | 28.4036 | +1.1031 [+0.3378, +1.8542] | +0.4497 [-0.0449, +0.9277] |
| hc2 / dev96 | 51.0240 | 50.4222 | 29.5378 | 29.3265 | +0.6018 [-0.2473, +1.4458] | +0.2113 [-0.3728, +0.7784] |
| hc2 / confirm32 | 48.0989 | 45.4918 | 26.7998 | 25.6347 | +2.6071 [+1.1779, +4.1552] | +1.1651 [+0.3682, +2.0595] |
| vidstg / all128 | 40.8402 | 40.7986 | 19.8303 | 19.8332 | +0.0416 [-0.4844, +0.5410] | -0.0028 [-0.2986, +0.2641] |
| vidstg / dev96 | 41.7507 | 41.4329 | 20.8381 | 20.7310 | +0.3177 [-0.2076, +0.8442] | +0.1072 [-0.1807, +0.3885] |
| vidstg / confirm32 | 38.1089 | 38.8957 | 16.8069 | 17.1398 | -0.7868 [-2.3031, +0.4290] | -0.3328 [-1.1777, +0.2942] |

## Severe declines

A severe query declines by more than5pp in tIoU OR complete vIoU; union counts a query once. Separate counts are not additive. NativeSpatial uses the same complete spatial method with the native interval and no Native2D temporal feedback.

| Direction / panel | Arm | vsSource t / v / union | vsNativeSpatial t / v / union |
|---|---|---|---|
| hc2 / all128 | K16 | 16 / 16 / 26 of 128 | 16 / 9 / 16 of 128 |
| hc2 / all128 | K64 | 6 / 14 / 19 of 128 | 6 / 3 / 6 of 128 |
| hc2 / dev96 | K16 | 13 / 12 / 19 of 96 | 13 / 9 / 13 of 96 |
| hc2 / dev96 | K64 | 4 / 9 / 12 of 96 | 4 / 3 / 4 of 96 |
| hc2 / confirm32 | K16 | 3 / 4 / 7 of 32 | 3 / 0 / 3 of 32 |
| hc2 / confirm32 | K64 | 2 / 5 / 7 of 32 | 2 / 0 / 2 of 32 |
| vidstg / all128 | K16 | 10 / 19 / 25 of 128 | 10 / 5 / 10 of 128 |
| vidstg / all128 | K64 | 4 / 18 / 21 of 128 | 4 / 1 / 4 of 128 |
| vidstg / dev96 | K16 | 5 / 15 / 16 of 96 | 5 / 3 / 5 of 96 |
| vidstg / dev96 | K64 | 2 / 14 / 15 of 96 | 2 / 1 / 2 of 96 |
| vidstg / confirm32 | K16 | 5 / 4 / 9 of 32 | 5 / 2 / 5 of 32 |
| vidstg / confirm32 | K64 | 2 / 4 / 6 of 32 | 2 / 0 / 2 of 32 |

## Short and long events

Duration is GT event duration, evaluation only: short<=5seconds, long>5seconds.

| Direction / panel / duration | N | K16-Source t / v pp | K64-Source t / v pp |
|---|---:|---|---|
| hc2 / all128 / short_le5s | 28 | +1.1129 / +2.0485 | +0.4605 / +1.7964 |
| hc2 / all128 / long_gt5s | 100 | +2.7316 / +4.3458 | +1.5023 / +3.8407 |
| hc2 / dev96 / short_le5s | 22 | +0.5837 / +1.7016 | +0.4323 / +1.5123 |
| hc2 / dev96 / long_gt5s | 74 | +2.1089 / +4.6483 | +1.3733 / +4.4305 |
| hc2 / confirm32 / short_le5s | 6 | +3.0529 / +3.3203 | +0.5642 / +2.8381 |
| hc2 / confirm32 / long_gt5s | 26 | +4.5040 / +3.4848 | +1.8696 / +2.1622 |
| vidstg / all128 / short_le5s | 61 | +0.0666 / +1.2908 | +0.2332 / +1.4271 |
| vidstg / all128 / long_gt5s | 67 | +3.4749 / +6.8207 | +3.2437 / +6.7020 |
| vidstg / dev96 / short_le5s | 49 | +0.5054 / +1.4625 | +0.2745 / +1.3826 |
| vidstg / dev96 / long_gt5s | 47 | +3.8689 / +6.0661 | +3.4606 / +5.9306 |
| vidstg / confirm32 / short_le5s | 12 | -1.7251 / +0.5896 | +0.0647 / +1.6090 |
| vidstg / confirm32 / long_gt5s | 20 | +2.5492 / +8.5940 | +2.7342 / +8.5149 |

## Interpretation and execution

HC pooled128 favors K16 in mean temporal accuracy, with positive paired temporal CI, but severe temporal declines versusSource rise6->16 and versusNativeSpatial also6->16. In Vid128, mean t/v differences are near zero, while severe temporal declines rise4->10. Vid oldconfirm32 short events favorK64 (K16-Source t-1.7251pp; K64+.0647pp). No arm dominates both mean accuracy and tail risk across both directions.

Smaller kappa changes evidence strength, not visual information or model capacity. The sealed source-only posterior TV audit shows larger updates forK16: meanTV .29265 versus.13283 HC, .32458 versus.19860 Vid. This supports a stronger-correction/stronger-error tradeoff, not a claim that additional event information was acquired or generalization fixed.

K64 posterior/time/finalboxes exactly reproduce all256 parent outputs. Source/NativeSpatial/Global6/K64 metric dictionaries exactly reproduce sealed parent metrics. 1280 official-versus-dense metric checks maxerror1.33e-15. Cached replay41.40s and independent CPU scoring14.38s;0GPU/model/RGB/encoder/expert/training. These are research replay costs, not deployment latency.

Production remains kappa64. No queue switch, no new confirmation claim and no full-result parameter promotion. [Complete signed outcomes](signed.tsv), [all severe declines](severe.tsv), [report with all groups and CIs](report.json), [audit](AUDIT.json), [protocol](protocol.json).
