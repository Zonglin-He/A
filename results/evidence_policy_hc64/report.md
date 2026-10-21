# HC64: evidence-conditioned A/B policy adaptation

Fixed first64 development parents, one query per parent, seed41. All methods share the exact sealed Strong time interval and the same complete-video metric. Six previous outputs are reused byte-for-byte; the six A/B outputs are independently executed and sealed before GT. No network, decoder, expert or RGB call is made.

| Method | Full vIoU % | tIoU % | Observed sIoU % | Unobserved sIoU % |
|---|---:|---:|---:|---:|
| ERefStrong | 31.0918 | 58.6043 | 52.2680 | 50.7055 |
| GeometryOPD | 31.3868 | 58.6043 | 53.1654 | 51.2469 |
| BoxFixed | 31.3985 | 58.6043 | 55.3363 | 50.9672 |
| BoxOPD | 31.1909 | 58.6043 | 52.8269 | 50.7916 |
| BoxFixedKD | 31.0459 | 58.6043 | 52.1779 | 50.6770 |
| BoxDirect | 31.1997 | 58.6043 | 52.8269 | 50.8086 |
| A_OPD | 31.0930 | 58.6043 | 54.6190 | 50.6948 |
| A_FixedKD | 31.3064 | 58.6043 | 54.7613 | 50.9100 |
| A_Direct | 31.0989 | 58.6043 | 54.5886 | 50.7475 |
| B_OPD | 31.1227 | 58.6043 | 54.6227 | 50.7581 |
| B_FixedKD | 31.2453 | 58.6043 | 54.7242 | 50.8315 |
| B_Direct | 31.1048 | 58.6043 | 54.6003 | 50.8170 |

| Comparison | vIoU delta pp [95% CI] | Improve / decline / same | vIoU drops >5 pp | Unobserved sIoU delta pp [95% CI] |
|---|---:|---:|---:|---:|
| A_OPD − ERefStrong | +0.0012 [-1.6188, 1.5710] | 26 / 20 / 18 | 7 | -0.0107 [-2.2751, 2.6186] |
| A_OPD − GeometryOPD | -0.2938 [-1.8394, 1.1651] | 27 / 36 / 1 | 6 | -0.5521 [-2.7310, 1.9578] |
| A_OPD − BoxFixed | -0.3056 [-0.8521, 0.2124] | 26 / 32 / 6 | 2 | -0.2725 [-1.0588, 0.5148] |
| A_OPD − BoxOPD | -0.0980 [-1.7661, 1.4886] | 26 / 20 / 18 | 7 | -0.0968 [-2.3533, 2.5204] |
| A_FixedKD − ERefStrong | +0.2147 [-1.5107, 1.8955] | 25 / 20 / 19 | 7 | +0.2045 [-2.1189, 2.8942] |
| A_FixedKD − GeometryOPD | -0.0803 [-1.7308, 1.4834] | 28 / 35 / 1 | 7 | -0.3369 [-2.5775, 2.2072] |
| A_FixedKD − BoxFixed | -0.0921 [-0.5094, 0.3133] | 26 / 30 / 8 | 1 | -0.0572 [-0.7155, 0.5794] |
| A_FixedKD − BoxFixedKD | +0.2605 [-1.4372, 1.9158] | 26 / 19 / 19 | 6 | +0.2330 [-2.0826, 2.8975] |
| A_Direct − ERefStrong | +0.0072 [-1.5993, 1.5556] | 25 / 21 / 18 | 6 | +0.0420 [-2.1785, 2.6334] |
| A_Direct − GeometryOPD | -0.2878 [-1.8175, 1.1501] | 28 / 35 / 1 | 6 | -0.4994 [-2.6268, 1.9720] |
| A_Direct − BoxFixed | -0.2996 [-0.8899, 0.2389] | 26 / 32 / 6 | 2 | -0.2197 [-1.0455, 0.5710] |
| A_Direct − BoxDirect | -0.1008 [-1.7592, 1.4880] | 25 / 21 / 18 | 6 | -0.0611 [-2.2717, 2.5272] |
| B_OPD − ERefStrong | +0.0310 [-1.5901, 1.6082] | 27 / 20 / 17 | 6 | +0.0525 [-2.2424, 2.6901] |
| B_OPD − GeometryOPD | -0.2640 [-1.8138, 1.1923] | 28 / 35 / 1 | 6 | -0.4889 [-2.6834, 2.0288] |
| B_OPD − BoxFixed | -0.2758 [-0.8221, 0.2377] | 26 / 32 / 6 | 2 | -0.2092 [-0.9940, 0.5652] |
| B_OPD − BoxOPD | -0.0682 [-1.7480, 1.5215] | 27 / 20 / 17 | 6 | -0.0336 [-2.3104, 2.5858] |
| B_FixedKD − ERefStrong | +0.1536 [-1.5606, 1.8225] | 25 / 20 / 19 | 7 | +0.1260 [-2.1794, 2.7996] |
| B_FixedKD − GeometryOPD | -0.1414 [-1.7849, 1.3950] | 27 / 36 / 1 | 7 | -0.4154 [-2.6219, 2.1238] |
| B_FixedKD − BoxFixed | -0.1532 [-0.5659, 0.2577] | 25 / 31 / 8 | 1 | -0.1357 [-0.8011, 0.5271] |
| B_FixedKD − BoxFixedKD | +0.1994 [-1.4860, 1.8493] | 25 / 20 / 19 | 6 | +0.1545 [-2.1390, 2.8269] |
| B_Direct − ERefStrong | +0.0131 [-1.5895, 1.5629] | 27 / 19 / 18 | 6 | +0.1115 [-2.1248, 2.7213] |
| B_Direct − GeometryOPD | -0.2819 [-1.8089, 1.1693] | 29 / 34 / 1 | 6 | -0.4299 [-2.5808, 2.0643] |
| B_Direct − BoxFixed | -0.2937 [-0.8753, 0.2442] | 26 / 31 / 7 | 2 | -0.1502 [-0.9657, 0.6347] |
| B_Direct − BoxDirect | -0.0948 [-1.7509, 1.4971] | 27 / 19 / 18 | 6 | +0.0084 [-2.2158, 2.6137] |
| B_OPD − A_OPD | +0.0298 [-0.0080, 0.0804] | 3 / 2 / 59 | 0 | +0.0633 [0.0033, 0.1584] |
| B_FixedKD − A_FixedKD | -0.0611 [-0.1463, -0.0016] | 2 / 6 / 56 | 0 | -0.0785 [-0.2099, 0.0226] |
| B_Direct − A_Direct | +0.0059 [-0.0493, 0.0698] | 4 / 3 / 57 | 0 | +0.0695 [-0.0671, 0.2361] |
| A_OPD − A_FixedKD | -0.2134 [-0.6468, 0.1491] | 10 / 13 / 41 | 2 | -0.2152 [-0.7767, 0.2914] |
| A_OPD − A_Direct | -0.0060 [-0.1482, 0.1723] | 5 / 9 / 50 | 0 | -0.0527 [-0.3464, 0.2528] |
| B_OPD − B_FixedKD | -0.1226 [-0.5456, 0.2217] | 13 / 11 / 40 | 1 | -0.0735 [-0.6219, 0.4115] |
| B_OPD − B_Direct | +0.0179 [-0.1111, 0.1875] | 6 / 7 / 51 | 0 | -0.0589 [-0.3731, 0.2324] |

Intervals use10000 paired parent draws. All signed per-query changes and severe declines are retained. Observed/unobserved IoU uses fixed actual sampled nodes with GT, independently of temporal truncation; the eight expert positions are excluded from unobserved IoU.

## Actual optimization and final action margins

| Method | Seconds/query | Updates/query | Accepted/query | Active unobserved positive-margin nodes | All unobserved replacements / nodes | Recovery / initially below0.5 | Damage / initially at least0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_OPD | 0.14605 | 13.594 | 12.734 | 518 / 1478 | 518 / 3648 | 47 / 560 | 51 / 805 |
| A_FixedKD | 0.18809 | 13.781 | 12.922 | 544 / 1478 | 544 / 3648 | 48 / 560 | 54 / 805 |
| A_Direct | 0.48179 | 13.344 | 12.484 | 501 / 1478 | 501 / 3648 | 48 / 560 | 50 / 805 |
| B_OPD | 0.14896 | 13.625 | 12.766 | 529 / 1478 | 529 / 3648 | 48 / 560 | 51 / 805 |
| B_FixedKD | 0.22399 | 13.922 | 13.078 | 547 / 1478 | 547 / 3648 | 46 / 560 | 54 / 805 |
| B_Direct | 0.70178 | 13.781 | 12.953 | 505 / 1478 | 505 / 3648 | 48 / 560 | 50 / 805 |

The8 matching weights are the only updated state. All arms receive the same allowance of3 outer rounds×2 sampled histories×10 inner updates. A rejected hard-readout proposal restores weights and Adam and stops; actual accepted updates can differ. Margins compare the best eligible alternative with E on the actual action support; inactive and E-only nodes remain in replacement denominators. Recovery/damage thresholds are offline descriptions of final-node GT IoU, not online acceptance rules.

Observed incremental trial costs with existing Source/teacher caches; not standalone end-to-end inference latency

## Consequence-aware target and saved-history changes

| Method | Active unobserved saved-state evaluations | Q versus immediate-r argmax differences |
|---|---:|---:|
| A_OPD | 4670 | 0 |
| A_FixedKD | 4682 | 0 |
| A_Direct | 4522 | 0 |
| B_OPD | 4578 | 379 |
| B_FixedKD | 4582 | 406 |
| B_Direct | 4504 | 361 |

B OPD versus fixed-KD: 3768 matched-node evaluations have different saved histories; 132 of these also change the saved teacher-target argmax. Actual saved OPD and fixed-KD targets use the same frozen immediate-teacher reference continuation rule conditioned on different incoming histories. The rule does not use student-weight continuation. This is not an isolated causal memory ablation; identical complete incoming states must have identical targets.

Initial/final margins, prior barriers, every saved loss/update curve, target-selection changes and all anonymous final-node corrections are in report.json. All results are from an exposed development panel; no per-query GT selection or further configuration choice is performed by this scorer.
