# Controlled RGB shifts: completed screening

Screening includes64fixed independent parents per direction, one query/parent, each evaluated under3fixed RGB transformations.384transformed-video/query pairs produced paired Source/Full predictions. The whole128query stage was sealed before768independent official CPU metric checks (maximum error1.6653345369377348e-15). All384predictionSHA verified; all384expert calls actually ran the same ReferDINO on8observations with0cachehits. No cleanRGBteacher output was reused as shifted evidence. Main method parameters are unchanged.

Blur: OpenCV5x5 Gaussian sigma1.5, BORDER_REFLECT_101. JPEG:quality30/subsampling2 then RGB decode at the original size. Dim:RGBfloat32×0.5, clip and truncateuint8. These are controlled synthetic input shifts, not a new natural-domain benchmark. A transform need not decrease Source quality on every cohort.

## HC2 —64 parents

| Shift | Source t / v (%) | Full t / v (%) | Δt / Δv vs shiftedSource (pp) | Δv95% CI (pp) | Severe t / v / union | Full−CleanFull v (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Blur | 49.8614 / 24.2845 | 51.4534 / 25.9127 | +1.5920 / +1.6281 | [-2.1265, 5.5704] | 9 / 10 / 14 | -1.4655 |
| JPEG | 49.0914 / 23.8205 | 50.7205 / 26.0979 | +1.6291 / +2.2774 | [-0.9829, 5.9019] | 5 / 7 / 11 | -1.2802 |
| Dim | 49.1267 / 23.0700 | 50.1656 / 24.8365 | +1.0389 / +1.7664 | [-1.8924, 5.5351] | 11 / 9 / 15 | -2.5417 |

Severe union counts each query once if tIoU or vIoU drops by more than5pp against the Source prediction on the same shiftedRGB. No harmed query is removed. CleanSource/CleanFull are labeled references from the same selected queries, not substitute shifted predictions.

## VIDSTG —64 parents

| Shift | Source t / v (%) | Full t / v (%) | Δt / Δv vs shiftedSource (pp) | Δv95% CI (pp) | Severe t / v / union | Full−CleanFull v (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Blur | 27.8233 / 12.0163 | 29.6054 / 13.7114 | +1.7821 / +1.6951 | [-0.4255, 3.7879] | 4 / 5 / 8 | -1.3726 |
| JPEG | 28.0589 / 12.0683 | 29.8615 / 14.5649 | +1.8026 / +2.4966 | [0.1589, 4.8336] | 4 / 4 / 6 | -0.5191 |
| Dim | 27.0721 / 11.8164 | 30.1694 / 15.3634 | +3.0973 / +3.5471 | [0.7756, 6.5069] | 5 / 7 / 9 | +0.2794 |

Severe union counts each query once if tIoU or vIoU drops by more than5pp against the Source prediction on the same shiftedRGB. No harmed query is removed. CleanSource/CleanFull are labeled references from the same selected queries, not substitute shifted predictions.

## Interpretation and fixed continuation

All six shifted cohorts have positive Full−Source meanvIoU. All three HC vIoU intervals include zero. Vid JPEG and Dim pointwise vIoU intervals exclude zero, while Blur crosses zero; these are screening observations, not corrected family-wise tests or a claim of universal robustness. Full still losesvIoU against its own clean-input result for all three HC views and for Vid Blur/JPEG. Vid Dim Full gainsvIoU relative to clean. Clean and shifted outcomes, including negative changes and all severe tails, remain visible.

Everytransform advances unchanged to the same fixed128parent confirmation per direction. No transform/severity/method parameter was selected on accuracy. Screen and confirm parents are disjoint, but full official target sets were historically exposed; this is not a historically unseen dataset. Confirmation is already running. CG-STVG remainsLAST and has not constructed a model. Fig1 and original full-method results are unchanged.

## Costs and complete data

| Domain | Shift | Mean transform seconds | Mean per-view pipeline seconds | New same-expert joint8 calls |
|---|---|---:|---:|---:|
| hc2 | Blur | 0.111150 | 4.134169 | 64 |
| hc2 | JPEG | 0.239808 | 4.159434 | 64 |
| hc2 | Dim | 0.474448 | 4.341113 | 64 |
| vidstg | Blur | 0.087097 | 3.505594 | 64 |
| vidstg | JPEG | 0.206697 | 3.503358 | 64 |
| vidstg | Dim | 0.260554 | 3.557252 | 64 |

Per-view pipeline timing starts beforetransform and includes capture/finish; decoding, subject parsing and some initialization are outside it. Stagewall is1633.286596seconds and is not a resident deployment benchmark. The tails_timegroups_cost.json files retain every recorded numeric capture/finish/expert/load/transform component with mean/sum/coverage. Components can be nested and must not be added indiscriminately. Actual expert-forward counts and cache flags retain their original acquisition meaning.

Each domain/screen/transform folder contains report.json, full-precision signed.tsv, all_signed_severe_timegroups.tsv and tails_timegroups_cost.json. The latter reports Full−shiftedSource, shiftedSource−CleanSource and shiftedFull−CleanFull, including separate t/v/severeuniononce, short/long <=5s groups and parent-cluster CIs. Postsealshifts_reporting.py only consumes already scored metrics and verifies predictionSHA/reportmeans; it runs no model, prediction or newGTscoring.
