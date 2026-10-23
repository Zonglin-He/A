# Controlled RGB shifts: fixed confirmation complete

The same three predeclared transformations were evaluated on128fixed parents per direction, disjoint from screening, one query per parent.256queries×3views=768paired Source/Full predictions.768actual Sourcequery evaluations and768actual sameReferDINOjoint8calls,0teacher cachehits. All768predictionSHA and transformedRGBSHA verified. Whole stage sealed before1536independent officialdense metric checks, maxerror1.5543122344752192e-15. Targetfullsets were historically exposed; this is a frozen new contrast, not an historically unseen dataset.

Blur5x5/sigma1.5/reflect101; JPEGquality30/subsampling2; DimRGBfloat32×0.5 thenuint8. Source and Full consume exactly the same transformedRGB. No cleanexpert output reused as shifted evidence. No severity or method parameter selection followed screening.

## HC2 —128parents

| Shift | Source t / v (%) | Full t / v (%) | Δt / Δv vs shiftedSource (pp) | Δv95%CI (pp) | Severe t / v / union | Full−CleanFull v (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Blur | 51.3633 / 28.0173 | 52.6649 / 30.6886 | +1.3016 / +2.6713 | [0.6569, 4.7608] | 9 / 16 / 23 | -0.0012 |
| JPEG | 50.8911 / 27.4414 | 52.4206 / 30.2527 | +1.5295 / +2.8113 | [0.8794, 4.7250] | 7 / 19 / 23 | -0.4371 |
| Dim | 52.2996 / 28.3172 | 53.3679 / 30.0259 | +1.0683 / +1.7087 | [-0.4235, 3.9271] | 12 / 20 / 28 | -0.6640 |

Severe union counts a query once if tIoU or vIoU drops by more than5pp versus shiftedSource. All128queries remain. Pointwise95% parent-cluster CIs are not family-wise corrected.

## VIDSTG —128parents

| Shift | Source t / v (%) | Full t / v (%) | Δt / Δv vs shiftedSource (pp) | Δv95%CI (pp) | Severe t / v / union | Full−CleanFull v (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Blur | 34.8258 / 13.5883 | 35.6094 / 14.6569 | +0.7836 / +1.0686 | [-1.2231, 3.4358] | 8 / 20 / 26 | -0.5485 |
| JPEG | 34.8061 / 14.1438 | 35.1251 / 14.3720 | +0.3190 / +0.2282 | [-1.8008, 2.2709] | 14 / 21 / 32 | -0.8334 |
| Dim | 33.5573 / 14.0002 | 34.5650 / 14.5956 | +1.0077 / +0.5954 | [-1.6229, 2.7438] | 6 / 19 / 22 | -0.6098 |

Severe union counts a query once if tIoU or vIoU drops by more than5pp versus shiftedSource. All128queries remain. Pointwise95% parent-cluster CIs are not family-wise corrected.

## Interpretation

Full retains positive meanvIoU differences relative to Source in all six confirmation cohorts. HCBlur/JPEG pointwisevIoU intervals exclude zero; HCDim and all three Vid intervals cross zero. This does not establish universal or uniformly significant robustness. All six FullvIoU means are lower than their cleanFull reference, including the small HCBlur change. A transform can improve Source in a cohort, so these are fixed input-shift comparisons, not guaranteed quality-degradation tests.

Screening and confirmation are both retained, including changed signs and smaller gains. No transformation, severity or main parameter is reselected. Currentκ64β1/context1.5/eight/SimplePath remains unchanged. These are synthetic robustness controls and do not replace a third naturally shifted fully annotated STVG benchmark.

## Complete data and actual cost

Each domain/confirm/transform directory contains report.json and fullprecision signed.tsv, plus all_signed_severe_timegroups.tsv and tails_timegroups_cost.json. The latter reports Full−shiftedSource, shiftedSource−CleanSource and shiftedFull−CleanFull, allsigned severe t/v/uniononce, short<=5s/long parentCIs, and every recorded numeric capture/finish/load/expert cost. Nested components must not be added.

Confirmation stagewall: 2804.051809s. Per-view pipeline wall includes transform/capture/finish but excludes video decode/subject parsing and some initialization. It is not a matched resident deployment benchmark. Acrossscreen+confirm,1152actual Sourcequeries and1152sameexpertjoint8calls are recorded exactly once.

| Domain | Shift | Mean transform seconds | Mean per-view pipeline seconds | New expert joint8 calls |
|---|---|---:|---:|---:|
| hc2 | Blur | 0.088770 | 3.186657 | 128 |
| hc2 | JPEG | 0.199381 | 3.264408 | 128 |
| hc2 | Dim | 0.271607 | 3.311981 | 128 |
| vidstg | Blur | 0.076116 | 3.386100 | 128 |
| vidstg | JPEG | 0.180925 | 3.432123 | 128 |
| vidstg | Dim | 0.205630 | 3.444265 | 128 |

AllTA consistency, budget and shift experiments now have complete sealed/scored stages and their workers exited. CG-STVG was started LAST after these receipts and GPUlockfree verification. Its first native420FP32 executable attempt OOMed before any completed Sourceprediction or expert call. The failure is preserved; an independently frozen, unit-bitwise-tested allocation-only v4 recovery is now running without reducedframes/resolution/precision or changed equations. See [CG memory recovery](../CGSTVG_MEMORY_RECOVERY.md). No CG effectiveness conclusion is available yet.
