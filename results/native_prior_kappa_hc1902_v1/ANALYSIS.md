# Fixed kappa16 versus kappa64 on official HC1902 caches

The user requests an accuracy/tail-risk/short-long comparison while the active kappa64 dual-domain queue continues. This independent CPU study changes only kappa16 versus64; native final-layer two-offset prior for BOTH prototype seeds and posterior, beta1, ONEround, context1.5, Source, Native2D pools, eight expert observations, complete geometry, original MBR and spatial DP are identical. All1902 predictions seal before independent CPU official labels.

This is a POST-HOC descriptive comparison after official kappa64 HC results were read. It is not new heldout validation, a parameter search or a production selection. The production plan/fingerprint/parameters and ongoing Vid worker remain unchanged.

## Accuracy

| Arm | tIoU% | Complete vIoU% | Delta Source t pp | Delta Source v pp |
|---|---:|---:|---:|---:|
| Source | 48.647345 | 24.994752 | +0.000000 | +0.000000 |
| NativeSpatial | 48.647345 | 26.814658 | +0.000000 | +1.819906 |
| Global6 | 48.580789 | 26.759675 | -0.066556 | +1.764922 |
| K16 | 49.920129 | 27.422330 | +1.272784 | +2.427578 |
| K64 | 49.394417 | 27.156100 | +0.747072 | +2.161348 |

K16-K64 paired tIoU+.525712pp, parent-cluster10000-resample95%CI[+.280072,+.773891]; complete vIoU+.266230pp, CI[+.117457,+.416605]. K16-Source tIoU+1.272784pp, CI[+.862824,+1.703847]. The measured HC mean exceeds Source+.9, but that does not retroactively pass the frozen K64 run or establish a fresh two-direction qualification.

## Severe declines

Severe means tIoU OR complete vIoU drop>5pp versus the named baseline. Union counts a query once, not the sum of metric counts. NativeSpatial=the same complete spatial method at native temporal output without Native2D feedback.

| Baseline | Arm | Severe t | Severe complete v | Union | Union percent |
|---|---|---:|---:|---:|---:|
| Source | K16 | 346 | 305 | 508 | 26.7087% |
| Source | K64 | 148 | 245 | 341 | 17.9285% |
| NativeSpatial | K16 | 346 | 159 | 348 | 18.2965% |
| NativeSpatial | K64 | 148 | 60 | 148 | 7.7813% |
| Global6 | K16 | 347 | 195 | 349 | 18.3491% |
| Global6 | K64 | 259 | 122 | 260 | 13.6698% |

## Short and long events

| Duration | Queries | Arm | Delta Source t pp [95% CI] | Delta Source complete v pp | Severe Source t / v / union |
|---|---:|---|---|---:|---|
| short_le5s | 544 | K16 | -0.049373 [-0.685639, +0.590129] | +1.831192 | 119 / 72 / 146 |
| short_le5s | 544 | K64 | +0.105014 [-0.211494, +0.431175] | +1.910113 | 48 / 55 / 86 |
| long_gt5s | 1358 | K16 | +1.802425 [+1.269798, +2.335050] | +2.666484 | 227 / 233 / 362 |
| long_gt5s | 1358 | K64 | +1.004273 [+0.700932, +1.302953] | +2.261990 | 100 / 190 / 255 |

Short<=5seconds544: K16 t-Source-.049373pp versusK64+.105014pp; the small negative K16 mean has a CI spanning zero. Long>5seconds1358: K16+1.802425pp versusK64+1.004273pp. K16-K64 longt+.798152pp, CI[+.483903,+1.113031]; shortt-.154387pp, CI[-.540651,+.226404]. The extra HC mean gain is driven by long events; no claim of a reliable short-event improvement.

## Interpretation

K16 increases HC mean precision but increases severe time harms148->346 and Source severe-union341->508. K16 time improvement/harm/tie versusSource959/699/244, K64 845/558/499. K16-K64 time800/662/440, with172 queries losing>5pp. Thus greater mean reward comes with substantially more individual damage.

The matched exposed128/domain study also finds no useful mean gain in Vid128 (K16-K64 t+.041605/v-.002844pp) and more severe temporal harms4->10. Vid oldconfirm32 short events are K16-Source-1.725104pp versusK64+.064690pp. [Complete dual-direction small-panel comparison](../native_prior_kappa_compare_v1/ANALYSIS.md).

Given the user request to compare only and preserve the running full, production stays kappa64. These observations do not automatically change the method, claim that overfitting is solved, or transfer HC gains to Vid. No new parameter values, gates or sample selection.

K64 intervals/finalboxes reproduce official Full bitwise1902/1902, all Source/NativeSpatial/Global6/K64 metric dictionaries exactly match saved official rows. 9510 official-versus-dense checks maxerror1.33e-15. Source code/configuration/evidence remain unchanged. See [audit](AUDIT.json), [all per-query signed outcomes](signed.tsv), [all severe negatives](severe.tsv), [full report and CIs](report.json), [protocol](protocol.json). Private media/captions/GT coordinates/features/weights are not published.
