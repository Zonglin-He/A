# Frozen source24: synchronized sigma .213 initialization and schedule

Fixed repaired HC epoch30 / Vid epoch20, original source-val12 each, original4 epsilon paths. No training, observation guidance, new Teacher/Student or target runs. Only initial sigma and its16-level log schedule start changed from2 to exactly.213, end.03 and Euler rule unchanged. Old Source/sigma2 references reused and independently verified.

Source origins are official train and were held out only for the auxiliary prior; not an independent evaluation unseen to pretrained Student. This is a fixed source-development inference comparison, not target confirmation or method promotion.

## Fixed first-path results

Percent; entire12-source denominator/domain. R03/R05 are query-level vIoU thresholds.

| Method | HC vIoU | HC tIoU | HC sIoU | HC R03 | HC R05 | Vid vIoU | Vid tIoU | Vid sIoU | Vid R03 | Vid R05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native | 43.280488 | 67.739714 | 64.016598 | 83.333333 | 50.000000 | 18.195709 | 35.521063 | 50.654046 | 25.000000 | 16.666667 |
| Native_codec | 43.367793 | 67.739714 | 64.120651 | 83.333333 | 50.000000 | 18.224082 | 35.521063 | 50.788860 | 25.000000 | 16.666667 |
| sigma2 | 21.212924 | 48.321120 | 50.512354 | 33.333333 | 0.000000 | 12.479204 | 39.216143 | 32.977724 | 25.000000 | 8.333333 |
| sigma213 | 38.329163 | 65.918976 | 59.296228 | 83.333333 | 16.666667 | 17.679277 | 39.501725 | 48.631961 | 16.666667 | 16.666667 |

## Paired first-path vIoU differences

Mean and source-query paired95% interval in percentage points. Confidence intervals do not define an automatic next stage.

| Comparison | HC mean / CI95 / positive-negative-neutral | Vid mean / CI95 / positive-negative-neutral |
|---|---|---|
| sigma213 - Native | -4.951324 / [-10.011089,+0.108440] / 4-8-0 | -0.516432 / [-3.609408,+2.576543] / 7-4-1 |
| sigma213 - Native_codec | -5.038630 / [-10.109765,+0.032506] / 4-8-0 | -0.544805 / [-3.665516,+2.575907] / 7-4-1 |
| sigma213 - sigma2 | +17.116240 / [+11.431075,+22.801405] / 11-1-0 | +5.200073 / [-5.169574,+15.569719] / 9-3-0 |

All4 per-path metrics and mean-of-path metrics are in RESULTS.json. No coordinate averaging, GT path selection, source removal or checkpoint reselection. Native_codec is the same32-knot readout control; raw Native is primary. Noisy initialization is correctly synchronized to the same new .213 starting sigma used by the model and update schedule. It is not the earlier no-initial-noise/sigma2 counterfactual.

## Scope and cost

This is still a finite Native-warm-start approximation, not an exact conditional posterior sampler. No new claims about identity reliability, dense reward, target adaptation or pretrained-Student-unseen generalization. No current-method promotion or automatic observation/target/OPD follow-up. Cached feature inference GPU reservation: 1.478318 seconds including2 fixed sigma2 interface replays; cumulative 38.086873 seconds under the original14400s cap. CPU audit/publication and original upstream video/feature/expert work are separate; not online E2E latency.

Independent scalar/reference/codec/mean/paired checks: 1333. All old model/reference seals preserved. Private source-pixel inspection is engineering only, not semantic identity labeling.
