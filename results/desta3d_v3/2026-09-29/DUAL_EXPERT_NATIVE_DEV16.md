# DESTA dual-expert native adaptation: Dev16 completed, further experiments paused

2026-09-29: the user requested a pause. All experimental workers and the continuation controller were stopped; the 30-minute monitor is paused. No automatic resumption. Dev64 contains only three newly completed configuration-query endpoints and other partial artifacts; it has no complete seal or score.

## Complete first-phase result

Fixed exposed source development panel: 16 parents, one query each, 27 configurations. PTD4B/B1/QC frozen; temporary R16 fields only, no model-weight optimizer. UniversalVTG and GroundingDINO+SAM2 provide fixed sparse-observation hard pseudo-labels. GroundingDINO+SAM2 is a detector/tracker fallback, not a task-trained RVOS specialist. Teacher-training overlap is unknown.

All 27 configurations have lower mean vIoU than B1. The selected best is K=1, radius=.03, temporal/spatial weight=2/1. This is an exposed-development result, not a fresh evaluation or a universal impossibility claim.

| Configuration | tIoU % | sIoU % | vIoU % | v change vs B1 (pp) | Query v harm >5pp |
|---|---:|---:|---:|---:|---:|
| B1 | 39.517189 | 36.243945 | 23.041295 | +0.000000 | 0 |
| GT-R16 | 41.894407 | 46.695016 | 35.806546 | +12.765252 | 0 |
| K1_R0.03_T2 | 40.605776 | 30.200662 | 17.856283 | -5.185012 | 5 |
| K1_R0.03_T1 | 40.462184 | 31.032848 | 16.992466 | -6.048828 | 4 |
| K1_R0.03_T0.5 | 39.588401 | 29.161884 | 16.855630 | -6.185665 | 4 |
| K5_R0.03_T2 | 39.560669 | 37.034678 | 16.475637 | -6.565658 | 4 |
| K1_R0.07_T2 | 37.946137 | 28.163803 | 16.297656 | -6.743639 | 4 |
| K3_R0.03_T0.5 | 38.824372 | 27.628559 | 16.162838 | -6.878457 | 5 |
| K3_R0.03_T1 | 37.062124 | 29.118953 | 15.790740 | -7.250555 | 4 |
| K1_R0.135_T0.5 | 40.305647 | 25.862745 | 15.476102 | -7.565193 | 5 |
| K1_R0.07_T0.5 | 37.859659 | 27.256340 | 15.083473 | -7.957822 | 5 |
| K3_R0.03_T2 | 37.343318 | 29.972719 | 15.057177 | -7.984118 | 5 |
| K3_R0.07_T1 | 38.130933 | 30.936243 | 14.925644 | -8.115651 | 4 |
| K3_R0.07_T2 | 40.184398 | 34.115991 | 14.860987 | -8.180308 | 4 |
| K1_R0.07_T1 | 39.264805 | 27.001696 | 14.659466 | -8.381829 | 6 |
| K5_R0.03_T1 | 37.806127 | 28.042316 | 14.177120 | -8.864175 | 5 |
| K3_R0.135_T0.5 | 38.204178 | 27.205563 | 14.057656 | -8.983639 | 6 |
| K1_R0.135_T2 | 38.897225 | 25.655266 | 13.972683 | -9.068612 | 6 |
| K1_R0.135_T1 | 39.215077 | 25.188705 | 13.966873 | -9.074421 | 7 |
| K5_R0.07_T0.5 | 38.943492 | 26.683165 | 13.585095 | -9.456200 | 6 |
| K5_R0.07_T2 | 37.421164 | 29.156309 | 13.418228 | -9.623067 | 6 |
| K3_R0.07_T0.5 | 38.658088 | 24.894452 | 13.409343 | -9.631951 | 7 |
| K3_R0.135_T2 | 38.755900 | 27.950771 | 13.164402 | -9.876892 | 7 |
| K5_R0.07_T1 | 35.810479 | 28.032934 | 13.003888 | -10.037406 | 7 |
| K5_R0.03_T0.5 | 35.843771 | 23.828483 | 12.553735 | -10.487559 | 5 |
| K5_R0.135_T0.5 | 37.653213 | 31.147178 | 11.187466 | -11.853828 | 8 |
| K3_R0.135_T1 | 33.149281 | 24.702144 | 10.019212 | -13.022083 | 6 |
| K5_R0.135_T2 | 30.961621 | 28.697052 | 9.765609 | -13.275686 | 6 |
| K5_R0.135_T1 | 35.389414 | 26.184144 | 9.415775 | -13.625520 | 8 |

Best configuration v delta: -5.185012 pp, descriptive unadjusted paired-parent 95% CI [-14.901582694581034, 1.8907530149811353]. Best v tail: 5 positive / 7 negative / 4 unchanged, five losses >5pp, best +20.494164 pp, worst -65.859151 pp. B1-good retention using metric >.5: temporal 6/6, spatial 4/5, tube 2/3.

GT-R16 is a previously saved source-GT diagnostic at historical radius .135455804, not a label-free competitor or an upper bound.

## Parameter influence within the complete Dev16 factorial

Values below average over the other two factors and all parents; they are not the best score at each parameter value.

| Factor | Marginal mean vIoU % by level | Marginal span (pp) | High minus low (pp), paired parent CI |
|---|---|---:|---|
| radius | 0.03: 15.769070, 0.07: 14.360420, 0.135: 12.336198 | 3.432872 | -3.432872; [-6.8576486199703295, -0.5213789801627067] |
| steps | 1: 15.684515, 3: 14.160889, 5: 12.620284 | 3.064231 | -3.064231; [-5.9841805224651585, -0.8374558646460096] |
| temporal_weight | 0.5: 14.263482, 1.0: 13.661243, 2.0: 14.540962 | 0.879720 | +0.277480; [-0.8390399272567476, 1.4094732397690044] |

For vIoU marginal span in this grid, radius > update count > temporal weight. Larger radius and more steps are worse on average; this does not uniquely identify expert noise as the cause and does not establish smaller untested radii as a fix. All confidence intervals are descriptive and unadjusted after development selection.

## Verification and remaining work

432 new native endpoints and 1296 latent updates were sealed before offline scoring. NumPy update reconstruction relative error: 0; norm relative maximum 1.1134e-15; radius overshoot 3.9839e-9. Scalar/tensor geometry maximum difference 4.4409e-16. Independent root parent/CI/tail/retention/ranking/factor aggregation maximum difference 7.1054e-15. Full-F gradients were saved only for the first query; projected gradients/full norms support the remaining update reconstruction.

Selection before the pause: K1_R0.03_T2, K1_R0.03_T1, K1_R0.03_T0.5, K5_R0.03_T2, K1_R0.07_T2, K3_R0.03_T0.5.

Dev64 expansion, single-expert ablations and the final complete trajectory projection-retention diagnostic remain unfinished. No new precision, radius, expert-input, model or OPD experiment was started. All negative cases, missing support, old failures and interrupted partial outputs are preserved. Fresh388 and target remain untouched.

Root aggregation implementation: `scripts/crosscheck_desta_native_phase.py`. Expert-input limitations: `docs/desta3d/EXPERT_INPUT_REVIEW.md`.
