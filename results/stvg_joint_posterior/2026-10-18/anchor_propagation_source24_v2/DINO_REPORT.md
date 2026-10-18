# Cached DINO follow-up after strong-anchor source qualification

Parent source-GT gate passed before this branch. No new expert calls, training, target inference or Student fits. Same source24, Native time, frozen priors and deterministic no-added-noise propagation.

DINO_all retains all cached admitted anchors. DINO_common and GT_common use identical original DINO positions with source GT; no IoU filtering. Comparing common arms changes reference box values only. DINO_all versus GT_planned changes support too and must not be called a pure reference-quality comparison. All queries, including empty sets, remain. A is constrained anchor correction plus existing interpolation; AP adds frozen denoiser propagation with the same final anchor precision. These are diagnostics, not deployable rules.

## Full-cohort vIoU

Percent,12/domain. N/P are the original no-anchor controls; no rerun.

| Reference set | HC A | HC AP | Vid A | Vid AP |
|---|---:|---:|---:|---:|
| DINO_all | 43.420465 | 46.082688 | 17.946859 | 18.368515 |
| DINO_common | 43.420001 | 46.075772 | 17.931446 | 18.336308 |
| GT_common | 47.350648 | 50.101516 | 20.751115 | 21.124600 |

## Additional propagation over the same anchor correction

Source-paired difference AP-A in percentage points,95% t interval and positive/negative/neutral counts. Native decoder and benchmark interpolation supports are separate; outside-footprint support is reference-set-specific, identical between the two common sets.

| Set / metric | HC | Vid |
|---|---|---|
| DINO_all / v | +2.662223 [+1.215670,+4.108775] 12/0/0 | +0.421656 [-0.154032,+0.997345] 8/2/2 |
| DINO_all / unobserved_native_s | +4.465821 [+1.943317,+6.988324] 12/0/0 | +0.965607 [-0.649466,+2.580680] 9/1/0 |
| DINO_all / outside_footprint_native_s | +4.662677 [+2.530673,+6.794680] 12/0/0 | +1.221478 [-0.948821,+3.391777] 7/2/0 |
| DINO_all / unobserved_interpolated_s | +3.917132 [+1.865288,+5.968976] 12/0/0 | +0.803405 [-0.725357,+2.332166] 8/2/0 |
| DINO_common / v | +2.655771 [+1.201675,+4.109868] 12/0/0 | +0.404862 [-0.150419,+0.960144] 8/2/2 |
| DINO_common / unobserved_native_s | +4.449841 [+1.900439,+6.999243] 12/0/0 | +0.905596 [-0.670138,+2.481331] 9/1/0 |
| DINO_common / outside_footprint_native_s | +4.635841 [+2.482812,+6.788870] 12/0/0 | +1.146041 [-0.956898,+3.248980] 6/3/0 |
| DINO_common / unobserved_interpolated_s | +3.900360 [+1.827238,+5.973482] 12/0/0 | +0.760234 [-0.735173,+2.255641] 8/2/0 |
| GT_common / v | +2.750868 [+1.197067,+4.304670] 12/0/0 | +0.373485 [-0.207698,+0.954668] 7/3/2 |
| GT_common / unobserved_native_s | +4.285665 [+1.936346,+6.634984] 12/0/0 | +0.795508 [-0.833189,+2.424206] 8/2/0 |
| GT_common / outside_footprint_native_s | +4.670322 [+2.416948,+6.923696] 12/0/0 | +0.867853 [-1.050885,+2.786591] 6/3/0 |
| GT_common / unobserved_interpolated_s | +4.106924 [+1.711895,+6.501953] 12/0/0 | +0.706561 [-0.841348,+2.254469] 7/3/0 |

## Same-support GT minus DINO reference-quality intervention

| Arm / metric | HC | Vid |
|---|---|---|
| A / v | +3.930647 [+1.251737,+6.609557] | +2.819670 [-0.628255,+6.267595] |
| A / unobserved_native_s | +2.536295 [+0.322242,+4.750348] | +4.374769 [-1.977845,+10.727383] |
| A / outside_footprint_native_s | +0.000000 [+0.000000,+0.000000] | +0.000000 [+0.000000,+0.000000] |
| AP / v | +4.025744 [+1.237567,+6.813921] | +2.788292 [-0.593442,+6.170027] |
| AP / unobserved_native_s | +2.372119 [+0.129061,+4.615178] | +4.264681 [-1.878813,+10.408175] |
| AP / outside_footprint_native_s | +0.034482 [-0.107996,+0.176959] | -0.278188 [-0.759039,+0.202663] |

## Interpretation boundary

A/AP anchor errors relative to their own supervision are <2e-6, including DINO. Therefore differences are not caused by one arm failing to apply its supplied sparse boxes. DINO reference geometry against source GT is separately recorded in per-case anchor metrics; low IoU does not identify wrong persons without actual semantic pixel review. Hard anchoring of erroneous DINO is an intervention, not a proposed deployment method or proof about the old soft mixture guidance.

Unanchored P can itself improve trajectories: AP-A is the total benefit of adding the frozen propagation procedure to exact anchoring, not solely the causal contribution of sparse conditions. The parent interaction (AP-P)-(A-N) remains necessary for interpreting whether anchors add propagation beyond the prior alone. No inference that diffusion is uniquely necessary; no deterministic learned corrector was compared here. No training, new target TTA, dense OPD or production promotion follows.
