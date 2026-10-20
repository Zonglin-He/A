# Harmful videos and parameter links

All analyses use already sealed coarse predictions, saved metric receipts and saved update processes. All128 parents per dataset remain in selection. Harmful-parent groups below are retrospective explanations, never online gates. Teacher quality means actual available-anchor IoU; low IoU alone does not identify an identity error. Time counterfactuals use the exact same saved boxes with Native versus Strong; neither time branch is changed.

## hc2: baseline C02_S40_A100

54/128 parents have negative full-vIoU adaptation relative to the same E+Strong. Flags: `{'teacher_mean_no_better_than_E': 18, 'teacher_mean_IoU_below_half': 17, 'reference_fit_improved_while_full_worse': 54, 'unobserved_motion_exceeds_observed': 21, 'no_Strong_GT_overlap': 0, 'halving_output_reduces_harm': 53, 'smaller_lr_reduces_harm': 44, 'earlier_stop_reduces_harm': 41, 'softer_tau_reduces_harm': 27}`.

| Alias | baseline−E pp | earlier20 | half-output | smaller-lr | softer-tau | larger-lr | coarse-winner | teacher−E anchor pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RC036 | -3.4926 | -0.4870 | -1.5877 | -0.5221 | -7.0163 | -8.5646 | -12.4287 | -8.1395 |
| RC097 | -2.4445 | -2.4274 | -0.5872 | -0.5739 | -2.3654 | -2.6198 | -2.6522 | +60.3319 |
| RC122 | -1.6264 | -1.2804 | -0.8585 | -0.7517 | -1.6563 | -1.7478 | -1.4519 | -10.7552 |
| RC125 | -1.0846 | -1.5236 | -0.0048 | -1.6375 | -1.2258 | -0.3694 | -0.4133 | +27.5808 |
| RC063 | -0.9376 | -1.0069 | -0.3471 | -0.9517 | -0.8185 | -1.0618 | -0.9240 | +10.1451 |
| RC080 | -0.8530 | -0.0102 | -0.4602 | +0.0149 | -0.9983 | -1.9980 | -2.0914 | +37.6970 |
| RC109 | -0.8170 | -0.5185 | -0.4013 | -0.4067 | -0.8129 | -1.2215 | -1.0290 | +4.9762 |
| RC070 | -0.5094 | -0.4754 | -0.4144 | -0.3968 | -0.5070 | -0.4763 | -0.4699 | +26.0904 |
| RC114 | -0.4820 | -0.5204 | -0.0051 | -0.2034 | -0.3060 | +0.9263 | -0.2276 | +26.7973 |
| RC025 | -0.4441 | -0.6745 | -0.2241 | -0.4689 | -0.4435 | -0.4549 | -0.4787 | +11.9431 |
| RC088 | -0.4003 | -0.3323 | -0.1978 | -0.2475 | -0.3986 | -0.4446 | -0.4660 | +52.7990 |
| RC060 | -0.2298 | -0.2788 | -0.1078 | -0.1588 | -0.6380 | -0.3196 | -0.9588 | -6.0273 |

| Matched change vs original baseline | all128 mean pp | baseline harmed mean pp | remaining parents mean pp |
|---|---:|---:|---:|
| same_path_earlier20 | -0.115149 | +0.081709 | -0.258803 |
| same_path_later40 | +0.000000 | +0.000000 | +0.000000 |
| same_path_output_half | -0.155665 | +0.182385 | -0.402350 |
| same_lr_steps_softer_tau | -0.019975 | -0.075285 | +0.020386 |
| same_steps_tau_smaller_lr | -0.173493 | +0.155736 | -0.413741 |
| same_steps_tau_larger_lr | +0.110806 | -0.166560 | +0.313209 |
| whole_panel_coarse_winner | +0.116275 | -0.323170 | +0.436950 |
## vidstg: baseline C02_S30_A100

39/128 parents have negative full-vIoU adaptation relative to the same E+Strong. Flags: `{'teacher_mean_no_better_than_E': 13, 'teacher_mean_IoU_below_half': 13, 'reference_fit_improved_while_full_worse': 35, 'unobserved_motion_exceeds_observed': 20, 'no_Strong_GT_overlap': 0, 'halving_output_reduces_harm': 38, 'smaller_lr_reduces_harm': 36, 'earlier_stop_reduces_harm': 30, 'softer_tau_reduces_harm': 20}`.

| Alias | baseline−E pp | earlier20 | half-output | smaller-lr | softer-tau | larger-lr | coarse-winner | teacher−E anchor pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BV030 | -7.1584 | -7.2512 | -6.7949 | -7.0095 | -7.1164 | -7.0952 | +0.0216 | +26.4219 |
| BV022 | -4.5344 | -2.3742 | -2.4857 | -1.7479 | -1.8183 | -5.2047 | -0.8248 | -5.5273 |
| BV064 | -2.8003 | -2.4484 | -2.0320 | -0.2534 | -2.8766 | -2.9974 | -0.1617 | +1.5806 |
| BV083 | -1.6475 | -1.5459 | -0.9776 | -1.5843 | -1.6313 | -1.8198 | -2.2871 | +67.4357 |
| BV023 | -1.4425 | -1.6862 | -0.8535 | -0.6620 | +0.4984 | -1.1793 | +0.3152 | +29.5370 |
| BV114 | -1.3929 | -1.0447 | -0.5531 | -0.0028 | -1.6682 | -2.4464 | +0.0954 | +50.3598 |
| BV108 | -1.0681 | -0.5569 | -0.4884 | -0.3147 | -2.9177 | -2.4116 | -0.3636 | +31.0332 |
| BV074 | -0.7640 | -1.0117 | -0.5431 | -0.6097 | -0.5072 | -0.5279 | -0.0080 | -0.4325 |
| BV047 | -0.5797 | -0.2827 | -0.2779 | -0.0435 | -0.5905 | -0.5916 | -0.0043 | +15.8244 |
| BV098 | -0.5499 | -0.3360 | -0.2864 | -0.1351 | -0.4491 | -1.2495 | -0.0491 | NA |
| BV103 | -0.4340 | -0.0251 | -0.2462 | +0.0178 | -0.2323 | +1.0840 | +0.0138 | +8.1246 |
| BV072 | -0.4228 | -0.5230 | -0.2125 | -0.4603 | -0.3542 | +0.2744 | -0.4335 | -1.5581 |

| Matched change vs original baseline | all128 mean pp | baseline harmed mean pp | remaining parents mean pp |
|---|---:|---:|---:|
| same_path_earlier20 | +0.012668 | +0.133715 | -0.040375 |
| same_path_later40 | -0.009006 | -0.075425 | +0.020099 |
| same_path_output_half | -0.102295 | +0.208510 | -0.238490 |
| same_lr_steps_softer_tau | +0.035778 | +0.096937 | +0.008978 |
| same_steps_tau_smaller_lr | +0.028539 | +0.318056 | -0.098328 |
| same_steps_tau_larger_lr | -0.011281 | -0.084094 | +0.020626 |
| whole_panel_coarse_winner | +0.076360 | +0.554973 | -0.133369 |

The whole-panel tradeoff determines the dataset-wide parameter, not the individually best recovery shown in a row. Lower reference loss is not task correctness. If less update rescues harmful videos but removes larger gains elsewhere, shrinking globally need not be better. Initialization-only gamma curves are labeled separately from actual adapted results; nonunit gamma must receive actual OPD before entering the final adaptive comparison.
