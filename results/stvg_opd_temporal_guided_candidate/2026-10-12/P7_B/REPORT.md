# P7-B: temporal support guides sparse spatial observations

All512 clean cells reuse the same original Full initial state in each paired comparison. A uses old OPD boxes/time I0/observations I0; B uses those exact same boxes with projected time I1. C uses unchanged OPD and observations reallocated into I1; D uses unchanged Direct L1/GIoU5/2 in I1. All1,024 complete C/D fits globally sealed before new B scoring. No learning rate, reward, temperature, roster, new gate or teacher was selected from B results.

The 256 unique four-observation evidence packs are shared across two orders and C/D. Actual paid DINO calls and planned/admitted positions are reported from the real evidence receipts; no claim that sharing a cached benchmark pack eliminates a candidate deployment expert cost. TA-STVG and DINO were loaded in separate GPU processes; the original baseline PIDs/optimizer/prefix were preserved and resumed after the GPU phase.

| Target | Subset | Paired contrast | Valid parents | Mean (pp) | 95% CI (pp) | >5 pp harm parents | >20 pp harm parents |
|---|---|---|---:|---:|---|---:|---:|
| hc2 | all128 | coupling_C_minus_B_v | 128 | -0.457 | [-1.937, +0.767] | 12 | 1 |
| hc2 | all128 | D_minus_B_v | 128 | -1.084 | [-2.925, +0.601] | 20 | 3 |
| hc2 | all128 | C_minus_D_v | 128 | +0.627 | [-0.428, +1.783] | 7 | 1 |
| hc2 | all128 | C_minus_B_common_unobserved_real_IoU | 128 | -0.630 | [-2.745, +1.353] | 17 | 7 |
| hc2 | all128 | C_minus_B_gained_event_unobserved_real_IoU | 78 | -0.605 | [-3.952, +2.443] | 12 | 6 |
| hc2 | previous_P6_32 | coupling_C_minus_B_v | 32 | +0.180 | [-1.708, +2.103] | 2 | 0 |
| hc2 | previous_P6_32 | D_minus_B_v | 32 | -1.597 | [-5.143, +1.561] | 5 | 1 |
| hc2 | previous_P6_32 | C_minus_D_v | 32 | +1.778 | [-0.774, +5.021] | 2 | 0 |
| hc2 | previous_P6_32 | C_minus_B_common_unobserved_real_IoU | 32 | +0.447 | [-3.227, +4.463] | 7 | 1 |
| hc2 | previous_P6_32 | C_minus_B_gained_event_unobserved_real_IoU | 23 | +0.482 | [-3.370, +4.627] | 4 | 1 |
| hc2 | added96 | coupling_C_minus_B_v | 96 | -0.670 | [-2.530, +0.798] | 10 | 1 |
| hc2 | added96 | D_minus_B_v | 96 | -0.913 | [-3.112, +1.059] | 15 | 2 |
| hc2 | added96 | C_minus_D_v | 96 | +0.243 | [-0.844, +1.322] | 5 | 1 |
| hc2 | added96 | C_minus_B_common_unobserved_real_IoU | 96 | -0.989 | [-3.519, +1.266] | 10 | 6 |
| hc2 | added96 | C_minus_B_gained_event_unobserved_real_IoU | 55 | -1.059 | [-5.450, +2.899] | 8 | 5 |
| vidstg | all128 | coupling_C_minus_B_v | 128 | +0.099 | [-0.662, +0.848] | 4 | 1 |
| vidstg | all128 | D_minus_B_v | 128 | +0.227 | [-0.362, +0.849] | 7 | 0 |
| vidstg | all128 | C_minus_D_v | 128 | -0.128 | [-0.777, +0.492] | 4 | 1 |
| vidstg | all128 | C_minus_B_common_unobserved_real_IoU | 128 | -0.378 | [-2.138, +1.065] | 10 | 3 |
| vidstg | all128 | C_minus_B_gained_event_unobserved_real_IoU | 60 | -0.008 | [-1.499, +1.416] | 7 | 0 |
| vidstg | previous_P6_32 | coupling_C_minus_B_v | 32 | +0.560 | [-0.308, +1.751] | 0 | 0 |
| vidstg | previous_P6_32 | D_minus_B_v | 32 | -0.028 | [-0.565, +0.443] | 0 | 0 |
| vidstg | previous_P6_32 | C_minus_D_v | 32 | +0.588 | [-0.482, +2.102] | 0 | 0 |
| vidstg | previous_P6_32 | C_minus_B_common_unobserved_real_IoU | 32 | +0.562 | [-1.211, +2.693] | 2 | 0 |
| vidstg | previous_P6_32 | C_minus_B_gained_event_unobserved_real_IoU | 14 | +0.739 | [-1.325, +3.310] | 0 | 0 |
| vidstg | added96 | coupling_C_minus_B_v | 96 | -0.055 | [-1.032, +0.851] | 4 | 1 |
| vidstg | added96 | D_minus_B_v | 96 | +0.312 | [-0.452, +1.104] | 7 | 0 |
| vidstg | added96 | C_minus_D_v | 96 | -0.367 | [-1.168, +0.288] | 4 | 1 |
| vidstg | added96 | C_minus_B_common_unobserved_real_IoU | 96 | -0.691 | [-2.924, +1.048] | 8 | 3 |
| vidstg | added96 | C_minus_B_gained_event_unobserved_real_IoU | 46 | -0.236 | [-1.989, +1.431] | 7 | 0 |

Actual new DINO calls: **936**, unique evidence packs256, planned observations1024. Qualification used16 complete repeated fits; formal used1,024 new complete fits, A/B reuse recorded separately.

Common unobserved real frames exclude the union of both arms' actually visited DINO frames; they are genuine original decoder frames within the GT event, not interpolation-only evidence. Empty partitions remain undefined with counts. New/lost event support, event-outside visits, admitted reference quality and update geometry are strictly offline diagnostics; they never choose observations or gates.

Added96 mechanism decision: **both-target positive coupling plus unobserved-real improvement supported=False**. Even when useful, this is one-way temporal evidence -> observation allocation -> spatial update, not bidirectional joint optimization or temporal parameter TTA. Direct superiority is retained. Conditional original Full history is not an independently evolving online candidate stream. No production promotion or full official-query expansion is performed.

All four task readouts, R@.3/.5, all128/old32/new96, each order, entire positive/negative parent vectors, zero/undefined partitions and actual resource costs are available in the complete anonymous scalar files. Confidence intervals use10,000 paired parent resamples, seed20261012, unadjusted for multiplicity; all source cohorts have historical evaluation exposure.

## Support, before/after localization and negative cases

The offline groups below overlap and select cells by complete sealed GT diagnostics. Selected orders are averaged per parent inside each group. These are exploratory descriptions, not an online gate or a causal decomposition of detector identity versus coverage. Main all-parent harm counts remain in SUMMARY.

| Target | Offline group | Cells | Parents | C-B vIoU (pp) | 95% CI (pp) | C-B common unobserved real spatial IoU (pp) |
|---|---|---:|---:|---:|---|---:|
| hc2 | all_added96 | 192 | 96 | -0.670 | [-2.530,+0.798] | -0.989 |
| hc2 | gained_annotated_event_support | 110 | 55 | -0.865 | [-3.989,+1.613] | -1.053 |
| hc2 | lost_annotated_event_support | 22 | 11 | +0.204 | [-0.435,+0.820] | +1.101 |
| hc2 | increased_event_outside_duration | 116 | 58 | -1.822 | [-4.668,+0.348] | -2.768 |
| hc2 | new_observation_outside_event | 134 | 67 | +0.209 | [-1.020,+1.571] | -0.485 |
| hc2 | posthoc_temporal_harm_over5pp | 12 | 6 | -1.950 | [-4.141,+0.273] | -4.741 |
| hc2 | coupling_harm_over5pp | 19 | 12 | -15.168 | [-26.209,-8.050] | -26.628 |
| vidstg | all_added96 | 192 | 96 | -0.055 | [-1.032,+0.851] | -0.691 |
| vidstg | gained_annotated_event_support | 94 | 47 | -0.254 | [-1.876,+1.097] | -0.013 |
| vidstg | lost_annotated_event_support | 12 | 6 | -4.472 | [-16.216,+7.951] | -16.748 |
| vidstg | increased_event_outside_duration | 84 | 42 | -0.032 | [-1.718,+1.271] | +0.229 |
| vidstg | new_observation_outside_event | 104 | 52 | +0.069 | [-1.094,+1.148] | -1.070 |
| vidstg | posthoc_temporal_harm_over5pp | 9 | 5 | -3.518 | [-11.249,+1.495] | -16.018 |
| vidstg | coupling_harm_over5pp | 9 | 7 | -15.226 | [-28.891,-6.178] | -26.461 |

Every one of256 query budgets is matched:4 planned positions, the same per-query actual visit count,936 total visits per arm;22 queries have no call under the original parser rule. Visits move for205 parents. Original admission accepts812 positions versus814 at projected positions, a downstream consequence reported with quality and event membership rather than assumed to be pure coverage.

Among the complete512 paired cells, 108 have exactly identical admitted evidence (54 nonempty). New C final boxes and entire final parameter states match original Full bitwise for every such control. Changed-evidence C/D cases keep the same Full Before and original hyperparameters.

DIAGNOSTIC_EXTRA_ROWS retains the identical actual Full Before scores, B/C/D after-minus-before spatial localization, old versus new admitted-expert IoU and admission changes. Admitted-reference IoU does not establish object identity. Audited fit timings include recording and CPU math verification and are not isolated deployment latency; the originally authorized isolated cost experiment remains separate.

The final eight-page support-allocation case PDF shows both original orders of each target's best and worst added96 paired-parent effect, including identical Full Before, B/C/D actual curves, Native/projected/GT intervals and old/new visits. Every parent and severe negative remains in the complete scalar outputs. The initial four-page order1-only layout is preserved privately. These are offline selections; actual framewise payload and raw GT geometry stay private.

Actual case-view revision: support_allocation_cases_both_orders.pdf displays both original orders of each of the four paired-parent extremes (eight pages), including opposite signs across orders. The original order1-only export and all original receipts remain preserved privately.

Case interpretation: anonymous HC parent41 loses67.865 pp in the paired C-B mean although projected support recovers event coverage. Original admitted-reference IoU is28.732%, whereas all four new admitted references have zero overlap with annotated GT; C and D both collapse to zero mvIoU from the same strong Full Before. This associates the observed failure with harmful admitted supervision and is not proof of detector object identity. Anonymous Vid parent147 has almost unchanged high reference quality (91.444% to91.369%) yet C-B is+0.446 pp in order1 and-53.845 pp in order2; D remains beneficial in both. The same query and new evidence can therefore yield strongly different OPD behavior under the original inherited Full Before states. This is a saved conditional-state observation, not a controlled attribution to one optimizer variable or evidence that all temporal-guided adaptation is impossible.
