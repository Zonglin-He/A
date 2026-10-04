# Critic-DeCoTA: matched LN1/16 inheritance P1

The user explicitly authorized LN inheritance after the mixed P0 readback. This experiment freezes the P0 energy critic and adds only the original C1 LN1/16 writeback. It is a small, historically exposed research panel; production registries are unchanged.

| Corruption panel | Frozen vIoU % | Before − Frozen pp [95% CI] | Online − Episodic pp [95% CI] | Online − Frozen pp [95% CI] |
|---|---:|---:|---:|---:|
| vidstg search | 16.215 | 0.795 [0.297, 1.391] | -0.186 [-1.791, 1.009] | 1.959 [-0.484, 4.100] |
| vidstg confirm | 32.953 | 0.660 [0.213, 1.206] | 0.150 [0.012, 0.309] | 3.134 [-1.814, 8.777] |
| hc2 search | 29.867 | 2.707 [1.951, 3.553] | -0.311 [-1.097, 0.262] | 5.022 [3.058, 6.946] |
| hc2 confirm | 26.478 | 1.388 [0.845, 2.007] | 0.284 [-0.115, 0.795] | 4.259 [1.423, 7.540] |

The bootstrap resamples sources, averages both orders and the five corruptions within source, and uses 10,000 paired draws, seed20261004. Each dataset has32 development/16 confirmation sources, one query/source; all are historically exposed. All1,152 online arrivals actually run. P0 episodic readouts are hash-bound reused controls, not rerun or duplicated independent observations.

## Exact state lifecycle

- At each arrival inherit only1,536 last-spatial-block norm1/norm3/norm4 affine coordinates; reset the256-dimensional query residual and construct fresh Adam.
- Run the unchanged P0 critic: temperatures1/1, Adam lr.03,10 updates, earliest minimum of its own energy among steps0..10.
- Emit the selected current correction; commit `LN_next = LN_arrival + (LN_selected − LN_arrival)/16`. The selected query residual never enters the next state.
- Reset chains at dataset/development-or-confirmation/corruption/order boundaries. Native source I0 stays fixed; no temporal adaptation, new DINO calls, new full-backbone forward, gate or sweep.
- Every query still has four cached native-I0 expert observations. InheritedBefore is measured before current-query expert correction; it is not a stream with25% expert arrivals and not a formal nonexpert endpoint.
- First arrivals independently reproduce every P0 critic step bitwise. All predecessor hashes, coordinate arithmetic, resets, independent NumPy energies/Adam updates and official/vectorized dense metrics pass after the global prediction barrier.

## Clean, order effects and negative tails

| Confirmation comparison | Clean ΔvIoU pp [95% CI] | Corrupt >5/>20pp harms | Correct→wrong / wrong→correct at .3 | At .5 | Positive sources |
|---|---:|---:|---:|---:|---:|
| vidstg / inherited | 0.597 [0.130, 1.155] | 0/0 | 0/0 | 0/0 | 14/16 |
| vidstg / online_vs_episodic | 0.103 [-0.061, 0.289] | 1/0 | 0/0 | 0/0 | 12/16 |
| vidstg / online_vs_frozen | 2.367 [-4.454, 8.837] | 8/8 | 8/0 | 0/12 | 15/16 |
| hc2 / inherited | 1.310 [0.607, 2.042] | 0/0 | 0/3 | 0/10 | 15/16 |
| hc2 / online_vs_episodic | 0.085 [-0.316, 0.497] | 0/0 | 0/2 | 0/8 | 9/16 |
| hc2 / online_vs_frozen | 4.415 [1.696, 7.481] | 4/0 | 2/2 | 0/42 | 11/16 |

Harm counts are actual order-specific arrivals, correlated within source; bootstrap units remain sources. Current correction and inherited utility are separate endpoints. tIoU is exactly unchanged.

| Confirmation order | Before − Frozen pp | Online − Episodic pp | Online − Frozen pp |
|---|---:|---:|---:|
| vidstg/order1 | 0.743 [0.306, 1.260] | 0.193 [-0.030, 0.434] | 3.176 [-1.923, 8.975] |
| vidstg/order2 | 0.576 [0.010, 1.268] | 0.108 [-0.170, 0.386] | 3.092 [-1.728, 8.573] |
| hc2/order1 | 1.430 [0.840, 2.014] | 0.440 [0.059, 0.950] | 4.416 [1.556, 7.700] |
| hc2/order2 | 1.347 [0.644, 2.199] | 0.128 [-0.330, 0.655] | 4.103 [1.267, 7.372] |

## Positive and negative inherited cases

| Dataset | Anonymous source/condition/order/arrival | Before − Frozen pp | Online − Episodic pp |
|---|---|---:|---:|
| vidstg | 35/motion_blur_5/order2/10 | -1.352 | -2.090 |
| vidstg | 37/occlusion_5/order2/12 | 4.411 | -0.360 |
| hc2 | 41/frame_freeze_5/order2/8 | -0.292 | 0.411 |
| hc2 | 35/motion_blur_5/order2/15 | 5.826 | -0.055 |

Every source, condition, order, per-step energy/GT diagnostic, gross gain/loss and correctness transition is exported, including severe negative cases. No GT-selected state or new parameter is introduced.

## Actual cost and scoped decision

| Dataset | Actual online arrivals | Spatial backward calls | Worker wall minutes | New expert/full backbone calls |
|---|---:|---:|---:|---:|
| vidstg | 576 | 4440 | 4.25 | 0/0 |
| hc2 | 576 | 5760 | 4.76 | 0/0 |

Four extra cached smoke fits test source-start parity and second-arrival state inheritance before GT. Worker wall time includes cache IO, replay, parameter copying and logging; it is not pure GPU kernel time.

- vidstg: inherited before gain has positive CI; current online versus episodic has positive CI on this panel.
- hc2: inherited before gain has positive CI; current online versus episodic is inconclusive on this panel.

No method is promoted. Small-panel mean differences do not establish full-data, cross-domain or future25%-expert-stream benefit. P0 Vid uncertainty and severe harms remain part of the evidence.

Artifacts: [protocol](../protocols/tastvg_decota_critic_ln_p1_v1.md), [all anonymous results](../results/tastvg_decota_critic_ln_p1/2026-10-04/), [root audit](../results/tastvg_decota_critic_ln_p1/2026-10-04/ROOT_AUDIT.json).


## Root readback: inherited utility and current correction are different results

On confirmation corruption, before-current-correction inheritance improves
14/16 Vid and15/16 HC source means, with paired CIs above zero in each
original order. The largest source accounts for30.14%/20.70% of positive
inherited gain. This is evidence for this fixed LN-transfer mechanism on
this panel, beyond a single positive source; it is not a25%-expert-stream
or fresh/full-data qualification.

The online-after increment over episodic critic is small: Vid+0.1504pp
[+0.0116,+0.3091], HC+0.2838pp[−0.1151,+0.7950]. Vid's separately evaluated
order intervals both cross zero. Development increments are negative
point estimates with intervals crossing zero. Therefore a broadly stable
increase in current correction over P0 is not established.

Critically, LN transfer does not remove the current critic's Vid tail:
all8 confirmation corruption online-after harms above20pp are still the
same4 unique inputs from anonymous source35, evaluated in two orders.
Before-current-correction has zero harms above5pp in both datasets, while
current correction itself produces8 such severe Vid arrivals and11 HC
harms above5pp. On HC, net online versus Frozen has4 harms above5pp and
zero above20pp. These counts have different baselines and cannot be
interchanged. The current proxy-to-tube failure remains, even though
inherited LN has useful next-query effects.

Empty expert evidence never selects a new state: online-after exactly
equals inherited-before on those arrivals. Nevertheless inherited LN
can already have altered the tube; empty evidence is not a reset to
Frozen. ROOT_CASE_REVIEW.json verifies these no-op cells and preserves
positive and negative inherited/current examples.

No P0 scientific file, C1 lock or production registry was changed.
Do not describe the critic as a native-policy OPD mechanism or uniquely
attribute its result to a different teacher distribution. P0 changed
the supervision-construction block, and P1 specifically tests the extra
state lifecycle. Retain this positive LN-inheritance evidence while
keeping Vid current-correction uncertainty and severe harms visible.
