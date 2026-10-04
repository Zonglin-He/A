# Unchanged A under clean cross-dataset shift

Predeclared qualification decision: **GO in this exposed qualification scope**. No method promotion, target retuning, arm selection or automatic extension.

Both Full−Frozen contrasts are approximately one percentage point in this run. This is modest qualification evidence, not reproduction of the historical 4–8pp DeCoTA gains.

Official source checkpoints and source-domain development bundles; one query per historical target source; three fixed orders, clean inputs and nominal 25% specialist availability. Vid→HC2: 135 sources/405 arrivals; HC2→Vid: 384 sources/1152 arrivals. 1557 arrivals, 390 scheduled specialist arrivals. The cohorts and development history are exposed; this is not a fresh test.

| Source→target | Method | vIoU % | Δ vs source Frozen, pp [95% CI] | tIoU % | sIoU % | @.3 % | @.5 % |
|---|---|---:|---|---:|---:|---:|---:|
| vid_to_hc2 | Frozen | 20.559 | — | 40.312 | 51.012 | 24.444 | 9.630 |
| vid_to_hc2 | Fast-only | 21.429 | +0.871 [+0.397, +1.387] | 42.104 | 51.012 | 25.679 | 10.617 |
| vid_to_hc2 | Spatial-only | 20.758 | +0.199 [-0.165, +0.565] | 40.312 | 51.184 | 26.173 | 11.111 |
| vid_to_hc2 | Full A | 21.620 | +1.061 [+0.477, +1.665] | 42.104 | 51.184 | 26.914 | 11.111 |
| vid_to_hc2 | Target-trained reference* | 32.891 | +12.332 [+9.230, +15.509] | 61.002 | 52.133 | 53.333 | 23.704 |
| hc2_to_vid | Frozen | 14.107 | — | 32.592 | 39.625 | 18.490 | 4.427 |
| hc2_to_vid | Fast-only | 14.538 | +0.431 [+0.132, +0.730] | 34.058 | 39.625 | 18.403 | 4.774 |
| hc2_to_vid | Spatial-only | 14.633 | +0.526 [+0.316, +0.723] | 32.592 | 41.114 | 19.618 | 5.382 |
| hc2_to_vid | Full A | 15.103 | +0.996 [+0.639, +1.356] | 34.058 | 41.114 | 20.052 | 5.990 |
| hc2_to_vid | Target-trained reference* | 21.912 | +7.805 [+6.061, +9.576] | 44.146 | 47.353 | 33.854 | 15.625 |

*Target-trained is a supervised comparison reference, not a mathematical upper bound or fair TTA baseline. All methods use this run’s actual TA outputs; old TubeDETR/DeCoTA scores are not substituted. Native lowercase text preprocessing and original parent sampling grids are preserved.

## Persistent state versus current expert correction

| Direction | Full−Fast, all, pp [95% CI] | Full−Fast, nonexpert | Spatial−Frozen, nonexpert | Current Fast on inherited state |
|---|---|---|---|---|
| vid_to_hc2 | +0.191 [-0.181, +0.561] | +0.414 [-0.126, +1.123] | +0.414 [-0.126, +1.123] | +0.862 [+0.393, +1.374] |
| hc2_to_vid | +0.565 [+0.355, +0.766] | +0.581 [+0.417, +0.751] | +0.581 [+0.417, +0.751] | +0.470 [+0.151, +0.791] |

Current outputs precede the current spatial write. Full−Fast therefore measures consequences of prior spatial state, including its effects on native temporal candidates. It is not uniquely a box-coordinate causal effect. Fast-disabled live controls verify identical spatial states, outputs and gradients for two scheduled inputs per direction; all persistent and inner-step state chains are audited. No query or probe offset is carried in Frozen/Fast-only.

## Where correct outputs are lost

### vid_to_hc2

Fixed-time inherited boxes: +0.199 [-0.165, +0.565] pp; inherited native interval: +0.000 [+0.000, +0.000] pp; current Fast: +0.862 [+0.393, +1.374] pp. This is one ordered accounting path; the three summing contrasts do not establish independent causal modules.

The largest gross loss along this ordered three-stage path is **IB_minus_F** (0.713 pp). This identifies the largest measured damage on this accounting path, not a unique causal explanation.

| Transition | Gross gain pp | Gross loss pp | >5pp harm | >20pp harm | Correct→wrong at .3 / at .5 |
|---|---:|---:|---:|---:|---|
| A_minus_F | 1.885 | 0.824 | 18 | 2 | 6/99 / 3/39 |
| A_minus_T | 0.933 | 0.743 | 14 | 1 | 3/104 / 3/43 |
| IB_minus_F | 0.912 | 0.713 | 14 | 1 | 4/99 / 2/39 |
| A_minus_S | 1.078 | 0.216 | 4 | 1 | 5/106 / 3/45 |
| S_minus_IB | 0.000 | -0.000 | 0 | 0 | 0/106 / 0/45 |

On 102 scheduled arrivals: temporal selection regret +2.343 [+1.331, +3.531] pp; fixed-space GT-time minus candidate oracle +24.591 [+20.391, +29.180] pp. These conditional upper bounds locate available opportunity, not deployable gains.

| vIoU threshold | Good interval exists | Good interval missed | No good interval in support | Native correct destroyed |
|---|---:|---:|---:|---:|
| 0.3 | 45 | 8 | 57 | 5 |
| 0.5 | 20 | 9 | 82 | 3 |

The preceding vIoU thresholds require the fixed spatial trajectory to be correct as well. An absent correct tube does not prove an absent correct time interval. Temporal-only counts separate these cases:

| tIoU threshold | Good interval exists | Good interval missed | No good interval in support | Native time correct destroyed |
|---|---:|---:|---:|---:|
| 0.3 | 82 | 1 | 20 | 1 |
| 0.5 | 61 | 6 | 41 | 4 |

For final Full A output, replacing only time with GT at fixed A boxes adds +29.606 [+26.308, +32.960] pp; ideal GT boxes at fixed A time add +20.484 [+17.805, +23.313] pp. These are conditional evaluation replacements, not decoder inputs or additive independent contributions.

Spatial evidence at 102 scheduled arrivals: 0 empty; 9 nonempty but no reference on the GT event. expert_event_frame_precision: +39.437 [+33.636, +45.693]%; expert_event_GT_IoU: +64.310 [+57.854, +70.434]%. Nonempty/event-eligible denominators are saved separately; missing evidence is not assigned zero localization quality.


| Inner step | Calls | Empty evidence | Nonempty but no GT-event reference | Loss↓ / GT-time quality↓ | Useful top / harmful update | Net GT-time update pp |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 102 | 0 | 9 | 35 | 15 | 0.864 |

The useful-top count allows a tied top group; unique useful-top/harm counts are step 1: 15.

| Step / threshold | Correct support exists | No correct support | Reward-top group misses correct support | Correct central output destroyed by update |
|---|---:|---:|---:|---:|
| 1 / 0.3 | 82 | 20 | 0 | 0 |
| 1 / 0.5 | 64 | 38 | 0 | 1 |

| Step | Observed GT-event frame sIoU change, pp [95% CI] | Other dense GT-event frames |
|---|---|---|
| 1 | +1.204 [-0.228, +2.654] | +0.836 [-0.349, +1.871] |

Step counts are repeated online arrivals, not independent sample sizes. Updates are evaluated on fixed actual output time and separately GT time; loss decrease is not task correctness. Immediate post-update diagnosis does not count as a formal current prediction or establish future transfer. Correct-support miss counts and empty evidence are retained, including negative tails. The nine probe upper bound does not bound arbitrary gradient adaptation.

#### Dataset characteristics and limits of attribution

Net complete-arrival spatial update (all executed steps): actual A-time +0.617 [+0.071, +1.163] pp; GT-time +0.838 [-0.348, +1.875] pp. The latter removes current temporal masking; both are post-update diagnostics, not the sealed current output.

event_duration_seconds: 7.4530; event_fraction: 0.3756; GT_box_area_fraction: 0.2539; caption_words: 16.8667; observed_frames: 100.0000; input_grid_GT_frame_fraction: 0.3727.
- small_object: 1 sources; Full−Frozen +2.839 [+2.839, +2.839] pp; Full−Fast +0.678 [+0.678, +0.678] pp.
- larger_object: 134 sources; Full−Frozen +1.048 [+0.446, +1.666] pp; Full−Fast +0.187 [-0.193, +0.573] pp.
- short_event: 28 sources; Full−Frozen +1.124 [-0.443, +2.695] pp; Full−Fast -0.001 [-0.706, +0.521] pp.
- longer_event: 107 sources; Full−Frozen +1.045 [+0.409, +1.687] pp; Full−Fast +0.241 [-0.196, +0.688] pp.

Short-event (<25% observed clip) and small-box (<2% image area) slices are posthoc diagnostic associations, not online gates or causal effects. Exposure, one-query selection and available-media filtering limit population claims.

A one-source slice has no estimable between-source uncertainty; its degenerate bootstrap interval is not population evidence.

Domain-gap recovery point estimate: 8.61%; denominator 12.332 pp [9.230, 15.509]. Stable positive denominator: True. The saved ratio CI is conditional on positive denominator bootstrap draws; an unstable denominator prevents a reliable recovered-gap claim.
### hc2_to_vid

Fixed-time inherited boxes: +0.526 [+0.316, +0.723] pp; inherited native interval: +0.000 [+0.000, +0.000] pp; current Fast: +0.470 [+0.151, +0.791] pp. This is one ordered accounting path; the three summing contrasts do not establish independent causal modules.

The largest gross loss along this ordered three-stage path is **IB_minus_F** (0.362 pp). This identifies the largest measured damage on this accounting path, not a unique causal explanation.

| Transition | Gross gain pp | Gross loss pp | >5pp harm | >20pp harm | Correct→wrong at .3 / at .5 |
|---|---:|---:|---:|---:|---|
| A_minus_F | 1.604 | 0.608 | 27 | 7 | 19/213 / 5/51 |
| A_minus_T | 0.935 | 0.369 | 14 | 2 | 6/212 / 5/55 |
| IB_minus_F | 0.889 | 0.362 | 14 | 2 | 7/213 / 5/51 |
| A_minus_S | 0.812 | 0.342 | 16 | 6 | 13/226 / 2/62 |
| S_minus_IB | 0.000 | -0.000 | 0 | 0 | 0/226 / 0/62 |

On 288 scheduled arrivals: temporal selection regret +3.025 [+2.228, +3.899] pp; fixed-space GT-time minus candidate oracle +19.775 [+17.339, +22.258] pp. These conditional upper bounds locate available opportunity, not deployable gains.

| vIoU threshold | Good interval exists | Good interval missed | No good interval in support | Native correct destroyed |
|---|---:|---:|---:|---:|
| 0.3 | 76 | 18 | 212 | 13 |
| 0.5 | 26 | 5 | 262 | 2 |

The preceding vIoU thresholds require the fixed spatial trajectory to be correct as well. An absent correct tube does not prove an absent correct time interval. Temporal-only counts separate these cases:

| tIoU threshold | Good interval exists | Good interval missed | No good interval in support | Native time correct destroyed |
|---|---:|---:|---:|---:|
| 0.3 | 198 | 26 | 90 | 9 |
| 0.5 | 131 | 32 | 157 | 11 |

For final Full A output, replacing only time with GT at fixed A boxes adds +26.011 [+23.934, +28.111] pp; ideal GT boxes at fixed A time add +18.955 [+17.162, +20.800] pp. These are conditional evaluation replacements, not decoder inputs or additive independent contributions.

Spatial evidence at 288 scheduled arrivals: 21 empty; 68 nonempty but no reference on the GT event. expert_event_frame_precision: +43.685 [+38.693, +48.725]%; expert_event_GT_IoU: +66.459 [+61.741, +70.980]%. Nonempty/event-eligible denominators are saved separately; missing evidence is not assigned zero localization quality.


| Inner step | Calls | Empty evidence | Nonempty but no GT-event reference | Loss↓ / GT-time quality↓ | Useful top / harmful update | Net GT-time update pp |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 288 | 21 | 68 | 59 | 21 | 0.346 |
| 2 | 267 | 0 | 68 | 64 | 22 | 0.027 |
| 3 | 267 | 0 | 68 | 64 | 23 | 0.142 |
| 4 | 267 | 0 | 68 | 65 | 23 | 0.120 |
| 5 | 267 | 0 | 68 | 64 | 22 | 0.081 |
| 6 | 267 | 0 | 68 | 67 | 29 | 0.035 |
| 7 | 267 | 0 | 68 | 69 | 25 | 0.015 |
| 8 | 267 | 0 | 68 | 68 | 26 | 0.161 |

The useful-top count allows a tied top group; unique useful-top/harm counts are step 1: 19, step 2: 20, step 3: 21, step 4: 21, step 5: 20, step 6: 27, step 7: 23, step 8: 24.

| Step / threshold | Correct support exists | No correct support | Reward-top group misses correct support | Correct central output destroyed by update |
|---|---:|---:|---:|---:|
| 1 / 0.3 | 183 | 105 | 1 | 0 |
| 1 / 0.5 | 123 | 165 | 1 | 0 |
| 2 / 0.3 | 179 | 88 | 1 | 1 |
| 2 / 0.5 | 119 | 148 | 1 | 2 |
| 3 / 0.3 | 179 | 88 | 1 | 1 |
| 3 / 0.5 | 119 | 148 | 1 | 0 |
| 4 / 0.3 | 179 | 88 | 1 | 0 |
| 4 / 0.5 | 119 | 148 | 1 | 0 |
| 5 / 0.3 | 178 | 89 | 0 | 0 |
| 5 / 0.5 | 119 | 148 | 1 | 1 |
| 6 / 0.3 | 178 | 89 | 0 | 1 |
| 6 / 0.5 | 118 | 149 | 0 | 2 |
| 7 / 0.3 | 179 | 88 | 2 | 1 |
| 7 / 0.5 | 119 | 148 | 3 | 1 |
| 8 / 0.3 | 180 | 87 | 1 | 0 |
| 8 / 0.5 | 117 | 150 | 0 | 0 |

| Step | Observed GT-event frame sIoU change, pp [95% CI] | Other dense GT-event frames |
|---|---|---|
| 1 | +0.543 [+0.278, +0.869] | +0.355 [+0.197, +0.544] |
| 2 | +0.038 [-0.270, +0.258] | +0.015 [-0.197, +0.182] |
| 3 | +0.134 [+0.006, +0.261] | +0.192 [+0.074, +0.332] |
| 4 | +0.136 [-0.063, +0.390] | +0.119 [-0.056, +0.308] |
| 5 | +0.053 [-0.238, +0.283] | +0.068 [-0.101, +0.211] |
| 6 | +0.434 [+0.099, +0.966] | -0.066 [-0.628, +0.319] |
| 7 | -0.216 [-0.775, +0.140] | +0.058 [-0.195, +0.292] |
| 8 | +0.274 [+0.103, +0.502] | +0.155 [+0.074, +0.255] |

Step counts are repeated online arrivals, not independent sample sizes. Updates are evaluated on fixed actual output time and separately GT time; loss decrease is not task correctness. Immediate post-update diagnosis does not count as a formal current prediction or establish future transfer. Correct-support miss counts and empty evidence are retained, including negative tails. The nine probe upper bound does not bound arbitrary gradient adaptation.

#### Dataset characteristics and limits of attribution

Net complete-arrival spatial update (all executed steps): actual A-time +0.367 [+0.122, +0.606] pp; GT-time +0.863 [+0.359, +1.338] pp. The latter removes current temporal masking; both are post-update diagnostics, not the sealed current output.

event_duration_seconds: 10.2026; event_fraction: 0.4347; GT_box_area_fraction: 0.2226; caption_words: 10.1745; observed_frames: 112.2188; input_grid_GT_frame_fraction: 0.4286.
- small_object: 43 sources; Full−Frozen +0.315 [+0.041, +0.671] pp; Full−Fast +0.124 [-0.052, +0.347] pp.
- larger_object: 341 sources; Full−Frozen +1.082 [+0.682, +1.478] pp; Full−Fast +0.621 [+0.387, +0.844] pp.
- short_event: 181 sources; Full−Frozen +0.206 [-0.256, +0.630] pp; Full−Fast +0.286 [+0.132, +0.441] pp.
- longer_event: 203 sources; Full−Frozen +1.701 [+1.177, +2.252] pp; Full−Fast +0.815 [+0.446, +1.148] pp.

Short-event (<25% observed clip) and small-box (<2% image area) slices are posthoc diagnostic associations, not online gates or causal effects. Exposure, one-query selection and available-media filtering limit population claims.

A one-source slice has no estimable between-source uncertainty; its degenerate bootstrap interval is not population evidence.

Domain-gap recovery point estimate: 12.76%; denominator 7.805 pp [6.061, 9.576]. Stable positive denominator: True. The saved ratio CI is conditional on positive denominator bootstrap draws; an unstable denominator prevents a reliable recovered-gap claim.

The original [VidSTG paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Where_Does_It_Exist_Spatio-Temporal_Video_Grounding_for_Multi-Form_Sentences_CVPR_2020_paper.html) defines untrimmed object/relation grounding and multiple sentence forms. This selected historical pool is declarative; it does not evaluate unknown-object interrogatives. The official [HC-STVG dataset](https://github.com/tzhhhh123/HC-STVG) focuses on a person among multiple people in 20-second movie clips. These differences motivate checking identity ambiguity, event coverage and temporal extent, but do not by themselves prove why a particular update failed. Numerical evidence above takes priority over a dataset-level narrative.

## Cases, resources and verification

CASES.json saves the eight strongest positive and eight strongest negative arrivals for each specified pipeline contrast, using a fixed metric ordering. ROWS.json contains every anonymous source/order cell; STEP_ROWS.json contains every saved inner step. Summaries average eligible order cells within source and bootstrap sources with 10,000 paired draws. Recency bins distinguish no prior write; scheduled observations cannot imply a successful write when evidence is empty. Empty bins are N/A, not zero benefit.

RESOURCES.json records actual replay/backward/provider counts and worker/CPU wall times. Wall time is not pure GPU-kernel time. Exact metadata and receipt hashes permit historical expert-cache reuse; specialists remain frozen. Full H capture is not repeated per inner SGD step. Root audit independently checks dense metrics, rank/KL/SGD, masks, source state and target checkpoint bindings, temporal critic/selection, inner/persistent state chains, live arm parity and all anonymous aggregates. It does not recompute the complete model Jacobian on CPU.

Prediction barriers precede new GT exposure; GT is only for evaluation/diagnosis and never used to select updates, samples or parameters. No new backbone, loss, memory, teacher, scorer, corruption or followup job was launched.
