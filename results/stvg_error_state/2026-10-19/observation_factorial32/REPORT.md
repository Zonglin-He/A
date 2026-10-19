# Spatial RGB re-detection / temporal dense reading / factorial joint32

Status: completed / measurement valid. Original NH16/NV16 exposed development, one query per parent. This is an observation-capability experiment; no OPD fit, Fixed/Live comparison or trained new module was run. All three requested candidates were generated and sealed before any NEW GT scoring. No branch was skipped based on another branch result.

Frozen original TA-STVG cross-domain source checkpoints and same DINO-T. Original E, original four physical observation slots, target span/parsing, source calibration and evaluation rules retained. Formal FH/FV64 is excluded. Model checkpoint provenance and hashes are in MODEL_PROVENANCE.json.

S: original E proposes a 2x RGB context, clipped integer raster, then native DINO processor re-encodes and outputs actual detected boxes. Accepted crop pools replace the original same-slot pool; rejected/unsupported slots retain it. Original E is re-estimated relative to Native Source, never added twice to E. No confidence-energy coordinate gradient or IoU reward.

T: keep all original physical IDs and add one integer midpoint per adjacent pair. Same full temporal extent, original two offsets and positional-encoding limit. Actual video pixels and frozen appearance/motion/time decoder rerun. Actual FP32 logits form the original independent-offset envelope posterior; same physical expected-tIoU finalMBR. No logit/feature interpolation. E spatial boxes stay at original nodes and use original interpolation. The denser sequence also changes index-position and motion-window timescales; its effect cannot be attributed solely to newly observed RGB evidence.

ST combines exactly S boxes and T interval without extra calls; it measures factorial joint output, NOT joint conditional adaptation or OPD. GridOnly embeds the old posterior mass exactly in the dense decision grid, then MBR; it isolates expanded decision support, not dense model length/position effects.

## Main paired changes vs E, pp [parent bootstrap95%]

| Candidate | HC full vIoU | Vid full vIoU | HC tIoU | Vid tIoU | HC unobserved actual sIoU | Vid unobserved actual sIoU |
|---|---:|---:|---:|---:|---:|---:|
| S_crop | -0.2194 [-0.4599, +0.0130] | -1.6290 [-4.9018, +0.0148] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0748 [-0.5483, +0.4780] | -3.2841 [-9.0142, +0.0937] |
| T_dense | -9.2619 [-15.1431, -3.8120] | -0.6673 [-4.9055, +3.1791] | -10.1838 [-20.6267, +1.4876] | -0.8874 [-10.6869, +7.8429] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| ST | -9.3439 [-15.2513, -3.8624] | -2.7544 [-7.3597, +1.3968] | -10.1838 [-20.6267, +1.4876] | -0.8874 [-10.6869, +7.8429] | -0.0748 [-0.5483, +0.4780] | -3.2841 [-9.0142, +0.0937] |
| T_gridonly | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

## Dense posterior versus old-mass expanded-decision control

| Contrast | HC full vIoU | Vid full vIoU | HC tIoU | Vid tIoU |
|---|---:|---:|---:|---:|
| T_dense - T_gridonly | -9.2619 [-15.1431, -3.8120] | -0.6673 [-4.9055, +3.1791] | -10.1838 [-20.6267, +1.4876] | -0.8874 [-10.6869, +7.8429] |

## Absolute metrics and actual time changes

| Target | Output | full vIoU % | tIoU % | fixed-support sIoU % | unobserved actual sIoU % | changed final intervals /16 |
|---|---|---:|---:|---:|---:|---:|
| hc2 | E_MBR | 35.3055 | 58.7381 | 57.1719 | 56.6604 | 0 |
| hc2 | S_crop | 35.0861 | 58.7381 | 56.9707 | 56.5856 | 0 |
| hc2 | T_dense | 26.0435 | 48.5543 | 57.1719 | 56.6604 | 16 |
| hc2 | ST | 25.9616 | 48.5543 | 56.9707 | 56.5856 | 16 |
| hc2 | T_gridonly | 35.3055 | 58.7381 | 57.1719 | 56.6604 | 0 |
| vidstg | E_MBR | 15.7559 | 35.5542 | 45.4514 | 45.4485 | 0 |
| vidstg | S_crop | 14.1269 | 35.5542 | 42.1565 | 42.1644 | 0 |
| vidstg | T_dense | 15.0887 | 34.6668 | 45.4514 | 45.4485 | 16 |
| vidstg | ST | 13.0015 | 34.6668 | 42.1565 | 42.1644 | 16 |
| vidstg | T_gridonly | 15.7559 | 35.5542 | 45.4514 | 45.4485 | 0 |

Temporal-only actual s is exact E; S time exact E; ST time exact T and spatial exact S. Unknown GT stays unknown, no-reference/zero-overlap/no-change parents remain in the main denominator. Endpoint errors, lengths, zero overlaps, all signed changes and paired intervals are in RESULTS_PUBLIC.json.

## Factorial interaction, not evidence of learned cross-branch transfer

- hc2: V_ST-V_S-V_T+V_E = +0.1374 [+0.0304, +0.2574] pp. This includes the metric's temporal/spatial coupling.
- vidstg: V_ST-V_S-V_T+V_E = -0.4581 [-1.2933, +0.0500] pp. This includes the metric's temporal/spatial coupling.

## All signed full-v changes vs E, pp

| Alias | Target | S | T | ST | GridOnly |
|---|---|---:|---:|---:|---:|
| NH001 | hc2 | -0.5202 | +7.0934 | +6.4140 | +0.0000 |
| NH002 | hc2 | -1.1084 | -28.9621 | -29.4080 | +0.0000 |
| NH003 | hc2 | +0.0927 | -12.3936 | -12.0662 | +0.0000 |
| NH004 | hc2 | +0.2499 | -4.1593 | -3.9373 | +0.0000 |
| NH005 | hc2 | +0.0081 | +4.2556 | +4.2058 | +0.0000 |
| NH006 | hc2 | +0.6618 | +0.5055 | +1.3357 | +0.0000 |
| NH007 | hc2 | -0.0096 | -16.9646 | -16.9807 | +0.0000 |
| NH008 | hc2 | -0.5926 | -23.8103 | -24.0468 | +0.0000 |
| NH009 | hc2 | +0.0974 | -6.4916 | -6.4916 | +0.0000 |
| NH010 | hc2 | +0.0606 | -0.4923 | +0.1092 | +0.0000 |
| NH011 | hc2 | +0.0546 | -18.7272 | -18.7385 | +0.0000 |
| NH012 | hc2 | -0.5959 | -0.6992 | -1.1198 | +0.0000 |
| NH013 | hc2 | -0.0073 | -34.5916 | -34.5950 | +0.0000 |
| NH014 | hc2 | -1.0329 | -2.0325 | -2.6812 | +0.0000 |
| NH015 | hc2 | -0.0455 | -6.1814 | -6.2217 | +0.0000 |
| NH016 | hc2 | -0.8224 | -4.5400 | -5.2801 | +0.0000 |
| NV001 | vidstg | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| NV002 | vidstg | +0.0000 | +5.6321 | +5.5586 | +0.0000 |
| NV003 | vidstg | +0.0000 | -4.0878 | -4.0878 | +0.0000 |
| NV004 | vidstg | +0.0000 | +8.6551 | +6.9142 | +0.0000 |
| NV005 | vidstg | +0.1083 | -22.5233 | -21.9795 | +0.0000 |
| NV006 | vidstg | +0.0000 | +4.2483 | +4.2483 | +0.0000 |
| NV007 | vidstg | +0.0000 | +1.4632 | +1.4632 | +0.0000 |
| NV008 | vidstg | -0.0438 | +0.9833 | +0.9383 | +0.0000 |
| NV009 | vidstg | +0.0000 | +5.8248 | +5.8248 | +0.0000 |
| NV010 | vidstg | +0.0000 | +2.3720 | +2.3720 | +0.0000 |
| NV011 | vidstg | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| NV012 | vidstg | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| NV013 | vidstg | +0.0000 | -1.0614 | -1.0614 | +0.0000 |
| NV014 | vidstg | -0.0008 | -6.2165 | -6.2139 | +0.0000 |
| NV015 | vidstg | +0.0000 | -16.6112 | -16.6112 | +0.0000 |
| NV016 | vidstg | -26.1278 | +10.6450 | -21.4363 | +0.0000 |

## Postseal reference-versus-propagation diagnostic

This diagnosis was added after observing the main output scores; no outputs or choices were changed. Frame IoUs are first averaged within each query, then paired across parents with known support. These conditional diagnostic parents are not the main16-parent denominators. Raw top boxes are not automatically E-selected references.

| Diagnostic, IoU pp | HC | Vid |
|---|---:|---:|
| raw_crop_minus_raw_full_pp | -2.9525 [-17.3312, +10.8043] (n=15) | +1.1457 [-30.0993, +27.7477] (n=8) |
| raw_crop_minus_E_pp | +1.7187 [-3.5774, +6.4119] (n=15) | +9.5430 [-0.9690, +25.4628] (n=8) |
| accepted_crop_minus_E_pp | +3.3990 [+0.3334, +6.9213] (n=15) | +2.1737 [-2.0174, +7.6759] (n=7) |

The accepted crop-top versus E comparison has a positive HC conditional signal, but raw crop-top versus the old raw full-image top does not establish an improvement. This is not a new deployable tube result and does not establish improved supervision versus the old teacher. NV016 loses26.1278pp full-v under S; of its three known planned frames, two crop-top IoUs exceed E while all three propagated S boxes degrade. Candidate selection/aggregation/propagation therefore cannot be collapsed into the claim that every crop detection is bad. NV016 stays in the main result.

## Costs, validity and execution caveats

New DINO image forwards: 108; new dense TA queries:32 / native-offset full forwards64. Additional postseal engineering positive control:2 original-input TA queries /4 offset forwards, 22.401s GPU. Total new GPU lease including loading/hash verification/engineering: 287.619s. Source/E/full-image DINO cached costs excluded; this is not first-deployment latency. 0 new networks, heads, training or student updates.

Original unresolved target conditions remain unresolved, so <=128 DINO calls is not a claim that every query spent four. All crop acceptance, original fallback, estimator support and unchanged full-image crops are itemized in OBSERVATION_COUNTS_PUBLIC.json. FRAME_DIAGNOSTICS_PUBLIC records raw target-score top boxes, including rejected ones; these are NOT automatically the final E estimator selections and not evidence of identity probability calibration.

A static review before inference required hard stopping and narrowed GridOnly attribution. Pre-execution amendment001 preserved the original code/protocol and installed an owned systemd RuntimeMaxSec880/TimeoutStopSec5 resource boundary inside the900s GPU budget. No scientific setting or prediction was changed. Additional helper/processor/tokenizer bindings were read back AFTER worker start; they are honestly supplemental, not claimed original pre-pins. All were unmodified before the plan, historical pins matched where available, and actual DINO weights matched the declared SHA. This qualification is disclosed rather than retroactively labelled a complete original lock.

Independent audit: {'crop_geometry': 128, 'NMS_candidates': 300, 'E_outputs': 32, 'CDF_rectangles': 160, 'MBR_utility_checks': 256, 'baseline_metrics': 160, 'bootstrap_endpoints': 70}. Max errors: {'crop_error': 1.1102230246251565e-16, 'E_error': 2.220446049250313e-16, 'CDF_error': 8.881784197001252e-16, 'utility_error': 1.7763568394002505e-15}. Baseline saved metrics, actual crop geometry/NMS/target scores, exhaustive Source-relative E association, physical envelope rectangles, direct MBR utilities, all paired statistics and factorial invariants passed. Numerical CDF corrections are recorded per query.

The unexpectedly large HC dense-view regression prompted two original-input engineering checks AFTER all outputs sealed and initial scoring. NH001/NV001 using the unchanged original physical input reproduce all cached temporal logits and Source boxes bit-exactly. No candidate, configuration, output or metric was altered, and no extra task scoring occurred. These are engineering controls, not additional method trials or pre-scoring tests; cost is included above. See ORIGINAL_INPUT_ENGINEERING_CHECK.json.

## Decision scope

These are development capability results. They do not establish OPD, no matter which direct output improves. A future same-evidence/equal-budget Direct/Fixed/Live study requires a separately fixed student state, physical probability transport and actual Live observation/candidate mechanism. A positive small mean/CI does not meet the2–3pp complete-cohort goal by itself. Negative recipes are retained, not repaired by scanning crop size, frame density or prompt. CURRENT/MAIN/PAPER and historical paused queues remain unchanged; no automatic follow-up.
