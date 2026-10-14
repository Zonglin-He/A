# Temporal Caption/Query Adaptation with Spatial OPD: Eight-Case Post-Seal Readout

**Scope.** This is a post-seal development readout for eight historically exposed cases (`HC01`–`HC04`, `V01`–`V04`), four per dataset. It is not a fresh confirmation set. Ground-truth metrics were accessed after student and diagnostic prediction barriers were sealed. The current method was reported unchanged; teacher-training overlap remains unknown.

## Results

Temporal adaptation improves the aggregate m_vIoU and m_tIoU over both the frozen baseline (`F`) and `Projection` in this panel. The incremental temporal gain over Projection is modest on VidSTG. The spatial-only (`S`) and combined (`TS`) arms have lower dense-GT m_sIoU than F; the spatial signal came from a selected student pseudo-reference, not an independent teacher. This does not establish that OPD is generally ineffective. These results are limited to this exposed eight-case panel.

The fixed temporal KL+L2 objective ended above its initial value in 7 of 8 cases; only 1 case ended lower. The initial-minus-final delta is negative when the final objective increased. This is optimization instability for the fixed five-step configuration. Spatial loss is resampled on-policy each round, so it is not a fixed-target convergence trace.

### Dataset-level averages

| Dataset | Arm | m_vIoU (%) | m_tIoU (%) | dense-GT m_sIoU (%) | Δ m_vIoU vs F (pp) | Δ m_tIoU vs F (pp) | Δ m_sIoU vs F (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| HC | F | 22.210 | 46.415 | 47.895 | +0.000 | +0.000 | +0.000 |
| HC | Projection | 27.851 | 58.434 | 47.895 | +5.641 | +12.019 | +0.000 |
| HC | T | 31.286 | 60.486 | 47.895 | +9.076 | +14.071 | +0.000 |
| HC | S | 21.213 | 46.415 | 47.674 | -0.998 | +0.000 | -0.221 |
| HC | TS | 29.874 | 60.486 | 47.674 | +7.664 | +14.071 | -0.221 |
| VidSTG | F | 14.490 | 27.778 | 48.458 | +0.000 | +0.000 | +0.000 |
| VidSTG | Projection | 22.016 | 40.807 | 48.458 | +7.526 | +13.028 | +0.000 |
| VidSTG | T | 22.442 | 40.905 | 48.458 | +7.952 | +13.127 | +0.000 |
| VidSTG | S | 13.513 | 27.778 | 45.673 | -0.977 | +0.000 | -2.785 |
| VidSTG | TS | 21.140 | 40.905 | 45.673 | +6.649 | +13.127 | -2.785 |

### Per-case final readouts

The rows below preserve the positive and negative tails. Values are percentages; no case was selected by its ground-truth best step.

| Case | Dataset | Arm | m_vIoU (%) | m_tIoU (%) | dense-GT m_sIoU (%) |
|---|---|---:|---:|---:|---:|
| HC01 | HC | F | 16.958 | 40.693 | 34.730 |
| HC01 | HC | Projection | 15.423 | 40.157 | 34.730 |
| HC01 | HC | T | 6.916 | 15.459 | 34.730 |
| HC01 | HC | S | 16.691 | 40.693 | 39.599 |
| HC01 | HC | TS | 6.579 | 15.459 | 39.599 |
| HC02 | HC | F | 35.818 | 58.252 | 64.913 |
| HC02 | HC | Projection | 39.942 | 61.702 | 64.913 |
| HC02 | HC | T | 52.265 | 82.979 | 64.913 |
| HC02 | HC | S | 30.929 | 58.252 | 56.973 |
| HC02 | HC | TS | 45.367 | 82.979 | 56.973 |
| HC03 | HC | F | 12.677 | 35.632 | 36.744 |
| HC03 | HC | Projection | 23.780 | 64.935 | 36.744 |
| HC03 | HC | T | 23.780 | 64.935 | 36.744 |
| HC03 | HC | S | 13.206 | 35.632 | 38.451 |
| HC03 | HC | TS | 24.885 | 64.935 | 38.451 |
| HC04 | HC | F | 23.388 | 51.082 | 55.193 |
| HC04 | HC | Projection | 32.261 | 66.942 | 55.193 |
| HC04 | HC | T | 42.184 | 78.571 | 55.193 |
| HC04 | HC | S | 24.026 | 51.082 | 55.672 |
| HC04 | HC | TS | 42.666 | 78.571 | 55.672 |
| V01 | VidSTG | F | 36.181 | 54.615 | 63.491 |
| V01 | VidSTG | Projection | 49.282 | 74.815 | 63.491 |
| V01 | VidSTG | T | 51.733 | 78.519 | 63.491 |
| V01 | VidSTG | S | 33.951 | 54.615 | 59.601 |
| V01 | VidSTG | TS | 48.564 | 78.519 | 59.601 |
| V02 | VidSTG | F | 9.470 | 16.091 | 59.008 |
| V02 | VidSTG | Projection | 24.913 | 42.489 | 59.008 |
| V02 | VidSTG | T | 24.477 | 41.546 | 59.008 |
| V02 | VidSTG | S | 9.310 | 16.091 | 57.959 |
| V02 | VidSTG | TS | 24.062 | 41.546 | 57.959 |
| V03 | VidSTG | F | 6.007 | 14.043 | 48.260 |
| V03 | VidSTG | Projection | 6.892 | 15.670 | 48.260 |
| V03 | VidSTG | T | 6.892 | 15.670 | 48.260 |
| V03 | VidSTG | S | 5.762 | 14.043 | 46.290 |
| V03 | VidSTG | TS | 6.614 | 15.670 | 46.290 |
| V04 | VidSTG | F | 6.303 | 26.364 | 23.072 |
| V04 | VidSTG | Projection | 6.980 | 30.252 | 23.072 |
| V04 | VidSTG | T | 6.666 | 27.885 | 23.072 |
| V04 | VidSTG | S | 5.029 | 26.364 | 18.843 |
| V04 | VidSTG | TS | 5.319 | 27.885 | 18.843 |

### Per-case optimizer and support summaries

| Case | Dataset | T steps | S steps | Initial − final T loss | QA no information | Supervised / unobserved GT frames | Δ supervised sIoU (pp) | Δ unobserved sIoU (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HC01 | HC | 5 | 5 | -0.072668 | False | 3 / 205 | +0.474 | +4.933 |
| HC02 | HC | 5 | 5 | -0.387585 | False | 3 / 85 | -8.079 | -7.935 |
| HC03 | HC | 5 | 5 | -0.020360 | False | 1 / 200 | +2.702 | +1.702 |
| HC04 | HC | 5 | 5 | -0.115696 | False | 4 / 228 | +1.187 | +0.466 |
| V01 | VidSTG | 5 | 5 | +0.047127 | False | 3 / 107 | -4.139 | -3.882 |
| V02 | VidSTG | 5 | 5 | -0.058479 | False | 4 / 1587 | -1.039 | -1.049 |
| V03 | VidSTG | 5 | 5 | -0.029734 | False | 1 / 122 | -1.318 | -1.976 |
| V04 | VidSTG | 5 | 5 | -0.016087 | False | 1 / 35 | -1.827 | -4.298 |

## Adaptation traces

There are six scalar snapshots per case (initial state plus five updates). The JSON retains every temporal/spatial loss, temporal parameter norm, and T/S/TS metric. Spatial-loss values that were unavailable remain `null`. The readout uses 512 temporal parameters and five fixed optimizer steps. Spatial OPD uses 1,792 parameters and five actual steps in this run. The final five-step model is reported; no GT-best step was chosen. A negative initial-minus-final temporal-loss delta means the final KL+L2 objective rose; the observed 7/8 increase is reported explicitly.

| Case | Dataset | Step | Temporal loss | Spatial loss | Temporal parameter norm | T m_vIoU | S m_vIoU | TS m_vIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HC01 | HC | 0 | 0.001 | -0.402 | 0.000 | 0.154 | 0.170 | 0.154 |
| HC01 | HC | 1 | 0.165 | -0.152 | 0.226 | 0.223 | 0.177 | 0.236 |
| HC01 | HC | 2 | 0.005 | -0.158 | 0.135 | 0.190 | 0.171 | 0.186 |
| HC01 | HC | 3 | 0.049 | -0.331 | 0.183 | 0.096 | 0.166 | 0.094 |
| HC01 | HC | 4 | 0.097 | -0.258 | 0.220 | 0.069 | 0.183 | 0.064 |
| HC01 | HC | 5 | 0.074 | — | 0.210 | 0.069 | 0.167 | 0.066 |
| HC02 | HC | 0 | 0.010 | 0.177 | 0.000 | 0.399 | 0.358 | 0.399 |
| HC02 | HC | 1 | 0.777 | 0.163 | 0.226 | 0.206 | 0.348 | 0.200 |
| HC02 | HC | 2 | 0.048 | 0.052 | 0.085 | 0.249 | 0.331 | 0.232 |
| HC02 | HC | 3 | 0.136 | 0.268 | 0.086 | 0.477 | 0.318 | 0.425 |
| HC02 | HC | 4 | 0.355 | 0.304 | 0.157 | 0.523 | 0.311 | 0.456 |
| HC02 | HC | 5 | 0.398 | — | 0.172 | 0.523 | 0.309 | 0.454 |
| HC03 | HC | 0 | 0.005 | -0.285 | 0.000 | 0.238 | 0.127 | 0.238 |
| HC03 | HC | 1 | 0.085 | -0.096 | 0.226 | 0.161 | 0.149 | 0.191 |
| HC03 | HC | 2 | 0.008 | -0.212 | 0.109 | 0.210 | 0.131 | 0.218 |
| HC03 | HC | 3 | 0.018 | -0.165 | 0.080 | 0.238 | 0.115 | 0.214 |
| HC03 | HC | 4 | 0.045 | -0.353 | 0.108 | 0.215 | 0.122 | 0.209 |
| HC03 | HC | 5 | 0.025 | — | 0.075 | 0.238 | 0.132 | 0.249 |
| HC04 | HC | 0 | 0.002 | -0.612 | 0.000 | 0.323 | 0.234 | 0.323 |
| HC04 | HC | 1 | 0.505 | -0.152 | 0.226 | 0.057 | 0.243 | 0.059 |
| HC04 | HC | 2 | 0.045 | -0.211 | 0.091 | 0.170 | 0.230 | 0.167 |
| HC04 | HC | 3 | 0.166 | -0.220 | 0.131 | 0.432 | 0.221 | 0.416 |
| HC04 | HC | 4 | 0.278 | -0.280 | 0.155 | 0.432 | 0.230 | 0.425 |
| HC04 | HC | 5 | 0.117 | — | 0.113 | 0.422 | 0.240 | 0.427 |
| V01 | VidSTG | 0 | 0.168 | -0.325 | 0.000 | 0.493 | 0.362 | 0.493 |
| V01 | VidSTG | 1 | 0.118 | -0.153 | 0.226 | 0.517 | 0.388 | 0.560 |
| V01 | VidSTG | 2 | 0.101 | -0.351 | 0.435 | 0.517 | 0.393 | 0.563 |
| V01 | VidSTG | 3 | 0.105 | -0.262 | 0.608 | 0.517 | 0.355 | 0.508 |
| V01 | VidSTG | 4 | 0.115 | -0.357 | 0.727 | 0.517 | 0.340 | 0.486 |
| V01 | VidSTG | 5 | 0.121 | — | 0.792 | 0.517 | 0.340 | 0.486 |
| V02 | VidSTG | 0 | 0.000 | -0.321 | 0.000 | 0.249 | 0.095 | 0.249 |
| V02 | VidSTG | 1 | 0.139 | -0.068 | 0.226 | 0.051 | 0.102 | 0.054 |
| V02 | VidSTG | 2 | 0.008 | -0.125 | 0.164 | 0.136 | 0.097 | 0.140 |
| V02 | VidSTG | 3 | 0.046 | -0.185 | 0.206 | 0.293 | 0.092 | 0.282 |
| V02 | VidSTG | 4 | 0.079 | -0.123 | 0.235 | 0.278 | 0.091 | 0.266 |
| V02 | VidSTG | 5 | 0.058 | — | 0.230 | 0.245 | 0.093 | 0.241 |
| V03 | VidSTG | 0 | 0.010 | -0.262 | 0.000 | 0.069 | 0.060 | 0.069 |
| V03 | VidSTG | 1 | 0.099 | -0.089 | 0.226 | 0.081 | 0.055 | 0.074 |
| V03 | VidSTG | 2 | 0.006 | -0.168 | 0.187 | 0.070 | 0.062 | 0.071 |
| V03 | VidSTG | 3 | 0.021 | -0.139 | 0.230 | 0.069 | 0.062 | 0.071 |
| V03 | VidSTG | 4 | 0.052 | -0.157 | 0.283 | 0.069 | 0.058 | 0.067 |
| V03 | VidSTG | 5 | 0.040 | — | 0.307 | 0.069 | 0.058 | 0.066 |
| V04 | VidSTG | 0 | 0.000 | -0.225 | 0.000 | 0.070 | 0.063 | 0.070 |
| V04 | VidSTG | 1 | 0.041 | -0.117 | 0.226 | 0.070 | 0.047 | 0.054 |
| V04 | VidSTG | 2 | 0.003 | -0.234 | 0.064 | 0.070 | 0.059 | 0.067 |
| V04 | VidSTG | 3 | 0.009 | -0.186 | 0.095 | 0.067 | 0.058 | 0.061 |
| V04 | VidSTG | 4 | 0.024 | -0.232 | 0.154 | 0.067 | 0.052 | 0.055 |
| V04 | VidSTG | 5 | 0.016 | — | 0.124 | 0.067 | 0.050 | 0.053 |

## Caption/query and visual-information diagnostics

The 32 candidate rows contain support/contradict/insufficient probabilities, reward, token count, visual-information gain per token, and candidate tIoU/vIoU. G=(log P(caption|video) − log P(the same caption|no video))/token count. These categorical labels are not calibrated temporal IoU. Candidate rows are nested under only four sources per dataset.

| Dataset | Sources | Candidate rows | Reward–tIoU within-query centered r | Gain–tIoU within-query centered r |
|---|---:|---:|---:|---:|
| HC | 4 | 16 | -0.604 | +0.317 |
| VidSTG | 4 | 16 | +0.392 | -0.142 |

These correlations are descriptive, not significance tests. The pooled-candidate correlations and mean visual gains are provided in `ADDITIONAL_DIAGNOSTICS.json`; pooled candidates are not independent sources. Higher visual-information gain does not imply higher tIoU.

### Gradient comparisons

QA and the separate SSL-replay gradients were compared with the same-case endpoint-supervised NLL gradient on 512 frozen-hidden-state parameters. This is not the derivative of discrete vIoU, and GT gradients were not used for updates. Cosines vary in sign across cases; the data do not support a general claim that QA gradients align better with GT gradients.

| Case | Dataset | QA vs GT cosine (shared S0) | SSL replay vs same GT cosine | QA vs GT cosine (native score) | SSL replay vs GT cosine (native score) |
|---|---|---:|---:|---:|---:|
| HC01 | HC | +0.0467 | -0.3374 | +0.3371 | -0.1026 |
| HC02 | HC | -0.7638 | +0.1633 | -0.0690 | +0.8692 |
| HC03 | HC | -0.3290 | +0.4000 | -0.3068 | +0.3880 |
| HC04 | HC | -0.0589 | +0.5830 | -0.3736 | +0.5178 |
| V01 | VidSTG | +0.8719 | +0.8701 | +0.9049 | +0.9258 |
| V02 | VidSTG | -0.5360 | +0.4122 | -0.7156 | +0.4132 |
| V03 | VidSTG | -0.0046 | -0.0043 | +0.2342 | -0.2907 |
| V04 | VidSTG | +0.0923 | -0.0240 | +0.1088 | -0.0122 |

The SSL comparison is a 512-parameter replay control, not the historical 66,306-parameter/32-panel fit. The fixed five-step readout is not selected by GT. The 12 counterfactual description controls are listed as hashes and reward deltas only; no free text is included.

### Reward shuffling

| Dataset | Shuffled-T m_vIoU (%) | QA T minus shuffled T (pp) | QA–GT cosine mean | Negative QA–GT cosines | SSL replay cosine mean |
|---|---:|---:|---:|---:|---:|
| HC | 21.947 | +9.339 | -0.276 | 3 | +0.202 |
| VidSTG | 17.642 | +4.800 | +0.106 | 2 | +0.313 |

The eight shuffled controls are separate CPU replay fits using permuted rewards; they add no teacher calls or full-model forwards. They are diagnostics, not independent test data.

## Cost and verification

| Stage | Requests / cases | Tokens (in / out) | Wall time (s) | Peak allocated GPU memory (GiB) |
|---|---:|---:|---:|---:|
| Teacher method | 64 calls | 43724 / 1268 | 31.410 | 15.88 |
| Auxiliary teacher diagnostics | 96 calls | 66165 / 467 | 26.950 | 15.92 |
| Student readout | 8 cases; 40 T + 40 S steps | — | 56.394 | 5.35 |

The root readback records 184 metric readouts across 8 cases. Its official metric reconstruction maximum absolute error is 1.22e-15; teacher probability, shared-readout loss, and teacher-forced token reconstruction errors are separately preserved in `READBACK_METADATA.json`. The bounded CPU/source audit and its synthetic-contract counts are summarized in `SOURCE_AUDIT_METADATA.json`; this package does not claim a second inference run.

## Evidence boundaries

- The cohort was historically exposed and is not fresh confirmation. Teacher-training overlap is unknown.
- Ground-truth evaluation was post-seal; the run must not be described as no-GT.
- Spatial supervision came from a student-selected pseudo-reference.
- `G` is visual-information gain per token; it is not a tIoU measure.
- Gradient comparisons use endpoint-supervised NLL, not a discrete vIoU derivative.
- The 512-parameter SSL replay is not the older 66,306-parameter/32-panel fit.
- The report does not promote a method or modify the protected current method.

## Reproducibility package

The adjacent public-stage package contains the main and diagnostic protocol notes, anonymous scalar JSON, six-step trajectories, sanitized audit metadata, and byte-for-byte copies of the scoped implementation scripts. Root receipts are represented by hashes and selected verification metadata; `truth_binding_metadata`, raw outputs, descriptions, query text, coordinates, hidden states, weights, model assets, and private media are omitted.

Cost boundary: the recorded stage timers exclude model loading, canonical input decoding and asset preparation. They are not cold end-to-end TTA latency.
