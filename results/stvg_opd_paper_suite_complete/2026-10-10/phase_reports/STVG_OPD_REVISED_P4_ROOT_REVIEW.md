# Original fixed OPD P4: same-domain physical-burst robustness

Full exceeds Frozen in all sixteen conditions in each dataset, with positive paired parent 95% intervals. Clean gains are HC2 +3.766 [+2.578, +4.901] pp and VidSTG +2.456 [+1.666, +3.253] pp. On the method's own trajectories, the mean current-query update is negative in all HC2 conditions; inherited state supplies the positive total mean. The current-query mean is positive in VidSTG, with uncertainty reported separately. Severe individual harms and reward/task mismatches remain. This phase measures Full against its frozen same-domain source on clean input and fifteen physical-burst conditions. It tests robustness of the fixed method; it does not compare new algorithms or choose new parameters.

All 15,504 new formal fits globally sealed before GT: HC-STVG2 validation has 237 parent movies, one clip/query per movie (3,792 arrivals), and VidSTG test has 732 parent videos, one query per parent (11,712 arrivals). There are sixteen conditions: clean and frame drop, frame freeze, motion blur, occlusion and exposure, each at 2.5%, 5% and 10% physical burst coverage. Every condition starts independently from its own same-domain source. There are no historical aliases and no qualification predictions among formal arrivals.

TA-STVG source checkpoints are `TASTVG_HCSTVG2.pth` (47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036) and `TASTVG_VidSTG.pth` (5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83). HC2 lr/sigma/tau/rounds/writeback are .01/.025/.05/40/1/16; VidSTG .03/.1/.25/10/1/8. M=32 antithetic Gaussian actions, Uniform4, admitted Top1 frozen DINO, joint 1,792 active parameters, per-query Adam/query-residual reset, within-condition LN inheritance and final-round output stay fixed. Native WHEN remains unchanged.

Numbers below use parent macro averaging with 10,000 paired parent bootstrap draws (seed 20261006). One query per parent makes query macro and parent macro equal. The selected query roster and saved stream order remain fixed. These intervals are conditional on the saved histories and are not multiplicity adjusted. No formal result is used to retune. The efficacy plot averages the five corruption families at each nonzero coverage; its shaded bands are marginal intervals. The paired-effect plot and complete tables retain every individual condition. The ranked-tail plot shows every parent for clean and the five 5% conditions; all sixteen condition tails remain in the complete anonymous data.

| Dataset | Conditions with positive 95% interval | Negative interval | Interval includes zero |
|---|---:|---:|---:|
| P4_hc2 | 16 / 16 | 0 / 16 | 0 / 16 |
| P4_vidstg | 16 / 16 | 0 / 16 | 0 / 16 |

| Dataset / condition | Frozen vIoU (%) | Full vIoU (%) | Full − Frozen (pp), 95% CI | Current: After − Before | Inherited: Before − Frozen | >5 / >20 pp harmful parents |
|---|---:|---:|---:|---:|---:|---:|
| P4_hc2 / clean | 30.684 | 34.450 | +3.766 [+2.578, +4.901] | -0.722 [-1.909, +0.378] | +4.488 [+3.913, +5.058] | 18 / 7 |
| P4_hc2 / frame_drop_2.5 | 30.548 | 34.325 | +3.776 [+2.595, +4.930] | -0.632 [-1.819, +0.520] | +4.408 [+3.881, +4.941] | 20 / 6 |
| P4_hc2 / frame_freeze_2.5 | 30.617 | 34.022 | +3.405 [+2.138, +4.608] | -1.072 [-2.320, +0.111] | +4.478 [+3.918, +5.039] | 19 / 9 |
| P4_hc2 / motion_blur_2.5 | 30.842 | 34.148 | +3.307 [+2.072, +4.491] | -1.050 [-2.313, +0.170] | +4.357 [+3.800, +4.910] | 21 / 9 |
| P4_hc2 / occlusion_2.5 | 30.772 | 34.143 | +3.371 [+2.097, +4.599] | -1.094 [-2.373, +0.133] | +4.465 [+3.900, +5.025] | 23 / 10 |
| P4_hc2 / exposure_2.5 | 30.710 | 34.510 | +3.800 [+2.510, +5.020] | -0.645 [-1.947, +0.586] | +4.445 [+3.904, +4.987] | 16 / 9 |
| P4_hc2 / frame_drop_5 | 29.961 | 33.490 | +3.529 [+2.349, +4.670] | -0.739 [-1.910, +0.372] | +4.269 [+3.726, +4.816] | 20 / 7 |
| P4_hc2 / frame_freeze_5 | 30.631 | 34.555 | +3.925 [+2.779, +5.023] | -0.708 [-1.867, +0.359] | +4.633 [+4.071, +5.185] | 17 / 7 |
| P4_hc2 / motion_blur_5 | 30.524 | 33.617 | +3.093 [+1.880, +4.267] | -1.190 [-2.431, -0.009] | +4.283 [+3.731, +4.841] | 23 / 9 |
| P4_hc2 / occlusion_5 | 30.400 | 33.432 | +3.033 [+1.772, +4.248] | -1.375 [-2.652, -0.160] | +4.408 [+3.840, +4.977] | 25 / 9 |
| P4_hc2 / exposure_5 | 30.536 | 34.347 | +3.811 [+2.539, +5.062] | -0.640 [-1.945, +0.586] | +4.450 [+3.904, +4.999] | 22 / 9 |
| P4_hc2 / frame_drop_10 | 28.756 | 32.439 | +3.683 [+2.561, +4.766] | -0.433 [-1.542, +0.613] | +4.116 [+3.584, +4.657] | 16 / 5 |
| P4_hc2 / frame_freeze_10 | 30.338 | 33.830 | +3.492 [+2.361, +4.570] | -1.002 [-2.105, +0.052] | +4.495 [+3.947, +5.045] | 20 / 6 |
| P4_hc2 / motion_blur_10 | 29.676 | 32.795 | +3.119 [+1.996, +4.203] | -0.932 [-2.044, +0.131] | +4.051 [+3.517, +4.586] | 23 / 5 |
| P4_hc2 / occlusion_10 | 29.853 | 33.519 | +3.665 [+2.486, +4.818] | -0.607 [-1.752, +0.538] | +4.272 [+3.756, +4.782] | 20 / 6 |
| P4_hc2 / exposure_10 | 30.410 | 34.151 | +3.741 [+2.535, +4.885] | -0.763 [-1.963, +0.385] | +4.504 [+3.948, +5.067] | 19 / 8 |
| P4_vidstg / clean | 21.224 | 23.679 | +2.456 [+1.666, +3.253] | +0.799 [+0.120, +1.493] | +1.657 [+1.209, +2.096] | 70 / 25 |
| P4_vidstg / frame_drop_2.5 | 20.262 | 22.542 | +2.281 [+1.516, +3.041] | +0.714 [+0.075, +1.368] | +1.566 [+1.137, +1.978] | 70 / 23 |
| P4_vidstg / frame_freeze_2.5 | 21.228 | 23.522 | +2.294 [+1.516, +3.069] | +0.626 [-0.039, +1.312] | +1.667 [+1.222, +2.101] | 75 / 27 |
| P4_vidstg / motion_blur_2.5 | 21.104 | 23.345 | +2.241 [+1.495, +2.977] | +0.749 [+0.094, +1.419] | +1.492 [+1.052, +1.916] | 68 / 23 |
| P4_vidstg / occlusion_2.5 | 20.948 | 23.144 | +2.196 [+1.427, +2.950] | +0.564 [-0.088, +1.232] | +1.631 [+1.195, +2.062] | 71 / 24 |
| P4_vidstg / exposure_2.5 | 21.113 | 23.521 | +2.408 [+1.641, +3.183] | +0.708 [+0.052, +1.383] | +1.700 [+1.262, +2.127] | 67 / 24 |
| P4_vidstg / frame_drop_5 | 19.257 | 21.295 | +2.039 [+1.281, +2.792] | +0.575 [-0.075, +1.226] | +1.464 [+1.059, +1.864] | 65 / 24 |
| P4_vidstg / frame_freeze_5 | 21.269 | 23.357 | +2.088 [+1.347, +2.843] | +0.674 [+0.033, +1.334] | +1.414 [+0.981, +1.842] | 72 / 24 |
| P4_vidstg / motion_blur_5 | 20.979 | 23.009 | +2.030 [+1.274, +2.776] | +0.495 [-0.171, +1.170] | +1.536 [+1.081, +1.972] | 73 / 23 |
| P4_vidstg / occlusion_5 | 20.875 | 23.077 | +2.201 [+1.429, +2.970] | +0.576 [-0.088, +1.259] | +1.625 [+1.200, +2.042] | 70 / 25 |
| P4_vidstg / exposure_5 | 21.208 | 23.345 | +2.137 [+1.371, +2.909] | +0.746 [+0.081, +1.434] | +1.391 [+0.944, +1.823] | 74 / 24 |
| P4_vidstg / frame_drop_10 | 17.817 | 19.559 | +1.742 [+1.081, +2.417] | +0.426 [-0.140, +1.009] | +1.316 [+0.918, +1.709] | 68 / 23 |
| P4_vidstg / frame_freeze_10 | 20.816 | 23.347 | +2.531 [+1.762, +3.308] | +0.736 [+0.079, +1.411] | +1.795 [+1.357, +2.234] | 63 / 22 |
| P4_vidstg / motion_blur_10 | 20.582 | 22.714 | +2.132 [+1.391, +2.867] | +0.560 [-0.054, +1.199] | +1.572 [+1.144, +2.002] | 75 / 23 |
| P4_vidstg / occlusion_10 | 20.539 | 22.765 | +2.227 [+1.467, +2.981] | +0.308 [-0.339, +0.968] | +1.918 [+1.493, +2.330] | 63 / 24 |
| P4_vidstg / exposure_10 | 21.163 | 23.421 | +2.259 [+1.495, +3.021] | +0.512 [-0.158, +1.195] | +1.747 [+1.315, +2.180] | 68 / 23 |

Current and inherited effects decompose the method's own stream. Inherited Before−Frozen is not an alpha0 causal contrast. All negative parent effects, gross gain/loss and severe tails remain in the complete condition tables and anonymous rows. Temporal deltas are zero because WHEN is fixed. This phase cannot establish a benefit from temporal-head adaptation.

| Dataset / condition | Observed-frame IoU current delta (pp) | Valid queries | Unobserved-frame IoU current delta (pp) | Valid queries | Fit wall seconds / arrival | Shared capture seconds / arrival |
|---|---:|---:|---:|---:|---:|---:|
| hc2 / clean | -0.128 | 223 | -1.784 | 237 | 1.498889 | 1.797006 |
| hc2 / frame_drop_2.5 | -0.069 | 218 | -1.797 | 237 | 1.403510 | 1.732501 |
| hc2 / frame_freeze_2.5 | -0.568 | 225 | -2.628 | 237 | 1.420697 | 2.379320 |
| hc2 / motion_blur_2.5 | +0.003 | 219 | -2.305 | 237 | 1.420352 | 1.736531 |
| hc2 / occlusion_2.5 | +0.005 | 221 | -2.395 | 237 | 1.448127 | 1.740991 |
| hc2 / exposure_2.5 | +0.085 | 224 | -1.923 | 237 | 1.400801 | 1.700527 |
| hc2 / frame_drop_5 | -1.243 | 217 | -1.897 | 237 | 1.405544 | 1.445617 |
| hc2 / frame_freeze_5 | -0.100 | 222 | -1.592 | 237 | 1.398515 | 2.071627 |
| hc2 / motion_blur_5 | -1.169 | 218 | -2.559 | 237 | 1.400024 | 1.509283 |
| hc2 / occlusion_5 | -0.883 | 223 | -2.995 | 237 | 1.401712 | 1.469824 |
| hc2 / exposure_5 | -0.383 | 225 | -2.065 | 237 | 1.406796 | 1.461244 |
| hc2 / frame_drop_10 | +0.015 | 207 | -1.246 | 237 | 1.402269 | 1.701010 |
| hc2 / frame_freeze_10 | -0.063 | 218 | -2.267 | 237 | 1.397605 | 2.323665 |
| hc2 / motion_blur_10 | -1.868 | 213 | -2.041 | 237 | 1.392941 | 1.778826 |
| hc2 / occlusion_10 | -0.628 | 223 | -1.829 | 237 | 1.401743 | 1.695365 |
| hc2 / exposure_10 | -0.197 | 223 | -1.737 | 237 | 1.400883 | 1.699481 |
| vidstg / clean | +3.376 | 401 | +0.970 | 732 | 0.332248 | 1.189682 |
| vidstg / frame_drop_2.5 | +3.342 | 400 | +0.994 | 732 | 0.334442 | 1.191330 |
| vidstg / frame_freeze_2.5 | +2.982 | 403 | +0.678 | 732 | 0.335950 | 1.317754 |
| vidstg / motion_blur_2.5 | +3.395 | 403 | +0.836 | 732 | 0.334474 | 1.210273 |
| vidstg / occlusion_2.5 | +2.854 | 397 | +0.472 | 732 | 0.335280 | 1.185732 |
| vidstg / exposure_2.5 | +3.068 | 398 | +0.662 | 732 | 0.334566 | 1.184841 |
| vidstg / frame_drop_5 | +3.505 | 395 | +0.844 | 732 | 0.334719 | 1.116513 |
| vidstg / frame_freeze_5 | +3.164 | 402 | +0.647 | 732 | 0.336297 | 1.250158 |
| vidstg / motion_blur_5 | +3.074 | 393 | +0.667 | 732 | 0.335417 | 1.162038 |
| vidstg / occlusion_5 | +2.855 | 402 | +0.658 | 732 | 0.336179 | 1.117156 |
| vidstg / exposure_5 | +2.918 | 400 | +0.851 | 732 | 0.335736 | 1.119212 |
| vidstg / frame_drop_10 | +2.593 | 389 | +0.341 | 732 | 0.336349 | 1.181783 |
| vidstg / frame_freeze_10 | +3.477 | 404 | +0.856 | 732 | 0.335838 | 1.328441 |
| vidstg / motion_blur_10 | +2.974 | 392 | +0.702 | 732 | 0.336538 | 1.282049 |
| vidstg / occlusion_10 | +2.538 | 407 | +0.305 | 732 | 0.337585 | 1.187219 |
| vidstg / exposure_10 | +2.325 | 394 | +0.633 | 732 | 0.336252 | 1.193915 |

Observed and unobserved means use different frame populations; neither is a randomized causal intervention. The stored fit wall time includes synchronous numerical recording. Shared capture is reported separately. Recorded expert-forward time may include reuse and is not a cold expert latency measurement.

| Dataset | Fit wall seconds / arrival | Shared capture seconds / arrival | CPU mathematical audit seconds / arrival | Original execution new DINO calls |
|---|---:|---:|---:|---:|
| P4_hc2 | 1.412525 | 1.765176 | 0.114659 | 12456 |
| P4_vidstg | 0.335492 | 1.201131 | 0.022554 | 34616 |

The actual root read every 15504 saved fit and complete mathematical dictionary, 268800 rounds, 27783168 state coordinates, 15504 input bindings, 139536 independent dense scalars, 31008 observed/unobserved effects and 64 complete qualification/formal pairs. It also repeated 62016 prediction/input/receipt SHA checks. Maximum independent dense discrepancy was 3.21964677e-15. Each condition's source reset, per-query residual/Adam reset, LN inheritance/writeback, original Native intervals and receipt-aware numerical dictionaries were actually checked.

Six distinct post-hoc success/harm/expert-reward mismatch cases were selected only after full population readback. All original saved GPU action sample-vs-GT chains were independently checked (14720 comparisons). The six private RGB sheets re-decode and apply the original physical corruption, requiring exact original input pixel SHA and corruption-spec equality. Five report plot pairs and all six real case sheets were actually viewed. Offline best samples are diagnostic only and never select deployment outputs. Cases are not independent efficacy estimates.

Public implementation/configuration/protocols/anonymous rows/scalars/plots retain all negative findings. Private RGB/query/caption/GT geometry/weights/raw logits/boxes/actions/fit/gradient/Adam/cache payloads are excluded. The root makes no model or optimizer calls. Saved mathematical/state/dense integrity is checked without independently implementing the entire decoder Jacobian or proving CUDA transcendental kernels. Historical numerical failures remain preserved.

Decision: close only the fixed P4 phase after actual remote verification and archive maintenance, then continue original P5 budget/cost experiments. No retuning, method promotion or new scientific algorithm is introduced. P5/P6 require their own real qualification/global deployment seal/CPU/root/view/public/archive closing. EATA and historical paused queues remain paused; the paper suite remains incomplete.
