# TA-STVG: deployment corruption and temporal critic qualification

Completed, frozen-only development experiments. No adaptation, OPD, spatial critic, S1, cross-domain tuning, or production promotion.

C0.5 benchmark-development gate: **False**. C2-T next-OPD resource gate: **True**. These are finite resource decisions, not universal mechanism claims.

## Setting and protocol correction

The user corrected the initial GT-centered proposal before the full experiment. Three unscored black-frame smoke cells (10.352 s) from that earlier protocol are preserved and excluded. The primary protocol is now **Transient Deployment Corruption**, independent of query and GT.

For each full physical source video, one deterministic uniform random start and ceil(1/5/10% of full-video frames) duration define a burst. Only the original observed frames falling in the burst are changed. The seed depends on source identity only; changing a query or its observed clip does not change the burst. No re-sampling to hit an event, no result-dependent severity changes. Frame drop uses black missing-frame placeholders with unchanged timestamps; freeze repeats the exact preceding original frame. Blur, 25%-area occlusion and +100 RGB exposure use fixed registered operators. These are simulated deployment faults, not an official THUMOS14-C reproduction or measured real fault frequencies.

The original roster contains 32 previously exposed VidSTG-test parents, one query per parent, official VidSTG-source TA-STVG checkpoint (5ab12c86…), original 20–200 observed frames, two native offsets and precision. GT is used only after sealing predictions for metrics and retrospective overlap analysis. Old persistent noise/defocus/JPEG remains unchanged. No fresh confirmation claim.

## C0.5: fixed five-family, three-duration panel

480 corrupted cells plus 32 clean references. Values are corrupted-minus-clean percentage points; brackets are paired parent-bootstrap 95% intervals. Primary averages all15 cells within each parent before bootstrapping32 parents.

| Condition | Delta sIoU (pp) | Delta tIoU (pp) | Delta corrected vIoU (pp) | No observed hit |
|---|---:|---:|---:|---:|
| panel | -0.382 [-0.763, -0.095] | -0.116 [-1.998, +2.021] | -0.461 [-1.371, +0.460] | 60/480 |
| frame_drop_1 | -0.182 [-0.359, -0.038] | +0.950 [-2.621, +5.571] | -0.513 [-1.548, +0.303] | 8/32 |
| frame_drop_5 | -0.509 [-1.164, -0.068] | -1.116 [-7.023, +5.369] | -1.376 [-4.182, +1.421] | 2/32 |
| frame_drop_10 | -1.705 [-2.963, -0.650] | -2.514 [-10.638, +6.298] | -1.583 [-4.660, +1.621] | 2/32 |
| frame_freeze_1 | +0.007 [-0.026, +0.057] | -0.035 [-0.230, +0.134] | +0.068 [-0.015, +0.218] | 8/32 |
| frame_freeze_5 | -0.150 [-0.768, +0.399] | -0.173 [-0.814, +0.351] | -0.173 [-0.587, +0.192] | 2/32 |
| frame_freeze_10 | -0.967 [-2.031, -0.140] | -0.006 [-0.903, +0.787] | -0.456 [-1.077, +0.121] | 2/32 |
| motion_blur_1 | -0.089 [-0.244, +0.005] | -0.264 [-1.494, +0.689] | -0.261 [-0.894, +0.164] | 8/32 |
| motion_blur_5 | -0.345 [-0.822, -0.017] | +3.149 [-0.890, +8.955] | +0.402 [-0.856, +1.950] | 2/32 |
| motion_blur_10 | -0.585 [-1.678, +0.177] | +2.910 [-0.872, +7.589] | +0.317 [-1.141, +2.378] | 2/32 |
| occlusion_1 | -0.044 [-0.185, +0.061] | +0.010 [-0.307, +0.413] | +0.019 [-0.131, +0.209] | 8/32 |
| occlusion_5 | -0.149 [-0.806, +0.431] | -2.672 [-5.829, -0.421] | -1.626 [-3.860, -0.168] | 2/32 |
| occlusion_10 | -0.482 [-1.622, +0.524] | -1.841 [-5.795, +1.891] | -1.422 [-4.011, +0.904] | 2/32 |
| exposure_1 | +0.039 [-0.004, +0.099] | -0.022 [-0.563, +0.449] | -0.018 [-0.310, +0.218] | 8/32 |
| exposure_5 | -0.171 [-0.853, +0.257] | +0.084 [-0.557, +0.721] | -0.051 [-0.427, +0.293] | 2/32 |
| exposure_10 | -0.394 [-1.694, +0.401] | -0.199 [-1.697, +1.232] | -0.235 [-1.089, +0.549] | 2/32 |

Primary qualification requires both delta-tIoU and delta-vIoU 95% upper bounds below zero: **False**. Individual-condition patterns are descriptive; no worst-condition winner is promoted.

The full-video burst may fall outside the student query clip or between observed frames. Those cases remain in the primary denominator. Physical dose and actually observed/changed frames are saved per cell. `C05_OVERLAP.json` groups retrospective GT-event overlap; these composition-varying groups are associations, not a causal estimate of overlap.

Retrospective >10%-GT-event-overlap group: 115 cells / 16 parents, delta-tIoU -3.972 [-6.838, -1.201] pp and delta-vIoU -2.157 [-4.191, -0.354] pp. This subset cannot replace the primary population or guide burst re-generation.

## C2-T: fixed original candidate panel

This iteration deliberately reuses the original C1 first16 parents x clean/noise_medium/defocus_medium/jpeg_medium (64 cells), not C0.5 transient candidates. The student candidates and spatial boxes are unchanged. UniversalVTG best.pth with PE-Core-L14-336 is frozen, unifier disabled, and sees the same sampled corrupted pixels. Existing observations are mapped to physical 2fps slots. One cached expert pass per cell; no extra teacher-generated interval can become the final output.

The registered critic score is max over pre-NMS expert proposals of confidence times temporal IoU with each student interval. This is a specific proposal-to-candidate bridge, not a calibrated native candidate likelihood. A negative result cannot establish that all expert scoring rules fail. No GT tunes the bridge; score ties keep the native-first candidate.

| Condition | Native tIoU (%) | Expert tIoU (%) | Oracle tIoU (%) | Uniform tIoU (%) | Expert delta-tIoU (pp) | Expert delta-vIoU (pp) |
|---|---:|---:|---:|---:|---:|---:|
| clean | 36.175 | 43.738 | 50.985 | 36.693 | +7.563 [+1.714, +14.994] | +2.862 [+0.187, +6.153] |
| noise_medium | 35.576 | 41.138 | 51.107 | 35.721 | +5.562 [+1.460, +10.249] | +2.980 [+0.346, +6.087] |
| defocus_medium | 35.142 | 36.970 | 48.639 | 32.586 | +1.828 [-1.154, +5.336] | +0.986 [-0.688, +2.999] |
| jpeg_medium | 36.002 | 40.197 | 50.265 | 35.934 | +4.195 [+0.859, +8.243] | +2.046 [-0.082, +4.978] |
| corrupted_parent_macro | 35.573 | 39.435 | 50.004 | 34.747 | +3.862 [+0.936, +7.161] | +2.004 [+0.078, +4.331] |

Paired excess gain (corrupted minus clean): tIoU **-3.701 [-11.112, +1.313] pp**, corrected vIoU **-0.858 [-3.090, +0.986] pp**. Clean/corrupt headroom must not automatically be called corruption recovery.

Critic minus exact uniform-candidate expectation on corrupted parent macro: tIoU +4.688 [+0.549, +8.613] pp; vIoU +2.189 [+0.012, +4.456] pp.

| Pair group | Clean accuracy (%) [95% CI] | Corrupted parent-macro accuracy (%) [95% CI] | Corrupted valid pairs |
|---|---:|---:|---:|
| all | +69.938 [+54.747, +84.343] | +74.570 [+61.832, +86.123] | 1155 |
| low | +67.403 [+51.270, +82.158] | +63.874 [+50.020, +76.370] | 372 |
| middle | +66.974 [+47.853, +84.725] | +71.497 [+57.474, +84.911] | 365 |
| high | +76.016 [+53.883, +94.444] | +80.752 [+62.196, +96.528] | 418 |

GT ties are excluded; critic ties score0.5. Margin terciles are locked using all64 cells before scoring. Pair accuracy is averaged within cell and then within parent; CI resamples parents. High-margin accuracy is a diagnostic reliability signal, not a calibrated deployment gate.

Corrupted expert >5pp harm cells: tIoU 4/48, corrected vIoU 2/48. Worst changes: tIoU -9.821 pp, corrected vIoU -5.924 pp. Native retained 2/48.

Among the 3 original medium-corruption cells with >5pp temporal damage, the critic restores clean-level tIoU in 1. This is a small descriptive subset; all positive and negative examples are retained in C2_RECOVERY.json.

## Decision, reproducibility and resource scope

No OPD is implemented in this round. The predeclared next-OPD gate requires positive corrupted tIoU gain CI and high-margin accuracy CI above0.5; observed gate=True. Benchmark gate=False. Preserve the separation among corruption failure, student support, and critic error.

Three CPU semantic tests cover locality, deterministic severity, full-video query-independent bursts, and candidate-only ranking. New clean smoke is bitwise identical to old Frozen. Independent readback checks source-only burst generation, no-hit clean identities, scalar bootstrap values, critic scoring and selection, old-C1 oracle parity, and dual implementation s/t/v metrics. Model state hashes are unchanged. No backward or adaptation.

GPU-process wall time: C0.5 412.433s; C2-T 137.058s; superseded smoke10.352s; total 559.843s. Includes loading, input preprocessing, audit hashing within workers and any recorded failed allocations; excludes CPU coding/scoring/report work. Artifact cap8GiB, free-space floor8GiB, per-phase GPU-process cap3600s.

## Source basis and implementation differences

[Zeng et al., CVPR2024](https://arxiv.org/abs/2403.20254) motivates temporal corruption; [official configuration and operators](https://github.com/Alvin-Zeng/temporal-robustness-benchmark/tree/a46eee452222fa67958c81c49496e712dedefeea/extract_corrupted_feature_code/i3d/thumos) were inspected. Their action-centered GT-conditioned generation is a diagnostic reference, not our primary deployment protocol. Our blur/occlusion/drop/freeze operators and sampling are explicitly registered adaptations.

Private media, captions, original IDs, GT coordinates, weights, and raw feature/prediction caches are excluded from the public export. Public anonymous scalar rows and audits preserve all conditions and negative findings.
