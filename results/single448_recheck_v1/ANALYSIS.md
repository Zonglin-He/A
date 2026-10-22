# Single448 same-view hull: two-direction development recheck

Both offsets fixed separately: offset0 primary,offset1 sensitivity. Existing32 parents/domain were historically exposed. This is a development recheck, not untouched confirmation, per-query branch selection, or a statistical equivalence claim. BoxFixed remains unchanged.320 failed its earlier screen and was not run on these panels.

## hc2:32 parents

| Method | Full vIoU % | tIoU % | Unobserved sIoU % | Mean interval s |
|---|---:|---:|---:|---:|
| Source | 23.4655 | 50.4639 | 48.0826 | 8.1893 |
| Strong | 27.5582 | 50.0107 | 51.1795 | 9.1878 |
| Global6 | 27.8907 | 51.6010 | 51.3546 | 8.1430 |
| Single448_0 | 27.0421 | 49.5632 | 51.1593 | 8.9255 |
| Single448_1 | 27.8547 | 50.7275 | 50.8932 | 9.0133 |

| Comparison | vIoU pp,95%CI | tIoU pp,95%CI | Severe v loss >5pp |
|---|---:|---:|---:|
| Single448_0-Strong | -0.5162 [-1.2039,0.1529] | -0.4475 [-2.8543,2.9376] | 1 |
| Single448_0-Global6 | -0.8487 [-2.0750,0.3463] | -2.0378 [-4.7060,0.6194] | 4 |
| Single448_0-Source | +3.5766 [-4.4708,11.7015] | -0.9007 [-3.5334,1.7540] | 4 |
| Single448_1-Strong | +0.2965 [-0.1880,0.8515] | +0.7168 [-0.3099,1.8714] | 0 |
| Single448_1-Global6 | -0.0360 [-1.3470,1.2865] | -0.8735 [-6.1312,3.1903] | 4 |
| Single448_1-Source | +4.3892 [-3.5601,12.3908] | +0.2636 [-4.5054,4.2185] | 4 |
| Strong-Global6 | -0.3325 [-1.6862,0.9537] | -1.5903 [-6.5690,2.2266] | 5 |
| Global6-Source | +4.4253 [-3.3213,12.3936] | +1.1371 [-0.6253,3.0265] | 3 |

### Actual complete deployment timing,4 fixed queries

| Method | Mean all4 s | Median all4 s | Mean excluding first query s (n=3) | Extra temporal branch s,excluding first |
|---|---:|---:|---:|---:|
| Strong | 10.2754 | 8.9056 | 8.5611 | 3.3373 |
| Global6 | 5.5981 | 5.5064 | 5.2523 | 0.0000 |
| Single448_0 | 7.5817 | 7.3722 | 7.0918 | 2.0116 |
| Single448_1 | 7.5418 | 7.3274 | 7.0658 | 2.0096 |

Four FIXED queries/domain, eight total. Each timed arm independently executes rawRGB decode, native224 both offsets, one ReferDINO eight-observation request without evidence cache, ERef/complete-box propagation, its actual required extra high-res views, physical posterior/MBR/BoxFixed. Resident model and teacher processes; startup/load separately. Cyclic arm order by timing-query index, no RGB/visual/evidence cache substitution in timed runs. Other precision queries reuse existing224/evidence and are NOT called end-to-end timings. Actual query wall includes actual CPU/GPU transfers, parse and final DP; artifact persistence reported separately. First query startup and all-versus-excluding-first timing both reported. No diagnostic per-module synchronization.

All original Source and Global6 outputs recover previous fixed32 bitwise. Every actual timed complete pipeline independently recomputes224, eight-frame ReferDINO and geometry and must reproduce the cached precision prediction bitwise. Encoder hooks only count calls, with no module-level timing/synchronization. Startup is visible; all4 and matched excluding-first-query means are both reported. Three warm queries/domain are a small cost sample, not a full-stream throughput guarantee.

## vidstg:32 parents

| Method | Full vIoU % | tIoU % | Unobserved sIoU % | Mean interval s |
|---|---:|---:|---:|---:|
| Source | 15.6772 | 34.0110 | 44.0397 | 10.5907 |
| Strong | 23.0979 | 33.8827 | 59.9987 | 11.7943 |
| Global6 | 23.9576 | 35.6554 | 60.6281 | 11.6918 |
| Single448_0 | 22.5905 | 32.5598 | 60.3027 | 11.2054 |
| Single448_1 | 23.1900 | 33.9364 | 59.8246 | 11.5109 |

| Comparison | vIoU pp,95%CI | tIoU pp,95%CI | Severe v loss >5pp |
|---|---:|---:|---:|
| Single448_0-Strong | -0.5075 [-1.0763,-0.0242] | -1.3229 [-2.6413,-0.2400] | 0 |
| Single448_0-Global6 | -1.3671 [-3.0472,0.1439] | -3.0956 [-5.8166,-0.5791] | 4 |
| Single448_0-Source | +6.9132 [2.8698,11.1594] | -1.4513 [-3.5811,0.6027] | 2 |
| Single448_1-Strong | +0.0920 [-0.5438,0.7816] | +0.0537 [-0.9351,1.0628] | 0 |
| Single448_1-Global6 | -0.7676 [-2.5315,0.8326] | -1.7190 [-4.2465,0.6023] | 3 |
| Single448_1-Source | +7.5128 [3.1241,12.1701] | -0.0747 [-2.3794,2.1165] | 2 |
| Strong-Global6 | -0.8596 [-2.4761,0.6133] | -1.7727 [-4.1197,0.4235] | 3 |
| Global6-Source | +8.2804 [3.5570,13.5748] | +1.6443 [-0.0972,3.8050] | 3 |

### Actual complete deployment timing,4 fixed queries

| Method | Mean all4 s | Median all4 s | Mean excluding first query s (n=3) | Extra temporal branch s,excluding first |
|---|---:|---:|---:|---:|
| Strong | 8.3803 | 7.9415 | 8.8266 | 3.6899 |
| Global6 | 6.2432 | 5.4296 | 5.0925 | 0.0000 |
| Single448_0 | 6.9267 | 6.5969 | 7.2570 | 2.2213 |
| Single448_1 | 6.9883 | 6.5783 | 7.3144 | 2.2902 |

Four FIXED queries/domain, eight total. Each timed arm independently executes rawRGB decode, native224 both offsets, one ReferDINO eight-observation request without evidence cache, ERef/complete-box propagation, its actual required extra high-res views, physical posterior/MBR/BoxFixed. Resident model and teacher processes; startup/load separately. Cyclic arm order by timing-query index, no RGB/visual/evidence cache substitution in timed runs. Other precision queries reuse existing224/evidence and are NOT called end-to-end timings. Actual query wall includes actual CPU/GPU transfers, parse and final DP; artifact persistence reported separately. First query startup and all-versus-excluding-first timing both reported. No diagnostic per-module synchronization.

All original Source and Global6 outputs recover previous fixed32 bitwise. Every actual timed complete pipeline independently recomputes224, eight-frame ReferDINO and geometry and must reproduce the cached precision prediction bitwise. Encoder hooks only count calls, with no module-level timing/synchronization. Startup is visible; all4 and matched excluding-first-query means are both reported. Three warm queries/domain are a small cost sample, not a full-stream throughput guarantee.

## Decision boundary

Do not choose an offset per query or after reading GT; retain the complete offset0/offset1 signed panel. No automatic production adoption, full resume or new resolution/weight scan. Precision changes on these exposed32 panels do not establish equivalence; the complete method must be assessed against Source and temporal differences against fixed Global6/Strong. Entire fixed-space effects, thresholds, severe tails, individual timings and bootstrap CIs are public.


## Root outcome

Single448 offset0 is about0.52pp/0.51pp below original Strong in full vIoU onHC32/Vid32; its complete warmed pipeline is17.16%/17.78% faster, not50%. Offset1 is only the fixed sensitivity control: its point estimates near Strong do not establish equivalence or justify posthoc promotion.

Global6 has the best mean full vIoU AND tIoU on BOTH32 panels, at the lowest actual complete cost. This reverses the olderHC63 precision ranking, where Strong was better. Sparse developmental panels do not establish a universal resolution benefit. Source remains the overall unadapted baseline. No new candidate is registered as a general improvement; no production/default or full-queue switch follows these results.

Against Global6, severe full-v losses>5pp are4/4queries for single448offset0 (HC/Vid) and4/3foroffset1. Against Strong, the corresponding counts are1/0and0/0. All threshold harms/rescues and signed losses are published, including the unfavorable comparisons.
