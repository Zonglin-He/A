# HC64: global and tube-conditioned temporal posterior fusion

All64 predictions seal before GT. Source_Native224 is the unadapted Source spatial output and its native final-layer temporal MBR. Global6, GlobalTube6 and GlobalPermuted6 share unchanged original BoxFixed geometry/score/DP. Strong is a historical reference, not an acceptance gate. Each of six layer logits is normalized and mapped to the original physical interval axis separately before averaging. Fusion is fixed .5/.5.

| Method | Full vIoU % | tIoU % | Boundary MAE s | Predicted length s | Unobserved sIoU % |
|---|---:|---:|---:|---:|---:|
| Source_Native224 | 26.8938 | 54.8230 | 2.6821 | 8.5256 | 47.6990 |
| Global6 | 29.6952 | 55.9613 | 2.6034 | 8.6481 | 50.5762 |
| GlobalTube6 | 29.7235 | 55.8081 | 2.6163 | 8.6147 | 50.7215 |
| GlobalPermuted6 | 29.6977 | 55.9221 | 2.5990 | 8.6528 | 50.5547 |
| Strong_Reference | 31.3985 | 58.6043 | 2.5113 | 9.1533 | 50.9672 |

| Comparison | vIoU pp [95% CI] | tIoU pp [95% CI] | Boundary MAE seconds [95% CI] | v improve / harm / same | v decline >5pp |
|---|---:|---:|---:|---:|---:|
| GlobalTube6 - Source_Native224 | +2.8297 [0.9075,4.8356] | +0.9851 [-0.6004,2.5937] | -0.0658 [-0.1737,0.0323] | 41/22/1 | 8 |
| Global6 - Source_Native224 | +2.8014 [0.8963,4.7724] | +1.1383 [-0.4703,2.7668] | -0.0787 [-0.1890,0.0221] | 41/22/1 | 8 |
| GlobalPermuted6 - Source_Native224 | +2.8039 [0.8944,4.7641] | +1.0991 [-0.5044,2.7282] | -0.0832 [-0.1946,0.0177] | 42/21/1 | 8 |
| GlobalTube6 - Global6 | +0.0283 [-0.1076,0.1781] | -0.1532 [-0.4858,0.1557] | +0.0129 [-0.0067,0.0332] | 5/8/51 | 0 |
| GlobalTube6 - GlobalPermuted6 | +0.0258 [-0.1380,0.1979] | -0.1140 [-0.4309,0.2007] | +0.0174 [-0.0051,0.0406] | 7/9/48 | 0 |
| GlobalTube6 - Strong_Reference | -1.6750 [-3.1133,-0.3199] | -2.7962 [-5.0858,-0.5780] | +0.1050 [-0.0840,0.2865] | 19/38/7 | 15 |
| Global6 - Strong_Reference | -1.7033 [-3.1196,-0.3634] | -2.6430 [-4.9534,-0.4436] | +0.0921 [-0.0932,0.2740] | 20/38/6 | 15 |

## Identical spatial trajectory, different temporal intervals

Every row below uses the same historical A BoxFixed boxes. These are pure temporal results, including the Source interval row; they are not the overall unadapted Source baseline.

| Interval source | Fixed-space vIoU % | tIoU % | Boundary MAE s |
|---|---:|---:|---:|
| Source_Native224 | 29.0550 | 54.8230 | 2.6821 |
| Global6 | 29.8122 | 55.9613 | 2.6034 |
| GlobalTube6 | 29.7985 | 55.8081 | 2.6163 |
| GlobalPermuted6 | 29.8146 | 55.9221 | 2.5990 |
| Strong_Reference | 31.3985 | 58.6043 | 2.5113 |

## Cost boundary

Head-only cached validation. Input IO, original frozen head projection, posterior construction, three final BoxFixed DPs and engineering recovery checks are measured. Historical native224/expert/decoder work is not counted as new computation. Do not report this as deployment time.

No new RGB/encoder/decoder/ReferDINO calls or updates. Six original head batches per query read G/T/P at two offsets; all final-layer logits/posteriors/intervals recover previous results bitwise. The real deployment of a conditioned branch still requires two offset temporal decoder calls, plus native224, expert evidence, propagation, IO and final DP. This cached validation wall is not deployed latency. Strong remains a reference only. All signed and severe outcomes, parent paired intervals and costs remain in JSON/TSV. HC64 is exposed development data. No fusion-weight scan is authorized; a nonpositive real spatial contribution ends this feedback trial.
