# HC64: BoxFixed with native and tube-guided temporal decoding

A is the exact previously sealed BoxFixed+Strong reference. B, C and D share one fresh224 native capture and the same cached eight selected ReferDINO boxes/masks. B uses native224 endpoints; C replays the cached temporal decoder once using the full-timeline auxiliary tube; D uses the fixed permuted spatial prior. There is no448 call, new expert call, OPD or optimizer in the new paths. All64 predictions are sealed before labels.

| Method | Full vIoU % | tIoU % | Boundary MAE s ↓ | Predicted length s | Unobserved sIoU % |
|---|---:|---:|---:|---:|---:|
| A_Strong | 31.3985 | 58.6043 | 2.5113 | 9.1533 | 50.9672 |
| B_Native224 | 29.0568 | 54.8230 | 2.6821 | 8.5256 | 50.7992 |
| C_TubeReplay | 28.8196 | 54.4344 | 2.7075 | 8.4674 | 50.8219 |
| D_Permuted | 29.1586 | 55.1254 | 2.6754 | 8.6099 | 50.8118 |

| Comparison | vIoU delta pp [95% CI] | tIoU delta pp [95% CI] | Boundary MAE delta s [95% CI] | vIoU improve / harm / same | vIoU drops >5 pp |
|---|---:|---:|---:|---:|---:|
| C_TubeReplay − B_Native224 | -0.2372 [-0.4401, -0.0710] | -0.3886 [-0.6844, -0.1370] | +0.0254 [0.0054, 0.0480] | 4 / 9 / 51 | 0 |
| C_TubeReplay − D_Permuted | -0.3390 [-0.6621, -0.0404] | -0.6910 [-1.2361, -0.1699] | +0.0321 [-0.0040, 0.0687] | 14 / 17 / 33 | 0 |
| B_Native224 − A_Strong | -2.3417 [-4.0344, -0.8474] | -3.7813 [-6.4398, -1.2991] | +0.1708 [-0.0443, 0.3947] | 20 / 38 / 6 | 19 |
| C_TubeReplay − A_Strong | -2.5789 [-4.3471, -1.0191] | -4.1698 [-6.9414, -1.5953] | +0.1962 [-0.0256, 0.4288] | 20 / 39 / 5 | 19 |
| D_Permuted − A_Strong | -2.2400 [-3.9553, -0.7197] | -3.4789 [-6.1681, -0.9864] | +0.1641 [-0.0454, 0.3855] | 20 / 37 / 7 | 16 |
| D_Permuted − B_Native224 | +0.1017 [-0.1620, 0.3800] | +0.3024 [-0.1397, 0.7575] | -0.0068 [-0.0334, 0.0205] | 13 / 12 / 39 | 0 |

Negative boundary-MAE differences are improvements. All intervals retain their original physical frame/offset mapping, with boundary errors and lengths divided by the explicit input FPS. Paired intervals use10000 parent-query draws. Full signed positive/negative/unchanged and severe outcomes remain in JSON/TSV. A confidence interval spanning zero does not establish equal accuracy.

## Pure temporal contribution with A spatial trajectory fixed

| Interval source | Fixed-A full vIoU % | tIoU % | Boundary MAE s |
|---|---:|---:|---:|
| A_Strong | 31.3985 | 58.6043 | 2.5113 |
| B_Native224 | 29.0550 | 54.8230 | 2.6821 |
| C_TubeReplay | 28.8172 | 54.4344 | 2.7075 |
| D_Permuted | 29.1899 | 55.1254 | 2.6754 |

This second analysis uses exactly the same saved A box trajectory for every interval, including its existing fallback boxes outside the old interval. It generates no additional prediction and separates interval effects from the final interval-conditioned BoxFixed re-selection. All fixed-node spatial IoUs must remain identical in this analysis.

## Actual cost and reuse boundaries

Shared224 capture is measured once per query; component totals add that measured shared cost to each measured branch finish, excluding shared output-cache IO. Multi-branch trial wall includes all B/C/D work and IO; it is not one deployed method latency. Expert evidence hits and new native224 misses are reported separately. No predeclared speedup multiplier is assumed.

Predictions are reused; this run does not remeasure a64-query Strong end-to-end latency. Historical A timings are not treated as matched current costs.

Per-stage224/expert/evidence/auxiliary DP/extra decoder/postprocess/final DP and IO statistics are retained in report.json. Source model loading and transfers remain measured components. The existing-query raw-input production smoke, including its expert cache miss and Strong reference, is separate acceptance evidence; it is not silently averaged into this cache-hit panel.

Tube replay remains experimental. C−B tests whether tube feedback helps; C−D tests sensitivity to spatial placement. This exposed-development comparison does not validate other backbone adapters.
