# Stable trajectory-crop local224: fixed HC64 trial

HC64 repeatedly exposed development; fixed one configuration; no independent confirmation or production adoption.

| Method | Full vIoU % | tIoU % | Unobserved sIoU % | Boundary MAE s |
|---|---:|---:|---:|---:|
| Source | 26.8938 | 54.8230 | 47.6990 | 2.6821 |
| Global6 | 29.6952 | 55.9613 | 50.5762 | 2.6034 |
| GlobalPlusFull224 | 28.5082 | 53.2124 | 50.7085 | 2.6982 |
| GlobalPlusLocal224 | 28.5023 | 53.1137 | 50.6874 | 2.7007 |
| GlobalPlusShifted224 | 28.4349 | 53.1608 | 50.6847 | 2.6960 |
| Strong | 31.3985 | 58.6043 | 50.9672 | 2.5113 |

| Comparison | tIoU pp [95% CI] | Full vIoU pp [95% CI] | Fixed-space vIoU pp | t rescue / harm |
|---|---:|---:|---:|---:|
| GlobalPlusLocal224-Global6 | -2.8476 [-4.2146,-1.4851] | -1.1929 [-1.9711,-0.4430] | -1.2879 | 1 / 5 |
| GlobalPlusLocal224-GlobalPlusFull224 | -0.0986 [-0.3936,0.1623] | -0.0059 [-0.1372,0.1486] | -0.0059 | 0 / 0 |
| GlobalPlusLocal224-GlobalPlusShifted224 | -0.0471 [-0.4819,0.3626] | +0.0674 [-0.1888,0.3370] | +0.0610 | 0 / 0 |
| GlobalPlusLocal224-Strong | -5.4906 [-8.0513,-2.9650] | -2.8962 [-4.4616,-1.4307] | -2.8743 | 2 / 8 |
| GlobalPlusLocal224-Source | -1.7093 [-3.2001,-0.1896] | +1.6085 [-0.3558,3.6225] | -0.5307 | 1 / 5 |
| GlobalPlusFull224-Global6 | -2.7490 [-4.1182,-1.3965] | -1.1870 [-1.9550,-0.4438] | -1.2820 | 1 / 5 |
| GlobalPlusShifted224-Global6 | -2.8005 [-4.1516,-1.4585] | -1.2603 [-2.0356,-0.4929] | -1.3489 | 1 / 5 |

All signed changes, severe losses, boundary effects, stable-crop overlap and observed/unobserved space are in JSON/TSV. No per-query GT branch selection.

Actual128 local views with one prefix/two time passes each. Nested module timings must not be added. Fixed Global6/eight-expert/spatial cached costs excluded; deployment would also pay them. Model load is one-time; stages include host staging. Strong is a historical precision reference; no fresh matched448 timing, therefore no controlled speedup factor.
