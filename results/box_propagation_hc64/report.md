# HC64: matched component-bank and native-box propagation

The fixed first64 development parents are retained, one query per parent, seed41. All ten predictions share the saved Strong time interval and identical metric implementation. **OldOPD, OldFixedKD and OldDirect are newly executed in this run on the immutable old component bank.** The earlier v0 trial stopped before adaptation; it did not already produce those results. OldFixed alone reuses its sealed EvidenceDP output. The previous failed gate and all old artifacts remain unchanged.

| Method | Full vIoU % | tIoU % | Observed sIoU % | Unobserved sIoU % |
|---|---:|---:|---:|---:|---:|
| ERefStrong | 31.0918 | 58.6043 | 52.2680 | 50.7055 |
| GeometryOPD | 31.3868 | 58.6043 | 53.1654 | 51.2469 |
| OldFixed | 29.3919 | 58.6043 | 47.4528 | 49.1405 |
| OldOPD | 31.1156 | 58.6043 | 52.2680 | 50.7940 |
| OldFixedKD | 31.1705 | 58.6043 | 52.2680 | 50.9014 |
| OldDirect | 31.1068 | 58.6043 | 52.1358 | 50.7940 |
| BoxFixed | 31.3985 | 58.6043 | 55.3363 | 50.9672 |
| BoxOPD | 31.1909 | 58.6043 | 52.8269 | 50.7916 |
| BoxFixedKD | 31.0459 | 58.6043 | 52.1779 | 50.6770 |
| BoxDirect | 31.1997 | 58.6043 | 52.8269 | 50.8086 |

| Comparison | vIoU delta pp [95% CI] | Improve / decline / unchanged | Drops >5 pp | Unobserved sIoU delta pp [95% CI] |
|---|---:|---:|---:|---:|
| BoxFixed − OldFixed | +2.0066 [0.5748, 3.4517] | 39 / 23 / 2 | 4 | +1.8267 [-0.1831, 3.7958] |
| BoxOPD − OldOPD | +0.0754 [-0.2427, 0.3599] | 7 / 3 / 54 | 1 | -0.0024 [-0.3614, 0.2899] |
| BoxFixedKD − OldFixedKD | -0.1246 [-0.4088, 0.0925] | 4 / 6 / 54 | 1 | -0.2244 [-0.5895, 0.0550] |
| BoxDirect − OldDirect | +0.0929 [-0.2317, 0.3832] | 8 / 3 / 53 | 1 | +0.0146 [-0.3484, 0.3105] |
| BoxOPD − BoxFixed | -0.2076 [-1.9955, 1.6448] | 30 / 32 / 2 | 8 | -0.1756 [-2.9096, 2.2290] |
| BoxOPD − BoxFixedKD | +0.1450 [-0.0080, 0.3346] | 5 / 2 / 57 | 0 | +0.1146 [0.0000, 0.2600] |
| BoxOPD − BoxDirect | -0.0088 [-0.0263, 0.0000] | 0 / 1 / 63 | 0 | -0.0170 [-0.0509, 0.0000] |
| BoxOPD − ERefStrong | +0.0992 [-0.0536, 0.2897] | 4 / 2 / 58 | 0 | +0.0861 [0.0000, 0.1997] |
| OldOPD − ERefStrong | +0.0238 [-0.2113, 0.3201] | 3 / 3 / 58 | 0 | +0.0885 [-0.2084, 0.4612] |

All parent intervals use10000 paired draws. Observed and unobserved spatial IoU use the fixed actual sampled nodes with GT; all eight expert positions are excluded from unobserved IoU, independently of temporal truncation. Every positive, negative, unchanged and severe decline is retained in report.json and signed_changes.tsv.

## Actual updates and cached incremental costs

Observed incremental trial costs with existing Source/teacher caches; not standalone end-to-end inference latency

| Method | Seconds/query | Attempted updates | Accepted | Rejected | Backwards | Heavy forwards |
|---|---:|---:|---:|---:|---:|---:|
| OldOPD | 0.095847 | 3.000000 | 3.000000 | 0.000000 | 3.000000 | 0.000000 |
| OldFixedKD | 0.049480 | 3.000000 | 3.000000 | 0.000000 | 3.000000 | 0.000000 |
| OldDirect | 0.127118 | 3.000000 | 2.984375 | 0.015625 | 3.000000 | 0.000000 |
| BoxOPD | 0.089455 | 3.000000 | 3.000000 | 0.000000 | 3.000000 | 0.000000 |
| BoxFixedKD | 0.049903 | 3.000000 | 3.000000 | 0.000000 | 3.000000 | 0.000000 |
| BoxDirect | 0.127136 | 3.000000 | 3.000000 | 0.000000 | 3.000000 | 0.000000 |

Each adapted arm has the same locked v0 policy, optimizer, seed and allowance of at most3 rounds×2 paths. Fixed-score rejection restores weights/Adam and stops, so actual update counts can differ. The experiment uses existing Source/ReferDINO caches; these costs are not full deployed inference latency.

## OPD versus fixed-KD histories

Actual visited-history differences and policy differences on exactly the same saved OPD history are reported separately. Identical history nodes must give identical teacher feedback. Same-history probability evaluation only multiplies saved features by saved weights; it generates no additional candidate path. Cached fixed-KD studentq is not mistaken for its updated policy.

| Bank | Queries with different final actions | Different unobserved final actions | Same-state policy argmax differences |
|---|---:|---:|---:|
| Old | 9 | 13 | 0 |
| Box | 12 | 11 | 7 |

These are exposed-development results. No GT chooses per-query predictions, stopping steps, candidate slots or parameters. Fixed-score acceptance is not a guarantee of GT improvement; Direct uses the pre-existing fully differentiable soft-history relaxation and the same hard final decode.

## Offline candidate coverage

GT independently selects a candidate from the actual eligible action support only at fixed unobserved sampled nodes with GT, including all E-only fallback nodes. At inactive nodes every stored non-E proposal is excluded, so outside-event and confidence-rejected proposals do not count as actionable coverage. All other nodes retain E. This is an offline diagnostic, not an online method and not the exact global maximum of interpolated complete-video vIoU.

| Bank | Fixed unobserved GT nodes | E-only nodes | Mean oracle node-IoU gain pp | GT nodewise full-vIoU readout % |
|---|---:|---:|---:|---:|
| Old | 1365 | 545 | 5.4833 | 34.3111 |
| Box | 1365 | 397 | 6.6157 | 34.9185 |
