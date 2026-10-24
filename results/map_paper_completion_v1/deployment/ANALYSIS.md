# Actual current-MAP deployment timing

**Completed real uncached-query measurements, with every output bitwise verified.** This experiment compares original native Source and the current complete MAP method on the prior fixed 16-query/16-parent panel in each target domain, with three paired repeats. There are 192 measured actual Source calls and 96 eight-observation expert calls, plus four Source/two expert warmup calls excluded from the reported statistics. No GT is scored in timing or its read-only export.

| Target | Native Source seconds/query | Current FullMAP seconds/query | Paired extra seconds [95% CI] | Mean time ratio [95% CI] |
|---|---:|---:|---:|---:|
| hc2 | 3.399264 | 4.330661 | +0.931397 [0.881090, 0.984301] | 1.274000 [1.260573, 1.288440] |
| vidstg | 2.991875 | 4.025282 | +1.033408 [0.918846, 1.181426] | 1.345405 [1.316435, 1.378454] |

All measured calls and their nested component durations are in [all calls](all_measured_calls.tsv); repeat-averaged paired units and confidence intervals are in [paired timing](paired_timing.json). The bootstrap resamples the 16 paired parent/query averages, not the 48 individual repeats per arm, using 10,000 draws and seed20260920. These are pointwise intervals conditional on this panel and shared host, not familywise guarantees or an estimate of full-set throughput.

## What is included

Measurements run on NVIDIA GeForce RTX5090 (32,607MiB visible, driver580.178.04), Intel Core i5-14600KF, original native224 FP32/two-offset Source and the same eight-observation ReferDINO. The original Source resident-weight/CPU–GPU offload policy is retained. Cross-query RGB, visual, prediction, capture and teacher caches are disabled. Timers include RGB decode, parsing, CPU–GPU transfers, Source inference, actual expert inference and communication, shared initialization, geometry, auxiliary path DP, actual ROI repooling, one prototype update, joint-MAP readout, final interval-conditioned DP and per-call cleanup. Startup/model loading and warmup are recorded separately. Output verification and its I/O occur outside the timer.

Root temporarily stopped only this task's two CPU-control process groups during the matched measurement; both were resumed afterward. Other user services were neither signalled nor reconfigured. This is a measured shared-host condition, not proof of complete isolation.

| Mean component inside FullMAP | hc2 seconds | vidstg seconds |
|---|---:|---:|
| Source native forward | 2.112526 | 2.236248 |
| Expert request wall, including communication | 0.695397 | 0.683217 |
| Expert GPU forward, nested inside request wall | 0.190037 | 0.188686 |
| Shared initialization | 0.009384 | 0.009527 |
| Geometry | 0.092670 | 0.117834 |
| Auxiliary DP | 0.001189 | 0.001523 |
| Actual ROI pooling | 0.002890 | 0.003721 |
| Prototype/evidence update | 0.001326 | 0.001340 |
| MAP readout | 0.000019 | 0.000023 |
| Final DP | 0.001296 | 0.001634 |

**Nested values are not additive:** expert forward is inside expert request wall; postprocessing subcomponents are inside finish wall; Source transfer/forward/cleanup are inside the complete query wall. Different arms need not have identical Source-forward timings under shared-host execution, so component sums are not used to infer the paired total overhead. The source of the complete method's observed extra cost is measured directly by the paired outer wall time.

The measured additional wall cost is approximately0.93s/query on HC and1.03s/query on Vid in these conditions. The approximately0.19s expert GPU forward alone is not the complete expert-related or complete adaptation overhead. No cached reconstruction duration is substituted for deployment latency. Original parameters, observation selection and inference outputs remain unchanged; this study adds no accuracy-driven tuning.
