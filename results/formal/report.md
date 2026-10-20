# Formal evaluation

Only completely sealed, scored and hash-bound jobs contribute numbers. Pending, running, failed or integrity-blocked jobs remain unfilled. No new inference, scoring, GT reading, parameter selection or filtering is performed by this reporter.

## Main table

| Backbone | Target | Output | m_vIoU % | vIoU@0.3 % | vIoU@0.5 % | Status |
|---|---|---|---:|---:|---:|---|
| tastvg | hc2 | Source | 24.9948 | 37.8023 | 12.8812 | completed |
| tastvg | hc2 | Full | 26.8784 | 40.8517 | 14.5636 | completed |
| tastvg | vidstg | Source | — | — | — | running |
| tastvg | vidstg | Full | — | — | — | running |
| cgstvg | hc2 | Source | — | — | — | pending |
| cgstvg | hc2 | Full | — | — | — | pending |
| cgstvg | vidstg | Source | — | — | — | pending |
| cgstvg | vidstg | Full | — | — | — | pending |

Official vIoU@ thresholds use strict >0.3 and >0.5. m_sIoU uses the dense fixed-GT-frame spatial metric; actual feature-node metrics are separately named.

## TA-STVG five-row ablation

| Target | Output | m_vIoU % | vIoU@0.3 % | vIoU@0.5 % | m_tIoU % | m_sIoU % | Unobserved actual m_sIoU % |
|---|---|---:|---:|---:|---:|---:|---:|
| hc2 | Source | 24.9948 | 37.8023 | 12.8812 | 48.6473 | 50.6858 | 51.0497 |
| hc2 | Spatial | 26.1279 | 39.3270 | 14.5636 | 48.6473 | 52.4878 | 52.4640 |
| hc2 | Temporal | 25.7652 | 39.0641 | 12.1451 | 50.3561 | 50.6858 | 51.0497 |
| hc2 | Full | 26.8784 | 40.8517 | 14.5636 | 50.3561 | 52.4878 | 52.4640 |
| hc2 | WithoutOPD | 26.6088 | 40.4311 | 14.5110 | 50.3561 | 51.9112 | 51.9662 |
| vidstg | Source | — | — | — | — | — | — |
| vidstg | Spatial | — | — | — | — | — | — |
| vidstg | Temporal | — | — | — | — | — | — |
| vidstg | Full | — | — | — | — | — | — |
| vidstg | WithoutOPD | — | — | — | — | — | — |

| Target | Output | tIoU>=0.5 % | tIoU=0 % | Boundary MAE seconds | Observed actual m_sIoU % |
|---|---|---:|---:|---:|---:|
| hc2 | Source | 50.7886 | 7.6236 | 3.2907 | 51.4714 |
| hc2 | Spatial | 50.7886 | 7.6236 | 3.2907 | 54.1387 |
| hc2 | Temporal | 51.8402 | 4.8370 | 3.1712 | 51.4714 |
| hc2 | Full | 51.8402 | 4.8370 | 3.1712 | 54.1387 |
| hc2 | WithoutOPD | 51.8402 | 4.8370 | 3.1712 | 52.7022 |
| vidstg | Source | — | — | — | — |
| vidstg | Spatial | — | — | — | — |
| vidstg | Temporal | — | — | — | — |
| vidstg | Full | — | — | — | — |
| vidstg | WithoutOPD | — | — | — | — |

## Actual execution cost

| Backbone | Target | Full query wall mean s | Adaptation mean s | Complete allocated CUDA peak GiB | Standalone Source total s |
|---|---|---:|---:|---:|---:|
| tastvg | hc2 | 8.6510 | 1.1034 | 8.0851 | — |
| tastvg | vidstg | — | — | — | — |
| cgstvg | hc2 | — | — | — | — |
| cgstvg | vidstg | — | — | — | — |

These are the actual resident-run costs, including the ReferDINO work actually executed; cache reuse counts are disclosed per job in results.json. They are not cold uncached latency estimates. Source224 phase time is only a component and cannot stand in for standalone Source end-to-end latency; the separate verified benchmark below is the only source of that measurement. CUDA memory is the maximum allocated peak of the student and teacher processes, not the sum; missing complete instrumentation remains pending.

## Standalone Source versus Full end-to-end benchmark

| Backbone | Target | Parents | SourceStandalone mean s | Full mean s | Full / Source | Source allocated peak GiB | Full allocated peak GiB | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tastvg | hc2 | — | — | — | — | — | — | pending |
| tastvg | vidstg | — | — | — | — | — | — | pending |
| cgstvg | hc2 | — | — | — | — | — | — | pending |
| cgstvg | vidstg | — | — | — | — | — | — | pending |

Benchmark numbers are published only after status, manifest/runtime binding, unique measured parents, every measured and warmup bitwise prediction check, cache scope and recomputed summaries pass. They cover at most sixteen prespecified parents, exclude warmup and separate weight-load time. Missing or invalid benchmarks remain unfilled; Source224 is never substituted.

results.json retains all signed query outcomes and parent identifiers replaced by per-job anonymous ordinals. It contains no captions, original parent IDs, video paths, media, annotations, checkpoint tensors or raw caches. Parent-level paired intervals and query-weighted intervals remain separate; full-dataset results that include development parents are not labelled fully independent confirmation.
