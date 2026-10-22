# Complete two-direction evaluation: BoxFixed + Global6

Both official prediction sets sealed and CPU scoring completed on 2026-10-22. This is the selected frozen complete-box propagation method with native224 Global6 old hull and original physical MBR. Source is the unadapted cross-domain checkpoint. Final and Untuned are exact aliases; they are not independent methods or duplicate executions. No OPD, optimizer, 448 or additional conditional decoder is used.

## Complete metrics

All IoU and threshold rates below are percentages. Means are query weighted. HC2 uses the VidSTG-trained source; VidSTG uses the HC2-trained source.

| Target / source training | Method | Queries / parents | vIoU | vIoU > .3 | vIoU > .5 | tIoU | sIoU | Observed sIoU | Unobserved sIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HC2 / VidSTG | Source | 1902 / 227 | 24.9948 | 37.8023 | 12.8812 | 48.6473 | 50.6858 | 51.4714 | 51.0497 |
| HC2 / VidSTG | Final | 1902 / 227 | 26.7597 | 40.4311 | 15.0894 | 48.5808 | 53.5304 | 57.6122 | 52.8379 |
| VidSTG / HC2 | Source | 10303 / 732 | 11.7777 | 13.6368 | 3.5621 | 33.8046 | 32.7291 | 33.5366 | 33.0254 |
| VidSTG / HC2 | Final | 10303 / 732 | 14.2685 | 18.9653 | 8.3277 | 34.7663 | 35.8199 | 42.4228 | 35.3321 |

Observed/unobserved sIoU uses actual sampled GT-supported physical nodes, independently of the predicted interval. Empty groups are excluded per metric: observed n=1880/8507 and unobserved n=1902/10300 for HC2/VidSTG. These gains reflect ERef plus complete-box propagation together; this full table does not isolate propagation from ERef.

## Paired changes and all negative tails

Intervals below are 95% parent-cluster bootstrap intervals (10,000 resamples, seed20261022) for the query-weighted paired estimate. No full-result parameter selection was performed.

| Target | vIoU pp [CI] | tIoU pp [CI] | Unobserved sIoU pp [CI] | v improved / harmed / same | v decline >5pp |
|---|---:|---:|---:|---:|---:|
| hc2 | +1.7649 [1.2804, 2.2751] | -0.0666 [-0.3408, 0.2169] | +1.7881 [1.0863, 2.4902] | 1099/678/125 | 266 (13.99%) |
| vidstg | +2.4908 [2.1351, 2.8504] | +0.9617 [0.7930, 1.1401] | +2.3067 [1.6390, 2.9859] | 5055/2861/2387 | 1149 (11.15%) |

The full vIoU gain is positive in both directions, and supported by these parent-cluster intervals. HC2 temporal change is near zero with an interval crossing zero; VidSTG temporal gain is positive. This does not establish spatial-to-temporal feedback or an OPD benefit. All signed rows are retained, including 266 HC2 and1149 VidSTG severe vIoU declines. Anonymous query and parent keys are consistent within each target, without captions, video identities or GT coordinates.

- [HC2 complete report](hc2/report.json), all signed rows: [1](hc2/signed_01.tsv), [2](hc2/signed_02.tsv); [all severe declines](hc2/severe_declines.tsv).
- [VidSTG complete report](vidstg/report.json), all signed rows: [1](vidstg/signed_01.tsv), [2](vidstg/signed_02.tsv), [3](vidstg/signed_03.tsv), [4](vidstg/signed_04.tsv), [5](vidstg/signed_05.tsv), [6](vidstg/signed_06.tsv), [7](vidstg/signed_07.tsv), [8](vidstg/signed_08.tsv), [9](vidstg/signed_09.tsv), [10](vidstg/signed_10.tsv), [11](vidstg/signed_11.tsv); [all severe declines](vidstg/severe_declines.tsv).

## Temporal errors and length

| Target | Method | Boundary MAE s | Start MAE s | End MAE s | Mean interval s | Mean length / GT length | tIoU >= .5 % | tIoU = 0 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hc2 | Source | 3.2907 | 3.1236 | 3.4578 | 8.7294 | 1.3962 | 50.7886 | 7.6236 |
| hc2 | Final | 3.2826 | 3.1485 | 3.4168 | 8.8741 | 1.4273 | 50.1577 | 6.6246 |
| vidstg | Source | 7.6332 | 7.1063 | 8.1601 | 10.5494 | 3.3791 | 30.1174 | 20.4601 |
| vidstg | Final | 7.4914 | 7.0875 | 7.8954 | 11.8199 | 3.7014 | 31.6898 | 18.8198 |

The final interval is longer on average in both domains. VidSTG boundary MAE improves by0.1418s, but mean length error increases; stronger event coverage does not remove the overcoverage problem. HC2 temporal improvement is not established.

## Actual receipt costs and cache resets

These are current per-query work recorded during this full run, excluding the original computation of imported complete captures. Stage times overlap and cannot be summed. Process/service startup and loop housekeeping outside each receipt are not a standalone cold deployment benchmark. No matched full Strong timing was run, so no full-stream speedup over448 is inferred.

| Target | Segment | Queries | Fresh / imported capture | Mean query wall s | Mean fresh wall s | RGB hit / miss | Visual hit / miss | Visual hit % |
|---|---|---:|---:|---:|---:|---:|---:|
| hc2 | original_HC_worker | 1902 | 1839/63 | 4.1366 | 4.2775 | 0/1839 | 0/7356 | 0.00 |
| vidstg | before_pause_6088 | 6088 | 6026/62 | 3.8084 | 3.8452 | 3319/2707 | 13344/10760 | 55.36 |
| vidstg | resumed_remaining_4215 | 4215 | 4181/34 | 3.8972 | 3.9268 | 2292/1889 | 9232/7492 | 55.20 |

Combined all-receipt mean query wall is4.1366s on HC2 and3.8447s on VidSTG; fresh-capture means are4.2775s and3.8786s. These include shared Source, expert and spatial evidence work. A Source capture includes both original native224 offsets. See [efficiency.json](efficiency.json) for RGB hit/miss and full/partial/no-visual-hit query groups and per-stage Source224/RGB/expert/evidence/head/posterior/DP costs. Source/expert allocated peaks are separate-process values, not a simultaneous whole-device peak.

The immutable scorer report contains a generic scope sentence saying BOTH Untuned and Final were executed. That sentence is incorrect for this selected alias configuration: every raw prediction and receipt was independently checked, with exactly equal arrays/intervals and zero additional Final posterior/DP time. The original hash-bound report is preserved; this clarification and efficiency.json provide the correct interpretation.

## Protocol and reproducibility

- [Frozen profile, hashes and exact controller/scorer references](protocol.json).
- [Independent signed aggregation, bootstrap, seal, receipt and raw alias verification](verification.json).
- Existing official-versus-dense scorer validation:24,410 comparisons, maximum absolute error2.220446049250313e-15. Publication adds no GT reading or model calls.
- Reproduce the publication audit from a private archived runtime using `python tools/report_full.py --runtime-root /path/to/stvg-opd --output results/boxfixed_full_v1`. This reads existing sealed predictions and scored output only.
- Both splits include historically exposed development/tuning/recheck parents; they are complete official evaluations, not pristine unexposed confirmations. Profiles were frozen before full evaluation. Strong/448 remain historical precision/cost references;320 was rejected and offset1 was not promoted.
- The current main inference tree remains the single selected method. All previous research and signed negative findings remain in [the archived tree](https://github.com/Zonglin-He/A/tree/archive/research-pre-mainline-20261022). Private media, captions, labels, weights and raw intermediate tensors are not published.
