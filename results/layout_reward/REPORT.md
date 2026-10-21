# HC128 layout reward comparison

Completed exposed development panel: VidSTG to HC2, 128 parent videos, one query each, seed 41. All arms use the identical saved Strong interval. Geometry OPD reuses sealed predictions after exact Source, ERef and observation-position checks. No per-query method or stopping-step selection.

| Method | Full vIoU (%) | Unobserved-node sIoU (%) | Observed-node sIoU (%) |
|---|---:|---:|---:|
| ERefStrong | 28.5877 | 49.9315 | 51.5311 |
| GeometryOPD | 29.0669 | 50.9973 | 53.8285 |
| LayoutOPD | 28.8628 | 50.4255 | 52.2375 |
| LayoutDirect | 28.8058 | 50.2491 | 51.8749 |

Unobserved sIoU uses actual sampled frames with GT after excluding all eight expert observation positions, independently of the predicted temporal interval. It does not treat interpolated frames as observations.

| Comparison | vIoU delta pp [95% CI] | Improved / declined / unchanged | vIoU drops >5 pp | Unobserved sIoU delta pp [95% CI] | Unobserved drops >5 pp |
|---|---:|---:|---:|---:|---:|
| GeometryOPD-ERefStrong | +0.4793 [0.2069, 0.8123] | 92 / 33 / 3 | 0 | +1.0659 [0.4340, 1.7828] | 2 |
| LayoutOPD-ERefStrong | +0.2752 [0.0439, 0.5288] | 87 / 38 / 3 | 1 | +0.4941 [0.0517, 0.9898] | 1 |
| LayoutDirect-ERefStrong | +0.2181 [0.0209, 0.4495] | 88 / 37 / 3 | 0 | +0.3176 [-0.0806, 0.7360] | 3 |
| LayoutOPD-GeometryOPD | -0.2041 [-0.4444, -0.0188] | 57 / 68 / 3 | 1 | -0.5718 [-1.2599, -0.0158] | 6 |
| LayoutOPD-LayoutDirect | +0.0571 [-0.1000, 0.2367] | 53 / 72 / 3 | 0 | +0.1765 [-0.1335, 0.5012] | 1 |
| LayoutDirect-GeometryOPD | -0.2612 [-0.5416, -0.0259] | 64 / 61 / 3 | 2 | -0.7482 [-1.3997, -0.2067] | 7 |

Intervals use 10,000 paired parent-video bootstrap draws. This is an exposed development comparison, not independent confirmation. All 128 positive and negative anonymous query results are retained in hc128.json.

## Decision

At this fixed prespecified configuration, replacing geometry with layout does not improve the method. Layout OPD loses 0.2041 pp full vIoU and 0.5718 pp unobserved-node sIoU relative to Geometry OPD; both paired bootstrap intervals lie below zero on this exposed development panel. Layout OPD improves over no adaptation, but Geometry OPD retains the larger gain. Layout OPD minus same-layout Direct is only +0.0571 pp full vIoU with an interval crossing zero, so an independent advantage of OPD under this reward is not established. Keep the geometric registered method; this one configuration does not rule out all possible layout layers or calibrations. No further tuning was performed.

## Declining queries

All declines are included in the accompanying JSON. The following table shows the ten largest full-vIoU declines for each adaptation relative to ERef+Strong; values are percentage points.

| Query | Geometry OPD | Layout OPD | Layout Direct |
|---|---:|---:|---:|
| H015 | +0.1086 | -1.6326 | -0.2822 |
| H032 | -0.4499 | -0.0229 | -0.0412 |
| H036 | -3.6688 | -7.1764 | -4.3326 |
| H040 | +0.1281 | -0.2751 | -2.4776 |
| H045 | -1.6671 | -0.6354 | -1.5916 |
| H052 | +0.1571 | -0.7029 | -0.8447 |
| H060 | -0.2950 | -0.2001 | +0.5321 |
| H063 | -1.3078 | -1.3700 | -1.7203 |
| H070 | -1.1521 | -0.8930 | -0.9496 |
| H080 | -1.6810 | -1.0730 | -0.1042 |
| H097 | +1.8701 | +2.2568 | -1.8635 |
| H102 | -3.3664 | -0.0261 | +0.5325 |
| H109 | -1.5367 | +0.1122 | -1.3800 |
| H122 | -0.6793 | -0.6374 | -1.4635 |
| H123 | -0.1857 | -1.0729 | -1.0449 |
| H125 | +1.6410 | -1.9775 | -1.8726 |

## Execution and limits

Layout temperature 0.1 was prespecified; HC learning rate 0.014 and 40 updates were retained. No temperature, learning-rate, step or feature-layer search was performed. Direct differentiates the same layout reward through ROI coordinates. Only the per-query temporary content state changes; network weights and feature maps remain frozen.

Adaptation time only: Layout OPD 1.2832 seconds/query, Layout Direct 1.2309 seconds/query. These are observed trial step costs, not end-to-end standalone deployment latency; Source and ReferDINO feature export were shared and geometry predictions reused.

The original formal VidSTG evaluation remains paused at 8405/10303. No Vid128 layout experiment was run. Old methods/results are preserved.

## Missed corrections relative to geometric OPD

| Query | ERef+Strong vIoU (%) | Geometry OPD (%) | Layout OPD (%) | Layout Direct (%) | Layout minus geometry (pp) |
|---|---:|---:|---:|---:|---:|
| H084 | 2.7143 | 14.0159 | 3.0598 | 3.0317 | -10.9561 |
| H126 | 13.3187 | 19.2224 | 14.8604 | 15.2715 | -4.3620 |
| H125 | 23.5987 | 25.2398 | 21.6212 | 21.7262 | -3.6185 |
| H036 | 44.9383 | 41.2695 | 37.7620 | 40.6057 | -3.5075 |
| H120 | 0.2470 | 2.5953 | 0.7273 | 0.1501 | -1.8681 |
