# Spatial Self-Rank control, 2026-09-30

Self-derived spatial ranking did not reproduce the frozen specialist's future-transfer gain on the fixed development panel. With five shared-source orders, corruption future ΔvIoU (percentage points, mean ± sample SD) was:

| Spatial preference | Future ΔvIoU | Future ΔsIoU | Whole ΔvIoU |
|---|---:|---:|---:|
| Expert-Rank / frozen Final | +0.025505 ± 0.020209 | +0.094390 ± 0.071792 | +1.517681 ± 0.859943 |
| Self-Rank | −0.017059 ± 0.013361 | −0.053115 ± 0.032443 | +1.482004 ± 0.854531 |
| Random-Rank | −0.004011 ± 0.016776 | −0.009502 ± 0.047416 | +1.494774 ± 0.859934 |

Expert minus Self future ΔvIoU was +0.042563 pp on average and positive in all five paired orders. Self-Rank was negative in four of five orders. Clean future ΔvIoU was also negative for Self (−0.018686 pp), versus Expert +0.028776 pp. Whole-stream improvement is dominated by the unchanged temporal reranking and is not evidence that Self-Rank helped. All three spatial-ranking arms had zero source/cell >5 pp future vIoU harm in this panel; the small negative Self effect should still be reported.

The new arm completed 480 arrivals: the same 16 already-exposed VidSTG sources, one query per source, five fixed orders, clean and five 5% transient corruption conditions. The Vid-trained TA checkpoint, 1792 parameters, nine current-state probes, SGD .005, one step, persistent state and pre-update output are unchanged. Spatial q is softmax of negative descending average ranks of detached student p. The temporal specialist is retained. No Sa2VA output is read by the new arm. Expert and Random results are reused from their sealed previous runs, with matched row identities and reference scores verified.

A predeclared admission difference limits causal attribution: removing Sa2VA also removes its all-invalid-mask veto, so Self performs 120 updates versus Expert/Random 102. This is a completely spatial-teacher-free control, not an exact rank-only intervention. These five orders share the same 16 sources, and the endpoint uses the historical sampled-grid development definition, not the official dense Paper48 endpoint. The result supports keeping the frozen specialist-ranking method for the authorized evaluation; it does not prove universal necessity of external knowledge or reproduce OPSA.

Validation passed: 480 state links, 120 independent student-rank/probability/KL checks, 215,040 SGD-coordinate checks, 1,920 temporal critic scores, four full spatial reinsertions and two full six-layer temporal reinsertions. Three mathematical CPU tests passed. Public scalar audit passed 14,472 checks across 2,880 anonymous arm-arrival rows. Predictions were globally sealed before reading only the 16 previously exposed labels. Frozen Final and production registration were unchanged.

The successful GPU run took 210.617 seconds; total including a preserved 122.995-second engineering interruption was 333.612 seconds. The interruption corrected an overbroad file-access hook that would reject a legacy expert-barrier metadata hash; actual spatial outputs remained forbidden. All 264 preserved partial outputs/state hashes reproduced exactly after this metadata-only repair. No partial attempt was scored or selected by outcome. New spatial/temporal expert inference count was zero.

Paper48 was saved and temporarily paused at spatial 2,658/2,658 and temporal 465/2,658 receipts. After this result's public verification it resumes from saved receipts. The original deadline remains 2026-10-02 09:20:26 +08; old B1, external baselines and the superseded paper queue remain stopped. No Final retuning is triggered by this control.

See [protocol](../protocols/tastvg_self_rank_v1.md), [report](../results/tastvg_self_rank/2026-09-30/REPORT.md), [anonymous rows](../results/tastvg_self_rank/2026-09-30/ROWS.json), and [public audit](../results/tastvg_self_rank/2026-09-30/PUBLIC_AUDIT.json).
