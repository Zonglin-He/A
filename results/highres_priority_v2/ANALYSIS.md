# Native448 decomposition and why local224 did not replace it

All64 repeatedly exposed development; no untouched confirmation. Single high-view arms have63, common arms64; never compare unmatched means.

Common4 arms use64; all following resolution/layer/offset/local comparisons use exactly63. Single offset0 is fixed primary; offset1 sensitivity is reported without posthoc selection. No weight scan or production change.

| Method, matched63 | tIoU % | Full vIoU % | Boundary MAE s |
|---|---:|---:|---:|
| Source | 55.1576 | 27.1951 | 2.6737 |
| Only224six | 56.2981 | 30.0946 | 2.5960 |
| Only448 | 57.7307 | 31.4231 | 2.4886 |
| Fusion | 59.1483 | 31.8466 | 2.4704 |
| SingleHigh0 | 51.2004 | 28.5437 | 2.7563 |
| SingleHigh1 | 50.1973 | 27.3835 | 2.7067 |
| SingleFusion0 | 56.3604 | 30.4439 | 2.5313 |
| SingleFusion1 | 56.9716 | 30.6157 | 2.4789 |
| Only224Last | 55.1576 | 29.4482 | 2.6737 |
| High6 | 57.9956 | 31.6833 | 2.4440 |
| High6Fusion | 58.8882 | 31.7143 | 2.4830 |
| Local224 | 53.3935 | 28.8886 | 2.6992 |
| Full224SingleFusion | 53.4938 | 28.8946 | 2.6967 |

| Comparison, matched63 | tIoU pp [95% CI] | Full vIoU pp [95% CI] | Fixed-space vIoU pp |
|---|---:|---:|---:|
| Only448-Only224six | +1.4326 [-2.4478,5.4060] | +1.3285 [-1.3275,4.1719] | +1.2074 |
| Fusion-Only448 | +1.4176 [-1.1368,3.8829] | +0.4235 [-1.4160,2.1214] | +0.4257 |
| High6-Only448 | +0.2648 [-1.1879,1.7313] | +0.2602 [-0.7029,1.1869] | +0.2707 |
| High6Fusion-Fusion | -0.2601 [-1.1714,0.6036] | -0.1322 [-0.7560,0.4446] | -0.0793 |
| Only224six-Only224Last | +1.1405 [-0.4586,2.7893] | +0.6464 [-0.3708,1.6663] | +0.7671 |
| High6-Only224six | +1.6975 [-1.8606,5.4366] | +1.5887 [-0.8603,4.1859] | +1.4781 |
| SingleFusion0-Fusion | -2.7879 [-4.0048,-1.5652] | -1.4026 [-2.2211,-0.6057] | -1.4007 |
| SingleFusion1-Fusion | -2.1767 [-3.5991,-0.7853] | -1.2309 [-2.0831,-0.4091] | -1.2335 |
| SingleHigh0-Only448 | -6.5303 [-9.4050,-3.7095] | -2.8794 [-4.4730,-1.2954] | -2.8183 |
| SingleHigh1-Only448 | -7.5334 [-10.6284,-4.4179] | -4.0395 [-6.0279,-2.1343] | -3.9541 |
| Local224-Fusion | -5.7547 [-8.3001,-3.2726] | -2.9579 [-4.4977,-1.4342] | -2.9357 |
| Local224-SingleFusion0 | -2.9668 [-5.3452,-0.8201] | -1.5553 [-2.8880,-0.3286] | -1.5350 |

## All threshold-crossing cases

| Query | Category | G t% | H t% | Fusion t% | Local t% | H start error delta s | H end error delta s |
|---|---|---:|---:|---:|---:|---:|---:|
| H007 | fusion_rescue | 47.101 | 53.873 | 52.536 | 46.923 | +0.320 | -0.920 |
| H008 | fusion_harm | 51.613 | 36.458 | 39.773 | 48.387 | +2.840 | +0.280 |
| H018 | fusion_harm | 52.362 | 38.109 | 44.186 | 54.286 | -0.599 | +3.760 |
| H019 | high_rescue | 33.178 | 58.612 | 49.395 | 24.939 | -1.560 | -3.440 |
| H021 | high_rescue | 34.559 | 55.042 | 36.623 | 35.162 | -6.986 | +0.599 |
| H023 | fusion_rescue | 31.332 | 69.942 | 58.630 | 32.086 | -1.231 | -4.060 |
| H025 | high_rescue | 30.541 | 90.566 | 44.507 | 30.541 | -1.875 | -8.208 |
| H029 | high_rescue | 29.767 | 52.736 | 48.219 | 25.349 | -2.800 | -0.933 |
| H039 | high_harm | 54.321 | 49.351 | 55.844 | 48.148 | +0.000 | +0.249 |
| H043 | fusion_rescue | 49.856 | 58.790 | 56.484 | 40.634 | -1.880 | +0.640 |
| H052 | fusion_rescue | 49.548 | 64.444 | 61.991 | 38.018 | +0.319 | -2.834 |
| H062 | high_harm | 71.820 | 42.273 | 59.491 | 71.247 | +4.080 | +1.560 |
| H063 | high_harm | 63.438 | 30.420 | 61.702 | 65.529 | +0.867 | +1.867 |

## Input and model evidence

{
  "mean_cosine_by_layer": [
    0.954111972971568,
    0.9225449396504296,
    0.9000362172013238,
    0.8944135592097328,
    0.884643920830318,
    0.8885363042354584
  ],
  "start_rank_spearman_mean": 0.7324632198920623,
  "end_rank_spearman_mean": 0.76867661776343,
  "offset_start_argmax_changes": 98,
  "offset_end_argmax_changes": 113,
  "offset_comparisons": 126,
  "mean_spatial_tokens_ratio": 3.926984126984128,
  "median_GT_target_height_at224_pixels": 165.43681592039803,
  "target_pixel_height_vs_high_gain_spearman": 0.031155489450713628,
  "low13_oracle_mean_t": 0.6409951405159979,
  "high_above_low13_oracle_queries": [
    "H004",
    "H010",
    "H011",
    "H012",
    "H013",
    "H017",
    "H020",
    "H021",
    "H023",
    "H025",
    "H027",
    "H036",
    "H040",
    "H041",
    "H043",
    "H044",
    "H045",
    "H050",
    "H055",
    "H058"
  ],
  "high_half_success_missing_from_low13_queries": [
    "H021",
    "H025",
    "H043"
  ]
}

The resolution change recomputes visual/motion/text fusion and native query flow. Endpoint rank changes refute a pure scalar temperature account, but do not assign causal shares to small details, context or training-scale alignment. Latent cosine is descriptive, not a correctness signal. Local224 did not create comparable pixel scale; no actionness used as new algorithm input.

The local224 trial produced55 whole-frame windows and effective scale only1.0014x vertical/1.0090x horizontal on average. It changed single-offset/last-layer weighting while failing to add448-level pixels. Actual448 increases spatial tokens and recomputes the native upstream evidence; late224 replay cannot reconstitute downsampled RGB. This does not prove all224 representations lack event information.

Original paper reports random-resized-crop training with short side420 and ResNet101/VideoSwin/RoBERTa. Nearer training scale is plausible, not causally isolated here. [Primary paper](https://arxiv.org/html/2502.11168v1).

Full saved/paused queue remains6088/10303; original deferred plan retained untouched. All signed losses and severe tails remain in report.json/signed_queries.tsv. No GT-based query selection, offset selection or deployment promotion.

## Root finding: the cheaper branch changed both pixels and readout

The original448 branch is not a single high-resolution prediction: it uses two physical offsets and envelopes two independently drawn interval posteriors before fusion. Removing one view also removed this hull operation. In matched63, raw single448 offset0 tIoU51.2004% versus dual44857.7307%; mean decoded duration6.8269s versus8.6144s. This is not a clean resolution-only comparison.

A separately locked, explicitly posthoc cached control applies the SAME hull to two draws of one posterior. Offset0 recovers to56.8942% tIoU, +5.6938pp [3.4889,7.7332], with NO second visual view. Therefore much of the apparent one-view failure is readout/coverage, not evidence that the second view alone supplies all the lost knowledge. With global224, primary SelfFusion0 reaches58.2852% tIoU/31.2180% fullv versus old59.1483/31.8466; fullv gap-0.6285pp [-1.1756,-0.1107]. Sensitivity SelfFusion1 reaches59.0460/31.7769, gap-0.1022/-0.0697pp, CIs include zero. It was not the primary and must not be selected from exposed HC63 as a confirmed equivalent. No production adoption.

See [all control results](../highres_selfhull_v1/report.md) and signed/severe tails. This posterior transform is architecture independent and adds no new information; it preserves part of the old decision geometry. It can also reward overcoverage, so negative cases and a fixed follow-up are necessary.

## Why the local224 route remains short of448

1. No real zoom:55/64 windows were full-frame; fixed-height224 preprocessing mostly removed horizontal context instead of increasing target pixels. New GT diagnostics on63 give median target height165.44 pixels at224, and59/63 queries have mean target height above112 pixels. Retaining an entire target of330.87 pixels at448 inside a height224 view is impossible at the same vertical pixel scale, even with a perfect full-target crop. This is a pixel-budget constraint, not a training or optimizer failure. It does not rule out focused partial regions or a different input budget.
2. Different upstream evidence:448 has3.927x as many spatial tokens, with126 offset comparisons showing98 start-peak and113 end-peak changes, and logit rank correlations only.732/.769. A scalar temperature cannot change those ranks. The448 signal is not merely the same224 probability made sharper. This still does not isolate detail, full-scene context, spatial-position distribution or query initialization as a sole cause.
3. Mixed information and selection limits:448-only rescues8 but harms5 original Global6 cases. Fusion retains4 of these rescues and reduces harms to2; there are0 fusion-only threshold rescues. Among63,448 beats the fixed low-resolution13-decision GT oracle on20 queries and crosses.5 onH021/H025/H043 where the13 low candidates cannot. Yet the low13 oracle mean64.0995% exceeds448 mean57.7307%, so low-resolution selection is also unresolved. Neither '224 has no useful information' nor 'only tune posterior weights' matches these facts.
4. Six-layer averaging is not the principal gap: at448 it changes t by+0.2648pp and fullv+0.2602pp relative to the last layer, with intervals crossing zero; both retain the native high-resolution representation. The original code defaults to auxiliary supervision on intermediate layers, so calling them untrained would be wrong.
5. Training-scale match remains an unisolated explanation: the primary paper reports short-side420 training. A448 forward is closer in scale, but the present experiment changes several upstream components together. No causal percentage is assigned to this factor.

Actual single-view448 code was executed on existingH002 and recovers dual-capture offset0 logits bitwise (error0), one appearance/motion/text/fusion prefix and two native time passes. Its measured network wall2.5585s is one cold engineering case. The dual run mean3.7636s includes all63 and diagnostic posterior work; these are not matched population timing statistics, so no end-to-end speedup factor is asserted.

The evidence-backed next candidate is a single native448 view plus restored same-view hull, rather than a full-timeline224 crop that does not zoom. Keep the primary/sensitivity distinction, freeze a choice before any further evaluation, and require a separate fixed recheck. This work does not launch that recheck or restore the saved full queue.

## Final matched low-resolution control

The fixed same-operator low224 follow-up is also explicitly posthoc, with a new immutable plan and all63 new cached decisions sealed before its labels. It adds no model calls. LowSelfFusion0 t55.7175%/v29.7569%; LocalSelfFusion0 t55.6499%/v29.8829%; high SelfFusion0 t58.2852%/v31.2180%. Thus high versus low under the SAME offset0/hull/global fusion is +2.5677pp tIoU [0.2618,4.9661] and +1.4611pp fullv [0.0943,2.8837]. Offset1 sensitivity likewise favors high, but neither is independent confirmation. This rules out the claim that the hull alone recovers the high-resolution evidence in this panel.

See [all low/global/local controls](../highres_selfhull_low_v1/report.md); local minus low under the same readout t-0.0676pp/v+0.1260pp, both CIs cross0. Correct local position minus shifted t+0.2145pp/v+0.2673pp also crosses0. The crop's no-zoom failure remains after fixing the readout. These last controls are a mechanism audit, not another parameter search.

The last-layer/two-offset matched resolution comparison is224last55.1576% versus448last57.7307% tIoU. Six-layer averaging at224 brings56.2981%; it cannot be credited as recovering all high-resolution effect. Intermediate layers are auxiliary-supervised by the original default implementation, so a 'never trained' explanation is unsupported.
