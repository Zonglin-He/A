# Does optimizing a TTA surrogate improve STVG?

This figure uses actual paired network-parameter updates, not fabricated samples, arrows, confidence ranks, or relabeled prototype feedback. All selected queries and signs are retained.

| Target domain | Queries | Surrogate loss improves | Grounding declines among these | Conditional fraction (95% CI) | Mean ΔvIoU (pp) | vIoU win/harm/tie | Severe vIoU decline >5pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| HC-STVG2.0 | 128 | 128 | 45 | 35.16% [26.56, 43.75] | -0.4042 | 30/45/53 | 4 |
| VidSTG | 128 | 128 | 32 | 25.00% [17.97, 32.81] | -0.6069 | 17/32/79 | 5 |

## Figure and data

- [HC real paired scatter](hc2/fig1.svg), [paired data](hc2/paired.csv), [report](hc2/report.json).
- [Vid real paired scatter](vidstg/fig1.svg), [paired data](vidstg/paired.csv), [report](vidstg/report.json).
- PDF/300-dpi PNG versions remain in local outputs/entropy_tta_v2/figures/.

## Scope of the evidence

Each domain has 128 deterministic hash-selected parent videos, one query/video. These are sampled exposed official sets, not complete test domains or fresh unseen validation. Loss is the mean entropy of the four native endpoint distributions; the same loss defines the horizontal axis. The ordinate is the official complete vIoU change, in percentage points. Right-lower points are real episodes in which the optimized surrogate improves but grounding degrades. Ties remain in the conditional denominator.

The implementation adapts only 9,728 existing temporal-decoder LayerNorm affine scalars, using episodic Adam0.001/five fixed steps. Cached final-pass input context, first-pass gating, Source boxes and every other parameter are frozen. It is a temporal LN entropy-minimization adapter, not canonical BN-Tent or a fully re-evaluated adapted first pass. No hyperparameters/steps/cases are selected from outcomes. A single limited adapter does not establish failure of all TTA or all entropy-minimization recipes. Mean gains/losses and harmful fractions must be read together.

**Suggested caption.** Paired effects of entropy-minimization adaptation on STVG. Each point is one parent-video/query episode; the horizontal axis is the reduction in the optimized entropy objective and the vertical axis is the change in official complete vIoU. The shaded quadrant marks episodes where the surrogate improves but grounding degrades. Five fixed updates adapt only the native temporal decoder LayerNorm affine parameters, with episode-wise reset and fixed Source spatial predictions. Fractions report all harmful episodes among surrogate-improving episodes, including unchanged-quality episodes in the denominator.

## Verification

All256 predictions sealed before any new labels were read. Exact Source boxes/logits/prior/interval, frozen replay, parameter restoration and reset logits were checked per query. Source official metrics reproduce the prior full. Independent official-vs-dense evaluator checks and every raw prediction/trace/native-cache SHA are verified. No new expert, new observations or extra backbone was used. Main SimplePath production sources, parameters and full outcomes remain unchanged. V1 pre-optimization stride failure is retained, not counted as a scientific result.
