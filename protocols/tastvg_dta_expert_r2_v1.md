# R2: matched expert-interval Gaussian temporal adaptation

User authorization: 2026-10-04, following completed R1 commit
885762286dd75b806fae34b0bb712583d5088d84. Run only two new teacher arms.
No frozen scorer, mixture, PoE, weighting, gate, tuning or persistence experiment.

## Question and controls

Can a deployable temporal expert interval drive the same native temporal-head
gradient channel that worked with GT in R1? Spatial A, cached hidden and pixels,
official same-domain EMA checkpoints, original two offsets, incoming head,
strict i<j support and original MAP/envelope are fixed. Only teacher center changes.
Reuse R1's exact fit_query implementation: final temp_embed.layers.1 only,
514 nominal parameters, K=3 ordinary SGD, lr=.01 both datasets, beta=1,
per-offset sigma=one original cell (median adjacent physical-frame spacing),
KL(q||p_theta)+KL(p0||p_theta), query reset and step3 readout/discard.
No new source selection or target tuning. Prior KL remains a soft penalty.

Original design per dataset: 32 development + 16 confirmation sources,
one query/source, two orders, clean + five 5% transient corruptions, 25% experts.
Exactly 288 existing scheduled arrivals have cached final hidden; adapt both
arms there (576 query-arm runs), keep 864 other arrivals identically cached A.
Full-flow reporting is readout emulation, not a new online stream. All sources
have historical exposure; this is same-domain deployment corruption, not fresh
or cross-domain evaluation. A is a research baseline distinct from CURRENT_METHOD.

## Expert support and prelocked teacher selection

Reuse cached UniversalVTG raw valid proposals from the original uniform temporal
view, not Old8/Expanded32 student intervals or the second-view rule. The existing
cache already maps seconds to original physical frames and removes nonpositive
duration proposals. Preserve all remaining proposals, duplicates and order;
no NMS, deduplication, added support, rounding or clipping of Gaussian centers.

- E-Deploy: highest proposal_confidence in that existing list; exact ties choose
  the first original index. This is direct expert top1, not the student A8
  confidence-overlap reranker. Confidence chooses a deployable center; it is not
  interpreted as calibrated correctness or used as a Gaussian weight.
- E-Oracle: greatest continuous physical tIoU with GT span in the identical
  proposal list; exact ties choose first original index. This explicitly labelled
  support diagnostic is not a method and does not maximize post-update metrics.

Both use the exact R1 single-interval Gaussian. Empty/invalid support is a
preparation error, not a silent native fallback or skipped query. The current
cache must be validated before locking and execution. Deploy predictions are
sealed by a separate process with target-GT/result-reading guard before the
oracle teacher worker reads target labels. Oracle's labels cannot affect Deploy.
All two-arm predictions then seal before separate dense scoring.

## Evaluation and interpretation

Primary: E-Deploy minus zero-update Native, tIoU and vIoU, expert-corrupt.
Secondary: E-Deploy minus A8; E-Oracle minus Native/A8/R1; Oracle minus Deploy.
Reuse R1 actual GT adaptation results unchanged. Also score each raw teacher
interval directly and its Gaussian MAP as diagnostic controls, separating
support/selection from gradient execution. Raw-center support tIoU uses continuous
coordinates; official dense readout controls truncate endpoints to integers as
the existing scorer does. Gaussian centers stay fractional and unclipped.
No target-GT choice of step, LR, arm, threshold, source or schedule.

Report both datasets and development/confirmation independently, clean,
expert/nonexpert, both orders, equal-source means and 10000 paired source-bootstrap
intervals (seed20261004), gross gains/losses, >5pp harms, correct-to-wrong at .3,
loss-down/task-down cases, source concentration and leave-one-source-out ranges.
Keep positive controls, rescued cases and baseline-good destruction cases.

E-Oracle improvement with E-Deploy failure supports a deployable center-selection
bottleneck in this fixed experiment. Oracle failure does not prove expert support
is globally useless: imperfect center quality, native-grid projection, fixed-space
utility and the unchanged three-step optimization remain possible explanations.
Neither result by itself proves hard selection destroys useful uncertainty;
mixture and persistence remain untested. Positive versus Native does not establish
safe replacement of A8 or future online transfer. Do not silently promote a winner.

## Verification and public closure

Validate cache pixel/input/head/A bindings and deploy/oracle sealing order.
Independently recompute all 1728 gradients/SGD states/logits/native decodes,
selection maxima and every official dense metric; confirm R1 and nonexpert A
unchanged. Six matched-interface/selector/guard tests, public scalar/CI audit,
source concentration, cases and readable PNG/PDF figures accompany results.
No GPU/backbone/suffix/expert calls, temporal inheritance or spatial updates.
Export code/configuration/anonymous metrics/traces/negative findings to Zonglin-He/A,
exclude private proposals/GT coordinates/hidden/weights/media/raw caches, verify
remote exact contents, update RESEARCH_HISTORY with check/snapshot/check.

