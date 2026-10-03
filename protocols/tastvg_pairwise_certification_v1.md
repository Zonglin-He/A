# Anchor-free pairwise certification: locked CPU experiment

The question is whether a source-balanced, candidate-versus-candidate calibration can turn the unchanged frozen L32 ranking into useful target replacements. The predecessor b7a8be7a source-native calibration reduced harm on Vid but accepted no HC corrections. This reopening changes the source comparison population and tests one within-set scale normalization. It does not change the ranking or use target labels for fitting.

## Frozen evidence and data boundary

Reuse the anonymous SOURCE_ROWS, SCORE_ROWS and labelled metric rows from `results/tastvg_temporal_latent_quality/2026-10-03`. Official source train/validation counts remain Vid 95/31 and HC 48/16. Only the 31/16 validation sources fit the new calibration; these were already used to select the frozen ridge alpha and in the preceding calibration, so they are not independent validation. Frozen coefficients, standardization and ridge alpha 10/1 do not change.

Each target dataset keeps 32 development and 16 confirmation sources, one query/source, two orders, clean plus five 5% corruptions, and the same 25% expert schedule. Across datasets: 1,152 arrivals, 288 experts (240 corrupt, 48 clean), 864 nonexperts unchanged A. All target panels have historical exposure. Official same-domain TA-STVG checkpoints Vid fbb1ed88 / HC ee72f0d9, original two-offset sampling/pixels, Old8 contained in Expanded32, A's 1,792-parameter spatial trajectory (Vid K1 / HC K8) and temporal scorer outputs are frozen. Only expert temporal readout changes.

## Source-only pair construction and fit

For every source's 32 candidates, take each unordered pair once. A strict pair has absolute L score difference > 1e-12. Orient the pair toward the higher score, so its margin is positive; do not exclude negative or zero GT tIoU differences. Score ties are excluded and counted, duplicate intervals are retained. Every nonempty source has total fit weight one, split equally across its strict pairs. An empty source is explicitly reported, not fabricated as 496 independent observations.

Two increasing weighted isotonic mean-delta fits are locked:

- Pair-Raw: higher minus lower L score.
- Pair-Norm (primary): the same margin divided by the IQR of all 32 scores plus 1e-8. IQR uses NumPy's linear 25th/75th percentiles. The epsilon is fixed, not searched. Zero-IQR cases are retained using that denominator and counted. With nonzero epsilon, normalization is not exactly invariant to score multiplication.

Fit all source pairs at their distinct margin knots. Bootstrap 10,000 whole sources with replacement (seed 20261003); each sampled source retains the original within-source pair weights. Interpolate each bootstrap fit at the full-fit knots, using flat bootstrap endpoints where a resample lacks an extreme source. The fixed lower curve is the linear 5th percentile, capped by the central fitted mean. This is an approximate pointwise fitted-mean confidence heuristic, not an individual prediction bound, simultaneous band, conformal guarantee or target-domain safety probability.

The only operating rule, for BOTH arms, is: unique L32 winner strictly above A8; margin within the full source-calibration domain; lower fitted mean > 1e-12. Otherwise retain A8. No threshold/quantile/normalization family is searched. Source pair models are sealed before opening target score rows, then all 288 target decisions are globally sealed before parsing existing target labelled rows. Past GT exposure is disclosed; sealing prevents new outcome-dependent choices, not historical knowledge.

## Evaluation and decision

Use unchanged cached official dense vIoU/tIoU. Report full corruption, expert/nonexpert, clean, both orders, source-macro paired 10,000-source bootstrap intervals, accepting sources, conditional benefit precision, gross gain/loss, and severe harm (delta vIoU < -0.05). For proposed L32 moves, r=(absolute start change+absolute end change)/A8 duration; r<.5 / r>=.5 are diagnostic only, never gates. Accepted-source precision is a ratio of source-balanced quantities with zero-denominator draws disclosed.

Carry forward the previous development GO rule for the primary Pair-Norm: both confirmation full-corruption means > 0, at least three accepting sources/dataset, positive leave-one-source-out means, and lower gross loss and severe-harm count than L32. Apply the same rule descriptively to Pair-Raw. GO is not proof of statistical significance or deployment promotion. No automatic dataset-specific arm choice.

Success cannot uniquely identify anchor mismatch or score-scale shift: changing the comparison population, weighting and selected-winner conditioning is also material. All source pairs remove native-specific conditioning but target top1-vs-A8 and source arbitrary pairs can still differ, in addition to domain shift. Joint failure supports stopping these locked 1D margin recipes, not impossibility of all latent quality models. A structured relative P/R or listwise model is a possible next variable, not an authorized concrete training configuration here.

## Cost and delivery

No GPU, encoder/backbone, expert, replay, backward, candidate generation, parameter update, download, recurring monitor or historical queue restart. Preserve all previous outputs and CURRENT. Deliver isolated code/configuration, complete anonymous positive/negative results, independent source/calibration/choice/aggregation audit, figures, archive check/snapshot/check, and verified GitHub Zonglin-He/A publication.
