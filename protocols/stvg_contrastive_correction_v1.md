# Contrastive Referent Correction — fixed-pool development protocol

32 previously exposed development sources, 16 per target domain. TA-STVG Source checkpoints and frozen STVG-R1-7B; existing Weight-free candidates only. Current deployment and the human-paused full experiment remain unchanged.

## P0: relative referent correction

Reuse all 128 original new_crop Top-1 decisions with exact source/job/response/video hashes. Only the 49 valid non-Source selections (23 sources) with presence at least 0.6 receive a new request. The two original videos, all nine contact-sheet slots, candidate geometry, and clocks are identical; no candidate or frame generation. The model compares Source ID 1 with the selected Alternative ID, and emits exactly Source, Alternative, or Uncertain. Keep the original Top-1 reference only for Alternative. Never repair malformed responses using annotations. No prompt search or new model.

Hold the original Offset-Trust temporal results, Gaussian IoU OPD, 1792 Query/LN parameters, deterministic query seed, five rounds, 32 samples, and per-domain locked configuration fixed. All 32 sources remain in the denominator. Empty-reference arms reuse Frozen/Temporal; identical reference/configuration/seed arms may reuse the exact historical fit and readout with provenance, separately counted from new execution. Otherwise perform actual fitting, full Student redecoding, and Source restoration. Arms: Frozen, Temporal, Top1 Spatial, Temporal+Top1 Spatial, Pairwise Spatial, Temporal+Pairwise Spatial.

Seal every prediction and independent input/reference/Student arithmetic readback before official GT evaluation. Report format legality, reference retention/rejection, common-observed reference quality, full-tube vIoU, dense/observed/unobserved sIoU, positive/negative/neutral/unknown outcomes, and measured cost. No GT candidate construction, oracle fits, bootstrap, or post-score prompt/parameter selection. Historical development labels were previously exposed; this is not independent confirmation.

## Conditional later stages

First inspect saved Delta b_t = adapted minus Source geometry across time, including positive and negative cases. A propagation constraint requires a supported mechanism and a fixed P0 reference set; generic motion/L2/output preservation are not assumed effective. Boundary-quality temporal evidence is a separate later matched experiment with Spatial fixed. Independent confirmation requires a frozen complete recipe. If the simplified P0 still supplies almost no corrections, stop prompt search. No automatic full evaluation or CURRENT promotion.
