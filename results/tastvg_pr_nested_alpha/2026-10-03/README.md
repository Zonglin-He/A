# Nested source-level regularization audit

Completed CPU supervised diagnosis, not a deployed TTA method. Vid16 / HC14
search expert sources, 96cells/dataset; historically exposed. Original final6
Inside256->P / Endpoint512->R, A/W, all32/role2 populations and source-weighted
ridge fixed. Nested whole-source inner MAE chooses alpha on [1e-3,...,1e3],
without outer-source labels. Fixed nmax LOSO control reused exactly.

[Full report](../../../docs/TA_PR_NESTED_REGULARIZATION_REVIEW.md),
[protocol](../../../protocols/tastvg_pr_nested_alpha_v1.md).
COHORT/FOLDS, INNER_CV/FIT_SUMMARY/ALPHA_DISTRIBUTION, PREDICTIONS/seal,
ROWS/label join, all/clean/corrupt/orders/conditions SUMMARY, positive/negative
CASES, full alpha CSV, 3 PNG/PDF figure pairs and independent audit receipts
are included. Private features/models/normalizers/GT spans/media are excluded.

MAE and R2 are primary; fixed clipped analytic decision is secondary.
HC winner recall MAE improved; 13/16 outer R2 point values remain negative;
Vid role analytic decision worsened. No method promotion or new experiment.
Conditional paired source-bootstrap10000, overlapping fit folds, no bootstrap
refitting or multiplicity adjustment. Scientific CPU fits11936, GPU calls0.
