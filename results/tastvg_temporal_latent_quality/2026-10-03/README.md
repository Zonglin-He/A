# Candidate-conditioned temporal latent quality audit

This is a completed source-supervised linear probe, not latent TTA. The target
1152-arrival panel and A trajectory are unchanged. The 288 expert intervals are
scored before joining cached dense GT labels. Source training GT labels only the
fixed source candidate intervals; source validation selects ridge strength.

See ../../../docs/TA_TEMPORAL_LATENT_QUALITY_REVIEW.md and
../../../protocols/tastvg_temporal_latent_quality_v1.md.

SCORE_ROWS and GLOBAL_SCORE_SEAL contain all immutable target scores and choices.
SOURCE_ROWS contains anonymized source candidate labels and selected-probe scores.
SUMMARY/ROWS/DISCRIMINATION/BINARY_ROWS/CASES preserve both panels, clean/corrupt,
positive and negative outcomes. Source labels are derived tIoU, not GT coordinates.
TOP1 and PAIRWISE diagnostics are descriptive analyses after seal. CODE_PINS locks
the executed implementation and audit/report helpers. CALL_ACCOUNTING clarifies
the source barrier observer-only suffix counter without rewriting original barriers.

Validation: from repository root use an environment with NumPy, Torch and Matplotlib:

    python -B -m scripts.test_tastvg_latent_quality_v1
    python -B scripts/audit_tastvg_latent_quality_public_v1.py results/tastvg_temporal_latent_quality/2026-10-03
    python -B scripts/draw_tastvg_latent_quality_v1.py results/tastvg_temporal_latent_quality/2026-10-03

The public audit reproduces scores-to-decisions, candidate metrics, and source
bootstrap. It does not claim to reproduce excluded private latent extraction or
learned coefficients; the independent private root audit checks those from original
receipts. Pretrained/learned weights, raw latent arrays, media, captions and annotation
files are excluded under the established public-export policy. No method is promoted.
