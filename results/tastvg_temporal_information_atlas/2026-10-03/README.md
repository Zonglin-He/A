# TA-STVG temporal latent information atlas, CPU P0

Completed representation characterization of the already saved final temporal
decoder hidden interface. No new STVG predictions, candidates, model/expert
inference, GPU work or persistent adaptation. This is a source-supervised probe
diagnostic, not an unsupervised selector or deployed method.

See `../../../docs/TA_TEMPORAL_INFORMATION_ATLAS_REVIEW.md` for measured findings,
data exposure, position/geometry/prior controls and the no-promotion decision.
The two datasets use 190 source queries (143 fit, 47 validation), 136 frozen
probes, and 288 target expert cells with 45 independent cached target sources.
Validation selects regularization; target selects no model/configuration.

- CONFIG: complete labels, interfaces, splits and checkpoint SHA configuration.
- SOURCE_PATHS: all 136 source-only alpha paths and selected configurations.
- SOURCE_FIT_SEAL / GLOBAL_READOUT_SEAL / LABEL_JOIN: actual stage timing and hashes;
  four pre-lock cached target-span schema inspections are explicitly disclosed.
- ROWS: all 335 validation/target packets, 89 anonymous metric entries each;
  no raw per-frame GT, intervals, captions, features or probe weights.
- SUMMARY: all seven domain/panel groupings, source moments, R²/MAE/MSE/AUROC/AP,
  undefined counts, 10000 source-bootstrap intervals and paired controls.
- ORDER_DIAGNOSTICS: both arrival orders without creating missing latent caches.
- ENDPOINT_DIAGNOSTICS: position-removed query-level boundary errors.
- FIGURE_DATA / figures: exact plotted values and five PNG/PDF figures.
- ROOT_READBACK / PUBLIC_AUDIT: separate private-fitting and anonymous-aggregation
  verification scopes. PUBLIC_AUDIT can run without any private assets.
- CODE_BINDING / ENVIRONMENT / RESOURCES / CPU_TESTS: code, packages, measured
  CPU wall times and validation. Worker wall time is not GPU kernel time.
- PUBLIC_FORMAT: whitespace compaction preserves all parsed statistics exactly.
- AUDIT_NOTES: AP floating roundoff repair in the auditor, no experiment changes.
- DECISION: keep A and deployed method; no automatic GPU follow-up or promotion.

Target historical exposure, source-native/target-A8 relative-anchor difference,
GT-gated phase, preserved priors under within-source shuffle, and source-balanced
pooled versus within-cell R² must be considered before using these measurements.
They do not certify query-specific grounding or safe top-1 large corrections.
