# Latest update: 2026-09-29 A0.3 completed

A Train-only query-balanced shared channel basis passes the exposed Dev64 structure gate (rank32 median energy86.1936%). Norm-matched native Shared-R16 and Shared-R32 both pass: Δt/s/v vs B1 are +9.0271/+9.2265/+10.9384pp and +9.0739/+10.4231/+12.5314pp. R16/R32 retain83.39%/95.53% of Full Oracle vIoU gain. The registered priority selects **fixed shared-rank16 factorized predictor as the next candidate only**; no learned low-rank network was implemented or trained.

Negative tails remain: v harm>5pp on6/5 queries; B1-good v retained14/17 and13/17. Source-GT oracle,16 exposed parents, descriptive CIs; not fresh/target or ordinary-input learnability. All256 predictions sealed; full scalar/tensor and independent source-parent audits passed. Original CPU hash audit failed from OpenBLAS4 versus original20; isolated v2 reconstructed every delta hash without GPU rerun or changing predictions/tolerances. Original failure retained. No full/fresh/expert/gate/OPD expansion.

[Full anonymous A0.3 results](../../results/desta3d_v3/2026-09-29/A03_SHARED_CHANNEL_BASIS.md).

Earlier entries retain their historical state.

## Implementation and completed run

- CPU: `scripts/desta3d_v3_a03_shared_basis.py` and independent `audit_desta3d_v3_a03_shared_basis.py`.
- Native: `scripts/desta3d_v3_a03_native.py`; exact original execution readback uses **`audit_desta3d_v3_a03_native_v2.py`**. Original failed auditor preserved for provenance.
- Scoring: `score_desta3d_v3_a03_native.py`, `crosscheck_desta3d_v3_a03_native.py`, `summarize_desta3d_v3_a03.py`.
- Protocol: `protocols/desta3d_v3_a03_shared_channel_basis_v1.md`.

All runners are write-once. These are completed artifacts, not an instruction to rerun. Private caches, raw, labels, videos and weights are not published.
