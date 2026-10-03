# Source-held-out quality-readout diagnosis

The report is docs/TA_PR_SOURCE_HELD_OUT_REVIEW.md. This is target-search-GT
supervised CPU diagnosis, not unsupervised TTA or a promoted quality model.
192 expert cells from Vid16 and HC14 historically exposed search sources.
FOLDS.json locks nested4/8/12/remaining training sources, excluding the test
source and all its orders/conditions/roles. PREDICTIONS/ROWS contain anonymous
scalar readouts; private latents, model coefficients/normalizers, GT spans,
media and original weights are excluded. RUN reproduction requires those
private cached assets; the public scalar audit is standalone from these
published results plus the bound repository code.

Run CPU controls with scripts/test_tastvg_pr_loso_v1.py and the public audit:
`python -B scripts/audit_tastvg_pr_loso_v1.py public results/tastvg_pr_loso/2026-10-03`.
Dependencies are NumPy, SciPy and scikit-learn; figures use Matplotlib.
Private root auditing additionally uses PyTorch CPU to verify cached latents.
All bootstrap intervals condition on the fixed out-of-fold predictions and
one nested subset sequence, without refits. No fresh-test or causal root
identification claim. Preserve A/CURRENT. See RECOVERY.json for engineering
fixes, without changed scientific configuration.
