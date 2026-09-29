# A0.5 R16 label-free direction audit

A0.4 offline predictor fitting stopped. A0.5 isolates test-time direction quality in the already audited fixed R16 action space.

- `vg_tta/desta3d_v3_a05_signal.py`: GT-free complete-vocabulary losses, hash selection and equal-branch coefficient balancing.
- `scripts/desta3d_v3_a05_signal.py`: CPU registration/reference projection; isolated frozen GPU worker at zero coefficient field. Observed-to-mild teacher KL or observed entropy, no optimizer.
- `scripts/audit_desta3d_v3_a05_signal.py`: sealed full logits/gradients/reference projection independent NumPy readback, all16 gates.
- `scripts/crosscheck_desta3d_v3_a05_signal.py`: second Torch geometry/standard-library aggregation, independent hash selection/basis checks.
- `scripts/summarize_desta3d_v3_a05_signal.py`: anonymous failed-gate report; refuses to mark a passing signal complete without conditional native work.
- `tests/test_desta3d_v3_a05_signal.py`: complete152775-vocabulary, teacher-detach, coefficient-chain/frozen-scope and routing controls.

[Protocol](../../protocols/desta3d_v3_a05_signal_v1.md) · [Results](../../results/desta3d_v3/2026-09-29/A05_UNLABELED_SIGNAL.md) · [All anonymous cases](../../results/desta3d_v3/2026-09-29/A05_UNLABELED_SIGNAL.json).

Both signals failed all-three-gate qualification; no new native prediction, training or expert stage was run. Next proposal: expert pseudo-target gradient qualification on locked native support, not direct heterogeneous-logit KL or OPD. Old Oracle-R16 efficacy remains a separate source-GT result. Private videos, labels, weights, raw gradients and logits remain local.

Normalization order matters: requested unlabeled branches are unit-normalized in R16, historical GT oracle balanced in full F before R16 projection. Consistency differentiates mild-input F and transports its direction to observed THW coordinates for diagnostic GT dots. Neither distinction is hidden as exact gradient equivalence.
