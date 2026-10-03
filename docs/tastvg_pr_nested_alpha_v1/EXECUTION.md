# Finite CPU nested-alpha execution

Run from repository root with .conda/tubedetr/bin/python -B:

1. scripts/test_tastvg_pr_nested_alpha_v1.py
2. scripts/run_tastvg_pr_nested_alpha_v1.py prepare
3. scripts/run_tastvg_pr_nested_alpha_v1.py fit
4. scripts/run_tastvg_pr_nested_alpha_v1.py readout
5. scripts/run_tastvg_pr_nested_alpha_v1.py diagnose
6. scripts/audit_tastvg_pr_nested_alpha_v1.py root
7. scripts/audit_tastvg_pr_nested_alpha_v1.py public
8. scripts/draw_tastvg_pr_nested_alpha_v1.py
9. Review all PNG/PDF, write TA_PR_NESTED_REGULARIZATION_REVIEW.md and ROOT_REVIEW.
10. Archive check/snapshot/check; explicit public export, remote-byte verification;
    save FINAL_COMPLETION, then archive receipt and check/snapshot/check.

Private runtime: artifacts/tastvg_pr_nested_alpha_v1. Public anonymous results:
results/tastvg_pr_nested_alpha/2026-10-03. Reuse predecessor source packs read-only.
No controller, monitor, GPU jobs or historical queues are launched. Guard/model/
prediction seals are mandatory. A failed attempt is preserved with its revision;
never overwrite completed scientific records or pick alpha from outer outcomes.
