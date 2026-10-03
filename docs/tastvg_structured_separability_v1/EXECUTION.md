# Finite CPU execution

Run from the project root with `.conda/tubedetr/bin/python -B`:

1. `-m unittest scripts.test_tastvg_structured_separability_v1`
2. `scripts/run_tastvg_structured_separability_v1.py prepare`
3. `scripts/run_tastvg_structured_separability_v1.py extract`
4. `scripts/run_tastvg_structured_separability_v1.py diagnose`
5. `scripts/audit_tastvg_structured_separability_v1.py root`
6. Plot/report, then the same auditor in `public` mode on the exported directory.
7. Archive check/snapshot/check, GitHub publication and per-file remote readback.

Separate immutable stages. Never overwrite a failed attempt without preserving
it and an explicit repair receipt. Extraction opens only original unlabelled
score/decision metadata and SHA-verified locally generated atlas readouts;
diagnosis joins cached labels after FEATURE_SEAL. No model/weight loading or
probe fitting is implemented. Source-deletion analysis uses fixed readouts and
must not be reported as new probe LOSO training or independent confirmation.
