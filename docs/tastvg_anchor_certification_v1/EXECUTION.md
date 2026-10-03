# CPU execution

Protocol: `protocols/tastvg_anchor_certification_v1.md`. Private `artifacts/tastvg_anchor_certification_v1`; anonymous results `results/tastvg_anchor_certification/2026-10-03`.

1. Run CPU mathematical tests before runtime pinning.
2. `scripts/run_tastvg_anchor_certification_v1.py prepare`: hashes only, immutable configuration/runtime. No target labelled rows.
3. `... calibrate`:31/16 source-validation native-anchor winners, fixed isotonic/bootstrap models; no target GT.
4. `... seal`: both datasets'288 cached score decisions, including every predefined diagnostic operating point; no target GT.
5. `... diagnose`: join1152 existing anonymous GT-derived rows only after global seal, full and expert curves and source paired intervals.
6. Independent root/public audit, report and PNG/PDF rendering/readback, archive check/snapshot/check, publish narrow new files and verify remote contents. Final receipt only after these actual checks.

No GPU controller/background monitor is needed. Source native is intentionally distinguished from target A8 under the user's explicit response. Do not fit a target threshold or reopen query-swap/MLP/old queues. Preserve failed attempts and pin engineering-only revisions if required.
