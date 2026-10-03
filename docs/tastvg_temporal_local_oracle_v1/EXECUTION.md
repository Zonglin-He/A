# CPU local-oracle audit execution

Only the sealed anonymous public candidate metrics from 4ecf366 are inputs.
No controller, background queue, GPU work or new labels are required.

Run with the project's existing Python:

1. `scripts/test_tastvg_local_oracle_v1.py`
2. `scripts/run_tastvg_local_oracle_v1.py prepare`
3. `scripts/run_tastvg_local_oracle_v1.py compute`
4. `scripts/audit_tastvg_local_oracle_public_v1.py results/tastvg_temporal_local_oracle/2026-10-03`
5. `scripts/report_tastvg_local_oracle_v1.py`

The root verifies independent arithmetic/bootstrap, reviews the actual charts,
records decisions and limitations, then publishes the explicit allowlist to
Zonglin-He/A and verifies every remote file byte plus the standalone public audit.
Update RESEARCH_HISTORY and run research_archive.py check/snapshot/check before
completion. Keep all predecessor files and the production method unchanged.
