# Execution

Entry: `.conda/tubedetr/bin/python -B scripts/run_tastvg_query_swap_v1.py`.
Finite stages: `prepare`, `smoke vidstg`, `smoke hc2`; root checks true-query
bitwise parity and writes SMOKE_ROOT_ACCEPTANCE. Then `capture vidstg`,
`capture hc2`, `seal`, `diagnose`. Every GPU worker uses the established GPU
lease, clean constructor loader and annotation-open guard. Serial GPU work;
no monitor or historical queue is started. No fixed total deadline.

Worker completion requires independent root/public audits, report and plots,
research archive check/snapshot/check, publication to Zonglin-He/A and bytewise
remote verification. Read the runtime lock and dynamic STATUS for actual
progress, not this plan. Preserve errors in `failures`; engineering-only
changes require an explicit immutable revision pin. Scientific changes are
not silently substituted.
