# R1 execution

Finite CPU-only entry point: `.conda/tubedetr/bin/python -B
scripts/run_tastvg_dta_oracle_r1_v1.py`.
Private artifacts: `artifacts/tastvg_dta_oracle_r1_v1`.
Anonymous public result directory: `results/tastvg_dta_oracle_r1/2026-10-04`.

Order: input/code lock and CPU parity → source-validation five-lr paths → seal
both chosen learning rates → 288 episodic GT-supervised target adaptations →
prediction barrier → dense scoring, full-flow cached emulation → independent
analytic-gradient/root and public scalar audits → plots/report → archive
check/snapshot/check → public GitHub export and exact remote verification.

Every zero-update native interval must equal the cached native interval. Original
CUDA vs new CPU logits are compared with a declared 1e-4 absolute tolerance,
without overriding their values or bypassing interval parity. The original
MLP eval-mode dropout is inactive. Do not use the frozen scalar probes.
Final completion requires actual coverage, audits, remote receipt and archive.
No R2/R3, decoder/LN update or old queue is launched automatically.
