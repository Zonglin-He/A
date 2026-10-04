# R2b finite CPU execution

Entry: scripts/run_tastvg_dta_mixture_r2b_v1.py. Separate processes prepare →
GT-guarded mix → score. Exactly 288 new query fits/864 backwards, no GPU.
Private artifacts/tastvg_dta_mixture_r2b_v1 and anonymous
results/tastvg_dta_mixture_r2b/2026-10-04 are isolated from immutable R1/R2.
Never restart completed/prepared phases or overwrite failed scientific records.

The only changed variable is a per-offset equal average of all individually
normalized R2 raw-proposal Gaussian joint distributions. Loop/optimizer/decoder
are matched to R1; single-center parity tests precede target fitting.
Empirical duplicates retain their weight; no confidence-based subset/weighting.

After MIX_PREDICTION_BARRIER, score against official dense metrics and compare
immutable E-Deploy/E-Oracle/R1 controls. The worker's completed_pending_root_audit
status is not completion. Root must perform independent mixture-gradient/state/
dense and public scalar audits, report good/bad cases and concentration, draw and
inspect figures, publish code/results and verify GitHub, update research archive
with check/snapshot/check, then write FINAL_COMPLETION. No R3 or purification job.
