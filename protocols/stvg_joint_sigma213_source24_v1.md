# Fixed sigma .213 source24 comparison

Explicit user authorization: no new model training. Change only initial raw
Gaussian sigma AND starting value of the logarithmic16-level sampling schedule
from2 to exactly .213; end .03, Euler update, coordinate units, trained EDM
preconditioning, source statistics, seeds and cached conditions are fixed.

Freeze the repaired HC epoch30/Vid epoch20 checkpoints (failed previous
qualification, no reselection), original source-val12 each. Same four CUDA
epsilon paths per query; sample0 primary. Four path scores averaged only as
diagnostic, never latent coordinate averaging or GT path selection. Per-query
optimizer/Student/Teacher/association updates: zero. Observation guidance remains
OFF. Native inputs and old sigma2 predictions are exact saved references.

Compare raw Native Source, Native encoded/decoded32-knot control, old sigma2
terminal states, and new sigma.213 terminal states. Report v/t/s/R03/R05,
per-source paired deltas/CI95/positive-negative-neutral and all four samples.
Primary contrast new sigma.213 minus raw Native; secondary versus sigma2;
codec control isolates the small representation readout change. Twelve queries
per domain, not48 independent samples. Official source train origins are held
out only for auxiliary prior training, not established unseen to pretrained
Student; this remains exposed source development, not target confirmation.

No source GT enters the new model paths. Seal ALL24 new outputs before source
GT scoring. Source supervision is reused for evaluation only. Existing source
statistics/checkpoints are immutable. No target GT/inference/media, no new
Source capture, DINO/Qwen/SAM2, no temporal or spatial Student fit.

Before full run: CPU schedule/Euler/identity contracts. GPU interface check at
sigma2 on prelocked first source case in each domain must reproduce all four
old terminal states and epsilon; only2 diagnostic replays, no full old rerun.
Then run exactly24 new paths x4samples x16steps. No sigma scan or later method
promotion. No automated observation isolation or target follow-up after scores.

Use shared serial GPU lease, memory.high16GiB/max20GiB/swap1GiB and host6GiB/
disk30GiB floors. New GPU reservation <=300s; cumulative with previous repair
and path diagnosis starts36.608554596081376s, must remain within14400s. CPU
audit/publication measured separately; cached component cost is not first-use
online end-to-end latency. Any failure preserves partials; no retry science.

Independent official/DenseTube/source-binding/seal/epsilon/mean/paired-CI audit,
private actual first-source pixels per domain (Native,old,new) then curated
code/protocol/scalars including negatives published to Zonglin-He/A. No raw
coordinate arrays/Query/GT/media/weights/tensors/credentials in public export.
FreshCAS both main/research refs and actual remote bytes/SHA/blob/mode/tree.
CURRENT/MAIN/PAPER and all old negative predictions/receipts unchanged.
