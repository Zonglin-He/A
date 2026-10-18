# Frozen-decoder query-residual E feedback64

Human-authorized current-video adaptation, not external20 transfer. Exact prior
HC32/Vid32 cohort; all queries use existing four-frame DINO-derived fixed E.
Every arm/query starts at frozen Source with QR256=0 and fresh Adam. Only external
spatial.query_residual is differentiated/updated through the frozen six-block
spatial decoder. Every pretrained parameter including LN is frozen and unchanged.
No cross-query inheritance; no temporal parameter or LN update. Historical paused
LN fit outputs are not inputs to this run. Source/E outputs reuse sealed256.

Six outputs: Source_MBR, E_MBR, Sparse_OPD, Dense_OPD, Fixed_rollout, Dense_Direct.
Sparse uses fixed E at all four original planned positions; dense covers every
actual decoder frame without GT visibility selection and without extra sparse
loss. Fixed caches first actions/rewards/weights/rollout mean; it retains original
likelihood/control-variate formula at each updated Student. C/D sample current
Student every step. Common full-frame antithetic noise then restrict Sparse loss
provides matched initial samples per physical frame. Original chart, endpoint
finite native-logit extension, detached actions/weights and original per-frame
Gaussian objective retained. Not a learned uncertainty model.

Parameters are inherited unchanged: HC LR.01/sigma.025/tau.05/40 steps; Vid
LR.03/sigma.1/tau.25/10 steps; 32 candidates per frame, Adam(.9,.999),eps1e-8.
Direct mean(5 L1+2 GIoU). Same recipe across four arms per domain, no search,
acceptance/rollback/new trust gate, GT checkpoint choice or E refresh. Final
preplanned step; unsupported E remains no-op with full denominator. Native
physical-time final-distribution MBR remains identical for all outputs; final
spatial output is Student itself, no E overlay. This is spatial adaptation only.

Use frozen-prefix cache inputs where available; otherwise reproduce Source prefix
on selected64 only and check exact boxes/logits. No historical Before replay,
new DINO/Teacher/source training/second backbone. First two selected cases count
inside64 and record synchronized timings for bounded extrapolation. Up to256 fit
slots/6400 update slots, 60min GPU lease hard cap and 3h overall deadline, shared
GPU lease, memory16/20GiB swap1GiB, host6GiB/GPU4GiB/disk30GiB floors. Failure
preserves partials, never replaces a parent or resets deadline automatically.

Seal all64 six-output predictions before GT. Report vIoU, tIoU invariance,
unobserved actual-decoder sIoU excluding four planned positions, paired parent
CIs, all positive/negative/neutral/unknown and severe tails, parameter/output
movement and actual cached vs first-capture costs. Dense comparisons vs Sparse,
Fixed, Direct, E and Source; preserve known development exposure. Independent
formula/state/metric audit, fixed actual-pixel review, public privacy export,
actual remote bytes/SHA/blob/mode/tree/two-ref verification, archive closure.
No automatic expansion or production registry promotion based on this panel.
