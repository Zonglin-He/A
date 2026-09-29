# TA-STVG Round2: Causal Evidence and Correctability Audit

Authorized by the 2026-09-29 user proposal (archived locally) and explicit execution request.
Seven primary arms plus PartA, original64 historical exposed target-test parents,
32 Vid checkpoint -> HC1 and32 HC2 checkpoint -> Vid. No new cohort, experts,
PTD resumption, model training, production promotion or Round3. Follow completed
Round1 sealed caches and code; do not modify them.

## Pre-GT hypotheses and locked endpoints

Primary VA: for each Round1 budget, average ASA1-app cosine distance over the two
offsets at its already sealed strictly preserving selected state (self-vIoU>.95),
then max over rho .005/.01/.02; absent feasible state ->0. Mean over the three
budgets is secondary. Argmax uses ascending-budget earliest tie. dV is the saved
selected delta for that budget, appearance block, not an oracle-selected step.
Lock all64 scalar V and budget choices BEFORE reading this round's GT.

Primary associations: Spearman(VA,1-B0_sIoU), Spearman(VA,OS_sIoU-B0_sIoU),
Spearman(VA,OST_vIoU-B0_vIoU). Report each direction FIRST, pooled64 secondary;
bootstrap paired parents10000 seed20260929, descriptive95% CIs, ties retained.
Report both cos(dV,raw spatial loss gradient) and cos(dV,negative gradient);
positive raw-gradient alignment is NOT a corrective direction. Mean-V secondary.
Correlation>=.3 with positive lower CI in BOTH directions would support a common
WHEN hypothesis; inconsistent/wide CI is inconclusive or domain-specific, not
proof of no relation. Direction support requires positive descent-alignment CI;
neither rule licenses a final method. Primary outcomes/tails remain continuous.

## PartA: cached causal interventions, no GT

Use Round1 rho.02 selected state, all64, both stages and appearance/motion.
A stage is eligible only if its exact selected-frame list is identical in BOTH
offsets (not merely Jaccard>.5). Report coverage and changed-route cases separately.
Construct Q00=mean(A H), Q10=mean(A' H), Q01=mean(A H'), Q11=mean(A' H').
Preserve the actual official mean over frames AND patches, not a sum approximation.
Save all four vectors and deltaA,deltaH,deltaAH, bilinear interaction. Report
relative norms, cosine and cancellation ratio ||deltaAH||/(||deltaA||+||deltaH||).
Unchanged A/H replays must exactly match captured native Q values.

Frozen-context decoder: hold baseline H, positions, counterpart query and all other
inputs fixed; insert each changed Q into its matching frozen decoder stage.
Stage1 head output is an INTERMEDIATE diagnostic, not the final two-stage tube.
Stage2 is final output. Additionally propagate the stage1-only Q intervention
through actual actionness/second selection/ASA2/decoder2 using baseline H; this
isolates the route-mediated final effect. Stage1 with both later query and routing
artificially fixed would trivially have no final effect, so do not present that
as discovered redundancy. Compare boxes/interval/self-vIoU to the appropriate
baseline (stage1 intermediate or native final). No semantic wrong-object claims.

## PartB: native-Jacobian oracle and seven arms

B0, OS, OS_PT, OT, OT_PS, OST, Oselective. H=[Ha;Htext;Hm]; text delta zero.
All model weights frozen. TTS/ASA input detach and decoder reference detach remain
as official. Candidate gradients follow native differentiable paths, then every
finite candidate recomputes all TTS, hard selections, ASA/query and both decoders.
First query of each direction: compare cached forward AND raw H gradients with
full original-pipeline reinsertion under the historical FP16-prefix/FP32-suffix
contract. Hook-audit TTS/ASA input tensors are detached; direct H-weighted pooling
and decoder memory remain differentiable. No expanded Round1 evidence gradients.

Spatial final-head objective: average L1 sum-coordinates and GIoU loss over ALL
GT-valid sampled frames, regardless of current predicted interval. Official source
weights: Vid checkpoint L1=5/GIoU=3; HC2 checkpoint L1=5/GIoU=4.
Temporal: exact official loss_sted formula: Gaussian targets sigma2, eps1e-6,
KL(pred||Gaussian) implemented with official eps, mean across time and sum start/end,
then equal-offset average; source TEMP weights2(Vid)/10(HC2). GT start/end targets
are first/last GT event frame in each offset. If no event frame occurs, use nearest
physical start/end representable indices and disclose that exception, never silently
skip a query. Spatial no-valid support: zero spatial loss/gradient, keep and disclose.
Only these final output tasks are optimized (no auxiliary/TTS/ASA classification loss).

rho=.02 times joint baseline visual-H Frobenius norm across both offsets, SAME absolute
cap for every arm; K10 normalized gradient steps; nominal step2*cap/K. Zero start.
OS updates Ha, OT Hm, OST both with official weighted spatial+temporal loss gradient.
Project cumulative delta into allowed blocks and radius. Every arm uses the SAME
loss-descent backtracking alpha={1,.5,.25,.125}, largest feasible step; if none, no-op.
Descent tolerance1e-8*max(1,abs(current loss)), no selection by evaluated GT metrics.
OS_PT adds exact decoded physical interval=B0 interval. OT_PS adds box IoU>=.95
on B0's original interval compared with B0 boxes. Endpoint KL, all-frame box drift,
GT-supported frame loss and final s/t/v scores are reported separately.

Oselective uses5 spatial steps preserving the B0 interval, then5 temporal steps
preserving the spatial-phase terminal boxes on that phase's decoded interval.
Each step decreases its task loss AND the official weighted joint loss; preservation
reference is fixed within each phase, preventing cumulative within-phase drift.
Joint cap spans both phases. Total backwards10 matches OST, but each branch gets5;
state this budget difference rather than claim matched per-branch optimization.
Save terminal states, trials/rejections/no-ops; no best-GT-state selection.

To answer architecture alignment rather than assume it: save baseline full visual
spatial and temporal gradients (including off-branch components), per-block gradient
energy shares, and two secondary ONE-STEP swapped-branch probes (spatial objective in
Hm; temporal in Ha), with same nominal alpha/backtracking as the aligned first step.
These are initial local/one-step diagnostics, not additional tuned full arms.

## Evaluation, analysis and budget

Only after PartA and VA lock are sealed, stream the immutable existing label container
and retain only the64 authorized keys. No constructor annos access; model loader gets
empty dictionaries as in strict Round1. Round2 explicitly uses GT for oracle updates,
not an unlabeled TTA evaluation. Save GT subset/label-container hash and exposure record.
After all primary outputs seal, evaluate existing corrected sparse-frame s/t/v scores
with two independent implementations. Equal-parent means (one query per parent), paired
bootstrap CIs, negative tails >5pp, baseline-good retention at .5 (descriptive), interval
and spatial collateral drift, accepted fraction/no-op, and useful-gain retention for
OS_PT vs OS and OT_PS vs OT. Retention denominators include per-case positive oracle
opportunity; keep harms and zero denominators explicit. Native spatial sIoU and a
final-tube-supported spatial score are separate so temporal support loss is visible.

SVD is secondary: per-query gradient matrix tokens*time by256 channels, appearance
spatial / motion temporal / concatenated native joint gradient; retained energy at
r8/16/32/64. This describes channel compressibility, not a learned transferable basis
or low-rank native efficacy. No rank projection GPU sweep in this round.

Serial RTX5090; cumulative Round2 GPU-process cap7200s, imports/load/failures included.
Disk floor8GiB, new artifacts cap30GiB. Existing caches reused; selected terminal deltas,
initial full gradients and compact trial predictions retained, no giant all-step
activation archive. CPU analysis separately timed. First-query engineering audit is
part of the same fixed64 run, not a science-driven tuning pilot. Preserve any failure
and record repairs without changing the locked scientific quantities after GT access.
Finish with local archive check/snapshot/check, publish code/protocol/anonymous numeric
results to Zonglin-He/A, then fetch and verify the remote files. No new scheduler.
