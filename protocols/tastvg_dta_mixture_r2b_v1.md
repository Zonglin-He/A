# R2b: all-proposal equal-weight temporal teacher

Authorization: user's 2026-10-04 R2b request after verified R2 commit
324b3357dbd67b435d1fb507ae516a44ecf53961. Only E-Mix is newly fitted.
E-Deploy, E-Oracle and R1 are immutable reused controls, not rerun.

## Question, fixed conditions and scope

Can preserving the empirical raw UniversalVTG proposal distribution outperform
the confidence-top1 Gaussian through the same temporal parameter channel?
Reuse exactly R2's 288 scheduled cells, 235 input-matched expert caches and
original raw positive-duration proposal list (26--312 per cell). Preserve all
duplicates, order and fractional physical-frame endpoints. No top-K, purification,
confidence weighting, NMS/unifier, third view, new expert or new proposal.

Keep R1/R2's official same-domain EMA checkpoints, frozen final hidden/MLP first
layer, native temp_embed.layers.1 only (514 nominal parameters), K3 SGD,
lr .01 both datasets, beta1 prior KL, sigma one median original offset cell,
per-query reset and final-step readout/discard. Spatial A's 1792D trajectory,
Vid K1 / HC K8, boxes, pixels, sampling, expert schedule and source list are fixed.
No optimizer/subset/temperature/entropy/memory/gate/PoE or source/target tuning.

Each dataset uses original 32 development + 16 confirmation sources, one query
per source, two orders, clean + five 5% transient deployment corruptions. Fit only
288 cached expert arrivals (240 corruption + 48 clean); other 864 arrivals remain
cached A. Total1152 is readout emulation, not a new full stream or persistence.
Confirmation has historical exposure; independent expert sources are Vid16/8
and HC14/7 (development/confirmation), not the nominal full design counts.

## Prelocked joint mixture definition

For each original offset, legal support is L={(i,j):i<j}; a physical interval
is [frame_i, frame_j+1). For each raw proposal m, construct EXACTLY the R1/R2
single-Gaussian teacher on that support:

log q_m(i,j) = -((frame_i-s_m)^2+(frame_j+1-e_m)^2)/(2 sigma^2) - log Z_m.
q_mix(i,j) = (1/M) sum_m q_m(i,j).

Each component is normalized separately on the unchanged legal native support
BEFORE averaging, giving every raw proposal mass1/M. Mixing raw truncated
densities then globally normalizing would implicitly reweight components by
their support mass and is not this equal-mixture experiment. Stable FP64
logsumexp computes the joint mixture; centers/sigma remain unchanged. Preserve
the full joint teacher, NOT a product of independently mixed start/end marginals.
The student's original additive start/end head remains unchanged; projecting
a multimodal joint into that restricted family is an interpretation boundary.

Optimize mean across two offsets of KL(q_mix||p_theta)+KL(p0||p_theta).
Use original FP32 student MAP and two-offset physical envelope. Mixture-MAP
diagnostic instead selects first maximum of the full FP64 joint teacher per
offset then the same envelope; it is not a student performance or a new method.
No synthetic mean-center teacher is introduced. Single-proposal fit must match
R1/R2 bitwise; duplicate multiplicity must affect its empirical mixture weight.

## Execution and evaluation barriers

Separate preparation and E-Mix fitting processes do not read target labels or
R2 scored/oracle results. Pin R2's hashes and GT-label expected hashes as metadata.
E-Mix worker installs an additional read guard blocking labels, oracle runs,
old scored controls and public result files. After all 288 predictions seal,
another CPU process opens GT for scoring and immutable R2 controls for comparison.
No GT selection of proposal, arm, hyperparameter, step, source or threshold.

Primary: E-Mix minus Native, expert-corrupt tIoU and vIoU; E-Mix minus E-Deploy
is the matched aggregation contrast. Secondary: versus A8/R1/E-Oracle,
full-flow source means, clean, each order, unchanged nonexpert, gross gains/losses,
>5pp harms, .3 correct-to-wrong/rescue, loss-down/task-down, concentration and
leave-one-source-out ranges. Use official dense scorer, equal-source means and
10000 paired source-bootstrap draws, seed20261004. Preserve good and bad cases.
Post-seal teacher diagnostics may report unweighted raw proposal tIoU and
mixture entropy/expected offset-joint tIoU; none is used for fitting or selection.

## Interpretations fixed before results

- Mixture improves over Deploy and Native with credible paired evidence: useful
  all-proposal supervisory distribution in this scope; no automatic persistence.
- Improves over Deploy but not Native: aggregation mitigates this teacher's
  failure without establishing beneficial adaptation.
- Fails to improve: equal raw mass is not sufficient here. Support contamination,
  diffusion and the factorized student projection remain competing explanations;
  do not uniquely diagnose irrelevant proposal mass from a negative result.

Mixing also changes concentration/gradient strength. Improvement alone does not
prove uncertainty preservation is the unique cause. No follow-up weighting,
purification, PoE, new expert, R3, or paused queues starts automatically.

## Independent validation and required publication

Test joint mixture versus independent arithmetic, one-component fit parity,
duplicate weights, proposal-order invariance, multimodal joint preservation,
reset and GT guard. Independently recompute all 864 joint-marginal gradients,
SGD states/logits/MAP, Gaussian component normalization and official dense metrics;
check reused baselines, spatial A and input hashes. Public scalar/CI audit,
cases, concentration and PNG/PDF figures are required. Publish code/protocol/all
anonymous results and negative findings to Zonglin-He/A, verify remote bytes,
record completion in RESEARCH_HISTORY and run check/snapshot/check.
Exclude private GT/raw proposals, latents, head weights, media and credentials.
