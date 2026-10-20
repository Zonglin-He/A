# Pure DenseLocal supervised temporal OPD, HC128 development

2026-10-20. User explicitly requests an actual complete adaptation experiment,
not more prerequisite diagnostics. Earlier DenseLocal direct/mixed readouts and
the state-guided observation proxy did not establish gains over Strong; none
tested this exact DenseLocal teacher through the existing temporal OPD fitter.
No hard-readout, complementarity or refresh-necessity gate precedes this run.

Only supervision changes relative to the latest Strong-start temporal OPD.
Initial B is the exact saved HiResAuxMix posterior, not an after-results choice
between Native and Strong. Teacher T is the exact PURE DenseLocal final-layer
posterior from completed local-motion128 cache; no mixture, new normalization,
temperature, observation selector or support change. Source Vid checkpoint
5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83.
All65 physical endpoints/2080 legal intervals and existing finalMBR unchanged.

Reuse Policy and instrumented fit from stvg_temporal_conditional_opd128_v1 and
stvg_temporal_strong_action_opd128_v1 without changing arithmetic. Each video/fit
starts with fresh130 temporary start/end biases, p_eta proportional to
B(i,j) exp(u_i+v_j). No neural weights, new head, optimizer state across videos,
spatial state or network training. Zero eta exactly restores B. Forty Adam
steps, lr0.1, betas0.9/0.999, eps1e-8, no weight decay, M32, seeds41/42/43;
original0.5 loss scale and additive-gauge centering preserved.

For sampled student interval(i,j), query the fixed DenseLocal table for the two
full conditional labels T(start|j) and T(end|i), and minimize their mean KL to
the CURRENT student's corresponding conditionals. Teacher and sampling targets
stop gradient. These are derived posterior conditionals, not native conditional
decoder outputs, calibrated event scores or scalar bbox IoU feedback.

Run DirectJoint (original0.5 full joint KL), DirectConditional (teacher-forced
expectation of the same conditional loss), fresh Fixed (32 new samples from B
each step), Live (32 new samples from actual current p_eta each step). Fixed
and Live use paired random streams, same update/candidate budget and current
student likelihood. Their first-step samples/feedback/gradient/state must match.
Later Live samples must follow actual updated q, with all traces and conditional
target digests saved. Teacher RGB does not refresh and need not do so. Do not
call fixed initial samples Live. Two deterministic plus six stochastic fits per
parent =1024 real fits/40960 scheduled updates; no extra sweep or oracle branch.

Static Native, Strong, pure DenseLocal TeacherDirect and DenseLocalMix reuse
saved q/intervals exactly. Primary Live uses time seed paired to the SAME frozen
spatial Live41/42/43 then parent mean. Direct/static outputs average those three
spatial seeds; E and all cross-seed values retained as secondary. All128 unique
RC development parents are repeatedly exposed; no independent confirmation.
No parent, no-change case, zero-time overlap or severe loss excluded.

CPU cache-only actual policy fitting, four threads, finite900s per measurement
and audit/scoring stage, own nonblocking lock, host>=6GiB/disk>=30GiB. No GPU,
model loading/inference, RGB encoding, DINO/ReferDINO calls, weights downloads or
new training. First2 timing is engineering only and remains in the full128.
Existing teacher cache generation GPU cost is not zero deployment cost and will
be linked separately, not reallocated as this CPU adaptation latency.

All128 final posteriors, eta, real sampled intervals, uniforms, teacher digests,
gradients and updates seal before GT. Pre-GT independent NumPy audit replays
ALL actual gradient/Adam/sampling chains, verifies source posterior receipts,
first Fixed/Live exact, zero start, support, normalization and final MBR. Only
then official/DenseTube scoring: full-v, tIoU, boundaryMAE, fixed unobserved
actual-node sIoU, paired10000 parent bootstrap seed2026101909, signed counts,
positive/negative contribution, severe<-3/-5 and worst/LOO. Compare Live−Strong
(primary), Live−Native, Live−TeacherDirect, Live−DirectJoint/DirectConditional/
Fixed and Live−saved DenseLocalMix. Report each seed, no seed/parent selection.

No success is assumed. Report actual magnitudes and uncertainty, distinguish
DenseLocal knowledge transfer from independent Live value. Direct readout
failure does not preempt fitting, loss decrease does not establish task gain,
and this one configuration cannot invalidate all OPD. No hidden2pp gate, no
automatic parameter/teacher/observation search or confirmation. All outcomes
and limitations published by existing Zonglin-He/A allowlist/dual-ref CAS/byte
verification; archive check/snapshot/check. Strong, spatial config, all network
weights and CURRENT/MAIN/PAPER remain frozen. Old autonomous goal stays paused.
