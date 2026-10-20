# Strong-relative DenseLocal preference OPD

2026-10-20. USER-REVISED EXECUTION PROTOCOL: implementation and locking in progress; no task result yet. User asks
for one directly executable full-method experiment, not another prerequisite
diagnosis. No new expert/checkpoint/neural training, no spatial change, no
hyperparameter menu. Target: actual final full-v +2 to3pp or more versus saved
Strong HiResAuxMix, not versus old Native, and distinguish Live value from
better feedback/state representation. No gain promised.

## Why this is different

Previous full128 Strong-start conditional OPD imitated the entire pure DenseLocal
posterior. It fitted that teacher well (joint KL .879460 to about .03147) but
Live−Strong full-v was −.772521pp, CI crossing0, and Live−Fixed −.024952pp.
Dense candidate expectation−Strong +2.075542pp CI positive is real offline
evidence but not an obtainable point-decision gain. New hypothesis: use only
cross-offset consistent incremental interval preferences; anchor the entire
current policy to Strong continuously instead of reproducing a complete teacher.
Old semantic/query/native-grid trust experiments also existed; merely adding
KL is not claimed new. This proposal jointly changes the supervised object
(relative preferences) and perquery policy state (joint interval residuals).

## Frozen inputs and exact correspondence

All128 RC parents, one query per parent, repeated exposed HC2 development,
same frozen spatial E/Live41/42/43 arrays and fixed Source Vid checkpoint.
Keep original65 physical endpoints, all2080 legal MBR decisions, old positive
policy support, physical end=fid+1, zero-probability handling, and finalMBR.
No added epsilon support. No temporal crop, extra frames, GPU, Source forward,
DINO/ReferDINO, or feature re-encoding. Read the existing raw endpoint logits:

- Dense: artifacts/stvg_temporal_local_motion_completion128_v1/predictions/RCxxx.pt,
  logits.DenseLocal, the original two native offset grids.
- Native224 six second-pass layers:
  artifacts/stvg_temporal_native_stage128_v1/predictions/RCxxx.pt,
  pass_logits[1][offset][layer].
- Native448:
  artifacts/stvg_temporal_hires128_v1/predictions/RCxxx.pt,
  observation.logits plus observation.records.
- Global Strong B: exact saved outputs.HiResAuxMix.q; never reconstructed from
  a newly rearranged ensemble or from the per-offset reference defined below.

Apply the original float64 native endpoint-joint softmax separately on each
offset grid G_o. D_o is Dense's native interval distribution. The reference
B_o = .5 * P448_o + .5 * mean_{six layers} P224_o,layer.
B_o is a local reference with the same ensemble recipe BEFORE the old envelope
mapping. It is NOT claimed to be a marginal or exact factorization of global B:
mixture and two-offset min-start/max-end mapping do not commute. B itself stays
byte exact. Both offsets share a network/context and are correlated, not two
independent experts or a statistical reliability certificate.

For a physical candidate I, M_o(I) is the native legal interval in G_o with
maximum physical tIoU to I; exact ties take smallest physical start, then end.
This deterministic mapping uses no labels. All comparisons occur on the same
G_o. Use log-space mixture/normalization, no GT calibration or prompt change.

## Actual student and on-policy pairs

Temporary joint residual r(I) per B-positive interval, at most2080 slots;
pi_r(I)=B(I)*exp(r(I))/Z, r=0 at each video/seed/fit.
This is a transient probability table, not a new model/head, and no state is
shared across queries. It replaces the old u_start+v_end-only parameterization
so a preferred complete interval can be adjusted without obligatorily changing
every other interval that shares one boundary. Preserve the mean-zero additive
gauge after each step. Sampling support stays B>0; all2080 final MBR decisions
still allowed exactly as before. Zero r recovers Strong and its readout exactly.

Each Live update samples32 independent pairs:
I_m~stopgrad(pi_current), J_m~B. This is64 discrete interval draws, not the
old32 single-interval draws and not64 expert forwards. J is sampled on B's
positive support; do NOT use the Strong MBR action as a log-probability anchor,
because an MBR decision may itself have zero hypothesis mass. Same pairs may
repeat, I=J may occur, and no pair is resampled to fill an accepted-label quota.
Fixed samples both I and J freshly from B at each update; it fits the current
policy, not a frozen likelihood. Pair random streams match Fixed/Live seeds,
so first-step samples/labels/gradients/updates must be exact.

## Teacher preference: partial, relative and abstaining

For each pair (I,J), compute in each offset:
h_o=log D_o(M_o(I))−log D_o(M_o(J));
d_o=h_o−[log B_o(M_o(I))−log B_o(M_o(J))].

Only if all four numbers h_0,h_1,d_0,d_1 are positive, set a=min of the four.
Only if all four are negative, set a=max of the four (closest to0).
Otherwise abstain: w=0, target t=.5. For an admitted pair set t=sigmoid(a),
w=1; abstaining pairs have w=0. Natural log unit/sigmoid scale1, no extra temperature, learned
weights or threshold sweep. Reversal (J,I) must give 1−t and the same w.

Thus Dense must actually prefer the same interval on both offsets, and that
preference must also be an increment relative to the local Strong references.
Weak agreement has t near .5 and hence a small initial gradient; its admitted weight remains exactly1. Disagreement does not force .5 supervision. No additional strength-dependent weighting is applied.
This is a heuristic consistency criterion, NOT proof of a correct event or
calibrated preference probability. Wrong agreement can still damage Strong.
If D_o=B_o on both offsets, all d_o=0 and there is exactly no Dense update.

## Loss and complete update

s_r(I,J)=sigmoid(log(pi_r(I)/pi_r(J))−log(B(I)/B(J)))
          =sigmoid(r(I)−r(J)).

L = mean_{all32 sampled pairs} w * KL(Bernoulli(t)||Bernoulli(s_r))
    + KL(pi_r||B).

Stop gradients through sampling, M_o, D_o, B_o, teacher t/w and global B.
Only r receives gradients. Do not divide by the number/weight of admitted
pairs: it would amplify weak/rare feedback. Global KL uses all positive
support, not only sampled actions; coefficient1 predeclared, no search. The
first term teaches a change in preference RELATIVE to Strong; it does not
match Dense's complete probabilities. The second is an explicit continuing
penalty, not just initialization. Neither is a GT no-regression guarantee.
This objective is not ordinary DPO or its original RLHF guarantee: it is a
soft-preference distillation design using a reference-policy log-ratio form.

Fixed configuration:40 Adam updates, lr.1, betas(.9,.999), eps1e-8, no weight
decay/clipping/schedule, seeds41/42/43, same finalMBR. Constants w rule,
sigmoid scale1 and KL coefficient1 are one unvalidated design configuration,
not empirically optimal or selected on GT. No parameter/threshold/policy-budget
menu follows the result. Zero available feedback for an entire video must
recover Strong exactly. No hard-readout or admission-rate scientific gate.

## One full128 method comparison

Reuse Strong and old whole-Dense Live/TeacherDirect saved results exactly.
New fits: Preference-Direct, Preference-Fixed, Preference-Live, each3seeds.
This is1152 fits/46080 nominal optimizer steps, plus reused static outputs.

For each Direct/Fixed seed precompute the40x32 pair stream sampled from BxB.
Fixed consumes32 pairs per update from this fresh-initial-state stream.
Direct optimizes the SAME preference/reference loss over that entire1280-pair
bank on each of its40 updates, a stronger fixed-feedback control. Live uses
the same uniforms, replacing only I's sampling law by actual current pi.
Direct has the same maximum1280 distinct pair evaluations but more pair-loss
arithmetic per update; count this CPU difference, do not claim equal FLOPs.
Fixed and Live have identical pair/optimizer budgets. Direct's loss banks and
all teacher labels are fixed without GT, not selected according to the result.

Save r/pi before-after, actual I/J/uniforms, mapped native intervals, h/d/t/w,
admission mask, preference/global-KL losses, gradients and Adam states, final
q/interval and exact static/spatial hashes. All128 predictions seal before GT.
Required engineering: zero-state and no-feedback exact, offset physical mapping,
pair reversal, same evidence no-op, AD versus independent manual/finite
difference gradients, current-policy versus fixed sampling, final MBR and
all frozen spatial arrays. These are implementation checks, not task gates.

Evaluate final full-v against CURRENT Strong; tIoU, boundary errors, fixed
unobserved actual spatial sIoU, all signed/zero results, severe tails and parent
paired10000 bootstrap. Match temporal/spatial seeds then average within parent;
never treat384 video-seed pairs as independent. Report all3 seeds, gate statistics
as process metadata only, and measured complete CPU fitting/label/audit costs.
Finite900s per CPU stage, original host6GiB/disk30GiB guards and own lock; no
background successor or expanded study. No model/expert inference or training.

## Decision and boundaries

Target full-v31.5349%–32.5349% or above, namely+2–3pp versus currentStrong29.5349%,
not versus Native27.9285%. This is a goal, not a forecast. Mean gain, paired CI,
parent concentration and severe negatives determine support. If the same new
feedback works for Direct/Fixed equally well, attribute gain to feedback/state
design, not to on-policy sampling. Independent OPD contribution requires Live
to outperform these same-feedback controls with credible paired evidence.
If none reaches the intended gain, keep Strong, retain every failure and close
this exact configuration; do not sweep agreement/KL/lr/bank sizes. A positive
development result is frozen before any separately authorized confirmation;
RC128 itself is not confirmation. No CURRENT/MAIN/PAPER/spatial promotion.

The unproven assumption is cross-offset consistent relative preference having
higher usable task reliability than the whole Dense posterior. It cannot be
settled by KL decline or label counts; this full method run tests it directly.
Even positive candidate expectation gives no2–3pp lower bound. MBR is retained
to make any actual improvement arise from the adapted distribution, but that
also means the prior expectation-to-decision gap is NOT guaranteed solved.

Publish executed code/config/all signed results and limitations through the
existing Zonglin-He/A allowlist with dual-ref CAS and remote byte receipts;
archive check/snapshot/check. This document authorizes no execution by itself.

Mechanism references, not empirical guarantees for this project:
- DPO reference-policy odds: https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf
- GKD student-generated states and detached sampling: https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf
