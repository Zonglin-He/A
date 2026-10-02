# H-lite: event-conditioned spatial reward and geometry

Authorization: user attachments b498dd34 and 7b64d376, 2026-10-02.
Current batch P0 only: A versus H-lite. P1 event-conditioned specialist frames,
P2 critic quality and P3 uncertainty scaling are later conditional research,
not part of this job. No temporal parameter update, new loss, gate, temperature,
learning rate, cap or hard target. Event-support mismatch is a working hypothesis;
prior A/G findings do not uniquely identify it or rule out all scale/critic/state
effects. Negative development configurations are not universal impossibility.

Both original32 exposed development video sources/dataset, one query/source,
two original orders, clean+five5% transient deployment corruption conditions,
25% experts at positions0,4,...,28.384 new arrivals/arm/dataset,1536 total. Same
TA-STVG official same-domain checkpoints, original Paper48 sampling and pixels,
two offsets, cached H and cached five uniform Sa2VA frames/UniversalVTG inputs.
Nine source-relative probes,1792 spatial parameters, no additional candidates.
Vid lr=.033761698432507946,teacherT=.34902548789596055,K1;
HC2 lr=.006097133675874025,teacherT=1,K8;
rho=.05,studentT=1,directions4. No retuning or fresh/full cohort claim.

A keeps original Rank-RKL arithmetic and SGD and must bitwise reproduce all384
sealed preceding A states/predictions/gradients per dataset. H-lite changes only
the support used by the two spatial reductions. Native deduplicated temporal
candidate generator is unchanged (maximum8; denominator actual candidate count).
Observe central decoder layers at the current arm state on every inner step;
do not use UniversalVTG scores, chosen interval, GT or probe temporal intervals.
For original sampled physical frame ids f_tau and half-open candidate intervals
[start,end), define w_tau = mean_k 1[start_k <= f_tau < end_k].
Weights, targets and rewards are detached. No hard crop, duration weighting,
uniform-weight smoothing, entropy multiplier, fallback to full clip or new network.
Equal candidate consensus is not a calibrated posterior or reliable uncertainty
measure; duplicate removal is inherited, and correlation may concentrate support.

Cached valid Sa2VA boxes exist only at their original uniform reference indices.
r_i = sum_valid(w_r * IoU(student_candidate_i,r, expert_r))/sum_valid(w_r).
Invalid/empty masks are excluded exactly as before. If valid mass is zero, reward
is undefined and the arrival performs no spatial update, preserving original
empty-evidence early break; log empty_expert versus zero_weighted_reference_mass.
This is a mathematical empty-support case, no tuned admission threshold. Flat
nonempty rewards keep the original A loss semantics and may have a nonzero gradient.
Do not silently smooth weights or invent reference evidence.

D_i = BBOX_COEF * sum_tau(w_tau L1_tau)/sum_tau(w_tau)
    + GIOU_COEF * sum_tau(w_tau(1-GIoU_tau))/sum_tau(w_tau).
p=softmax(-D/studentT);q=softmax(-average_rank(r)/teacherT);
KL(p||q), ties tolerance1e-12, ordinary FP32 SGD, unchanged lr and K.
Target geometry coefficients inherited exactly, not silently changed to1/1.
Candidate consensus is refreshed in HC K8; expert fetched once/arrival. Current
output is sealed before any update and retains normal temporal Fast rerank;
updates only change spatial state used on future arrivals. Reset each condition/order.
Normal native routing unchanged. No new backbone/expert inference.

Meaningful CPU checks: hand-computed irregular-frame consensus/half-open intervals,
weighted reward and zero mass, weighted native geometry and analytical zero
off-support gradients, target/weight detachment, full-support parity, source-macro
statistics and tamper detection. Fixed first2 scheduled clean inputs/dataset
no-GT smoke: A bitwise, H support/reward/geometry/FP32 SGD/current output/providers,
HC inner-step refresh and full checkpoint restoration. Freeze runtime dependencies
and cache receipts. Finite serial Vid A/H then HC A/H. All four streams must seal
GLOBAL_PREDICTION_BARRIER before any new GT scoring. Original exposed GT then CPU only.

Primary: corrupt future nonexpert source-macro dense delta vIoU vsFrozen and
paired H-A10000 source-bootstrap seed20261001. Secondary whole/clean/expert,
tIoU/sIoU/correct@.3/.5, gross gain/loss, severe negative tails and positive/negative
cases. Preserve all steps, including zero-evidence cases. Log complete weights,
native intervals and frame indices privately; publish anonymous support scalars.
After seal, diagnose reference GT coverage/weight, student support GT coverage,
reward rankings, useful selected target but harmful execution, loss/targetdistance
versus GT, K8 step and net arrival effects. Conditional counts reflect different
on-policy trajectories, not same-state causal replacements. No GT diagnostic
subgroup is used to set online weights, admission or future thresholds.

Development priority H only if strictly larger primary mean; tie<=1e-12 retainsA.
Paired confidence intervals report uncertainty, and a negative mean alone is not
a statistically established harm. H-lite cannot create event reference frames;
its result cannot establish the benefit/failure of H-full. Full specialist
recollection is not automatically launched by this finite P0 controller. Later
resource decisions require explicit documented comparison to P0 evidence.

Log backward/replay/provider counts and formal worker wall including loads/I/O,
not pure GPU kernel time. One GPU serial, no total deadline, minimum8GiB disk.
Preserve all predecessor predictions, failures, paused queues and deployed registry.
Completed implementation, exact protocol, anonymous scalars, negatives, costs and
figures must be published to Zonglin-He/A and individually verified remotely;
archive check/snapshot/check and FINAL_COMPLETION precede monitor pause.
