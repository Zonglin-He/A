# Design completion: three fixed component interventions

The human request on 2026-10-11 authorizes the three experiments in the attached
recommendation and uses the attached manuscript to define the interventions.
The original P1–P6 suite remains completed and immutable. This is a separate
experiment, not a reopening of tuning, EATA, or any historical paused queue.

## Fixed population and comparisons

Copy the exact P3 stage definitions: 128 historical parent sources per target,
one fixed query per parent, cross-domain clean with both original orders, and
same-domain five physical burst families at 5% with the original order. Start
every arm/condition/order independently from its original source checkpoint.
Three new arms each contain 1,792 arrivals; total 5,376 new formal fits.
Reuse only the entire matched, sealed Full streams and anonymous P2/P3 rows.
Never insert qualification predictions into a formal stream.

1. `whole_video`: replace the observation extent by `[0, len(frame_ids)-1]`.
   Use the same physical-bin Uniform4 function, observed RGB frame grid, frozen
   Grounding DINO, query parses, admission/Top1, OPD, optimizer and state rules.
   Do not decode extra frames. Final temporal output remains the original native
   interval. Whole-video references have their own hash-bound input files.
2. `carry_query`: carry the final 256-coordinate query residual to the next
   arrival. Keep the original attenuated LN writeback. Adam is fresh for every
   query. This changes only the residual lifetime.
3. `full_LN_carry`: reset the query residual and Adam, but copy the final LN
   values exactly to the next arrival (`alpha=1`). Do not choose a new alpha
   from results. The original `alpha=0` control stays a reused P3 result.

All other main settings are fixed: HC target lr .01/sigma .025/tau .05/40 rounds/
LN writeback 1/16; Vid target .03/.1/.25/10 rounds/1/8; 32 antithetic Gaussian
actions, joint 1,792 parameters, final round, frozen native WHEN. The already
qualified P4 numerical recorder is reused without changing its science code,
thresholds, action/reward/loss/gradient/Adam computations. Numerical qualification
is distinct from support for the new scientific interventions.

## Qualification and barriers

CPU contracts check full-video positions, residual lifetime, exact alpha=1 LN
carry, invalid scopes/alphas/state schemas and the fixed population. Every
stage/condition/arm uses the first two order1 arrivals for real qualification:
72 arrivals, 144 complete fits (two repeats per arrival). Repeats must be
bitwise equal, with complete independent saved-math/Adam/state checks and actual
source/expert process hashes unchanged. The first two formal fits must match
the qualified fits before acceptance. Qualification GT reads and formal
prediction acceptance are zero. Report actual new DINO calls.

All three arms, conditions, orders and both targets must seal before new GT
access. Scores use the original official scorer and an independent dense
readback; annotations never affect admission, state, stopping, selection or
parameters. Save and explain errors; never skip queries or silently clamp boxes.

## Prespecified table and Figure 3

Table 3 combines all original P2/P3 component rows and these three new rows.
Report absolute parent-macro m_vIoU (%), query-macro, paired Full-minus-control
differences, 10,000 paired parent bootstrap CIs, both orders, all negative tails,
observed/unobserved changes, admitted expert quality and measured costs. The
same-domain main table averages all five families before parent aggregation.

Figure 3 uses only the two cross-domain clean streams. Native event coverage is
`intersection(native_half_open, GT_annotation_half_open) / GT_duration`, with the
GT annotation support defined as `[min(annotated_frame), max(annotated_frame)+1)`.
This is not tIoU; original task scores retain their original scorer conventions.
Fixed bins are `[0,.25)`, `[.25,.5)`, `[.5,.75)`, `[.75,1]`, including zero and one.
Native WHEN is identical between paired arms and orders. Average the two orders
within each parent, then resample parents 10,000 times with seed 20261011. Plot
Event-scoped Full minus Whole-video m_vIoU (pp), a dashed orange zero line, each
bin's parent count and 95% CI. Empty bins are missing, not zero. No bin, case,
threshold or configuration is selected after looking at new scores.

Coverage is an offline conditional association. A coverage-dependent difference
does not independently identify temporal coverage as the cause of the effect.
Attempted observations, actual DINO calls, admitted references and expert-GT IoU
on valid annotated observed frames are separate denominators. Frames without
annotations are unmeasured, never treated as wrong boxes or invented GT.

## Closing obligations

Complete saved input/math/state-chain/1792-coordinate/dense readbacks, matched
statistics and all negative findings; inspect actual plots and fixed positive/
negative cases; publish only anonymous code/configuration/results to Zonglin-He/A
and verify remote bytes/SHA/blob/tree. Preserve private RGB, query texts,
annotations, expert/fit payloads, weights and credentials locally. Update the
research archive with check/snapshot/check. Preparation, qualification, CPU
handoff and process exit do not constitute completion. TubeDETR adaptation is
an optional separate scope pending the human preference; no production promotion.
