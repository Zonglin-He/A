# Frozen method paper experiments v1

## Scope and method freeze

The user explicitly authorized attachment 8edaf940 on 2026-10-05. It replaces
the previous four-direction all-query corruption queue. That partial queue is
paused and preserved, with no partial scoring or automatic resumption.
The existing selected method is frozen: Native WHEN, Uniform4, one frozen
Grounding DINO, existing admitted Frame-Top1 IoU energy, 256 query-residual plus
1536 spatial-LN parameters, Adam .03, 10 steps, earliest minimum own label-free
loss in steps 0..10, selected LN delta/16 consolidation. Query residual/Adam
reset per query; LN carries within each condition/order/checkpoint stream.
No target-result retuning, temporal module addition, extra specialist or method
promotion. CURRENT_METHOD remains a separate production registration.

## Dataset unit correction

VidSTG official test: 10303 queries, 732 videos, 732 source clusters.
HC-STVG-v2 official validation: 3482 queries, 3482 clips, 237 parent-source
clusters. Every HC2 clip already has one query. One query per official video
therefore means 732 Vid queries and all 3482 HC2 queries, not 237 HC2 clips.
Same-source HC2 clips remain clustered during statistics. Validation is labeled
validation, not relabeled as an unavailable official test split. All inputs have
project exposure; outside-development is not an untouched test.

## Table 1: natural cross-domain main results

Vid checkpoint -> HC2; HC2 checkpoint -> Vid. Clean only, complete official
queries, three prelocked source-blocked query-complete target orders. Ours has
41355 actual online arrivals (10446 HC2, 30909 Vid). Source Only, matched
parameter-free DINO-Refine, and target-trained frozen TA-STVG are stateless;
exactly identical input results may be reused across orders. Target-trained is
a supervised reference, not a mathematical upper bound or fair TTA competitor.
Online baseline streams are independent for each method/order, including their
optimizer/moving statistics/recovery state. No episodic-vs-online benchmark row.

Main metrics: source-macro m-vIoU, vIoU>.3, vIoU>.5. Average queries/clips and
orders within source first, then equally average sources. Paired 10000 source
bootstrap; report official query-macro companion metrics rather than calling
source-macro the original official query-macro. Supplement: tIoU/sIoU, each
order, negative tails and correctness damage. Do not mix cross corruption into
the natural-shift main table.

## Public baseline qualification

TENT/EATA/SAR are task ports, labeled as STVG ports. Their original sources are
https://github.com/DequanWang/tent, https://github.com/mr-eggplant/EATA and
https://github.com/mr-eggplant/SAR. No old pending baseline result is inherited.
Existing native entropy interface supplies per-offset H(start)+H(end)+mean
binary actionness entropy, with one sigmoid, masks and physical grids checked.
Decoder LayerNorm affine scope and deterministic eval dropout are disclosed;
visual/text encoders and native heads remain frozen, with live decoder gradients.
Post-update original-input output is a disclosed common current-query readout.
Final baseline settings/closure are sealed before target predictions, using
paper/default settings and no target metric tuning. All baseline streams use
the same three orders. A source Fisher preparation is required for EATA;
fishers=None is ETA and cannot be silently labeled EATA. Source training inputs
must be separately locked, never replaced by these target evaluation inputs.
Actual finite-gradient/parity/no-update/restoration/recovery smoke and independent
optimizer checks are required before launching the full baseline. Until that
qualification exists, the baseline row is pending, not an invented number.

DINO-Refine uses exactly Ours' Native interval, Uniform4, parser, admission and
retained Top1 detections. No new DINO calls or parameter adaptation. Replace
admitted sparse boxes and interpolate absolute cxcywh in physical frame time,
with nearest admitted box beyond the first/last admitted frame. With no admitted
frame retain Frozen. This readout is fixed before scores are available; it is
not the old deployed method with its temporal TTA enabled.

## Table 2: same-domain robustness

One prediction-independent hash-selected query per official video/clip, covering
all 732 Vid videos and 3482 HC2 clips. Same roster for every method/condition.
One prelocked complete source-blocked order; every condition resets state.
Clean + five families at 5% forms panel (a). Panel (b) uses five-family averages
at 2.5/5/10% physical-frame burst coverage. Existing transformation magnitude
is unchanged: these percentages measure affected frame fraction, not identical
physical severity across families. ceil coverage, start/end, sampled-frame hit,
and no-hit inputs are reported; no forced GT-event hit. 16 conditions (clean
plus 15 family/coverage combinations), 67424 arrivals per online method.
All main method rows use the same cohort: Source/TENT/EATA/SAR/DINO-Refine/Ours.
Main metric m-vIoU; other metrics in matched supplement. Parent-source bootstrap
also applies here because HC2 clips share 237 parents.

## Tables 3 and 4: matched component and supervision ablations

Use the same all-video one-query roster, one fixed order. Two cross directions
clean and the same-domain five-family 5% panel. Variants: Frozen, DINO-only,
w/o query residual, w/o LN adaptation, w/o consolidation, Full. No query-only
arm secretly performs a separate LN write. w/o consolidation writes no slow
state; it is a component ablation, not an extra named episodic method.
Supervision arms: existing direct 5 L1 + 2 GIoU, existing all-proposal energy,
admitted Top1. Other settings stay frozen; no Track/Authority/optimizer sweep.
Reuse a full/control result only if roster, input, state trajectory and settings
match exactly. An all-query online Full trajectory cannot supply the one-query
Full ablation's inherited state.

## Temporal diagnosis, budgets, sensitivity, cost and analyses

Temporal diagnosis uses the same one-query roster/clean cross directions and
fixed spatial output: Native, representative previously executed actionness
adaptation, latest real-input transform consensus, GT-head oracle. Deployable
predictions seal before diagnostic GT; GT-head is an explicit label-aware oracle
run afterward, never a method candidate. Do not stitch old small-cohort numbers.
UVTG remains the optional differently-scoped audited appendix; no new UVTG run.
These mechanism rows do not restart old queues or develop a new temporal method.

Observation budgets 1/2/4/8 use the same Uniform physical-bin rule, same admission,
lr and steps; alpha sensitivity is 0,1/32,1/16,1/8. Use the fixed clean cross
one-query roster/order. No promotion or best-budget/alpha selection. Report
actual calls (including empty/unavailable evidence), actual backward steps,
parameter counts, synchronized model/expert latency, CPU-audit overhead separately,
and CUDA peak allocation/reservation; no cached latency passed off as uncached.

All-label analyses are post-seal only: admitted expert GT-IoU terciles, event
duration terciles, GT-track motion normalized by frame size and physical elapsed
time, Vid declarative/interrogative, Frozen/Before/After pipeline damage and
recovery, >5/>20pp tails and positive/failure cases. Empty/unknown GT evidence
is separate from Low quality. Terciles use per-video clean GT/statistics once,
not repeated conditions; report ties/empty groups without forced balancing.
An expected quality/gain correlation is a hypothesis, not a promised conclusion.

## Execution and completion

Priority: Table1 -> Table2 -> components/supervision -> temporal diagnosis ->
budget/alpha/cost -> sealed analyses. One finite GPU controller/lease, no total
deadline, resource floor 8GiB, bounded RAM feature/frame cache, durable compact
float32 prediction/state receipts. Preserve every failure, revision and negative
result. Each separately locked stage seals every deployed arm before that
stage's evaluation labels are used. No final-results feedback to configuration.

Ours' Table1 implementation and real smoke are the first runnable stage; future
ports/stages are tracked explicitly until implemented and qualified. A stage
ending pending_root is a handoff, not completed paper experiments. Root continues
remaining already-authorized work, audit, figures, complete public anonymous rows
and exact GitHub-byte verification plus RESEARCH_HISTORY check/snapshot/check.
FINAL_COMPLETION and monitor pause require every requested stage, not just Ours.
