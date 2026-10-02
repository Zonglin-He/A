# Student-routed Sa2VA frame-acquisition qualification

Question: on exactly the same sealed A state and nine spatial candidates, does
student temporal support choose more useful five-frame observations than uniform
full-clip sampling? This is a critic qualification, not an online trajectory.

Keep spatial A Rank-RKL, official same-domain TA-STVG checkpoints, original
Paper48 pixels/frame grid, temporal critic,1792 parameters,D4,rho.05,student T1,
Vid lr.033761698432507946/teacher T.34902548789596055/K1,
HC lr.006097133675874025/teacher T1/K8 unchanged. Use the first inner-step
candidates in each sealed A arrival; later HC steps are not independent samples.

Within each old32 historically exposed development pool, eligible sources are
those scheduled at arrival modulo4=0 in either A order. Sort by
sha256(dataset+":"+source), take first10. Prefer order1 if eligible, else order2.
Every source has clean and two corrupted conditions: from the original ordered
five-corruption list take source-hash-rank j modulo5 and (j+2) modulo5. Thus each
dataset has30 cells/10 sources/one query per source/20 corrupted/10 clean and
each corruption has4 cells. Selection precedes new label interpretation.

Uniform reuses the old five-frame Sa2VA cache. Student-routed uses equal votes
from the original at most8 deduplicated native temporal candidates, physical
half-open intervals. CDF10/30/50/70/90% points, retain unique frames, fill to5 by
maximum CDF-midpoint separation among unused positive-support frames, then
larger support weight, then earliest index; if exhausted use all unused frames.
Sort positions chronologically. Record raw duplicate count and actual5distinct
frames. No E/SE, reward weighting, geometry weighting, new temporal parameters,
prototype or state-memory changes.

Local frozen Sa2VA-4B, original prompt, BF16, model preprocessing and256 output
tokens;5PIL frames per call.60 new routed calls plus one old uniform clean parity
call per dataset: hard cap62 attempts including parity. Original uniform masks,
boxes, text and input hash must reproduce bitwise before accepting comparability.
New rewards and selections are computed without GT. All outputs globally sealed
before new offline GT interpretation. GT never influences observations/ranking.

Spatial rewards are the original unweighted mean candidate/expert box IoU over
nonempty observed masks. Reward ties: first candidate index; empty feedback:
native index0 plus explicit missing-evidence flag. Two task-utility conventions
are reported: PRIMARY candidate boxes at the COMMON original A native interval
(isolates spatial judgment), SECONDARY candidate-own original native temporal
interval (joint variation). Same nine candidates for both observation strategies.
Candidate oracle is max utility including native. All-frame geometry or teacher
IoU on differing reference subsets is not the primary task-quality readout.

Pairwise ranking: all36 unordered pairs, GT utility tie tolerance1e-12; GT ties
excluded from strict-pair denominator; expert ties and empty feedback get.5.
Report strict-pair availability, decisive coverage, accuracy, candidate-native
gain, oracle regret, gross gain/loss, .3/.5 missed correct/support absence, empty
mask arrival rate, no event-inside valid reference (all arrivals and conditional
on nonempty evidence), sampled event-hit proportion and actual distinct frames.
The existing HC half-open event endpoint convention is preserved; actual GT-box
support is separately reported. No labels invented outside the event.

Primary summaries: source macro of condition cells, each source equal weight;
paired10000 source bootstrap seed20261001. Clean and corruption separate.
Missing pairwise denominators are explicit, matched availability also reported.
Keep positive/negative examples and all anonymous scalar rows/pairs. Small exposed
development signal cannot establish fresh generalization or online gain.

If only reference/event coverage improves, do not enter online adaptation. If
candidate judgment improves, report the signal and uncertainty; the proposed
subsequent online A vs reference-position-only A is a separate next round.
This qualification does not automatically start that run or promote production.
No deadline, no historical queue restart. Complete root/independent numerical
audits, scientific report/figures, archive check/snapshot/check and verified
public GitHub export. No raw media/labels/weights/masks/tubes/personal records.
