# R130: measured findings and development decision

The prelocked original-rho primary and all endpoints are reported. Nothing in
this document changes labels, radius, acceptance or predictions after scoring.
R129's protected-primary failure remains unchanged.

## 1. Retain the original working component, without overstating confirmation

Original `partial_F` remains +0.115181 pp / +0.079678 pp over its own zero,
and +0.096817 / +0.061572 pp over the same-information direct partial reader,
on DEV/EXP. These original predictions are reused, not a new-data replication.
Twelve engineering original-rho updates were replayed with exactly zero state
error to verify compatibility of the new implementation.

The new matched within-S wrong-set update gives 17.293569% / 16.255043% mIoU,
against correct 17.394852% / 16.498101%. Correct wins raw six-direction means
by +0.101283 / +0.243057 pp. This is evidence against these gains being entirely
independent of retained-set identity. It is not conclusive identity reasoning.

Scene-frequency weighting changes correct-minus-wrong to -0.004291 pp on DEV
and +0.145341 pp on EXP. That tiny DEV reversal fails the prelocked requirement
of positive differences under both aggregations. It must be disclosed, but it
does not erase the positive raw results or establish a large meaningful wrong-set
advantage. No threshold or primary endpoint is changed to reverse that outcome.

## 2. The tested radius neighborhood is narrow

| Radius factor | DEV correct mIoU | EXP correct mIoU | DEV delta vs zero | EXP delta vs zero |
|---|---:|---:|---:|---:|
| 0.5 | 17.292880 | 16.411944 | +0.013209 pp | -0.006479 pp |
| 1.0, fixed primary | 17.394852 | 16.498101 | +0.115181 pp | +0.079678 pp |
| 2.0 | 17.196333 | 16.411401 | -0.083338 pp | -0.007022 pp |

Original rho is the best of the three tested values on both raw cohorts; it
was already primary, so this is not a new choice. The endpoints do not establish
a broad positive plateau. This is sensitivity evidence, not an extra failure
criterion requiring every radius or every direction to improve.

Double steps lower the auxiliary task further but do not improve grounding.
On DEV/EXP their mean task term ends at .020920/.018089 versus half-step
.072941/.064989, while double-step grounding falls below zero. Greater task
fit alone remains an insufficient acceptance principle.

## 3. What the matched control actually isolates

S, label cardinality, group multiplicity bins and nominal optimization budgets
are identical. 252/257 active inputs have an effective retained/excluded swap;
the other five are transparently identical controls, not fabricated mismatches.
Effective-only results are included alongside the full denominator.

Initial conditional mass in C is .880629/.891044 for correct sets versus
.028157/.049019 for shuffled sets. Among active inputs q0 is in correct C on
50 DEV and 122 EXP records, versus 2 DEV and 10 EXP under shuffle. Thus this control also changes whether
the initial plausible/high-scoring candidates are retained. Winning it supports
the usefulness of that plausibility structure, not a unique causal claim about
registered RGB-LiDAR identity evidence. This confound was documented before
scoring. All actual parameter norms and Armijo trials are retained.

## 4. All 18 old changing records were reviewed

They comprise 11 improved and 7 worsened direction-inputs, 18 physical frames
and 6 target-scene clusters. The audit asks where candidates with better IoU
than the zero prediction reside; this is post-seal geometry diagnosis, not GT
used by an update and not certification of semantic object identity.

| Observed location | Improved | Worsened |
|---|---:|---:|
| A better candidate remains inside C; C does not specify its internal order | 11 | 4 |
| S has better candidates, but C excludes all of them | 0 | 1 |
| Zero already selects an A256 IoU-oracle candidate | 0 | 2 |

The largest category here is unresolved competition within C, not missing S.
Nine of eleven improved winners and six of seven worsened winners lie in C.
Merely selecting from C can therefore accompany both gains and losses. Two
improved winners lie outside C, consistent with shared-parameter propagation
rather than an explicit positive identity label for that winner.

This selected changing-case audit cannot replace full-cohort set recall. The
unchanged R129 broader diagnosis still has an Acc25-capable C candidate on
only 8/28 DEV and 17/70 EXP recoverable zero errors in partial-active scope.
The case-level continuous-IoU question and that broader thresholded recall
question have different denominators. No case-specific rule follows from them.

## Decision

Keep unguarded `partial_F` at original rho as the bounded development candidate;
do not replace architecture, add a safety loss or choose half/double by target.
Do not claim the strict mechanism gate passed, a broad stable radius interval,
or independent confirmation. No larger hyperparameter sweep is justified here.
A locked-method evaluation on additional data would be more informative than
another local amplitude search. Candidate-set development should distinguish
coverage/exclusion from the missing within-C ranking information. Such a new
experiment requires separate authorization; R130 ends with no queued follow-up.
