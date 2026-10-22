# Parent-isolated128 development: one-round semantic prototype regularization

Status: acquisition/prediction protocol locked; **no new development or confirmation
metrics yet**. The user's07deadline and original-paper fallback were explicitly
cancelled. OriginalpaperHC1902/Vid1373 remains paused. No temporal default is changed.

The main candidate remains complete-trajectory-conditioned, target-support learned
word semantics. This is not cross-query network training: prototypes are estimated
inside each video. Previously observed instability can reflect pseudo-label feedback,
domain transfer and repeated development selection; it cannot be called established
supervised-training overfitting from the HC64/fixed-panel gap alone.

## Four families, no stacked corrections

| Family | Fixed change | Finite strengths |
|---|---|---|
| Original two-round | original TargetSemanticLoop2 | none |
| Original one-round | only the first original update | none |
| **One-round prototype shrinkage** | both means shrink to opposite-parity unconditional mean | kappa1,4,16 |
| One-round weaker evidence | same original means; beta multiplies interval evidence | beta.25,.5,.75 |

Kappa is a common-mean pseudo-count in units of fitted frames. Both prototypes and
the unconditional mean use only the opposite temporal parity. Formula preserves
the original Bernoulli class-sum likelihood, inclusive interval sum, native prior,
physical MBR and unchanged complete-box DP. There is no shrink+beta combination,
new layer/head/capacity, ROI radius choice, confidence rejection or extra query step.
The baseline restoration checks preserve original one- and two-round results.

A weaker-feedback control is independently selected with the same three-value budget.
Shrinkage beating the original loop does not prove better prototype estimation if
matched weaker evidence explains the gain. The protocol separately reports that
mechanism comparison and matched random/time/position controls; less movement alone
is never a successful result.

## Larger fixed cohort and selection boundary

Input SHA chooses128 distinct official parents from each complete227/732parent pool,
one query per parent, without labels or candidate outcomes. Within each direction,
64 parents are development and64 are confirmation. Confirmation is prospective for
this candidate's selection: all official full results and older panels have historical
exposure, so it is not renamed an untouched test. One query/parent makes this a
parent-balanced development estimand; final official full remains query weighted.

Only development labels choose kappa and the separate beta control. Each family has
three strengths. Among configurations passing predeclared per-domain Source/G6/time/
complete-v/short-long/win–harm/serious-tail conditions, maximize equally weighted
per-domain complete-v gain; ties prefer the smaller numeric strength. If none passes,
the maximum utility configuration is explicitly frozen only for a diagnostic
confirmation, never as automatic full admission. Confirmation runs once after the
configuration file is written and hashed. Unselected confirmation predictions remain
sealed and are not GT scored. Full rules and matched-control conditions are in the
locked code and design; no post-result threshold change or control promotion.

## Source-only acquisition and retained failures

Compact paper caches preserve geometry and temporal logits but not every native
word-head input. Necessary Source224 features are reconstructed with identical
inputs, checkpoint, seeds, precision and offsets. Existing paper or content-bound
complete-capture geometry is reused. Otherwise the original capture is allowed only
with an old expert payload: a hard-failing cached-teacher adapter cannot launch a
ReferDINO forward. Source/E/Full/interval/eight-position outputs must all equal the
already sealed official outputs bitwise before semantic acquisition is accepted.
Only target/global/reflected/wrong-time learned heads are read; original head all-key
no-op is exact. No spatial parameter, extra expert model or observation changes.
Actual RGB/Source/reuse/semantic costs are recorded by real worker identity.

Acquisitionv1 safely stopped on a missing raw expert payload after7 saved queries;
there were0 new expert calls. The same selected member has a valid paper geometry
receipt. v2 introduced Source-only reconstruction with that geometry but lacked the
compact `native_interval` field; it stopped before accepting any new member. Its
Source/code/logs remain intact. v3 derives the interval from original final-layer
native logits using the unchanged native rule; it does not copy a GT interval.
The7 completed v1 semantic/capture/receipt files are migrated byte-for-byte under a
new plan, with explicit hashes and no recomputation. These are engineering failures,
not negative method outcomes or fingerprint exemptions. See the separate collection
protocols; all remain isolated from frozen48maintained/434provider source files.

Five CPU tests pass: original one-round exact restoration, formula/opposite-parity
fit, monotonic prototype shrinkage, constant/no-feedback no-op and axis permutation/
no-stack constraints. All256 candidate predictions must seal before new GT scoring.
Labels are accessed only by an independent CPU evaluator for the requested split;
confirmation coordinates are not JSON-parsed during development. No final results
or success are asserted before sealed evaluation.
