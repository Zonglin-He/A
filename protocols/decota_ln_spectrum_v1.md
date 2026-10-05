# Top1 DeCoTA LN correction spectrum and isolated-write audit

User-authorized on 2026-10-05 after verified public commit 38810f55.
This is a finite CPU analysis of sealed trajectories, not a new online method.
Production registries and all preceding predictions remain unchanged.

## Fixed method and data

One frozen Grounding DINO only. Native TA-STVG temporal interval is unchanged
and supplies the four existing physical-time uniform observation positions.
The existing admitted-frame Top1 spatial critic, Adam .03, 10-step own-loss
selection, joint 256 query + 1536 LN parameters and query/Adam reset are fixed.
The current output uses the complete selected correction; historical online
writes are 1/16 of the selected LN proposal. No Sa2VA, second expert, temporal
adaptation, new evidence, model forward, decoder replay, backward or training.

Reuse all 13824 sealed logical arrivals: each dataset has 32 development and 16
source-disjoint but historically exposed confirmation sources, one query per
source, two fixed orders, clean and five fixed 5% corruption inputs; episodic,
100% online and five nested 25/50% schedules. Independent schedules are not
independent new sources. Both official same-domain checkpoints and original
sampling remain fixed. The two parameter spaces are analyzed separately.

## Label-blind extraction and geometry

Read only receipt-verified saved parameter states and metadata. The vector is
selected fit.state LN minus fit.initial LN, not the final optimization step and
not the already-scaled committed delta. The exact six named LN blocks and their
order are locked. Empty evidence, selection at step 0 and zero writes remain in
coverage accounting. Nonexpert positions are not imaginary correction samples.
Deduplicate the identical two episodic orders by source/input; online orders
remain distinct trajectories. Audit every source/initial/selected/committed
state and previous-file chain. Canonical episodic and online100 spectra use
source-balanced raw columns, unit-column controls and centered controls.

Report uncentered energy E_r for ranks 1,2,4,8,16,32. Also report effective rank,
mean-direction energy, raw norm concentration, within/between-source cosine,
and source-held-out projection. A hash-defined two-fold split of development
sources fits a basis on one half and evaluates the other; all-development basis
evaluates confirmation. Neither labels nor utility choose any rank or basis.

For each actual online stream, independently reset geometry at the original
dataset/split/condition/order boundary. At a current nonzero write, measure
projection energy onto the uncentered basis of strictly PREVIOUS nonzero writes,
for all six ranks. No current or future vector enters its own prefix basis.
Cold starts and fewer-than-rank histories are explicit. These are parameter
energy diagnostics, not projected-model predictions or performance gains.

## Exact isolated utility available without inference

For each online stream, use its first actual nonzero LN commit. Before the next
actual nonzero commit, each subsequent arrival's initial LN equals that single
donor's committed state and its query is zero. The matching episodic recipient
has source LN, zero query, identical pixels and native interval. Thus the sealed
online Before minus Frozen metric is an exact single-write effect at scale 1/16
on this limited prefix. Audit those conditions instead of interpreting general
cumulative Before-Frozen as U(i->j). Include the Before of the next-write arrival,
since it precedes that second write. Exclude self-source pairs. Deduplicate
identical donor/recipient/input pairs across schedules and orders; retain their
full membership receipts. Both donor and recipient correction vectors for the
cosine are their independent source-initialized episodic selected proposals.
Recipients with a zero selected correction retain utility but have undefined
cosine and are counted separately.

This measures U^(1/16), not a full unscaled DeltaLN intervention. It covers early
prefix recipients and first active donors, not all donor-recipient pairs or
late-state transfer. The recipient correction is a post-hoc diagnostic and
cannot be an online admission rule for nonexpert arrivals.

Seal all geometry, prefix definitions and pair selections before joining old
anonymous GT-derived metrics. This is historically exposed post-hoc analysis,
not first exposure to GT. No annotations/media/weights are opened or rescored.
Analyze utility vs cosine as continuous correlation and prelocked cosine sign
groups (<0, >0; exact zero separate), with bins [-1,-.5,-.25,0,.25,.5,1].
Use recipient-source macro means and 10000 source-node bootstrap draws, seed
20261005. A single common resampling count per source weights both its donor
and recipient appearances; pairs/schedules/conditions are not independent n.
Intervals are conditional on cached vectors and fixed prefix coverage, without
re-fitting bases. Report clean, corruption, each order and donor/source influence.

## Interpretation and conditional follow-up

Low in-sample E8 alone does not identify useful shared memory: shared optimizer,
common mean, same source/pixels and dominant norms can produce low rank.
Source-held-out and strict-prefix retention address geometry only. Cosine-
utility association would be evidence for a limited write mechanism; it would
not prove that low-rank projection improves actual online output.

Only after reviewing BOTH datasets' independent geometry and utility evidence
can a separately locked projected-memory comparison be justified. No rank is
selected by GT, no projected state is executed here, no alpha is retuned, and
no future-informed global SVD is used as a deployed online basis. If the evidence
does not establish the proposed connection, report that limit and do not launch
the conditional method. Preserve positive and negative cases and development
vs confirmation distinctions. The present task closes with a complete audit,
report, figures, public anonymous scalar geometry/results and remote verification.
For the narrow cosine-to-utility connection, a positive result requires both
confirmation corruption panels' source-node 95% intervals for the continuous
correlation and positive-minus-negative utility contrast to lie above zero.
Missing sign-group coverage is inconclusive. This criterion motivates a later
matched projection experiment; it does not validate a deployed memory mechanism.
