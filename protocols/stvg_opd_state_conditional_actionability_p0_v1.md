# P9-P0: same evidence, different inherited student states

Question: do the same frozen expert observations produce different spatial
update consequences across the two original online orders, and how large are
the reversals relative to changes in the inherited LN state?

Use exactly the complete P7 historical clean roster:128 parents per target,
two original orders,512 cells. Analyze Omega0/original observations and
Omega1/projected observations separately,512 parent-scope pairs/1024 saved
updates. Both scopes within an order start from the original Full state.
P7-B did not propagate its own new C states. No new model calls, GPU fits,
DINO calls, optimizer steps, hyperparameter search, gate, or manuscript.
The already running C Expanded-State GPU task is left untouched.

Before any new lookup of GT-derived P9 labels, seal the full paired cache:
same query ordinal, source checkpoint hash, input pixels, physical frame IDs,
Native support, expert pack and actual visited/admitted anchors, configuration,
and source-bound initial state. Query residual must be reset to exact zero;
the six LN vectors have1536 coordinates. Original source initial states must
agree across orders. Keep every parent, including empty expert/zero-update
episodes; do not redefine admission or select a favorable cohort.

Read every saved rollout seed, shape, mean and candidate sample. Reconstruct
the original CPU FP32 generator's16 normal samples and16 antithetic samples.
Require exact cross-order standardized noise and exact replay of saved sample
coordinates from the respective mean and original sigma. Equal noise does
not mean equal actions: actions intentionally depend on the student state.
All source files receive SHA256 checks before and after extraction.

Offline outcomes reuse the already independently verified P9 labels after
the new paired seal. No fresh raw GT access or GT scoring. Report both
own-U (exclude the current allocation's visited positions) and common-U
(exclude the union of both allocations' visited positions), on annotated
GT-event genuine decoder frames. Same physical support/query annotation
identity and exclusion sets must match between orders. Undefined U stays
unknown, never zero. Teacher quality is an offline diagnostic, not a signal
available for online selection.

For each target/scope separately, report all128, previous32, added96, and
both individual orders. Sign tolerance is the original1e-12. Report the full
positive/negative/neutral/unknown2x2 transition matrix, both-nonneutral
denominators, benefit/harm flips, and flip rates on all128 and on eligible
pairs. Predeclared descriptive magnitude tiers are1pp and5pp on BOTH sides
of zero, order-effect differences of5/20pp, and severe negative outcomes
below-5/-20pp. These tiers are not used to train or tune a gate.

Measure cross-order inherited LN L2/RMS difference, each state distance to
the source initialization, and before-box displacement. Correlate LN RMS
with absolute effect difference using Pearson and Spearman; also retain
the relation between Before teacher agreement difference and signed effect
difference. Use10000 parent bootstrap samples, seed20261013, with the same
parent draws for all paired quantities. Retain undefined constant resamples.
Confidence intervals are exploratory and unadjusted for multiple diagnostics.

This is a naturally occurring state/order contrast. Matched evidence and
common random numbers remove two confounds but do not establish causal
attribution to LN inheritance or a safe unlabeled actionability predictor.
A controlled state intervention remains a later decision contingent on the
complete C experiment and independent root. No automatic new GPU work.

Completion requires an independent full source/state/randomness/scalar
readback, actual scientific PDF pixel view, every anonymous positive and
negative parent-scope row, verified GitHub bytes/SHA/Gitblob/tree and both
refs, and a real archive check-snapshot-check and scoped FINAL.
