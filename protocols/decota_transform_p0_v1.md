# Real-input temporal equivariance and spatial correction-stability P0

User-authorized isolated successor to 53c76f7; no old job resumes. Two P0s,
not a new deployed model. Each dataset retains exactly the preceding32 development
and16 historically exposed, source-disjoint confirmation sources; one query/source,
two locked orders, clean and the same five5% corruption pixels. Expert rate of the
working point is100%. Official same-domain TA-STVG EMA checkpoints and Paper48
sampling unchanged. Old Uniform4/single frozen DINO/admitted Top1 energy/Adam.03,
1792 query+LN parameters/10 steps/first minimum own loss/query reset/LN1/16 states
are reused with exact receipts. No new DINO, optimizer, gradient, memory, or writes.

## Temporal P0

For each of576 unique original source-condition inputs, decode the original
sampled RGB and verify the old corruption hash. Two genuine new input forwards:

* Shift: prepend ceil(N/8), at least2, repeats of the first sampled RGB. Median
  integral frame gap g defines a virtual pad grid; real frames' relative IDs shift
  by Delta=p*g. Inverse is new_time-Delta+original_first_frame. Real frames are
  preserved, no existing logits or H are merely translated.
* Crop: if native starts at index s and ends inclusively at e, retain half the
  available sampled context on both sides, [floor(s/2),e+1+ceil((N-e-1)/2)).
  At least4 frames, otherwise identity. Never remove a sampled Native-support
  frame. IDs subtract the crop's first original frame; inverse adds it back.

Every view uses both official offsets and their original envelope MAP readout.
Inverse intervals clip to original sampled support; disjoint/invalid ones fall
back to Native and are counted. Identity crop is explicitly counted and reuses
the original result. Fixed consensus is coordinatewise median of Native and both
inverse mapped view intervals. No scorer/teacher/LR/transform strength search.
Official dense tIoU/vIoU uses fixed old episodic corrected spatial boxes (primary)
and actual online100 corrected boxes (secondary). Native-spatial control is also
reported. Individual views are diagnostic controls, not post-hoc selectable arms.

Coordinate transport is exact; semantic invariance is not guaranteed. Repeated
padding changes motion context, and containing predicted Native support does not
guarantee containing the true event. GT never chooses or changes a crop.

## Spatial P0

Two real student inputs: horizontal RGB flip and deterministic round-to-nearest
brightness multiplier.95. Same query text, time sampling and corruption. Both
run the full frozen encoder anew. Reuse each new normalized prefix only for its
different old1792 parameter states; unchanged model source weights. Undo flip with
cx=1-cx, retaining cy,w,h. Existing current correction is never refit or selected.

Primary consistency is mean pairwise box IoU on sampled frames inside fixed Native
interval; full-clip consistency is secondary. DeltaC=C(selected)-C(prearrival).
For episodic prearrival is Native; actual online100 uses its saved prearrival
LN state, not source Native. Assess both streams separately; online state chains
remain sealed old ones. Cache identical episodic order states only after bitwise
verification. Both orders retained as logical cells, not independent sources.
Directional query words are flagged prior to scoring; flip may change referent
semantics, so non-directional results are additionally reported, not filtered.
Loss gain, parameter/hash provenance, zero corrections, clean and negative tails
are retained. No gate or GT-selected optimization step runs in P0.

## Sealing, evaluation and conditional decision

Global barrier covering every new view prediction, interval and DeltaC precedes
GT access. CPU official dense and independent private arithmetic verification,
paired10000 source bootstrap, search/confirmation/clean/orders/conditions reported.
Temporal qualifies only if corruption confirmation Delta t and Delta v lower95%
CI>0 on both datasets and development means>0. Spatial qualifies only if on BOTH
corruption confirmations episodic DeltaC-help/harm AUC lower CI>.5 and correlation
lower CI>0, development point signs positive, and online100 confirms those signs
and lower-CI conditions. ROC excludes exact zero utility and reports denominator;
correlation retains zero. Correlation is measured with a fixed unlabeled score,
no trained predictor: development and confirmation source rosters are disjoint.
Undefined stats never qualify. Secondary subgroups cannot override primary gates.

Only qualification allows a separately locked, matched conditional temporal514
SGD.01/K3 or correction acceptance trial. Until then zero gradients. Failed P0s
explicitly skip those stages and retain the research working point; no CURRENT
promotion, full benchmark, new expert, old queue, new proxy or memory starts.

Run no-GT native/full-replay parity for first two development clean inputs per
dataset; pixel/coordinate/zero-update contracts before full execution. Single GPU,
finite serial dataset sequence, disk floor8GiB, no total deadline. Do not store
new full encoder H after extracting predictions. Resource counts distinguish
full two-offset view passes, offset forwards, suffix replays and wall time.
Failures preserve original outputs and pinned repair revisions; no result-based
configuration changes. Publish all anonymous rows/negative findings/code/protocol
and verify exact GitHub bytes; private RGB/query/annotations/H/state remain local.
