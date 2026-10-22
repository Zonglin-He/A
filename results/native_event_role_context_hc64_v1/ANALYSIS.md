# Native actor–argument temporal context: completed negative feasibility test

The fixed actor-plus-argument context does not qualify. On the exposed HC64 panel,
relative to NativeG6, temporal IoU changes by **+0.1847 pp** (95% paired parent CI
[-0.7047,+1.0367]) and complete vIoU by **+0.1799 pp** ([-0.1729,+0.5922]).
Temporal changes:21 improved,12 harmed,31 tied, with2 losses worse than5pp.
Complete-v changes:19 improved,13 harmed,32 tied, no loss worse than5pp.
Relative to Source, short-event tIoU is **−0.3016 pp**, long-event **+1.7372 pp**.
No fixed-panel recheck or promotion follows this failed development screen.

## What actually changed

The existing Source-only Stanza rule selects the first main-event argument noun phrase:
59/64 syntactically supported queries;5 structural aliases remain NativeG6 in every arm.
The **same frozen Source** receives that auxiliary noun phrase and predicts its tube.
No new ReferDINO model, forward, observation, RGB or visual encoder is used. The
main read preserves the Full caption, actual temporal query, six layers, original
positions and full physical grid. Only motion cross-attention keys change: union of
1.5-context complete BoxFixed actor support and1.5-context auxiliary Source argument.
Final posterior is the predeclared half NativeG6/half context posterior, original
physical MBR and unchanged complete-box spatial DP. This is one actual spatial-to-
temporal-to-final-spatial round, without network training or new query fitting.
Original global fusion/motion already contains full-clip context; this is not pure
causal removal or newly recovered high-resolution visual information.

This differs from prior actor-only prefusion masks and score/prototype transforms:
it actually queries the same Source for an event argument and changes the trained
TimeDecoder's attention context, while retaining the original Full temporal state.

## Same-information and equal-size controls

All values are percentage points relative to NativeG6, complete64query population.

| Read | ΔtIoU | Δcomplete vIoU |
|---|---:|---:|
| **Actor+argument primary** | **+0.1847** | **+0.1799** |
| Direct same context, no posterior mixture |−0.2117|−0.0048|
| Target only, identical key-mask mechanism |−0.3622|+0.0862|
| Reflected argument, exact extra key quota |+0.1328|+0.2272|
| Wrong-time argument, exact extra key quota |−0.1663|+0.0899|
| Original Source actor as auxiliary, same quota |−0.0125|+0.0882|
| Auxiliary object time posterior mixture |−3.0207|−1.2134|
| Original posterior sharpen |−2.1353|−0.9507|
| Native all-keys no-op |0|0|

The three placement controls retain exactly the primary number of additional
non-actor cells per frame. Ties/fill are deterministic row-major; they are not known
negative objects. Primary−reflected tIoU has no reliable positive paired interval;
reflected complete-v mean is higher. The test does not establish event-object-specific
boundary evidence. Controls are not promoted after seeing their scores.

## Concrete failure and rescue audit

Independent postseal numerical and private visual review separates three issues.
First, a syntactic argument is not a verified auxiliary object: mean Source-argument
versus original Source-actor box IoU is0.3927;34/59 are below0.5, but difference alone
does not establish correct identity. Four metric-selected private cases show static-
context requests still produce regions on people or small unrelated regions. No
auxiliary-object GT or reliable object-localization accuracy claim is supplied.

Second, apparent time rescues are often not placement-specific. RC014 improves
12.3405pp but the reflected, wrong-time and Sourceactor controls give the same final
interval. RC049 improves8.5185pp, also matched by Sourceactor context. These are real
metric changes, but do not demonstrate that grounding the event argument caused them.

Third, context can still harm boundaries. RC015 loses12.3418pp while Sourceactor
context stays native; its auxiliary region visually follows other people. RC052
loses12.2172pp, also matched by TargetOnly and Sourceactor context. Correctly retaining
more context cannot be inferred merely from selecting a noun phrase or adding keys.
The four reviewed cases are selected metric extremes, not prevalence estimates.
Private frames/query text remain unpublished; all anonymous signed outcomes remain.

Across59 supported queries, actor context covers48.65% of valid motion cells and
actor-plus-argument60.09%, adding7.90 cells/frame on average. There are11 per-frame
empty-ROI fallbacks to original valid grid across the evaluated context constructions;
no query is rejected. Reflected-support overlap averages0.3208. No radius, score,
mixture or semantic-template sweep follows this result.

## Engineering, evaluation and cost

Original Full text/fusion replay restores all64 original hidden/logits/posteriors/
Source boxes bitwise. Restored native TimeDecoder kwargs/strides also reproduce
all64 original offset hidden/logits; Native/full-key outputs and final spatial boxes
match. Three CPU tests verify exact extra-key quota, preserved text keys/no-op, and
physical offset indexing. Independent CPU scoring begins only after all64 predictions
seal:1408 official/dense metric comparisons, maximum error below1e-14. See [signed
outcomes](signed.tsv), [severe tails](severe_declines.tsv), [full report](report.json),
[qualification](QUALIFICATION.json) and [context audit](CONTEXT_AUDIT.json).

Actual prediction worker33.12s includes loading/research bookkeeping:59 additional
Source text encodings,246 fusion-offset reads (including64 Full no-op replays),123
native two-pass replay queries,718 additional TimeDecoder reads. The native replays
also contain their original first/second temporal calls; the718 counter excludes
those nested native calls and must not be mistaken for total decoder cost. No new
RGB/visual encoder/expert/observation/optimizer. This is cached research cost,
**not a fair deployment latency claim**. Independent private visual inspection adds
CPU video decoding after scoring, no new model or final predictions.

Original48 maintained and434 provider source bindings remain unchanged. Original
paperHC1902/Vid1373 remains paused before07:00; this branch cannot authorize a new
method's full run. Under the user's deadline instruction, if no other fixed dual-
domain-qualified method exists at07:00, the original same-plan queue is resumed
only after fresh source/resource/output/process/lock verification.

## Source grounding of the intervention

The original [TA-STVG query decoder](https://github.com/HengLan/TA-STVG/blob/904ad0c344eb12fe34513f97a33f93118f6d3be4/models/grounding_model/query_decoder.py)
passes text-plus-motion encoded features to the TimeDecoder. This motivates testing
that actual context interface; it does not prove that an object-only noun phrase
matches its training distribution, that the auxiliary Source can ground arbitrary
objects, or that a masked context is a calibrated event signal. The original mask
layout is not repaired or rewritten; the explicit intervention operates on the
actual cached feature order. All-source no-op is a compatibility check, not evidence
that the new feedback works.
