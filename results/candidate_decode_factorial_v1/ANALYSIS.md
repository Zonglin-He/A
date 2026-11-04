# Full candidate generation × trajectory decoding

Fixed actual Full event interval, same raw8expert inputs and original E/feature/geometry/score weights. Whole HC1902/227parents and Vid10303/732parents. Current Full remains unchanged, with original soft E/expert choices at observed positions. Only no-transport anchor controls are newly predicted; independent/full/E original outputs and official values are reused by array identity.

|Spatial strategy|HC vIoU|HC conditional Unobs. IoU|Vid vIoU|Vid conditional Unobs. IoU|
|---|---:|---:|---:|---:|
|Shared correction E only|26.4897|52.7616|13.9487|37.8617|
|E + expert anchors, independent|26.7551|53.2056|14.2049|38.4247|
|E + expert anchors + DP|26.6267|52.9875|14.1596|38.3212|
|E + bidirectional candidates, independent|26.2815|52.1094|13.9978|37.9335|
|Full spatial inference|27.3490|54.3549|14.5582|39.5049|

|Spatial strategy|HC ΔvIoU|HC ΔUnobs.|Vid ΔvIoU|Vid ΔUnobs.|
|---|---:|---:|---:|---:|
|Shared correction E only|0|0|0|0|
|E + expert anchors, independent|+0.2653|+0.4440|+0.2562|+0.5630|
|E + expert anchors + DP|+0.1369|+0.2259|+0.2109|+0.4596|
|E + bidirectional candidates, independent|-0.2082|-0.6522|+0.0491|+0.0719|
|Full spatial inference|+0.8593|+1.5933|+0.6094|+1.6432|

All deltas above are percentage points relative to E. Detailed paired parent10000bootstrap95%CIs, every signed decline/rescue, short/long strata, exact support counts and actual process/start/CPU stage costs are in the two ROOT_report.json files.

Unobs. IoU here is CONDITIONAL on annotated dense frames inside the common predicted∩GT event, excluding exact eight expert physical frame IDs, per-query average. Empty event support is null; primary vIoU retains the full query denominator. This is not the earlier128 all-video native-sampled-frame Unobs definition.

Transport off removes only transported unobserved candidates from the same original bank, preserving E/expert observed candidates and original node/link tables, support and all normalization weights. This candidate-inclusion control must not be represented as a timing test with all geometry computation removed. Independent selection is per-node unary argmax, not interpolation. DP is the unchanged full global decoder.

A no-transport path has E at every unobserved NATIVE sample. Its dense unobserved IoU may still change near observed points because official densification interpolates between native predictions. This distinction is independently checked and is not hidden candidate transport.

The reported 2x2 interaction is between candidate inclusion and trajectory decoding under fixed temporal inference. It does not establish positive interaction between spatial and temporal branches, action understanding, or recurrent mutual reinforcement. Saved interpolation comparisons, including negative full Vid outcomes, remain separate and intact. No target score is used to change parameters or adopt a new method.

No new Source, expert, model, GPU or optimizer call. Original acquisition costs are nonzero; cached CPU stage walls are not deployment latency and concurrent walls are not summed. No manuscript/Fig1/production/legacy frozen sources, protected drafts or old paused queues changed.
