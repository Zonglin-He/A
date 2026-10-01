# CPU post-hoc pipeline attribution

User authorization 2026-10-01: inspect GT to locate correct samples rejected/damaged along the pipeline, without GPU if existing artifacts suffice; repeat on the eventual tuned full evaluation after tuning, not during model selection.

Immediate scope: immutable completed Paper48 P1 (670 VidSTG sources, two orders, 8040 arrivals) and P5 (128 HC2 sources, one order, 768 arrivals), original frozen J01/default parameters, clean plus five 5% transient corruptions, 25% scheduled experts. These are historical exposed diagnostic sources, not final tuned full evaluation or fresh confirmation. No new model execution, GPU work, changed selection, new candidates, method changes or resumed historical queue.

Verify all panel receipt/prediction hashes before reading GT. Reuse the original dense GT binding and evaluator conventions. Compute Frozen -> inherited boxes with frozen interval -> inherited full prediction -> temporal expert selected final prediction. The first intermediate is a CPU output substitution, not an actual rerun or attribution to a neural submodule; its ordering fixes interactions. Inherited-total plus current temporal-rerank delta exactly equals Ours-minus-Frozen. Current spatial update follows output sealing: its own-query causal effect is not measurable from the deployed output.

Report source-macro signed effects, 10000 source bootstrap conditional on existing orders, cell gross harm/rescue, threshold transitions using the existing strict vIoU>.3/.5 readouts, and explicit denominators. The most frequently wrong stage, gross damage and net improvement are distinct quantities. Conditional rates are descriptive repeated-cell rates, not independent sample confidence intervals.

For each scheduled arrival, evaluate all saved temporal candidate intervals with the same inherited boxes; compare expert selection to best candidate by tIoU and by vIoU separately. Report absent correct candidates, present-but-missed correct candidates, native-correct destruction, ties/continuous regret. Oracle selections are offline diagnostics only, not a new deployable method score.

Spatial rejection: code only skips update when no valid expert frames (reward=None) or zero gradient; there is no learned correctness rejection gate. Report skips and flat rewards. Old compact predictions and current v3 omit spatial candidate box tubes and post-update predictions: their GT ranking, rejected-correct spatial candidates, and attribution to a particular past update cannot be recovered without new inference. Do not fabricate them or infer correctness from decreasing pseudo loss. No extra GPU job is authorized by this CPU pass.

Follow-up after tuning: retain current v3 science/selection lock. Once tuned full-evaluation configuration/cohort is concretely locked and all predictions sealed, run the same diagnostic on that output and identify any remaining logging gaps before execution. This document does not define or start a new full benchmark. Current GT case inspection is development exposure and must not be fed into ongoing search.

Public output: code/protocol, anonymous scalar rows, aggregate diagnosis, representative cases by anonymous indices; no media, captions, labels, raw tubes, weights or state caches. Update research archive and publish/verify GitHub after completion.
