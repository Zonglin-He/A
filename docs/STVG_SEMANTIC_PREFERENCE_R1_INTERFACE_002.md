# R1 semantic preference: fixed8 interface feasibility

HC4/Vid4 first parents of the exposed32, slot0, same frozen weight-free E1 candidate pool (first4 legal existing centers), original4 context frames plus labelled full anchor. Ordinary and reverse arbitrary labels; 0 explicitly permits abstention. Frozen STVG-R1-7B/nativeBF16, exact original checkpoint and clock, no new model or Prompt search. This changes the scheduling of the R1 half only; Qwen download continues and its CPU controller was identity-checked before a bounded priority handoff.

Read actual full-vocabulary next-token log probabilities for single-token labels0..K. The reported posterior is conditional on the allowed label vocabulary. Allowed-label probability mass and full-vocabulary argmax are recorded separately, so normalization cannot hide failure to use the requested labels. No verbal confidence is used. These are preferences, not calibrated correctness probabilities.

| Domain | Parents | Forward calls | Mean order TV | Max order TV | Changed top choice | Averaged abstentions | Min allowed mass |
|---|---:|---:|---:|---:|---:|---:|---:|
|hc2|4|8|0.217671|0.345312|1|0|0.996871|
|vidstg|4|8|0.232685|0.401014|0|1|0.945593|

Engineering qualification passed: the native model can supply finite, aligned candidate/abstention probabilities. Label-order sensitivity remains and is retained; permutation averaging does not prove semantic correctness. Candidate coverage, identity/role quality and Student improvement are untested. No GT, free generation, Student fit, optimizer/response change or official fullset occurred. This report is not a completed2x2 factorial or a method promotion.

R1 output hashes, input/template/checkpoint and runtime are sealed privately. Raw query, model token logits, private anchor/context images, GT and weights are excluded. Anonymous scalar records include all eight parents and the abstention. Next: lock same-support geometric/preference Student conversion and same-information propagation; fixed-support weighted Gaussian NLL must not be described as on-policy resampling. Qwen half will reuse these R1 outputs after its own direct32 comparison; no repeated R1 request.
