# Fixed-caption pairwise revision: decision and limits

Original exposed development256, HC128/Vid128; same old512 R1 captions, two frozen text judges, both display orders, no new video/caption generation. This finite experiment is complete; CURRENT/PAPER unchanged.

|Method / vIoU %|HC|VidSTG|
|---|---:|---:|
|Source marginal MAP, no update|24.945094|10.299485|
|Old independent R1 B3|24.832876|10.290674|
|Pairwise R1, actual Student|24.897626|10.143823|
|Pairwise Qwen, actual Student|24.859644|10.228260|

The direct pairwise revision does not establish an incremental adaptation benefit over the same-decoder no-update comparator: R1 -0.047468/-0.155662pp, Qwen -0.085449/-0.071225pp. R1 improves the HC point mean over its old independent judge (+0.064749pp) but loses Vid (-0.146851pp); Qwen also does not establish a replacement. Paired intervals cross zero; this alone is not the decision rule. Negative observed task increments, imperfect feedback direction, raw-feedback hard-readout harm and the finite single-revision budget jointly motivate stopping small text-judge tweaks. This is not a universal rejection of language supervision.

On matched GT interval-order supports: HC old53/110, pairwise R1 and Qwen59/110; Vid old41/95 vs R1 46/95, and old41/94 vs Qwen46/94. These are modest localization-order proxy changes, not identity/role correctness or a guarantee of positive mean localization effect. Full denominator, all ties and invalids remain.

Actual FP32 conditional movement follows feedback in R1 243/243 and Qwen242/242 informative cases. Candidate KL falls and median target-logodds completion is about0.84-0.85. Thus this cannot be explained by a universally inactive/incorrect parameter-update interface. Nor are all fits certified converged: the original50-step/Armijo/gradient-tolerance rules remain and final norms/terminations are retained.

Raw reward hard choice remains below Source marginal: HC23.491004/23.735125 for R1/Qwen versus24.945094; Vid10.253575/10.173439 versus10.299485. Prior-tilted hard choice and actual Student are separate measured readouts, none an upper bound. Ranking agreement counts cannot substitute for effect magnitudes or the marginal no-update comparator.

Pairwise judges often assign both-insufficient: mean R1 HC0.731508/Vid0.483122, Qwen0.963235/0.827641. These uncalibrated posteriors do not prove the captions objectively lack information. Top-label order agreement is high in some cells because D dominates; the smaller A-vs-B directional margin still flips under display reversal in R1 HC67/127 Vid56/121 and Qwen61/127 64/121. Both mapped orders were averaged as prelocked; no order disagreement filter or confidence threshold was introduced.

Next unresolved question is whether the fixed two video hypotheses and their existing descriptions contain different, sufficiently informative event evidence. This round does not launch new hypothesis construction, new captions, prompt search, stronger updates, Actionness or a new backbone. Existing512-window semantic review remains44 model-assisted/468pending; no new semantic labels are claimed. No independent confirmation, production promotion or fullset is inferred from development256.

Readout engineering recovery: original49 prediction files retained exact, remaining207 only resumed, same deadline. Two known retired Source prefixes were replayed from original RGB and checked against sealed Source pixels/records/logits/hidden/boxes. Teacher and optimizer were not retried; original failure and costs remain in COST/READOUT_RECOVERY.
