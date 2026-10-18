# Original32 matched conditional-DINO content intervention

Question: does copying the highest native target-token-score query content,
instead of the query nearest the student's reference, improve conditional
feedback coverage and correction quality? Only content initialization changes.

The original exposed HC16/Vid16 roster, Source and E receipts, original four
positions and text, FP32 student references, model weights, native900 context,
attention mask, threshold .35, whole-trajectory legality, source scales and
final-MBR are fixed. No threshold search, new training or data expansion.
Use the exact 24 original fixed-rollout actions per active query, not a new seed
or feedback-dependent trajectory. Unsupported queries remain in the denominator.
The highest native slot is the first argmax of mean sigmoid target-token logits
from the same ordinary native forward. No target label or GT enters selection.
This is learned query content, not a new ROI encoder. Its score is not a calibrated
identity probability.

Both arms exclude the ORIGINAL geometry-nearest slot from added-query attention.
Native queries cannot read extras; extras cannot read each other. This identical
mask avoids changing attention support alongside content. Hash every other tensor
input, references, native content and mask and assert equality. Replay/clone
qualification includes boxes and target scores. The nearest arm must reproduce
the previous fixed24 outputs within 1e-5 numeric tolerance, and exact .35 admission.
This mask rule isolates the content intervention; it is not a separate claim
that this is the best deployment attention rule.

All32 matched outputs seal before new GT evaluation. GT is offline diagnosis
only, on the same known planned anchors. Missing labels are unknown, not failure.
Report raw and threshold-gated box IoU changes relative to identical input refs,
and query-level paired summaries (10000 bootstrap, seed20261018), including all
no-reference/no-feedback queries as zero correction. Multiple candidates from a
query are not independent samples. Coverage separately counts query/candidate/
anchor denominators, illegal refs and unsupported queries. No identity conclusion
from IoU alone. No Student/OPD fit in stage1.

Conditional resource decision, locked before outcomes: proceed to the existing
repaired OPD only if BOTH domains gain admitted-anchor coverage AND the target-
high threshold-gated correction improves mean known-anchor IoU over unchanged
references AND over nearest-content. Report uncertainty without requiring an
arbitrary significance threshold. Otherwise stop this interface variant, retain
negative/unknown results, do not tune thresholds or enlarge the roster.

If stage1 passes, use the SAME repaired mu <- mu+mean(y-a) estimator, E initial
mean, source scales, seed and 24 epsilon, fixed24 and dynamic3x8, no extra factors.
Recompute teacher responses at new dynamic states. Keep MBR and frozen weights,
seal all32 complete outputs before final GT, compare E, fixed and dynamic.
No second backbone, Qwen, SAM, critic training, QR/LN or historical Before replay.

Hard bounds: stage1 GPU900s, creation-to-prediction2400s, stage2 if qualified a
separate prelocked GPU900s/wall2400s maximum. MemoryHigh16GiB/Max20GiB/Swap1GiB,
shared GPU lease, host6GiB/GPU4GiB free/disk30GiB guards. First two original cases
one/domain are timed and qualified. Costs exclude existing Source/E construction.
Publish implementation, protocol, anonymous scalar results and limitations only;
raw media, query text/parent IDs, boxes/GT, weights and tensors stay private.
