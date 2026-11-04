# Fixed-time candidate generation × decoding: vidstg

10303 complete original queries/732parent clusters; same8expert inputs, current E, native feature/geometry sufficient-state and immutable Full interval. No hard anchor clamp, temporal update, new Source/expert/GPU/model/optimizer or parameter selection.

|Spatial strategy|Transport|DP|vIoU|Event-unobserved IoU|ΔvIoU vs E, pp [95% CI]|ΔUnobs vs E, pp [95% CI]|
|---|---|---|---:|---:|---|---|
|Shared correction E only|no|no|13.948734|37.861669|0|0|
|E + expert anchors, independent|no|no|14.204946|38.424716|+0.256212 [+0.217539, +0.297811]|+0.563047 [+0.480845, +0.649280]|
|E + expert anchors + DP|no|yes|14.159635|38.321242|+0.210901 [+0.177589, +0.245977]|+0.459573 [+0.389732, +0.530596]|
|E + bidirectional candidates, independent|yes|no|13.997841|37.933547|+0.049107 [-0.097059, +0.198573]|+0.071879 [-0.332398, +0.478492]|
|Full spatial inference|yes|yes|14.558151|39.504891|+0.609417 [+0.471700, +0.755210]|+1.643222 [+1.271581, +2.031733]|

The no-transport intervention changes only unobserved candidate legality, preserving original E/expert node scores, edge scores, geometric support and all normalization denominators. It tests the added value of transported candidate boxes under common scoring, not removal of every computation used to prepare these scores. Independent selection is node-wise argmax with original first-index ties, not interpolation.

No-transport unobserved NATIVE outputs are exactly E. Its conditional DENSE unobserved IoU can still change because the official evaluator interpolates between sampled predictions, including changed expert-observation nodes. These effects do not indicate an unintended transported candidate.

Conditional Unobserved IoU: same annotated dense frames within predicted∩GT event, excluding exact8expert physical times. Query means first, parent10000pointwiseCI next. Empty-support queries remain null only for conditional metrics, all queries including zeros remain in vIoU. This is distinct from the original128 all-video native-frame Unobs column.

|Key controlled comparison|ΔvIoU, pp [95% CI]|ΔUnobs, pp [95% CI]|
|---|---|---|
|AnchorsIndependent-E|+0.256212 [+0.217539, +0.297811]|+0.563047 [+0.480845, +0.649280]|
|TransportIndependent-AnchorsIndependent|-0.207105 [-0.330563, -0.083147]|-0.491169 [-0.848391, -0.130958]|
|AnchorsDP-AnchorsIndependent|-0.045311 [-0.057619, -0.033339]|-0.103475 [-0.135384, -0.072943]|
|Full-TransportIndependent|+0.560311 [+0.512204, +0.612344]|+1.571343 [+1.432749, +1.715868]|
|Full-AnchorsDP|+0.398517 [+0.282519, +0.519225]|+1.183649 [+0.850966, +1.532856]|

|Nearest expert distance|Supported queries|Annotated dense unobserved frames|E IoU|Transport independent IoU|Full IoU|
|---|---:|---:|---:|---:|---:|
|0_0.5s|7704|542118|38.568939|39.551809|41.033657|
|0.5_1s|6756|314750|38.286167|37.194284|39.036374|
|1_2s|4140|247594|38.360968|36.506874|38.312751|
|2_4s|1997|170663|38.607169|36.831734|38.682622|
|ge4s|769|111138|38.367158|36.572787|38.639218|

All signed losses/rescues/severe tails and short/long outcomes are in ROOT_report.json and all_signed_scalars.tsv. Interaction is (Full−TransportIndependent)−(AnchorsDP−AnchorsIndependent), with paired query/parent CI. Positive/null/negative outcomes retain the frozen parameters. Full target sets were previously exposed; not unknown confirmation or familywise inference. These controls do not replace or remove separately preserved interpolation comparisons. No manuscript/Fig1 or original method/scientific hash change.
