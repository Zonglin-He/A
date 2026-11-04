# Fixed-time candidate generation × decoding: hc2

1902 complete original queries/227parent clusters; same8expert inputs, current E, native feature/geometry sufficient-state and immutable Full interval. No hard anchor clamp, temporal update, new Source/expert/GPU/model/optimizer or parameter selection.

|Spatial strategy|Transport|DP|vIoU|Event-unobserved IoU|ΔvIoU vs E, pp [95% CI]|ΔUnobs vs E, pp [95% CI]|
|---|---|---|---:|---:|---|---|
|Shared correction E only|no|no|26.489749|52.761593|0|0|
|E + expert anchors, independent|no|no|26.755059|53.205638|+0.265310 [+0.181578, +0.348559]|+0.444045 [+0.300091, +0.586243]|
|E + expert anchors + DP|no|yes|26.626652|52.987479|+0.136903 [+0.055663, +0.219755]|+0.225886 [+0.083626, +0.365420]|
|E + bidirectional candidates, independent|yes|no|26.281532|52.109389|-0.208217 [-0.575857, +0.160342]|-0.652205 [-1.314105, +0.024191]|
|Full spatial inference|yes|yes|27.349044|54.354908|+0.859295 [+0.502377, +1.226670]|+1.593315 [+0.941953, +2.270148]|

The no-transport intervention changes only unobserved candidate legality, preserving original E/expert node scores, edge scores, geometric support and all normalization denominators. It tests the added value of transported candidate boxes under common scoring, not removal of every computation used to prepare these scores. Independent selection is node-wise argmax with original first-index ties, not interpolation.

No-transport unobserved NATIVE outputs are exactly E. Its conditional DENSE unobserved IoU can still change because the official evaluator interpolates between sampled predictions, including changed expert-observation nodes. These effects do not indicate an unintended transported candidate.

Conditional Unobserved IoU: same annotated dense frames within predicted∩GT event, excluding exact8expert physical times. Query means first, parent10000pointwiseCI next. Empty-support queries remain null only for conditional metrics, all queries including zeros remain in vIoU. This is distinct from the original128 all-video native-frame Unobs column.

|Key controlled comparison|ΔvIoU, pp [95% CI]|ΔUnobs, pp [95% CI]|
|---|---|---|
|AnchorsIndependent-E|+0.265310 [+0.181578, +0.348559]|+0.444045 [+0.300091, +0.586243]|
|TransportIndependent-AnchorsIndependent|-0.473526 [-0.785602, -0.158172]|-1.096250 [-1.682648, -0.502822]|
|AnchorsDP-AnchorsIndependent|-0.128407 [-0.151004, -0.106752]|-0.218159 [-0.255499, -0.183273]|
|Full-TransportIndependent|+1.067512 [+0.958161, +1.183652]|+2.245519 [+2.020370, +2.484704]|
|Full-AnchorsDP|+0.722392 [+0.421658, +1.037573]|+1.367429 [+0.791736, +1.965449]|

|Nearest expert distance|Supported queries|Annotated dense unobserved frames|E IoU|Transport independent IoU|Full IoU|
|---|---:|---:|---:|---:|---:|
|0_0.5s|1672|115451|53.303999|54.511259|56.322909|
|0.5_1s|1696|73639|52.753986|51.064031|53.937989|
|1_2s|1536|58645|51.728555|49.281774|51.989688|
|2_4s|456|6603|51.579288|47.778355|50.579201|
|ge4s|0|0|unscorable|unscorable|unscorable|

All signed losses/rescues/severe tails and short/long outcomes are in ROOT_report.json and all_signed_scalars.tsv. Interaction is (Full−TransportIndependent)−(AnchorsDP−AnchorsIndependent), with paired query/parent CI. Positive/null/negative outcomes retain the frozen parameters. Full target sets were previously exposed; not unknown confirmation or familywise inference. These controls do not replace or remove separately preserved interpolation comparisons. No manuscript/Fig1 or original method/scientific hash change.
