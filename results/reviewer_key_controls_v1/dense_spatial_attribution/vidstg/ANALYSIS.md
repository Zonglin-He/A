# Exact matched-time spatial vIoU attribution: vidstg

10303 original complete audited query paths. Independent audit recomputed pixel interpolation/IoU with a separate coordinate formula. Each six-bin sum reproduces original scored Ours-minus-interpolation vIoU within1e-12pp. No new predictions/model/Source/expert. Offline annotations only.

|Distance/support|Queries with support|Annotated dense frames inside prediction and GT event|Contribution to whole mean vIoU pp [95% CI]|
|---|---:|---:|---|
|observed|7243|22459|-0.010054 [-0.012220, -0.008023]|
|unobserved_0_05|7704|542118|-0.296012 [-0.393365, -0.202210]|
|unobserved_05_1|6756|314750|+0.048980 [+0.006730, +0.092438]|
|unobserved_1_2|4140|247594|+0.076440 [+0.050441, +0.103473]|
|unobserved_2_4|1997|170663|+0.051287 [+0.033409, +0.069274]|
|unobserved_4_inf|769|111138|+0.022403 [+0.007064, +0.039555]|

Sum = -0.106955368pp, the original full matched-time spatial comparison. Each bin contribution retains ALL queries and zero-support queries, unlike the conditional per-bin IoU table. Confidence intervals are parent10000pointwise intervals; do not sum CI endpoints. Dense GT annotations are real; only predicted boxes are interpolated as required by the official metric. Observed means exactly one of the8physical expert IDs, not a radius. This is an arithmetic decomposition of existing performance, not a causal intervention or evidence of action understanding. Full target data exposed, no model/default changes.
