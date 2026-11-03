# Exact matched-time spatial vIoU attribution: hc2

1902 original complete audited query paths. Independent audit recomputed pixel interpolation/IoU with a separate coordinate formula. Each six-bin sum reproduces original scored Ours-minus-interpolation vIoU within1e-12pp. No new predictions/model/Source/expert. Offline annotations only.

|Distance/support|Queries with support|Annotated dense frames inside prediction and GT event|Contribution to whole mean vIoU pp [95% CI]|
|---|---:|---:|---|
|observed|1630|5036|-0.019056 [-0.022522, -0.015571]|
|unobserved_0_05|1672|115451|+0.057518 [-0.054573, +0.174260]|
|unobserved_05_1|1696|73639|+0.729569 [+0.623837, +0.839513]|
|unobserved_1_2|1536|58645|+1.158859 [+1.018391, +1.307637]|
|unobserved_2_4|456|6603|+0.207863 [+0.153749, +0.269332]|
|unobserved_4_inf|0|0|+0.000000 [+0.000000, +0.000000]|

Sum = +2.134753068pp, the original full matched-time spatial comparison. Each bin contribution retains ALL queries and zero-support queries, unlike the conditional per-bin IoU table. Confidence intervals are parent10000pointwise intervals; do not sum CI endpoints. Dense GT annotations are real; only predicted boxes are interpolated as required by the official metric. Observed means exactly one of the8physical expert IDs, not a radius. This is an arithmetic decomposition of existing performance, not a causal intervention or evidence of action understanding. Full target data exposed, no model/default changes.
