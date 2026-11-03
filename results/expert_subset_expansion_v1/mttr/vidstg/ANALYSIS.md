# Expert subset: mttr, vidstg

128queries/128distinctparents, same original Source and eight physical expert frames; no target tuning/new Source or original ReferDINO. PureSource uses Source native event; PureOwn and Current share the expert-specific recomputed event. Pure is raw-eight center/logsize + nearest tails, no shared E tails. Current is unchanged soft-anchor implementation, not a hard-anchor fix.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|36.412936|
|PureSource|32.121535|12.469068|16.406250|6.250000|32.799300|
|PureOwn|35.748597|13.456923|17.968750|6.250000|32.799300|
|Current|35.748597|13.276196|16.406250|7.031250|32.736169|

All paired pointwise parent10000CIs, signed positives/negatives, severe/rescue union once and short<=5s/long groups retained. No claim of expert independence, equal compute/knowledge, unseen confirmation or familywise significance. Source/expert historical acquisition remains nonzero; request/forward/checkpoint/finish/query clocks nest, not uncached deployment. ReferFormer-R101 is a same-family backbone variation, not an independent method family. MTTR uses an explicit mask-to-box/eight-frame adapter; empty masks produce disclosed full-frame neutral boxes, no hidden Source or detector fallback.
