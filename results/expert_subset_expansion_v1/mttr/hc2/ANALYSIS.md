# Expert subset: mttr, hc2

128queries/128distinctparents, same original Source and eight physical expert frames; no target tuning/new Source or original ReferDINO. PureSource uses Source native event; PureOwn and Current share the expert-specific recomputed event. Pure is raw-eight center/logsize + nearest tails, no shared E tails. Current is unchanged soft-anchor implementation, not a hard-anchor fix.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|53.067325|
|PureSource|46.509478|17.776314|20.312500|6.250000|34.347638|
|PureOwn|54.540914|20.357462|24.218750|7.031250|34.347638|
|Current|54.540914|22.030943|29.687500|12.500000|40.043675|

All paired pointwise parent10000CIs, signed positives/negatives, severe/rescue union once and short<=5s/long groups retained. No claim of expert independence, equal compute/knowledge, unseen confirmation or familywise significance. Source/expert historical acquisition remains nonzero; request/forward/checkpoint/finish/query clocks nest, not uncached deployment. ReferFormer-R101 is a same-family backbone variation, not an independent method family. MTTR uses an explicit mask-to-box/eight-frame adapter; empty masks produce disclosed full-frame neutral boxes, no hidden Source or detector fallback.
