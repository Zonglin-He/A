# Expert subset: referdino, vidstg

128queries/128distinctparents, same original Source and eight physical expert frames; no target tuning/new Source or original ReferDINO. PureSource uses Source native event; PureOwn and Current share the expert-specific recomputed event. Pure is raw-eight center/logsize + nearest tails, no shared E tails. Current is unchanged soft-anchor implementation, not a hard-anchor fix.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|36.412936|
|PureSource|32.121535|13.896215|17.187500|7.812500|35.917760|
|PureOwn|35.202435|14.912114|17.968750|8.593750|35.917760|
|Current|35.202435|15.339259|19.531250|10.156250|38.207575|

All paired pointwise parent10000CIs, signed positives/negatives, severe/rescue union once and short<=5s/long groups retained. No claim of expert independence, equal compute/knowledge, unseen confirmation or familywise significance. Source/expert historical acquisition remains nonzero; request/forward/checkpoint/finish/query clocks nest, not uncached deployment. ReferFormer-R101 is a same-family backbone variation, not an independent method family. MTTR uses an explicit mask-to-box/eight-frame adapter; empty masks produce disclosed full-frame neutral boxes, no hidden Source or detector fallback.
