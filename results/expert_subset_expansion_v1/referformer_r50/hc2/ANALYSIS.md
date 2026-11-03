# Expert subset: referformer_r50, hc2

128queries/128distinctparents, same original Source and eight physical expert frames; no target tuning/new Source or original ReferDINO. PureSource uses Source native event; PureOwn and Current share the expert-specific recomputed event. Pure is raw-eight center/logsize + nearest tails, no shared E tails. Current is unchanged soft-anchor implementation, not a hard-anchor fix.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|53.067325|
|PureSource|46.509478|23.022299|36.718750|13.281250|44.239530|
|PureOwn|54.769180|25.823182|41.406250|14.062500|44.239530|
|Current|54.769180|27.552349|43.750000|17.187500|48.970451|

All paired pointwise parent10000CIs, signed positives/negatives, severe/rescue union once and short<=5s/long groups retained. No claim of expert independence, equal compute/knowledge, unseen confirmation or familywise significance. Source/expert historical acquisition remains nonzero; request/forward/checkpoint/finish/query clocks nest, not uncached deployment. ReferFormer-R101 is a same-family backbone variation, not an independent method family. MTTR uses an explicit mask-to-box/eight-frame adapter; empty masks produce disclosed full-frame neutral boxes, no hidden Source or detector fallback.
