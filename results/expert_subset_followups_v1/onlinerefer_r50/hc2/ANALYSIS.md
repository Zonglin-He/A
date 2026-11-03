# Expert subset: onlinerefer_r50, hc2

128queries/128distinctparents, same original Source and eight physical expert frames; no target tuning/new Source or original ReferDINO. PureSource uses Source native event; PureOwn and Current share the expert-specific recomputed event. Pure is raw-eight center/logsize + nearest tails, no shared E tails. Current is unchanged soft-anchor implementation, not a hard-anchor fix.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|53.067325|
|PureSource|46.509478|22.920533|37.500000|10.937500|43.833675|
|PureOwn|54.124795|25.344176|39.062500|14.062500|43.833675|
|Current|54.124795|27.509405|42.968750|20.312500|49.559679|

All paired pointwise parent10000CIs, signed positives/negatives, severe/rescue union once and short<=5s/long groups retained. No claim of expert independence, equal compute/knowledge, unseen confirmation or familywise significance. Source/expert historical acquisition remains nonzero; request/forward/checkpoint/finish/query clocks nest, not uncached deployment. OnlineRefer uses eight native causal steps with original query/reference-state carry and direct boxhead. SgMg uses one native jointly selected refined-mask clip and an explicit halfopen mask-to-box adapter; empty masks become disclosed full-frame neutral boxes. Both use FP32/noAMP/noTF32; SgMg official example AMP is not enabled here. No extra frame, hidden Source fill, detector or optimizer. Existing R50/R101 remain same family.
