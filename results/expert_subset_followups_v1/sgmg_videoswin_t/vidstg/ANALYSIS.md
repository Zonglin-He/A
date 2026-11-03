# Expert subset: sgmg_videoswin_t, vidstg

128queries/128distinctparents, same original Source and eight physical expert frames; no target tuning/new Source or original ReferDINO. PureSource uses Source native event; PureOwn and Current share the expert-specific recomputed event. Pure is raw-eight center/logsize + nearest tails, no shared E tails. Current is unchanged soft-anchor implementation, not a hard-anchor fix.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|36.412936|
|PureSource|32.121535|14.373415|19.531250|8.593750|35.522242|
|PureOwn|35.656312|15.323081|22.656250|9.375000|35.522242|
|Current|35.656312|15.673441|21.875000|8.593750|36.781833|

All paired pointwise parent10000CIs, signed positives/negatives, severe/rescue union once and short<=5s/long groups retained. No claim of expert independence, equal compute/knowledge, unseen confirmation or familywise significance. Source/expert historical acquisition remains nonzero; request/forward/checkpoint/finish/query clocks nest, not uncached deployment. OnlineRefer uses eight native causal steps with original query/reference-state carry and direct boxhead. SgMg uses one native jointly selected refined-mask clip and an explicit halfopen mask-to-box adapter; empty masks become disclosed full-frame neutral boxes. Both use FP32/noAMP/noTF32; SgMg official example AMP is not enabled here. No extra frame, hidden Source fill, detector or optimizer. Existing R50/R101 remain same family.
