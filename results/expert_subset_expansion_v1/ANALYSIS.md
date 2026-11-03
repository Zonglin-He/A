# Fixed128 expert replacement: all four experts

Same original128distinctparents/domain and eight original expert RGB frames/physical indices. One referring expert per branch, Source and current parameters fixed; no new Source/original ReferDINO/training. ReferFormer-R101 is a same-family capacity variant; MTTR is a distinct mask-output model with a declared mask-to-box/eight-frame adapter. Unequal pretraining/parameters/resolution/compute remain. Current uses unchanged soft observed candidates, not a hard-anchor correction.

|Dataset|Expert|Pure@Source t / v %|Current own t / v %|R.3 / R.5 %|Unobserved IoU %|
|---|---|---|---|---|---|
|hc2|referdino|46.509478 / 26.867332|54.968548 / 31.791298|50.781250 / 22.656250|54.637004|
|hc2|referformer_r50|46.509478 / 23.022299|54.769180 / 27.552349|43.750000 / 17.187500|48.970451|
|hc2|referformer_r101|46.509478 / 23.853887|55.450541 / 28.890920|44.531250 / 23.437500|49.883523|
|hc2|mttr|46.509478 / 17.776314|54.540914 / 22.030943|29.687500 / 12.500000|40.043675|
|vidstg|referdino|32.121535 / 13.896215|35.202435 / 15.339259|19.531250 / 10.156250|38.207575|
|vidstg|referformer_r50|32.121535 / 14.379322|34.603561 / 15.639079|20.312500 / 10.156250|38.507350|
|vidstg|referformer_r101|32.121535 / 15.246319|35.297444 / 16.713032|19.531250 / 11.718750|40.197912|
|vidstg|mttr|32.121535 / 12.469068|35.748597 / 13.276196|16.406250 / 7.031250|32.736169|

Each own-expert Current-minus-PureSource and fixed-own-event Current-minus-PureOwn comparison, all signed severe/rescue outcomes, short/long parent confidence intervals and actual expert/cached costs are in the per-expert reports. The newly computed pure baselines have nearest-expert tails; previous E-tail hybrid values remain preserved separately, not renamed. Source/previous Current scores are reused, not rescored. New-model masks, E geometry, actual ROI features and own posterior/interval/final DP were recomputed. No expert/setting chosen using target scores. All confidence intervals are pointwise paired-parent10000 bootstrap on exposed panels, not independent unknown-test validation or proof of expert independence.
