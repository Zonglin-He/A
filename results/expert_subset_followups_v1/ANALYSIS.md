# Six expert configurations on the same exposed128parent panels

Current is the unchanged soft E/expert observed-candidate implementation; this is not a hard-anchor fix. ReferFormerR50/R101 are one family. OnlineRefer carries native state over only eight sparse observations; SgMg has the declared joint-eight refined-mask-to-box adapter. No equal knowledge/compute or expert independence claim. All old expert scores are reused, not rescored.

|Dataset|Expert|Pure@Source t / v %|Pure@Own v %|Current t / v %|R.3 / R.5 %|Unobserved IoU %|
|---|---|---|---:|---|---|---:|
|hc2|referdino|46.509478 / 26.867332|29.346216|54.968548 / 31.791298|50.781250 / 22.656250|54.637004|
|hc2|referformer_r50|46.509478 / 23.022299|25.823182|54.769180 / 27.552349|43.750000 / 17.187500|48.970451|
|hc2|referformer_r101|46.509478 / 23.853887|27.213483|55.450541 / 28.890920|44.531250 / 23.437500|49.883523|
|hc2|mttr|46.509478 / 17.776314|20.357462|54.540914 / 22.030943|29.687500 / 12.500000|40.043675|
|hc2|onlinerefer_r50|46.509478 / 22.920533|25.344176|54.124795 / 27.509405|42.968750 / 20.312500|49.559679|
|hc2|sgmg_videoswin_t|46.509478 / 26.884454|29.904228|54.482223 / 31.495791|50.000000 / 25.000000|53.974782|
|vidstg|referdino|32.121535 / 13.896215|14.912114|35.202435 / 15.339259|19.531250 / 10.156250|38.207575|
|vidstg|referformer_r50|32.121535 / 14.379322|15.488434|34.603561 / 15.639079|20.312500 / 10.156250|38.507350|
|vidstg|referformer_r101|32.121535 / 15.246319|16.643761|35.297444 / 16.713032|19.531250 / 11.718750|40.197912|
|vidstg|mttr|32.121535 / 12.469068|13.456923|35.748597 / 13.276196|16.406250 / 7.031250|32.736169|
|vidstg|onlinerefer_r50|32.121535 / 15.272925|16.287149|35.129945 / 16.708918|22.656250 / 10.156250|39.237880|
|vidstg|sgmg_videoswin_t|32.121535 / 14.373415|15.323081|35.656312 / 15.673441|21.875000 / 8.593750|36.781833|

Current-minus-PureOwn isolates spatial strategy within each expert own interval; across experts that interval can change. Current-minus-PureSource includes temporal and spatial changes. Gain differences across experts are descriptive paired effects, not a causal isolation of expert quality. All signed gains/harms, severe/rescue union once, short<=5s/long groups, source-relative negatives and pointwise10000parent bootstrap intervals remain in each report; no target tuning/unknown-test/familywise/equivalence claim.

Actual OnlineRefer2048native causal steps and SgMg256joint-eight calls are512logical expert queries, not512equal-cost forward steps. Source/original ReferDINO/optimizer new calls0; inherited acquisition remains nonzero. Load once per actual PID/start, query/forward/checkpoint/finish costs nest and cannot be added or called uncached deployment. OnlineRefer step checkpoint writes lie within its measured forward window. All private raw masks/features/media/captions/coordinates/weights stay private.
