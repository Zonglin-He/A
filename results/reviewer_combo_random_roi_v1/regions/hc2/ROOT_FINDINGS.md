# Root reviewer control: regions, hc2

Whole128queries/128parents sealed, independent official CPU score of ONLY the new arm(128checks,maxerror1.33226762955e-15), separate input/prediction/receipt/allcoldarray/reusedcontrol/allmean audit passed. Existing controls are reused unchanged, not rescored. Root independently rederived90 paired/group bootstrap statistics exactly. No new Source/expert/GPU/model/optimizer; inherited acquisition nonzero, cached CPU clocks not deployment.

|Arm|Expert inputs needed|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---|---:|---:|---:|---:|---:|
|Source|no|46.509478|25.124253|42.968750|14.062500|53.067325|
|SourceROI|yes|54.833053|30.470815|51.562500|24.218750|53.413953|
|CorrectedROI|yes|54.968548|30.453799|50.000000|23.437500|53.413953|
|GlobalAppearance|yes|54.479768|30.045855|51.562500|21.875000|53.413953|
|BackgroundAppearance|yes|53.499251|29.179448|49.218750|20.312500|53.413953|
|ShuffledBoxes|yes|53.563567|29.910041|50.000000|21.093750|53.413953|
|RandomRegion|yes|52.658147|29.098598|49.218750|18.750000|53.413953|

All region arms use the SAME expert-derived E spatial path, SAME q0/Source grids and unchanged pooling/centering/normalization/prototype update/readout. Only appearance pooling support changes; own event intervals recomputed. Thus this is a temporal-evidence diagnostic, not an expert-free pipeline comparison. Global pools all valid cells, background uses valid complement of original expanded target ROI with prespecified whole-valid fallback, and shuffled boxes are the earlier whole-box temporal permutation. RandomRegion is NEW: per-frame translation of the exact original expanded1.5ROI fractional weight template to a SHA-selected feasible valid-grid location. Per-frame support shape/area/weight multiset bitwise retained, overlap and identity allowed. No caption/GT/predicted event/prototype evidence/target outcomes select locations. Full private manifest was committed before new labels.

Actual random intervention:8320nativeframes,1576identity placements,0queries with unchanged pooled features,22target-global-fallback frames; all stay in denominator. Mean support overlap0.524866, feasible placements range1..70. Grid coordinates and raw features remain private, scalar diagnostics/manifestSHA public.

## All signed paired evidence

### all

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|CorrectedROI-SourceROI|v|-0.017017|[-0.232810,+0.217617]|128 / 128|
|CorrectedROI-SourceROI|t|0.135495|[-0.479292,+0.770855]|128 / 128|
|CorrectedROI-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|126 / 126|
|CorrectedROI-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-SourceROI|boundary_MAE_seconds|0.035059|[-0.034381,+0.118238]|128 / 128|
|CorrectedROI-GlobalAppearance|v|0.407944|[-0.223229,+1.084151]|128 / 128|
|CorrectedROI-GlobalAppearance|t|0.488781|[-0.705089,+1.671980]|128 / 128|
|CorrectedROI-GlobalAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|126 / 126|
|CorrectedROI-GlobalAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-GlobalAppearance|boundary_MAE_seconds|-0.077910|[-0.171872,+0.015704]|128 / 128|
|CorrectedROI-BackgroundAppearance|v|1.274351|[+0.191013,+2.523046]|128 / 128|
|CorrectedROI-BackgroundAppearance|t|1.469298|[-0.168165,+3.233836]|128 / 128|
|CorrectedROI-BackgroundAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|126 / 126|
|CorrectedROI-BackgroundAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-BackgroundAppearance|boundary_MAE_seconds|-0.136379|[-0.259630,-0.023108]|128 / 128|
|CorrectedROI-ShuffledBoxes|v|0.543757|[-0.125456,+1.230776]|128 / 128|
|CorrectedROI-ShuffledBoxes|t|1.404981|[+0.174056,+2.660755]|128 / 128|
|CorrectedROI-ShuffledBoxes|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|126 / 126|
|CorrectedROI-ShuffledBoxes|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-ShuffledBoxes|boundary_MAE_seconds|-0.079549|[-0.183143,+0.017290]|128 / 128|
|CorrectedROI-RandomRegion|v|1.355201|[+0.290359,+2.697710]|128 / 128|
|CorrectedROI-RandomRegion|t|2.310402|[+0.693279,+4.190258]|128 / 128|
|CorrectedROI-RandomRegion|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|126 / 126|
|CorrectedROI-RandomRegion|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-RandomRegion|boundary_MAE_seconds|-0.178581|[-0.318969,-0.056401]|128 / 128|
|RandomRegion-SourceROI|v|-1.372218|[-2.698648,-0.315683]|128 / 128|
|RandomRegion-SourceROI|t|-2.174906|[-4.048834,-0.510169]|128 / 128|
|RandomRegion-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|126 / 126|
|RandomRegion-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|RandomRegion-SourceROI|boundary_MAE_seconds|0.213640|[+0.072692,+0.369386]|128 / 128|
### short_le5s

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|CorrectedROI-SourceROI|v|0.102002|[-0.109272,+0.395107]|42 / 42|
|CorrectedROI-SourceROI|t|0.478163|[-0.649461,+1.856214]|42 / 42|
|CorrectedROI-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|40 / 40|
|CorrectedROI-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|42 / 42|
|CorrectedROI-SourceROI|boundary_MAE_seconds|0.109473|[-0.067845,+0.344743]|42 / 42|
|CorrectedROI-GlobalAppearance|v|0.032531|[-0.630512,+0.646460]|42 / 42|
|CorrectedROI-GlobalAppearance|t|0.424168|[-1.099598,+1.944006]|42 / 42|
|CorrectedROI-GlobalAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|40 / 40|
|CorrectedROI-GlobalAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|42 / 42|
|CorrectedROI-GlobalAppearance|boundary_MAE_seconds|-0.045703|[-0.199419,+0.111305]|42 / 42|
|CorrectedROI-BackgroundAppearance|v|0.361142|[-0.912568,+1.631444]|42 / 42|
|CorrectedROI-BackgroundAppearance|t|0.753400|[-1.445431,+3.026843]|42 / 42|
|CorrectedROI-BackgroundAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|40 / 40|
|CorrectedROI-BackgroundAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|42 / 42|
|CorrectedROI-BackgroundAppearance|boundary_MAE_seconds|-0.108895|[-0.276813,+0.062356]|42 / 42|
|CorrectedROI-ShuffledBoxes|v|-0.036807|[-0.693738,+0.694570]|42 / 42|
|CorrectedROI-ShuffledBoxes|t|0.369322|[-1.212717,+2.139971]|42 / 42|
|CorrectedROI-ShuffledBoxes|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|40 / 40|
|CorrectedROI-ShuffledBoxes|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|42 / 42|
|CorrectedROI-ShuffledBoxes|boundary_MAE_seconds|0.002540|[-0.144968,+0.147221]|42 / 42|
|CorrectedROI-RandomRegion|v|0.550103|[-0.272174,+1.479686]|42 / 42|
|CorrectedROI-RandomRegion|t|0.858678|[-0.467547,+2.251091]|42 / 42|
|CorrectedROI-RandomRegion|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|40 / 40|
|CorrectedROI-RandomRegion|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|42 / 42|
|CorrectedROI-RandomRegion|boundary_MAE_seconds|-0.057747|[-0.167454,+0.057485]|42 / 42|
|RandomRegion-SourceROI|v|-0.448101|[-1.358193,+0.360632]|42 / 42|
|RandomRegion-SourceROI|t|-0.380515|[-2.029213,+1.341872]|42 / 42|
|RandomRegion-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|40 / 40|
|RandomRegion-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|42 / 42|
|RandomRegion-SourceROI|boundary_MAE_seconds|0.167220|[-0.050121,+0.448991]|42 / 42|
### long_gt5s

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|CorrectedROI-SourceROI|v|-0.075142|[-0.366329,+0.246398]|86 / 86|
|CorrectedROI-SourceROI|t|-0.031854|[-0.697191,+0.683294]|86 / 86|
|CorrectedROI-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-SourceROI|boundary_MAE_seconds|-0.001283|[-0.055103,+0.046826]|86 / 86|
|CorrectedROI-GlobalAppearance|v|0.591285|[-0.300759,+1.520682]|86 / 86|
|CorrectedROI-GlobalAppearance|t|0.520335|[-1.106335,+2.151661]|86 / 86|
|CorrectedROI-GlobalAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-GlobalAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-GlobalAppearance|boundary_MAE_seconds|-0.093639|[-0.214916,+0.020175]|86 / 86|
|CorrectedROI-BackgroundAppearance|v|1.720337|[+0.255382,+3.405631]|86 / 86|
|CorrectedROI-BackgroundAppearance|t|1.818922|[-0.364509,+4.184855]|86 / 86|
|CorrectedROI-BackgroundAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-BackgroundAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-BackgroundAppearance|boundary_MAE_seconds|-0.149801|[-0.312096,+0.001885]|86 / 86|
|CorrectedROI-ShuffledBoxes|v|0.827288|[-0.122309,+1.761063]|86 / 86|
|CorrectedROI-ShuffledBoxes|t|1.910768|[+0.301282,+3.536555]|86 / 86|
|CorrectedROI-ShuffledBoxes|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-ShuffledBoxes|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-ShuffledBoxes|boundary_MAE_seconds|-0.119638|[-0.249817,+0.002198]|86 / 86|
|CorrectedROI-RandomRegion|v|1.748389|[+0.222273,+3.707892]|86 / 86|
|CorrectedROI-RandomRegion|t|3.019383|[+0.665050,+5.624794]|86 / 86|
|CorrectedROI-RandomRegion|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-RandomRegion|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|CorrectedROI-RandomRegion|boundary_MAE_seconds|-0.237593|[-0.436874,-0.060096]|86 / 86|
|RandomRegion-SourceROI|v|-1.823531|[-3.746951,-0.309850]|86 / 86|
|RandomRegion-SourceROI|t|-3.051237|[-5.691124,-0.681892]|86 / 86|
|RandomRegion-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|RandomRegion-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|86 / 86|
|RandomRegion-SourceROI|boundary_MAE_seconds|0.236311|[+0.057890,+0.432737]|86 / 86|

## Severe harm and rescue (strictly over5pp, union once)

|Pair|v gain/harm/tie|Severe t/v/union|Rescue union|
|---|---|---|---:|
|CorrectedROI-SourceROI|11/17/100|8/1/8|9|
|CorrectedROI-GlobalAppearance|38/32/58|19/8/19|25|
|CorrectedROI-BackgroundAppearance|47/37/44|25/11/25|34|
|CorrectedROI-ShuffledBoxes|41/25/62|19/6/19|25|
|CorrectedROI-RandomRegion|47/30/51|16/8/16|30|
|RandomRegion-SourceROI|27/50/51|33/15/33|16|

All intervals pointwise10000paired-parent bootstrap, queryequal metric means, not independent-frame/familywise/unknown-confirmation/equivalence claims. HC full uses1902queries/227parents and Vid10303/732; regional128parent panels are previously exposed. <=5s/long GT groups are post-seal analysis only; missing observations not0. All observed-frame metrics include all GT-supported native observed frames, including outside predicted events; do not equate every observed loss with tube-inside harm. Current observed-anchor contract remains unchanged. No method selection from results, no claim of independent action-semantic recognition even if local evidence wins.
