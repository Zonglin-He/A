# Root reviewer control: regions, vidstg

Whole128queries/128parents sealed, independent official CPU score of ONLY the new arm(128checks,maxerror1.44328993201e-15), separate input/prediction/receipt/allcoldarray/reusedcontrol/allmean audit passed. Existing controls are reused unchanged, not rescored. Root independently rederived90 paired/group bootstrap statistics exactly. No new Source/expert/GPU/model/optimizer; inherited acquisition nonzero, cached CPU clocks not deployment.

|Arm|Expert inputs needed|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---|---:|---:|---:|---:|---:|
|Source|no|32.121535|13.322903|16.406250|3.906250|36.412936|
|SourceROI|yes|35.528603|14.912319|19.531250|10.156250|37.597649|
|CorrectedROI|yes|35.202435|14.815209|18.750000|10.156250|37.597649|
|GlobalAppearance|yes|35.560990|14.995460|19.531250|10.156250|37.597649|
|BackgroundAppearance|yes|35.070563|14.666547|20.312500|10.156250|37.597649|
|ShuffledBoxes|yes|35.675272|14.905602|20.312500|10.156250|37.597649|
|RandomRegion|yes|35.515154|14.836525|20.312500|9.375000|37.597649|

All region arms use the SAME expert-derived E spatial path, SAME q0/Source grids and unchanged pooling/centering/normalization/prototype update/readout. Only appearance pooling support changes; own event intervals recomputed. Thus this is a temporal-evidence diagnostic, not an expert-free pipeline comparison. Global pools all valid cells, background uses valid complement of original expanded target ROI with prespecified whole-valid fallback, and shuffled boxes are the earlier whole-box temporal permutation. RandomRegion is NEW: per-frame translation of the exact original expanded1.5ROI fractional weight template to a SHA-selected feasible valid-grid location. Per-frame support shape/area/weight multiset bitwise retained, overlap and identity allowed. No caption/GT/predicted event/prototype evidence/target outcomes select locations. Full private manifest was committed before new labels.

Actual random intervention:13502nativeframes,3507identity placements,7queries with unchanged pooled features,386target-global-fallback frames; all stay in denominator. Mean support overlap0.523905, feasible placements range1..70. Grid coordinates and raw features remain private, scalar diagnostics/manifestSHA public.

## All signed paired evidence

### all

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|CorrectedROI-SourceROI|v|-0.097110|[-0.355101,+0.149646]|128 / 128|
|CorrectedROI-SourceROI|t|-0.326168|[-1.025934,+0.461678]|128 / 128|
|CorrectedROI-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|103 / 103|
|CorrectedROI-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-SourceROI|boundary_MAE_seconds|0.014075|[-0.152053,+0.181120]|128 / 128|
|CorrectedROI-GlobalAppearance|v|-0.180251|[-0.528901,+0.121262]|128 / 128|
|CorrectedROI-GlobalAppearance|t|-0.358556|[-1.334083,+0.685216]|128 / 128|
|CorrectedROI-GlobalAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|103 / 103|
|CorrectedROI-GlobalAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-GlobalAppearance|boundary_MAE_seconds|0.015920|[-0.240014,+0.268299]|128 / 128|
|CorrectedROI-BackgroundAppearance|v|0.148662|[-0.356672,+0.617729]|128 / 128|
|CorrectedROI-BackgroundAppearance|t|0.131871|[-1.072934,+1.385601]|128 / 128|
|CorrectedROI-BackgroundAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|103 / 103|
|CorrectedROI-BackgroundAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-BackgroundAppearance|boundary_MAE_seconds|-0.044388|[-0.357859,+0.251641]|128 / 128|
|CorrectedROI-ShuffledBoxes|v|-0.090393|[-0.381584,+0.197301]|128 / 128|
|CorrectedROI-ShuffledBoxes|t|-0.472838|[-1.148537,+0.194076]|128 / 128|
|CorrectedROI-ShuffledBoxes|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|103 / 103|
|CorrectedROI-ShuffledBoxes|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-ShuffledBoxes|boundary_MAE_seconds|0.014767|[-0.135588,+0.174307]|128 / 128|
|CorrectedROI-RandomRegion|v|-0.021316|[-0.508807,+0.483006]|128 / 128|
|CorrectedROI-RandomRegion|t|-0.312720|[-1.447338,+0.889676]|128 / 128|
|CorrectedROI-RandomRegion|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|103 / 103|
|CorrectedROI-RandomRegion|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|CorrectedROI-RandomRegion|boundary_MAE_seconds|-0.141785|[-0.508405,+0.170858]|128 / 128|
|RandomRegion-SourceROI|v|-0.075795|[-0.676155,+0.469819]|128 / 128|
|RandomRegion-SourceROI|t|-0.013448|[-1.270172,+1.182705]|128 / 128|
|RandomRegion-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|103 / 103|
|RandomRegion-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|128 / 128|
|RandomRegion-SourceROI|boundary_MAE_seconds|0.155860|[-0.161888,+0.540020]|128 / 128|
### short_le5s

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|CorrectedROI-SourceROI|v|0.148539|[-0.106014,+0.414080]|78 / 78|
|CorrectedROI-SourceROI|t|0.092551|[-0.713344,+1.082189]|78 / 78|
|CorrectedROI-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|54 / 54|
|CorrectedROI-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|78 / 78|
|CorrectedROI-SourceROI|boundary_MAE_seconds|-0.016041|[-0.274121,+0.244582]|78 / 78|
|CorrectedROI-GlobalAppearance|v|-0.186388|[-0.692920,+0.224523]|78 / 78|
|CorrectedROI-GlobalAppearance|t|-0.311756|[-1.244755,+0.610226]|78 / 78|
|CorrectedROI-GlobalAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|54 / 54|
|CorrectedROI-GlobalAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|78 / 78|
|CorrectedROI-GlobalAppearance|boundary_MAE_seconds|0.109651|[-0.194396,+0.457897]|78 / 78|
|CorrectedROI-BackgroundAppearance|v|0.321512|[-0.279227,+0.863711]|78 / 78|
|CorrectedROI-BackgroundAppearance|t|0.615448|[-0.601202,+1.943819]|78 / 78|
|CorrectedROI-BackgroundAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|54 / 54|
|CorrectedROI-BackgroundAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|78 / 78|
|CorrectedROI-BackgroundAppearance|boundary_MAE_seconds|0.039817|[-0.369741,+0.455253]|78 / 78|
|CorrectedROI-ShuffledBoxes|v|0.078779|[-0.211941,+0.410684]|78 / 78|
|CorrectedROI-ShuffledBoxes|t|-0.156936|[-0.956007,+0.592992]|78 / 78|
|CorrectedROI-ShuffledBoxes|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|54 / 54|
|CorrectedROI-ShuffledBoxes|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|78 / 78|
|CorrectedROI-ShuffledBoxes|boundary_MAE_seconds|-0.001052|[-0.214516,+0.237031]|78 / 78|
|CorrectedROI-RandomRegion|v|-0.064962|[-0.495152,+0.359647]|78 / 78|
|CorrectedROI-RandomRegion|t|-0.472369|[-1.485805,+0.506519]|78 / 78|
|CorrectedROI-RandomRegion|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|54 / 54|
|CorrectedROI-RandomRegion|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|78 / 78|
|CorrectedROI-RandomRegion|boundary_MAE_seconds|0.015133|[-0.299231,+0.364985]|78 / 78|
|RandomRegion-SourceROI|v|0.213501|[-0.250007,+0.670765]|78 / 78|
|RandomRegion-SourceROI|t|0.564920|[-0.534631,+1.701937]|78 / 78|
|RandomRegion-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|54 / 54|
|RandomRegion-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|78 / 78|
|RandomRegion-SourceROI|boundary_MAE_seconds|-0.031174|[-0.395869,+0.318903]|78 / 78|
### long_gt5s

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|CorrectedROI-SourceROI|v|-0.480323|[-0.970674,-0.046474]|50 / 50|
|CorrectedROI-SourceROI|t|-0.979369|[-2.157619,+0.182405]|50 / 50|
|CorrectedROI-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|49 / 49|
|CorrectedROI-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|50 / 50|
|CorrectedROI-SourceROI|boundary_MAE_seconds|0.061055|[-0.085317,+0.186667]|50 / 50|
|CorrectedROI-GlobalAppearance|v|-0.170678|[-0.607280,+0.232301]|50 / 50|
|CorrectedROI-GlobalAppearance|t|-0.431564|[-2.455016,+1.773618]|50 / 50|
|CorrectedROI-GlobalAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|49 / 49|
|CorrectedROI-GlobalAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|50 / 50|
|CorrectedROI-GlobalAppearance|boundary_MAE_seconds|-0.130301|[-0.604026,+0.194593]|50 / 50|
|CorrectedROI-BackgroundAppearance|v|-0.120983|[-1.003694,+0.760562]|50 / 50|
|CorrectedROI-BackgroundAppearance|t|-0.622508|[-3.019623,+1.855897]|50 / 50|
|CorrectedROI-BackgroundAppearance|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|49 / 49|
|CorrectedROI-BackgroundAppearance|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|50 / 50|
|CorrectedROI-BackgroundAppearance|boundary_MAE_seconds|-0.175748|[-0.660345,+0.192563]|50 / 50|
|CorrectedROI-ShuffledBoxes|v|-0.354302|[-0.922338,+0.180048]|50 / 50|
|CorrectedROI-ShuffledBoxes|t|-0.965645|[-2.243948,+0.246880]|50 / 50|
|CorrectedROI-ShuffledBoxes|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|49 / 49|
|CorrectedROI-ShuffledBoxes|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|50 / 50|
|CorrectedROI-ShuffledBoxes|boundary_MAE_seconds|0.039444|[-0.166281,+0.220462]|50 / 50|
|CorrectedROI-RandomRegion|v|0.046772|[-1.012234,+1.183840]|50 / 50|
|CorrectedROI-RandomRegion|t|-0.063667|[-2.462848,+2.736748]|50 / 50|
|CorrectedROI-RandomRegion|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|49 / 49|
|CorrectedROI-RandomRegion|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|50 / 50|
|CorrectedROI-RandomRegion|boundary_MAE_seconds|-0.386577|[-1.208756,+0.142836]|50 / 50|
|RandomRegion-SourceROI|v|-0.527095|[-1.906710,+0.625689]|50 / 50|
|RandomRegion-SourceROI|t|-0.915702|[-3.634572,+1.402455]|50 / 50|
|RandomRegion-SourceROI|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|49 / 49|
|RandomRegion-SourceROI|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|50 / 50|
|RandomRegion-SourceROI|boundary_MAE_seconds|0.447632|[-0.071878,+1.273951]|50 / 50|

## Severe harm and rescue (strictly over5pp, union once)

|Pair|v gain/harm/tie|Severe t/v/union|Rescue union|
|---|---|---|---:|
|CorrectedROI-SourceROI|19/24/85|14/2/14|6|
|CorrectedROI-GlobalAppearance|26/26/76|15/3/15|6|
|CorrectedROI-BackgroundAppearance|36/33/59|18/5/18|12|
|CorrectedROI-ShuffledBoxes|19/24/85|12/3/12|7|
|CorrectedROI-RandomRegion|32/32/64|25/4/25|13|
|RandomRegion-SourceROI|35/33/60|17/4/17|23|

All intervals pointwise10000paired-parent bootstrap, queryequal metric means, not independent-frame/familywise/unknown-confirmation/equivalence claims. HC full uses1902queries/227parents and Vid10303/732; regional128parent panels are previously exposed. <=5s/long GT groups are post-seal analysis only; missing observations not0. All observed-frame metrics include all GT-supported native observed frames, including outside predicted events; do not equate every observed loss with tube-inside harm. Current observed-anchor contract remains unchanged. No method selection from results, no claim of independent action-semantic recognition even if local evidence wins.
