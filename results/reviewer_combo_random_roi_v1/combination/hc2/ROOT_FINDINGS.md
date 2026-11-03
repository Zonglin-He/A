# Root reviewer control: combination, hc2

Whole1902queries/227parents sealed, independent official CPU score of ONLY the new arm(1902checks,maxerror1.11022302463e-15), separate input/prediction/receipt/allcoldarray/reusedcontrol/allmean audit passed. Existing controls are reused unchanged, not rescored. Root independently rederived75 paired/group bootstrap statistics exactly. No new Source/expert/GPU/model/optimizer; inherited acquisition nonzero, cached CPU clocks not deployment.

|Arm|Expert inputs needed|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---|---:|---:|---:|---:|---:|
|Source|no|40.141475|20.704678|29.022082|8.937960|51.049723|
|SourceTemporal|no|49.689566|25.558305|39.221872|14.616193|51.049723|
|SimpleCombination|yes|49.689566|25.140858|37.802313|13.827550|47.262804|
|InitializedTemporal|yes|49.764548|26.489749|40.063091|15.772871|51.966189|
|Full|yes|49.764548|27.349044|41.219769|16.876972|52.942787|

SourceTemporal: native Source boxes and Source-box pre-text appearance, same q0/kappa0/beta1/one prototype update/jointMAP; no expert correction/transport/DP. SimpleCombination: strict raw-eight expert center/logsize interpolation with nearest-expert endpoint holding, paired with SourceTemporal own interval; no E tails/transport/DP. InitializedTemporal: original shared-corrected E path and E-ROI temporal interval, no spatial transport/DP. Full: original shared-corrected-ROI current finalDP, no first auxiliary DP. All expert configurations use the original same8 observations. Current remains soft E/expert at observed frames; this is not a hard-anchor repair or intended-hard-anchor validation.

## Existing full-domain fixed same refined interval (no new scoring)

|Spatial trajectory|tIoU %|vIoU %|
|---|---:|---:|
|Source|49.764548|25.604197|
|PureExpertFixed|49.764548|25.177364|
|InitializedFixed|49.764548|26.489749|
|Full|49.764548|27.349044|

Per-query tIoU exactly identical for all four fixed-time paths. Source in this fixed-time table means original Source boxes with expert-derived Full time and is a diagnostic, not a no-expert system. InitializedFixed is the E path and shares Full time; strict pure interpolated boxes retain original anchors and nearest tails. The unchanged old interpolation/E-tail hybrid remains in its own historical files and is never relabeled pure.

SourceTemporal-versus-native Source includes the switch from original offset-hull readout to legal jointMAP as well as visual update. Existing SourceJointMAP and fixed-E visual-update controls are reused separately; total temporal gains are not attributed entirely to appearance or action semantics.

## All signed paired evidence

### all

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|Full-SimpleCombination|v|2.208185|[+1.878230,+2.553212]|1902 / 227|
|Full-SimpleCombination|t|0.074982|[-0.094384,+0.236031]|1902 / 227|
|Full-SimpleCombination|observed_actual_sIoU|-3.581774|[-4.253204,-2.914003]|1880 / 226|
|Full-SimpleCombination|unobserved_actual_sIoU|5.679982|[+4.841201,+6.569601]|1902 / 227|
|Full-SimpleCombination|boundary_MAE_seconds|-0.002250|[-0.015483,+0.012022]|1902 / 227|
|SimpleCombination-SourceTemporal|v|-0.417447|[-1.032160,+0.206990]|1902 / 227|
|SimpleCombination-SourceTemporal|t|0.000000|[+0.000000,+0.000000]|1902 / 227|
|SimpleCombination-SourceTemporal|observed_actual_sIoU|8.319039|[+7.168853,+9.476111]|1880 / 226|
|SimpleCombination-SourceTemporal|unobserved_actual_sIoU|-3.786919|[-4.846088,-2.721960]|1902 / 227|
|SimpleCombination-SourceTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|1902 / 227|
|Full-InitializedTemporal|v|0.859295|[+0.502377,+1.226670]|1902 / 227|
|Full-InitializedTemporal|t|0.000000|[+0.000000,+0.000000]|1902 / 227|
|Full-InitializedTemporal|observed_actual_sIoU|3.506529|[+2.779059,+4.237851]|1880 / 226|
|Full-InitializedTemporal|unobserved_actual_sIoU|0.976598|[+0.475310,+1.493149]|1902 / 227|
|Full-InitializedTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|1902 / 227|
|InitializedTemporal-SourceTemporal|v|0.931444|[+0.621421,+1.243332]|1902 / 227|
|InitializedTemporal-SourceTemporal|t|0.074982|[-0.094384,+0.236031]|1902 / 227|
|InitializedTemporal-SourceTemporal|observed_actual_sIoU|1.230735|[+0.654206,+1.821503]|1880 / 226|
|InitializedTemporal-SourceTemporal|unobserved_actual_sIoU|0.916466|[+0.382726,+1.434468]|1902 / 227|
|InitializedTemporal-SourceTemporal|boundary_MAE_seconds|-0.002250|[-0.015483,+0.012022]|1902 / 227|
|SourceTemporal-Source|v|4.853626|[+4.274975,+5.455478]|1902 / 227|
|SourceTemporal-Source|t|9.548090|[+8.514222,+10.620953]|1902 / 227|
|SourceTemporal-Source|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|1880 / 226|
|SourceTemporal-Source|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|1902 / 227|
|SourceTemporal-Source|boundary_MAE_seconds|-0.251712|[-0.317631,-0.185554]|1902 / 227|
### short_le5s

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|Full-SimpleCombination|v|1.966138|[+1.422809,+2.532865]|544 / 176|
|Full-SimpleCombination|t|0.086879|[-0.227539,+0.394597]|544 / 176|
|Full-SimpleCombination|observed_actual_sIoU|-4.030228|[-5.522613,-2.531644]|523 / 174|
|Full-SimpleCombination|unobserved_actual_sIoU|6.256666|[+4.631368,+7.942627]|544 / 176|
|Full-SimpleCombination|boundary_MAE_seconds|-0.001267|[-0.028702,+0.028577]|544 / 176|
|SimpleCombination-SourceTemporal|v|-0.149226|[-1.102915,+0.785226]|544 / 176|
|SimpleCombination-SourceTemporal|t|0.000000|[+0.000000,+0.000000]|544 / 176|
|SimpleCombination-SourceTemporal|observed_actual_sIoU|9.660566|[+7.087121,+12.090391]|523 / 174|
|SimpleCombination-SourceTemporal|unobserved_actual_sIoU|-3.712705|[-5.860021,-1.636755]|544 / 176|
|SimpleCombination-SourceTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|544 / 176|
|Full-InitializedTemporal|v|1.174988|[+0.627267,+1.719856]|544 / 176|
|Full-InitializedTemporal|t|0.000000|[+0.000000,+0.000000]|544 / 176|
|Full-InitializedTemporal|observed_actual_sIoU|4.754890|[+3.091710,+6.430117]|523 / 174|
|Full-InitializedTemporal|unobserved_actual_sIoU|2.055634|[+0.951813,+3.124432]|544 / 176|
|Full-InitializedTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|544 / 176|
|InitializedTemporal-SourceTemporal|v|0.641924|[+0.221969,+1.045163]|544 / 176|
|InitializedTemporal-SourceTemporal|t|0.086879|[-0.227539,+0.394597]|544 / 176|
|InitializedTemporal-SourceTemporal|observed_actual_sIoU|0.875449|[-0.261569,+2.018712]|523 / 174|
|InitializedTemporal-SourceTemporal|unobserved_actual_sIoU|0.488327|[-0.451133,+1.412116]|544 / 176|
|InitializedTemporal-SourceTemporal|boundary_MAE_seconds|-0.001267|[-0.028702,+0.028577]|544 / 176|
|SourceTemporal-Source|v|1.617630|[+0.581645,+2.687702]|544 / 176|
|SourceTemporal-Source|t|3.452380|[+1.613816,+5.304164]|544 / 176|
|SourceTemporal-Source|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|523 / 174|
|SourceTemporal-Source|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|544 / 176|
|SourceTemporal-Source|boundary_MAE_seconds|0.173431|[+0.059351,+0.285028]|544 / 176|
### long_gt5s

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|Full-SimpleCombination|v|2.305147|[+1.912735,+2.716288]|1358 / 217|
|Full-SimpleCombination|t|0.070216|[-0.123939,+0.268090]|1358 / 217|
|Full-SimpleCombination|observed_actual_sIoU|-3.408936|[-4.156069,-2.653039]|1357 / 217|
|Full-SimpleCombination|unobserved_actual_sIoU|5.448969|[+4.489359,+6.421025]|1358 / 217|
|Full-SimpleCombination|boundary_MAE_seconds|-0.002644|[-0.017179,+0.011830]|1358 / 217|
|SimpleCombination-SourceTemporal|v|-0.524893|[-1.286011,+0.226881]|1358 / 217|
|SimpleCombination-SourceTemporal|t|0.000000|[+0.000000,+0.000000]|1358 / 217|
|SimpleCombination-SourceTemporal|observed_actual_sIoU|7.802002|[+6.511836,+9.099518]|1357 / 217|
|SimpleCombination-SourceTemporal|unobserved_actual_sIoU|-3.816648|[-5.051956,-2.592307]|1358 / 217|
|SimpleCombination-SourceTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|1358 / 217|
|Full-InitializedTemporal|v|0.732831|[+0.298838,+1.192886]|1358 / 217|
|Full-InitializedTemporal|t|0.000000|[+0.000000,+0.000000]|1358 / 217|
|Full-InitializedTemporal|observed_actual_sIoU|3.025400|[+2.221602,+3.905811]|1357 / 217|
|Full-InitializedTemporal|unobserved_actual_sIoU|0.544348|[+0.005296,+1.116435]|1358 / 217|
|Full-InitializedTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|1358 / 217|
|InitializedTemporal-SourceTemporal|v|1.047423|[+0.635229,+1.441094]|1358 / 217|
|InitializedTemporal-SourceTemporal|t|0.070216|[-0.123939,+0.268090]|1358 / 217|
|InitializedTemporal-SourceTemporal|observed_actual_sIoU|1.367666|[+0.640360,+2.062247]|1357 / 217|
|InitializedTemporal-SourceTemporal|unobserved_actual_sIoU|1.087974|[+0.438160,+1.730756]|1358 / 217|
|InitializedTemporal-SourceTemporal|boundary_MAE_seconds|-0.002644|[-0.017179,+0.011830]|1358 / 217|
|SourceTemporal-Source|v|6.149931|[+5.450730,+6.857661]|1358 / 217|
|SourceTemporal-Source|t|11.989966|[+10.792441,+13.217353]|1358 / 217|
|SourceTemporal-Source|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|1357 / 217|
|SourceTemporal-Source|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|1358 / 217|
|SourceTemporal-Source|boundary_MAE_seconds|-0.422020|[-0.495437,-0.345752]|1358 / 217|

## Severe harm and rescue (strictly over5pp, union once)

|Pair|v gain/harm/tie|Severe t/v/union|Rescue union|
|---|---|---|---:|
|Full-SimpleCombination|1072/646/184|88/138/198|476|
|SimpleCombination-SourceTemporal|858/868/176|0/461/461|422|
|Full-InitializedTemporal|822/775/305|0/198/198|260|
|InitializedTemporal-SourceTemporal|1277/443/182|88/160/222|331|
|SourceTemporal-Source|1143/507/252|345/218/353|989|

All intervals pointwise10000paired-parent bootstrap, queryequal metric means, not independent-frame/familywise/unknown-confirmation/equivalence claims. HC full uses1902queries/227parents and Vid10303/732; regional128parent panels are previously exposed. <=5s/long GT groups are post-seal analysis only; missing observations not0. All observed-frame metrics include all GT-supported native observed frames, including outside predicted events; do not equate every observed loss with tube-inside harm. Current observed-anchor contract remains unchanged. No method selection from results, no claim of independent action-semantic recognition even if local evidence wins.
