# Root reviewer control: combination, vidstg

Whole10303queries/732parents sealed, independent official CPU score of ONLY the new arm(10303checks,maxerror2.99760216649e-15), separate input/prediction/receipt/allcoldarray/reusedcontrol/allmean audit passed. Existing controls are reused unchanged, not rescored. Root independently rederived75 paired/group bootstrap statistics exactly. No new Source/expert/GPU/model/optimizer; inherited acquisition nonzero, cached CPU clocks not deployment.

|Arm|Expert inputs needed|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---|---:|---:|---:|---:|---:|
|Source|no|31.728454|11.138484|12.831214|3.348539|33.025378|
|SourceTemporal|no|34.908528|12.224898|15.005338|4.387072|33.025378|
|SimpleCombination|yes|34.908528|14.709589|19.819470|8.696496|35.542132|
|InitializedTemporal|yes|34.918192|13.948734|18.722702|8.230612|35.000813|
|Full|yes|34.918192|14.558151|19.411822|8.463554|35.785762|

SourceTemporal: native Source boxes and Source-box pre-text appearance, same q0/kappa0/beta1/one prototype update/jointMAP; no expert correction/transport/DP. SimpleCombination: strict raw-eight expert center/logsize interpolation with nearest-expert endpoint holding, paired with SourceTemporal own interval; no E tails/transport/DP. InitializedTemporal: original shared-corrected E path and E-ROI temporal interval, no spatial transport/DP. Full: original shared-corrected-ROI current finalDP, no first auxiliary DP. All expert configurations use the original same8 observations. Current remains soft E/expert at observed frames; this is not a hard-anchor repair or intended-hard-anchor validation.

## Existing full-domain fixed same refined interval (no new scoring)

|Spatial trajectory|tIoU %|vIoU %|
|---|---:|---:|
|Source|34.918192|12.227364|
|PureExpertFixed|34.918192|14.704394|
|InitializedFixed|34.918192|13.948734|
|Full|34.918192|14.558151|

Per-query tIoU exactly identical for all four fixed-time paths. Source in this fixed-time table means original Source boxes with expert-derived Full time and is a diagnostic, not a no-expert system. InitializedFixed is the E path and shares Full time; strict pure interpolated boxes retain original anchors and nearest tails. The unchanged old interpolation/E-tail hybrid remains in its own historical files and is never relabeled pure.

SourceTemporal-versus-native Source includes the switch from original offset-hull readout to legal jointMAP as well as visual update. Existing SourceJointMAP and fixed-E visual-update controls are reused separately; total temporal gains are not attributed entirely to appearance or action semantics.

## All signed paired evidence

### all

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|Full-SimpleCombination|v|-0.151437|[-0.305699,-0.000675]|10303 / 732|
|Full-SimpleCombination|t|0.009665|[-0.116164,+0.132882]|10303 / 732|
|Full-SimpleCombination|observed_actual_sIoU|-4.054089|[-4.477224,-3.660726]|8507 / 715|
|Full-SimpleCombination|unobserved_actual_sIoU|0.243630|[-0.305862,+0.790250]|10300 / 732|
|Full-SimpleCombination|boundary_MAE_seconds|-0.013220|[-0.057389,+0.025766]|10303 / 732|
|SimpleCombination-SourceTemporal|v|2.484691|[+2.083793,+2.909491]|10303 / 732|
|SimpleCombination-SourceTemporal|t|0.000000|[+0.000000,+0.000000]|10303 / 732|
|SimpleCombination-SourceTemporal|observed_actual_sIoU|12.403987|[+11.442497,+13.378308]|8507 / 715|
|SimpleCombination-SourceTemporal|unobserved_actual_sIoU|2.516754|[+1.577533,+3.476892]|10300 / 732|
|SimpleCombination-SourceTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|10303 / 732|
|Full-InitializedTemporal|v|0.609417|[+0.471700,+0.755210]|10303 / 732|
|Full-InitializedTemporal|t|0.000000|[+0.000000,+0.000000]|10303 / 732|
|Full-InitializedTemporal|observed_actual_sIoU|4.735687|[+4.203801,+5.296365]|8507 / 715|
|Full-InitializedTemporal|unobserved_actual_sIoU|0.784949|[+0.541043,+1.041068]|10300 / 732|
|Full-InitializedTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|10303 / 732|
|InitializedTemporal-SourceTemporal|v|1.723836|[+1.392222,+2.064571]|10303 / 732|
|InitializedTemporal-SourceTemporal|t|0.009665|[-0.116164,+0.132882]|10303 / 732|
|InitializedTemporal-SourceTemporal|observed_actual_sIoU|3.614211|[+2.858251,+4.378506]|8507 / 715|
|InitializedTemporal-SourceTemporal|unobserved_actual_sIoU|1.975435|[+1.289495,+2.672464]|10300 / 732|
|InitializedTemporal-SourceTemporal|boundary_MAE_seconds|-0.013220|[-0.057389,+0.025766]|10303 / 732|
|SourceTemporal-Source|v|1.086414|[+0.902995,+1.278769]|10303 / 732|
|SourceTemporal-Source|t|3.180073|[+2.730764,+3.650768]|10303 / 732|
|SourceTemporal-Source|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|8507 / 715|
|SourceTemporal-Source|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|10300 / 732|
|SourceTemporal-Source|boundary_MAE_seconds|-0.288308|[-0.433407,-0.151597]|10303 / 732|
### short_le5s

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|Full-SimpleCombination|v|-0.160329|[-0.310729,-0.020332]|6060 / 588|
|Full-SimpleCombination|t|-0.027566|[-0.154578,+0.102475]|6060 / 588|
|Full-SimpleCombination|observed_actual_sIoU|-3.671992|[-4.249440,-3.131758]|4404 / 556|
|Full-SimpleCombination|unobserved_actual_sIoU|0.750407|[+0.092926,+1.383249]|6057 / 588|
|Full-SimpleCombination|boundary_MAE_seconds|-0.012371|[-0.051034,+0.023257]|6060 / 588|
|SimpleCombination-SourceTemporal|v|1.387327|[+1.001022,+1.802168]|6060 / 588|
|SimpleCombination-SourceTemporal|t|0.000000|[+0.000000,+0.000000]|6060 / 588|
|SimpleCombination-SourceTemporal|observed_actual_sIoU|10.755329|[+9.487510,+12.025080]|4404 / 556|
|SimpleCombination-SourceTemporal|unobserved_actual_sIoU|0.247694|[-0.877486,+1.364863]|6057 / 588|
|SimpleCombination-SourceTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|6060 / 588|
|Full-InitializedTemporal|v|0.686711|[+0.525217,+0.849912]|6060 / 588|
|Full-InitializedTemporal|t|0.000000|[+0.000000,+0.000000]|6060 / 588|
|Full-InitializedTemporal|observed_actual_sIoU|5.695955|[+4.882909,+6.522328]|4404 / 556|
|Full-InitializedTemporal|unobserved_actual_sIoU|1.056179|[+0.702797,+1.400718]|6057 / 588|
|Full-InitializedTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|6060 / 588|
|InitializedTemporal-SourceTemporal|v|0.540288|[+0.233157,+0.866944]|6060 / 588|
|InitializedTemporal-SourceTemporal|t|-0.027566|[-0.154578,+0.102475]|6060 / 588|
|InitializedTemporal-SourceTemporal|observed_actual_sIoU|1.387381|[+0.432893,+2.360172]|4404 / 556|
|InitializedTemporal-SourceTemporal|unobserved_actual_sIoU|-0.058079|[-0.863806,+0.753778]|6057 / 588|
|InitializedTemporal-SourceTemporal|boundary_MAE_seconds|-0.012371|[-0.051034,+0.023257]|6060 / 588|
|SourceTemporal-Source|v|-0.091194|[-0.221030,+0.037713]|6060 / 588|
|SourceTemporal-Source|t|-0.102081|[-0.417235,+0.217802]|6060 / 588|
|SourceTemporal-Source|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|4404 / 556|
|SourceTemporal-Source|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|6057 / 588|
|SourceTemporal-Source|boundary_MAE_seconds|0.300158|[+0.181980,+0.422425]|6060 / 588|
### long_gt5s

|Pair|Metric|Mean difference|95% parent CI|Scorable queries / parents|
|---|---|---:|---|---:|
|Full-SimpleCombination|v|-0.138739|[-0.417618,+0.144339]|4243 / 540|
|Full-SimpleCombination|t|0.062838|[-0.180383,+0.310264]|4243 / 540|
|Full-SimpleCombination|observed_actual_sIoU|-4.464216|[-4.960394,-3.973148]|4103 / 538|
|Full-SimpleCombination|unobserved_actual_sIoU|-0.479808|[-1.194320,+0.246740]|4243 / 540|
|Full-SimpleCombination|boundary_MAE_seconds|-0.014431|[-0.086875,+0.055783]|4243 / 540|
|SimpleCombination-SourceTemporal|v|4.051984|[+3.394305,+4.736042]|4243 / 540|
|SimpleCombination-SourceTemporal|t|0.000000|[+0.000000,+0.000000]|4243 / 540|
|SimpleCombination-SourceTemporal|observed_actual_sIoU|14.173592|[+13.012234,+15.353910]|4103 / 538|
|SimpleCombination-SourceTemporal|unobserved_actual_sIoU|5.755900|[+4.544781,+6.962701]|4243 / 540|
|SimpleCombination-SourceTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|4243 / 540|
|Full-InitializedTemporal|v|0.499025|[+0.288674,+0.715895]|4243 / 540|
|Full-InitializedTemporal|t|0.000000|[+0.000000,+0.000000]|4243 / 540|
|Full-InitializedTemporal|observed_actual_sIoU|3.704973|[+3.153698,+4.265726]|4103 / 538|
|Full-InitializedTemporal|unobserved_actual_sIoU|0.397760|[+0.140394,+0.672674]|4243 / 540|
|Full-InitializedTemporal|boundary_MAE_seconds|0.000000|[+0.000000,+0.000000]|4243 / 540|
|InitializedTemporal-SourceTemporal|v|3.414221|[+2.863053,+3.982845]|4243 / 540|
|InitializedTemporal-SourceTemporal|t|0.062838|[-0.180383,+0.310264]|4243 / 540|
|InitializedTemporal-SourceTemporal|observed_actual_sIoU|6.004403|[+5.059366,+6.983756]|4103 / 538|
|InitializedTemporal-SourceTemporal|unobserved_actual_sIoU|4.878332|[+3.982097,+5.812249]|4243 / 540|
|InitializedTemporal-SourceTemporal|boundary_MAE_seconds|-0.014431|[-0.086875,+0.055783]|4243 / 540|
|SourceTemporal-Source|v|2.768314|[+2.390654,+3.160431]|4243 / 540|
|SourceTemporal-Source|t|7.867759|[+7.030749,+8.734391]|4243 / 540|
|SourceTemporal-Source|observed_actual_sIoU|0.000000|[+0.000000,+0.000000]|4103 / 538|
|SourceTemporal-Source|unobserved_actual_sIoU|0.000000|[+0.000000,+0.000000]|4243 / 540|
|SourceTemporal-Source|boundary_MAE_seconds|-1.128775|[-1.399131,-0.877656]|4243 / 540|

## Severe harm and rescue (strictly over5pp, union once)

|Pair|v gain/harm/tie|Severe t/v/union|Rescue union|
|---|---|---|---:|
|Full-SimpleCombination|3590/3917/2796|748/858/1347|1359|
|SimpleCombination-SourceTemporal|4683/3143/2477|0/1364/1364|2643|
|Full-InitializedTemporal|3636/3101/3566|0/375/375|809|
|InitializedTemporal-SourceTemporal|4736/3031/2536|748/1259/1758|2851|
|SourceTemporal-Source|4137/2655/3511|1531/526/1535|3152|

All intervals pointwise10000paired-parent bootstrap, queryequal metric means, not independent-frame/familywise/unknown-confirmation/equivalence claims. HC full uses1902queries/227parents and Vid10303/732; regional128parent panels are previously exposed. <=5s/long GT groups are post-seal analysis only; missing observations not0. All observed-frame metrics include all GT-supported native observed frames, including outside predicted events; do not equate every observed loss with tube-inside harm. Current observed-anchor contract remains unchanged. No method selection from results, no claim of independent action-semantic recognition even if local evidence wins.
