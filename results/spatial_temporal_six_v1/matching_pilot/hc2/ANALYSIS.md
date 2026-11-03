# Root matching-propagation cache128 pilot: hc2

128 queries/128 original confirm parents; all prediction arrays sealed before128new official matching tube checks, then independent complete-grid/raw/parent/prediction/receipt SHA, all arrays cold replay, frame-IoU distance groups and means audit. No new Source/expert/model/optimizer/GPU/media. Source-only selection fixed temp.05/topK10/context20 before target output scoring; no target tuning. Current, native Source and pure scalars reused by audited identity, with Source native interval explicitly separate.

Matching adapts the pinned official VideoWalk label-propagation kernel to the SAME Source postfusion token grid and SAME8teacher softmask/rawbox/physical positions used by Current. It is not the original learned encoder reproduction. The source validation already found the best adapter inferior to interpolation in BOTH directions; this negative result remains part of the interpretation, not hidden behind large Current-versus-adapter differences.

SourceMatched/PureMatched/Matching/Current use exactly the same Current interval. Pure/Matching preserve all8rawexpert anchor outputs; Current keeps its original soft candidate-selection behavior, which can choose E at anchors. No anchor clamp or method alteration was applied. The full observed-frame audit documents the difference. Therefore this controls evidence INPUT, feature layer and event interval, but does not fix Current anchor OUTPUT to the same boxes. The observed/unobserved IoUs below use the same actual GT-supported native sampled frames; observed support is not restricted to the predicted event.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
|Source|46.509478|25.124253|42.968750|14.062500|53.067325|
|SourceMatched|54.968548|30.123874|49.218750|23.437500|53.067325|
|PureMatched|54.968548|29.346216|45.312500|17.187500|47.870415|
|Matching|54.968548|24.995983|36.718750|17.968750|39.570804|
|Current|54.968548|31.791298|50.781250|22.656250|54.637004|

|Comparison|Delta v pp [95% CI]|Delta unobserved IoU pp [95% CI]|v improve/harm/tie|Severe v / rescue union|
|---|---|---|---|---|
|Current-Matching|+6.795315 [+5.195769, +8.476287]|+15.066200 [+12.368079, +17.782197]|96/19/13|2/55|
|Matching-PureMatched|-4.350233 [-6.060294, -2.696393]|-8.299611 [-11.122445, -5.424016]|33/82/13|45/8|
|Current-PureMatched|+2.445082 [+1.434673, +3.499023]|+6.766589 [+4.205956, +9.395139]|71/44/13|6/32|

## Short/long queries

|Group|Queries|Comparison|Delta v pp [95% CI]|
|---|---:|---|---|
|short_le5s|42|Current-Matching|+4.717649 [+2.786964, +6.867665]|
|short_le5s|42|Matching-PureMatched|-2.836528 [-5.630569, -0.055521]|
|short_le5s|42|Current-PureMatched|+1.881121 [+0.228022, +3.824590]|
|long_gt5s|86|Current-Matching|+7.809989 [+5.690124, +10.082104]|
|long_gt5s|86|Matching-PureMatched|-5.089484 [-7.303918, -3.079871]|
|long_gt5s|86|Current-PureMatched|+2.720505 [+1.519033, +4.026621]|

## Same annotated native-frame distance support

Frame counts are support counts; queries/parents, not frames, define the paired bootstrap. Empty query/bin support remains unscorable, not0. Distances use the original8physical expert frame times. No old E-tail-hybrid distance curve is renamed pure.

|Scope/bin|Scorable queries|Annotated frames|Pure IoU %|Matching IoU %|Current IoU %|Current-Matching pp [95% CI]|Current-Pure pp [95% CI]|
|---|---:|---:|---:|---:|---|---|
|GT_event/all_distances|128|2582|47.870485|39.580871|54.662684|+15.081813 [+12.384251, +17.802076]|+6.792199 [+4.225487, +9.418430]|
|GT_event/bin0|127|771|55.933805|42.085135|55.737267|+13.652131 [+10.883824, +16.522459]|-0.196538 [-2.543740, +2.236939]|
|GT_event/bin1|126|1035|47.089881|39.227990|56.489632|+17.261643 [+13.980376, +20.642557]|+9.399751 [+6.344535, +12.626322]|
|GT_event/bin2|111|652|39.946514|38.917347|53.639775|+14.722428 [+11.438358, +18.185830]|+13.693261 [+9.252346, +18.301690]|
|GT_event/bin3|38|124|39.569954|39.630119|52.989500|+13.359382 [+6.724503, +20.473425]|+13.419546 [+4.705504, +22.712178]|
|GT_event/bin4|0|0|unscorable|unscorable|unscorable|unscorable|unscorable|
|all_unobserved/all_distances|128|2594|47.870415|39.570804|54.637004|+15.066200 [+12.368079, +17.782197]|+6.766589 [+4.205956, +9.395139]|
|all_unobserved/bin0|127|774|55.827653|42.047685|55.678785|+13.631100 [+10.871689, +16.480386]|-0.148868 [-2.464153, +2.256818]|
|all_unobserved/bin1|126|1040|47.426253|39.356830|56.585077|+17.228247 [+13.976400, +20.592154]|+9.158824 [+6.118368, +12.333179]|
|all_unobserved/bin2|111|656|39.940051|38.921622|53.656908|+14.735287 [+11.457487, +18.170995]|+13.716858 [+9.264934, +18.321883]|
|all_unobserved/bin3|38|124|39.569954|39.630119|52.989500|+13.359382 [+6.724503, +20.473425]|+13.419546 [+4.705504, +22.712178]|
|all_unobserved/bin4|0|0|unscorable|unscorable|unscorable|unscorable|unscorable|
|common_predicted_event/all_distances|116|1942|50.826174|41.443896|56.100236|+14.656341 [+11.917255, +17.406684]|+5.274063 [+3.119434, +7.608388]|
|common_predicted_event/bin0|112|673|57.023851|43.403464|56.667978|+13.264515 [+10.443164, +16.033152]|-0.355873 [-1.970181, +1.267648]|
|common_predicted_event/bin1|115|832|49.406897|40.528880|57.682483|+17.153603 [+13.651750, +20.837130]|+8.275586 [+5.215774, +11.505107]|
|common_predicted_event/bin2|94|397|40.866648|39.300681|54.831820|+15.531139 [+11.680043, +19.521645]|+13.965172 [+9.083987, +19.198625]|
|common_predicted_event/bin3|18|40|41.056312|44.056467|59.117992|+15.061525 [+3.694826, +27.051332]|+18.061680 [+5.031726, +31.666870]|
|common_predicted_event/bin4|0|0|unscorable|unscorable|unscorable|unscorable|unscorable|

Selected matching outputs zero-area boxes on 2153/7296 unobserved native frames; no E/Source fallback, skipped query or new threshold. The source-defined settings are retained despite the negative target comparison against pure interpolation. All signed negative/severe/rescue/R.3/R.5/shortlong/distance outcomes public. This is the old exposed128panel, not an independent unseen cohort, and is not the full1902/10303 experiment.

Root interpretation: Current exceeds this particular weak module adapter, but that does not establish superiority to a strong learned VideoWalk model or every matched-feature propagation strategy. Pure interpolation is a stronger comparison in these source and target pilots. Full-target acquisition is NOT launched from this report; it still requires a separate frozen numerical/execution plan, current-method/Source exact compatibility, single-use paid checkpoints, actual first2root bridges and sufficient-state retention. No cancellation of the user-authorized six-item scope or resumption of old queues is implied. Actual five-arm ready-input/end-to-end timing and real Source density remain pending. Cached wall is not deployment latency; inherited acquisition cost is nonzero. Queryequal10000paired-parent95%CIs are pointwise, not familywise or equivalence. No Source/E/features/teacher/Current hyperparameters, manuscript or frozen source changed.
