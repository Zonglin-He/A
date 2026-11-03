# Root matching-propagation cache128 pilot: vidstg

128 queries/128 original confirm parents; all prediction arrays sealed before128new official matching tube checks, then independent complete-grid/raw/parent/prediction/receipt SHA, all arrays cold replay, frame-IoU distance groups and means audit. No new Source/expert/model/optimizer/GPU/media. Source-only selection fixed temp.05/topK10/context20 before target output scoring; no target tuning. Current, native Source and pure scalars reused by audited identity, with Source native interval explicitly separate.

Matching adapts the pinned official VideoWalk label-propagation kernel to the SAME Source postfusion token grid and SAME8teacher softmask/rawbox/physical positions used by Current. It is not the original learned encoder reproduction. The source validation already found the best adapter inferior to interpolation in BOTH directions; this negative result remains part of the interpretation, not hidden behind large Current-versus-adapter differences.

SourceMatched/PureMatched/Matching/Current use exactly the same Current interval. Pure/Matching preserve all8rawexpert anchor outputs; Current keeps its original soft candidate-selection behavior, which can choose E at anchors. No anchor clamp or method alteration was applied. The full observed-frame audit documents the difference. Therefore this controls evidence INPUT, feature layer and event interval, but does not fix Current anchor OUTPUT to the same boxes. The observed/unobserved IoUs below use the same actual GT-supported native sampled frames; observed support is not restricted to the predicted event.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
|Source|32.121535|13.322903|16.406250|3.906250|36.412936|
|SourceMatched|35.202435|14.521920|18.750000|7.812500|36.412936|
|PureMatched|35.202435|14.912114|17.968750|8.593750|35.917760|
|Matching|35.202435|10.383367|12.500000|6.250000|22.853901|
|Current|35.202435|15.339259|19.531250|10.156250|38.207575|

|Comparison|Delta v pp [95% CI]|Delta unobserved IoU pp [95% CI]|v improve/harm/tie|Severe v / rescue union|
|---|---|---|---|---|
|Current-Matching|+4.955893 [+3.534741, +6.613561]|+15.353674 [+12.197359, +18.796274]|85/11/32|1/35|
|Matching-PureMatched|-4.528747 [-6.145104, -3.130821]|-13.063859 [-16.518401, -9.800622]|14/81/33|37/2|
|Current-PureMatched|+0.427145 [-0.420869, +1.374636]|+2.289815 [-0.463692, +5.083786]|50/47/31|6/10|

## Short/long queries

|Group|Queries|Comparison|Delta v pp [95% CI]|
|---|---:|---|---|
|short_le5s|78|Current-Matching|+3.198403 [+1.815245, +4.825367]|
|short_le5s|78|Matching-PureMatched|-3.012790 [-4.520601, -1.672658]|
|short_le5s|78|Current-PureMatched|+0.185613 [-0.475009, +0.782640]|
|long_gt5s|50|Current-Matching|+7.697576 [+5.019677, +10.819534]|
|long_gt5s|50|Matching-PureMatched|-6.893641 [-10.197254, -4.186966]|
|long_gt5s|50|Current-PureMatched|+0.803935 [-1.119644, +2.988197]|

## Same annotated native-frame distance support

Frame counts are support counts; queries/parents, not frames, define the paired bootstrap. Empty query/bin support remains unscorable, not0. Distances use the original8physical expert frame times. No old E-tail-hybrid distance curve is renamed pure.

|Scope/bin|Scorable queries|Annotated frames|Pure IoU %|Matching IoU %|Current IoU %|Current-Matching pp [95% CI]|Current-Pure pp [95% CI]|
|---|---:|---:|---:|---:|---|---|
|GT_event/all_distances|128|4564|35.903977|22.851446|38.167594|+15.316148 [+12.160481, +18.743195]|+2.263617 [-0.472809, +5.044946]|
|GT_event/bin0|115|1361|40.893801|23.826111|39.616386|+15.790274 [+12.339044, +19.395513]|-1.277416 [-4.408923, +1.778739]|
|GT_event/bin1|99|860|37.888010|23.569938|39.829480|+16.259542 [+12.222417, +20.525005]|+1.941470 [-1.438354, +5.406465]|
|GT_event/bin2|73|1065|36.079437|23.209064|40.292368|+17.083304 [+12.427865, +22.093564]|+4.212930 [+0.492384, +8.028385]|
|GT_event/bin3|43|771|32.433550|22.816684|37.423563|+14.606879 [+8.520212, +21.244588]|+4.990013 [-0.142346, +10.299435]|
|GT_event/bin4|18|507|24.518274|14.801812|32.385212|+17.583400 [+8.364983, +27.776978]|+7.866938 [+2.157242, +15.220797]|
|all_unobserved/all_distances|128|4580|35.917760|22.853901|38.207575|+15.353674 [+12.197359, +18.796274]|+2.289815 [-0.463692, +5.083786]|
|all_unobserved/bin0|115|1368|40.860586|23.835143|39.640874|+15.805731 [+12.364723, +19.414832]|-1.219712 [-4.342314, +1.830064]|
|all_unobserved/bin1|100|865|37.583425|23.334238|39.436514|+16.102275 [+12.018188, +20.266726]|+1.853088 [-1.577909, +5.354617]|
|all_unobserved/bin2|73|1066|36.079437|23.209064|40.359473|+17.150409 [+12.451154, +22.146849]|+4.280036 [+0.529597, +8.135010]|
|all_unobserved/bin3|43|772|32.433550|22.816684|37.423563|+14.606879 [+8.520212, +21.244588]|+4.990013 [-0.142346, +10.299435]|
|all_unobserved/bin4|18|509|24.580586|14.773158|32.363064|+17.589906 [+8.393114, +27.777141]|+7.782477 [+2.037265, +15.182969]|
|common_predicted_event/all_distances|106|2518|37.422785|24.031907|39.738493|+15.706586 [+12.205619, +19.414410]|+2.315708 [-0.140278, +5.043264]|
|common_predicted_event/bin0|97|980|40.192843|24.533306|40.890874|+16.357567 [+12.690080, +20.375156]|+0.698030 [-2.019108, +3.532900]|
|common_predicted_event/bin1|77|541|39.379383|25.399598|41.857578|+16.457980 [+11.906905, +21.264505]|+2.478195 [-0.581293, +5.960476]|
|common_predicted_event/bin2|51|496|40.682335|28.790467|45.192033|+16.401566 [+10.547633, +22.848474]|+4.509698 [+0.770461, +8.815551]|
|common_predicted_event/bin3|19|288|35.894791|24.853648|38.838135|+13.984486 [+5.275665, +24.209833]|+2.943343 [-0.524882, +7.797223]|
|common_predicted_event/bin4|11|213|24.855358|14.385198|27.921082|+13.535885 [+4.391449, +24.321364]|+3.065724 [-1.171782, +7.733015]|

Selected matching outputs zero-area boxes on 4751/12478 unobserved native frames; no E/Source fallback, skipped query or new threshold. The source-defined settings are retained despite the negative target comparison against pure interpolation. All signed negative/severe/rescue/R.3/R.5/shortlong/distance outcomes public. This is the old exposed128panel, not an independent unseen cohort, and is not the full1902/10303 experiment.

Root interpretation: Current exceeds this particular weak module adapter, but that does not establish superiority to a strong learned VideoWalk model or every matched-feature propagation strategy. Pure interpolation is a stronger comparison in these source and target pilots. Full-target acquisition is NOT launched from this report; it still requires a separate frozen numerical/execution plan, current-method/Source exact compatibility, single-use paid checkpoints, actual first2root bridges and sufficient-state retention. No cancellation of the user-authorized six-item scope or resumption of old queues is implied. Actual five-arm ready-input/end-to-end timing and real Source density remain pending. Cached wall is not deployment latency; inherited acquisition cost is nonzero. Queryequal10000paired-parent95%CIs are pointwise, not familywise or equivalence. No Source/E/features/teacher/Current hyperparameters, manuscript or frozen source changed.
