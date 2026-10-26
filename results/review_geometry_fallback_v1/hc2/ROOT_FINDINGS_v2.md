# hc2: whole-target fixed-time geometry safety controls

All1902 current full queries sealed before official new-arm scores, separate exact all-array/parent SHA/mean audit passed. These are descriptive posthoc review controls after original target results were exposed, not unseen parameter selection or a production change.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|40.141475|20.704678|29.022082|8.937960|
|FullMAP|49.764548|27.349044|41.219769|16.876972|
|ClipToImage|49.764548|27.359375|41.219769|16.876972|
|SourceFallbackFullOutside|49.764548|27.350534|41.219769|16.876972|

Parent-paired10000 bootstrap pointwise95%CI:

- ClipToImage-FullMAP: v+0.010331pp CI[0.006462071454057234, 0.01533219588498687]; t+0.000000 CI[0.0, 0.0]; severe t/v/union 0/0/0, rescue union0.
- SourceFallbackFullOutside-FullMAP: v+0.001491pp CI[0.0, 0.004626916122844672]; t+0.000000 CI[0.0, 0.0]; severe t/v/union 0/0/0, rescue union0.
- FullMAP-Source: v+6.644365pp CI[5.90258819125556, 7.408450541469321]; t+9.623073 CI[8.594747497321746, 10.683117070277321]; severe t/v/union 346/305/458, rescue union1129.
- short_le5s, n544, ClipToImage-FullMAP: v+0.008354 CI[0.0025715639465997914, 0.015112547696821842].
- short_le5s, n544, SourceFallbackFullOutside-FullMAP: v+0.004823 CI[0.0, 0.015137506097182375].
- short_le5s, n544, FullMAP-Source: v+3.434542 CI[2.062776663390684, 4.774484234527192].
- long_gt5s, n1358, ClipToImage-FullMAP: v+0.011123 CI[0.006209494553353862, 0.01778628368001266].
- long_gt5s, n1358, SourceFallbackFullOutside-FullMAP: v+0.000156 CI[0.0, 0.0005118113523162165].
- long_gt5s, n1358, FullMAP-Source: v+7.930185 CI[7.010995123797677, 8.893365106282317].

Geometrically modified queries across allnativeframes: {'clipped_frames': 1059, 'fallback_frames': 19}; modified frame counts {'clipped_frames': 10093, 'fallback_frames': 112}. Clipping uses only the image rectangle. Fully-outside fallback uses original Source box iff original Full has zero image intersection/nonpositive size. Current event remains unchanged. Source fallback keeps every unselected box bitwise unchanged; ClipToImage explicitly roundtrips all boxes through xyxy and can introduce ULP changes even for in-image boxes (independent magnitude diagnostics retained), so these controls cannot repair wrong temporal boundaries. Current original trajectories remain immutable and unclipped in the original result. No low-IoU/GT/identity rule or score-selected threshold. The two controls are fully reported and no new default is selected. Partial crossing, fully off-image, identity switches and occlusion are distinct; only the first two are exactly measured here.

No model/Source/expert/optimizer calls; Source/Full metrics inherited only with exact original complete-array/SHA/audit lineage. New two-arm official scores and independent array audit are separate processes. Conditional replay/official/audit wall clocks include IO/checks and are not complete deployment speed; original Source/eight-expert acquisitions remain nonzero. All signed negatives, severe tails/rescues and short/long strata retained.
