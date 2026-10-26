# vidstg: whole-target fixed-time geometry safety controls

All10303 current full queries sealed before official new-arm scores, separate exact all-array/parent SHA/mean audit passed. These are descriptive posthoc review controls after original target results were exposed, not unseen parameter selection or a production change.

|Arm|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|31.728454|11.138484|12.831214|3.348539|
|FullMAP|34.918192|14.558151|19.411822|8.463554|
|ClipToImage|34.918192|14.629154|19.557410|8.570319|
|SourceFallbackFullOutside|34.918192|14.620222|19.450645|8.463554|

Parent-paired10000 bootstrap pointwise95%CI:

- ClipToImage-FullMAP: v+0.071003pp CI[0.060939163343269646, 0.08216795076369365]; t+0.000000 CI[0.0, 0.0]; severe t/v/union 0/0/0, rescue union9.
- SourceFallbackFullOutside-FullMAP: v+0.062071pp CI[0.045027617504478895, 0.0817330643783719]; t+0.000000 CI[0.0, 0.0]; severe t/v/union 0/0/0, rescue union36.
- FullMAP-Source: v+3.419668pp CI[3.0095824505869824, 3.851146170703118]; t+3.189738 CI[2.7535936953510007, 3.658697126846799]; severe t/v/union 1562/1222/2294, rescue union4460.
- short_le5s, n6060, ClipToImage-FullMAP: v+0.053779 CI[0.04258578061586983, 0.06704661324844027].
- short_le5s, n6060, SourceFallbackFullOutside-FullMAP: v+0.072282 CI[0.04872380848308648, 0.0991994131093341].
- short_le5s, n6060, FullMAP-Source: v+1.135805 CI[0.785926022061987, 1.5114475843031658].
- long_gt5s, n4243, ClipToImage-FullMAP: v+0.095603 CI[0.08017253783999322, 0.11209365309342616].
- long_gt5s, n4243, SourceFallbackFullOutside-FullMAP: v+0.047486 CI[0.028322917493088594, 0.07201482598291757].
- long_gt5s, n4243, FullMAP-Source: v+6.681559 CI[5.972095299072317, 7.420513187350265].

Geometrically modified queries across allnativeframes: {'clipped_frames': 7309, 'fallback_frames': 1029}; modified frame counts {'clipped_frames': 282090, 'fallback_frames': 33677}. Clipping uses only the image rectangle. Fully-outside fallback uses original Source box iff original Full has zero image intersection/nonpositive size. Current event remains unchanged. Source fallback keeps every unselected box bitwise unchanged; ClipToImage explicitly roundtrips all boxes through xyxy and can introduce ULP changes even for in-image boxes (independent magnitude diagnostics retained), so these controls cannot repair wrong temporal boundaries. Current original trajectories remain immutable and unclipped in the original result. No low-IoU/GT/identity rule or score-selected threshold. The two controls are fully reported and no new default is selected. Partial crossing, fully off-image, identity switches and occlusion are distinct; only the first two are exactly measured here.

No model/Source/expert/optimizer calls; Source/Full metrics inherited only with exact original complete-array/SHA/audit lineage. New two-arm official scores and independent array audit are separate processes. Conditional replay/official/audit wall clocks include IO/checks and are not complete deployment speed; original Source/eight-expert acquisitions remain nonzero. All signed negatives, severe tails/rescues and short/long strata retained.
