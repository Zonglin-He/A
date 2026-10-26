# Temporal pooling region isolation: hc2

Full 1902 queries / 227 parent videos. This is the exact requested Source-box appearance versus corrected-box appearance comparison, already independently officially scored and all-array audited as AuxSourceROI versus FullMAP in the selected structural study; the new namespace independently verifies the complete two-arm chain for every query. No new inference, experts, optimizer, GT scoring or invented predictions.

Only the temporal ROI changes. Both use the same original eight observations, pretext Source feature grids, 1.5 context, centering/normalization, kappa0/beta1/one update and main spatial geometry. Each arm reconstructs its own prototypes, posterior and jointMAP interval, and solves final DP in its own interval. This is NOT NoSharedCorrection, which changes spatial geometry too. All-query verification uses exact audited ROI sufficient features; four actual retained grids additionally re-pool both regions. Disposed original raw grids are not claimed recoverable.

|Temporal ROI|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|SourceBoxAppearance|49.689566|27.315220|41.114616|16.824395|
|CorrectedBoxAppearance|49.764548|27.349044|41.219769|16.876972|

Corrected minus Source ROI, percentage points, paired parent-video cluster10000bootstrap pointwise95CI:

- t: +0.074982 [-0.094384, +0.236031].
- v: +0.033824 [-0.046807, +0.114818].

Severe >5pp: t=88, v=27, union once=88; rescue union=86. All positive and negative outcomes and short/long groups are retained. Same candidate construction does not force identical final boxes: different intervals re-solve DP and may change paths. Internal sIoU/boundary checks remain in the report, outside the compact table.

Both tIoU and vIoU confidence intervals cross zero. These results do not support a stable average temporal-localization improvement caused by shared correction of the pooling boxes. They also do not establish equivalence or prove zero effect. The supported wording is limited to object-region appearance supporting temporal correction; this comparison does not justify an additional causal claim for shared-corrected ROI. No manuscript edits or parameter/default changes are made. All full targets were already exposed; this is not unseen confirmation.

Costs and inherited paid acquisition are kept separately in cost_segments.json; cached verification time is not deployment latency.
