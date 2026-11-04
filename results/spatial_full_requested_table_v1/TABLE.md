# Full-list companion of the original128 comparison

| Spatial trajectory | HC vIoU | HC Unobs. IoU | Vid vIoU | Vid Unobs. IoU |
|---|---:|---:|---:|---:|
|Expert interpolation|25.18|47.26|14.70|35.54|
|Our path|27.35|52.94|14.56|35.79|
|Δ, pp|+2.17|+5.68|-0.15|+0.24|

HC1902/227parents; Vid10303/732parents. Same complete query roster, original8ReferDINO observations and actual Full interval per query. Exact user-selected definition is center-linear/log-width-height interpolation with nearest rawexpert tails, no E tails and no corner-linear substitution. Original128/domain are exact scalar subsets of these complete results, verified per-query t/v/unobserved IoU.

Unobserved IoU follows the same ORIGINAL128 definition: GT-supported native sampled frames excluding exact8observed frames, query-equal over scorable queries, not restricted to predicted event. HC1902 and Vid10300queries have support; Vid3null queries are excluded identically from both conditional IoU means, but all10303remain in primaryv. No dense/native support mixing.

All means and10000parent pointwise CIs independently reproduced from sealed previously audited full predictions/scores. Every original interpolation and Current file SHA re-read, sameFullinterval checked, nearest-tail log-size interpolation independently reconstructed, Full equals the latest separate full-recalculation arrays. No new Source/expert/model/GPU/GT scoring/prediction, no hardanchor fix or manuscript change. Existing full negative/positive results are preserved. Full v gain positiveHC; Vid v point estimate-.146243pp CI[-.294717,-.000487], upper bound very close0; Unobs+.243630CI[-.305862,.790250]crosses0. Exposed query roster/pointwise statistics, no target-based tuning. Cached verification wall is not deployment time; inherited acquisition nonzero.
