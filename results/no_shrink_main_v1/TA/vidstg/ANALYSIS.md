# Selected no-shrinkage TA main: vidstg

10303 queries / 732 parents. All NoShrink predictions were fully sealed, officially scored and exact-replay audited before this read-only export. Current FullMAP is exactly that kappa0 arm, with its own MAP interval and actual interval-conditioned spatial DP. Source remains the original native per-offset MAP and physical hull.

User explicitly selected this main method after both full results were exposed. This is a post-exposure method selection, not new independent confirmation. Historical kappa64 results and failures remain immutable.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|
|---|---:|---:|---:|---:|---|
|Source|31.728454|11.138484|12.831214|3.348539|0/0/0|
|SourceJointMAP|33.450806|11.641627|13.345627|3.630011|1015/272/1015|
|FullMAP|34.927982|14.548371|19.450645|8.492672|1487/1221/2238|
|LegacyKappa64|34.664067|14.433693|18.926526|8.317966|1139/1154/1934|

Mean gains do not remove negative tails. Both domains have more severe losses versus native Source than the historical kappa64 FullMAP; all signed differences, rescues, observed/unobserved changes and short/long pointwise parent-cluster CIs remain published. No new model, Source, teacher or GT scoring in this export. Cached replay times are not deployment latency.
