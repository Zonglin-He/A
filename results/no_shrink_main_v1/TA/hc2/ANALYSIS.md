# Selected no-shrinkage TA main: hc2

1902 queries / 227 parents. All NoShrink predictions were fully sealed, officially scored and exact-replay audited before this read-only export. Current FullMAP is exactly that kappa0 arm, with its own MAP interval and actual interval-conditioned spatial DP. Source remains the original native per-offset MAP and physical hull.

User explicitly selected this main method after both full results were exposed. This is a post-exposure method selection, not new independent confirmation. Historical kappa64 results and failures remain immutable.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs native Source|
|---|---:|---:|---:|---:|---|
|Source|40.141475|20.704678|29.022082|8.937960|0/0/0|
|SourceJointMAP|46.693293|24.039097|36.225026|11.934805|220/118/221|
|FullMAP|49.514796|27.214726|41.062040|16.614090|364/305/467|
|LegacyKappa64|48.586628|26.731488|40.378549|16.929548|267/264/396|

Mean gains do not remove negative tails. Both domains have more severe losses versus native Source than the historical kappa64 FullMAP; all signed differences, rescues, observed/unobserved changes and short/long pointwise parent-cluster CIs remain published. No new model, Source, teacher or GT scoring in this export. Cached replay times are not deployment latency.
