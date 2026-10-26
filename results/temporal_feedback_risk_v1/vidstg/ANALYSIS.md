# Temporal feedback risk — vidstg

All 10303 queries sealed before independent official scoring; all arrays and means independently rederived. Pure semantic groups are operational text-cue strata, not gold labels. All mixed/uncertain queries remain in overall results.

|Temporal control|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|31.728454|11.138484|12.831214|3.348539|
|SourceJointMAP|33.450806|11.641627|13.345627|3.630011|
|NoVisualUpdate|33.450806|13.871399|17.790935|7.590022|
|FullMAP|34.918192|14.558151|19.411822|8.463554|
|LengthMeanEvidence|33.418197|13.873600|17.664758|7.648258|
|TemperatureEvidence2|34.629628|14.435483|19.023585|8.250024|
|SmoothPrior3|33.221131|13.807963|17.771523|7.619140|
|TemperaturePrior2|33.450806|13.871399|17.790935|7.590022|

Primary mechanistic contrast is FullMAP−NoVisualUpdate. All parent-paired t/v/boundary/R.3/R.5 pointwise95%CIs, all signed tails, short/long splits and initial-quality bins are retained in JSON/TSV. Pure-prior temperature must have identical MAP and boxes; it is an invariant calibration control. Fixed tau2/kernel3/mean evidence are not target-tuned.

Action/appearance operational contrast: supported_operational_text_strata, action n=1464, appearance n=2443. HC appearance coverage cannot establish a two-category conclusion. Joint-parent group-delta CI only reported with at least30queries and10parents in each category. Query-stratum differences are observational and can reflect event length/initial difficulty.

Severe native-good collapse count: 1; same-joint-decoder good-to-zero count: 1. Native decoder shift and appearance feedback are different components. Prior mass/confidence increase associated with wrong intervals is not causal proof of self-reinforcement. Pre-text features need not encode the full language event; these contrasts also cannot prove they contain no action information.

Cached conditional replay elapsed is not process CPU or uncached deployment. Original Source/expert acquisition remains nonzero; no main method, paper, score-based parameter selection or paused task is changed.
