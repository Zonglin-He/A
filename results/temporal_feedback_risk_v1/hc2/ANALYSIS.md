# Temporal feedback risk — hc2

All 1902 queries sealed before independent official scoring; all arrays and means independently rederived. Pure semantic groups are operational text-cue strata, not gold labels. All mixed/uncertain queries remain in overall results.

|Temporal control|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---:|---:|---:|---:|
|Source|40.141475|20.704678|29.022082|8.937960|
|SourceJointMAP|46.693293|24.039097|36.225026|11.934805|
|NoVisualUpdate|46.693293|25.847764|39.011567|15.036803|
|FullMAP|49.764548|27.349044|41.219769|16.876972|
|LengthMeanEvidence|46.483907|25.740517|38.380652|14.879075|
|TemperatureEvidence2|49.307396|27.051750|40.904311|16.929548|
|SmoothPrior3|46.326053|25.676727|38.696109|15.036803|
|TemperaturePrior2|46.693293|25.847764|39.011567|15.036803|

Primary mechanistic contrast is FullMAP−NoVisualUpdate. All parent-paired t/v/boundary/R.3/R.5 pointwise95%CIs, all signed tails, short/long splits and initial-quality bins are retained in JSON/TSV. Pure-prior temperature must have identical MAP and boxes; it is an invariant calibration control. Fixed tau2/kernel3/mean evidence are not target-tuned.

Action/appearance operational contrast: insufficient_pure_appearance_or_action_support, action n=401, appearance n=6. HC appearance coverage cannot establish a two-category conclusion. Joint-parent group-delta CI only reported with at least30queries and10parents in each category. Query-stratum differences are observational and can reflect event length/initial difficulty.

Severe native-good collapse count: 1; same-joint-decoder good-to-zero count: 1. Native decoder shift and appearance feedback are different components. Prior mass/confidence increase associated with wrong intervals is not causal proof of self-reinforcement. Pre-text features need not encode the full language event; these contrasts also cannot prove they contain no action information.

Cached conditional replay elapsed is not process CPU or uncached deployment. Original Source/expert acquisition remains nonzero; no main method, paper, score-based parameter selection or paused task is changed.
