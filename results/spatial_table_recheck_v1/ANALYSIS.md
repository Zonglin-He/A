# Independent full-cohort recheck of the two spatial comparison tables

Direct user suspects implementation or imbalanced comparisons. All previously published science/source/plans/predictions/results remain unchanged. Fresh pure trajectories reconstructed from raw original expert payloads by an independent interpolation implementation; full-list reseal, independent dense pixel scoring and official evaluator checks, separate normalized-coordinate bracket-interpolation audit. No new Source/expert/model/GPU/optimizer, no method/paper change.

## Pairing and full-list coverage

|Dataset|Queries per arm|Parents|Missing/extra/duplicates|Raw expert observations|Full not retaining raw expert at observed frames|
|---|---:|---:|---|---:|---:|
|hc2|1902|227|0/0/0|15216|10556 (69.3743%)|
|vidstg|10303|732|0/0/0|82424|50874 (61.7223%)|

All7arms within a domain have exactly the same query roster, parent binding, native physical grid,8observation indices and annotated GT support; same fixed-Full interval comparison has per-query tIoU exactly equal, and simple combinations share SourceTemporal time exactly. Query-equal means retain every query, including zeros; observed/unobserved missing-support denominators identical across arms. HC1902vsVid10303 is a difference between benchmark sizes, not within-domain unpaired/reweighted comparison. Official dataset-specific pixel clipping is applied equally to both arms(HC clips negative coordinates before dense interpolation, Vid preserves original coordinates). No cross-domain pooled score is used.

## Equal-parent secondary aggregation check

Official main tables above/below remain queryequal. This supplementary calculation gives each original parent video equal weight rather than weighting it by number of queries; no query is removed and this is not a replacement metric.

|Dataset|Pair|Queryequal delta v pp|Equalparent delta v pp [95% CI]|
|---|---|---:|---|
|hc2|Full-ComboLog|+2.208185|+2.398414 [+1.947720, +2.873232]|
|hc2|Full-PureLogFixed|+2.171679|+2.398522 [+1.949941, +2.861344]|
|hc2|Full-ComboLinear|+1.792276|+1.947576 [+1.519128, +2.391192]|
|hc2|Full-PureLinearFixed|+1.756307|+1.943105 [+1.527213, +2.372050]|
|vidstg|Full-ComboLog|-0.151437|-0.229099 [-0.432186, -0.028118]|
|vidstg|Full-PureLogFixed|-0.146243|-0.184036 [-0.371655, +0.003599]|
|vidstg|Full-ComboLinear|-0.251872|-0.331392 [-0.532949, -0.135216]|
|vidstg|Full-PureLinearFixed|-0.246251|-0.285613 [-0.471182, -0.101095]|
## Fresh main simple-combination table

|Dataset|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|
|---|---|---:|---:|---:|---:|
|hc2|Source|40.141475|20.704678|29.022082|8.937960|
|hc2|SourceTemporal|49.689566|25.558305|39.221872|14.616193|
|hc2|ComboLog|49.689566|25.140858|37.802313|13.827550|
|hc2|Full|49.764548|27.349044|41.219769|16.876972|
|vidstg|Source|31.728454|11.138484|12.831214|3.348539|
|vidstg|SourceTemporal|34.908528|12.224898|15.005338|4.387072|
|vidstg|ComboLog|34.908528|14.709589|19.819470|8.696496|
|vidstg|Full|34.918192|14.558151|19.411822|8.463554|

ComboLog is the independently rebuilt center/log-size pure interpolation with nearest rawexpert endpoint tails and SourceTemporal own interval. Full uses original corrected E-ROI interval and unchanged saved DP path; their pipeline difference changes both interval evidence and path and is not exclusively propagation gain.

## Same Full interval spatial table and prespecified alternative definition

|Dataset|Rule|Pure interpolation vIoU %|Unchanged Full vIoU %|Full minus pure pp [95% paired-parent CI]|
|---|---|---:|---:|---|
|hc2|Original center/log-size|25.177364|27.349044|+2.171679 [+1.839911, +2.519693]|
|hc2|Alternative linear xyxy-corners|25.592737|27.349044|+1.756307 [+1.445449, +2.071917]|
|vidstg|Original center/log-size|14.704394|14.558151|-0.146243 [-0.294717, -0.000487]|
|vidstg|Alternative linear xyxy-corners|14.804402|14.558151|-0.246251 [-0.392809, -0.101615]|

Both pure rules receive only the same8rawexpert boxes/physicaltimes, retain anchors exactly and hold nearest expert outside anchor span. Primary np.interp center/logsize is independent of the historical loop function; tiny arithmetic roundoff is checked rather than pretending an independent implementation must be bitwise identical. The alternative linearly interpolates four corners(equivalently centers and linear sizes). It was declared before new GT scoring; neither rule is selected by target score and no old result is relabeled. This alternate definition is a diagnostic, not a method parameter sweep.

|Dataset|Pipeline comparison|Delta vIoU pp [95% CI]|
|---|---|---|
|hc2|Full-ComboLog|+2.208185 [+1.878230, +2.553212]|
|hc2|Full-ComboLinear|+1.792276 [+1.487068, +2.104577]|
|vidstg|Full-ComboLog|-0.151437 [-0.305699, -0.000675]|
|vidstg|Full-ComboLinear|-0.251872 [-0.406188, -0.102585]|

## Implementation contract remains a real issue

Pure interpolation reproduces every original expert box at all eight observed positions. Full is unchanged and may choose E instead of the expert, including inside its event, while outside the event it returns E. Every saved observed Full box is exactly E or rawexpert, never an unrelated coordinate. The table therefore compares the actual soft E/expert output strategy to hard-preserving pure interpolation; it does not compare two paths with fixed identical observed outputs or solely unobserved propagation. If intended method requires original experts hard-locked, current implementation violates that requirement. Correct pairing and metric reproduction do not repair or certify intended-method correctness. No final overwrite, DP constraint change, new method/path or paper/main-table replacement is performed. Existing source comments and prior observed-contract diagnosis remain unchanged.

## Independent numerical evidence

36615fresh official t/v/s checks across Full/ComboLog/PureLogFixed on every query; maxdifference to independent scorer 2.99760216648792e-15. All7arms scored independently, separate reference uses normalized coordinates/bracket search/explicit IoU instead of shared helpers:756168scalar checks,maxerror2.33146835171283e-15;219690cold-array checks,126mean checks,ROOT120paired/timegroup bootstrap statistics exact. Historical per-query outputs compared,maxoldmetricdifference3.66373598126302e-15. Rawteacher payload/receiptSHA and complete caption/grid/positions/RGBcontext checked; current provider producer when present, exact archived producer/rawcoordinates otherwise(31HC,0Vid), all original hashes preserved. No GT-based teacher/prediction choice, no new expert/Source or cache repair. Raw output bodies retained private.

All signed query results/severe t/vuniononce>5pp/rescues/short<=5s-long and pointwise10000parentbootstrap seed20261022 remain in report/TSVs. Previously exposed benchmarks; notunknownconfirmation/familywise/equivalence claims. Freshscore walls are experimental CPU evaluation/IO, not deployment latency; inherited Source/expert acquisition nonzero.6actual child stages and controller exited0. Protected unrelated drafts, alloldscience/predictions/negative outcomes and paused/cancelled scopes untouched.
