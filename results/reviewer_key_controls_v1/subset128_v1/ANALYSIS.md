# Reviewer controls after requested subset reduction

All previously completed full CPU reviewer experiments remain complete and unchanged. The user cancelled the remaining full-target acquisition and requested subsets instead. Both fixed128-query/128-parent domains now have complete five-region and uniform-gamma controls, independent new gamma scores, array/input/score/mean audits and root signed/tail/short-long/cost reports. There are no new Source/expert/model/optimizer calls or GPU use in the subset successor.

## Fixed cross-domain gamma

|Domain|Gamma|tIoU %|vIoU %|Delta t versus current [95% CI]|Delta v versus current [95% CI]|
|---|---:|---:|---:|---|---|
|hc2|0.5|54.968548|31.791298|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|
|hc2|1.0|54.309866|31.570678|-0.658682 [-1.265610, -0.124017]|-0.220620 [-0.787487, +0.328363]|
|vidstg|0.5|34.988905|14.741875|-0.213530 [-0.769131, +0.317275]|-0.597384 [-1.252646, +0.063371]|
|vidstg|1.0|35.202435|15.339259|+0.000000 [+0.000000, +0.000000]|+0.000000 [+0.000000, +0.000000]|

The current default is gamma.5 on HC and gamma1 on Vid; matching-default rows are identity controls, not a statistical equivalence finding. Both fixed values are reported without adopting a new default. Existing historical gamma selection used exposed target development data; this subset does not retroactively make that selection source-only or an unknown-target validation.

## Regional temporal evidence at identical E space

|Domain|Corrected minus comparator|Delta t pp [95% CI]|Delta v pp [95% CI]|
|---|---|---|---|
|hc2|SourceROI|+0.135495 [-0.479292, +0.770855]|-0.017017 [-0.232810, +0.217617]|
|hc2|GlobalAppearance|+0.488781 [-0.705089, +1.671980]|+0.407944 [-0.223229, +1.084151]|
|hc2|BackgroundAppearance|+1.469298 [-0.168165, +3.233836]|+1.274351 [+0.191013, +2.523046]|
|hc2|ShuffledBoxes|+1.404981 [+0.174056, +2.660755]|+0.543757 [-0.125456, +1.230776]|
|vidstg|SourceROI|-0.326168 [-1.025934, +0.461678]|-0.097110 [-0.355101, +0.149646]|
|vidstg|GlobalAppearance|-0.358556 [-1.334083, +0.685216]|-0.180251 [-0.528901, +0.121262]|
|vidstg|BackgroundAppearance|+0.131871 [-1.072934, +1.385601]|+0.148662 [-0.356672, +0.617729]|
|vidstg|ShuffledBoxes|-0.472838 [-1.148537, +0.194076]|-0.090393 [-0.381584, +0.197301]|

The previously published192-parent region analyses are retained. These128 cases are the original matched-pipeline panel nested within those192, chosen through the user-requested scope change, not an independent replication or a replacement selected by significance. A pointwise positive subset contrast does not establish universal object-local necessity; nulls are not equivalence proofs. All methods, eight observations, kappa0/beta1/context1.5 and the last trajectory DP stay fixed.

The discontinued full run retains all289 committed HC predictions and their hashes/costs,260 actual new Source calls and29 raw-cache reuses, with no uncommitted paid attempt at stop. It is not scored or presented as full-domain accuracy. Original acquisition costs remain nonzero, partial-run whole-worker clocks unavailable/null, nested Source/query/IO costs not additive or deployment latency.

All signed gains/losses, severe t/v union-once/rescues, short/long and recall/spatial/boundary metrics, raw scalar costs and all numerical nulls/negatives are in the domain reports/TSVs. CIs are10000 paired-parent percentile pointwise95%, not familywise. Public outputs contain no captions/media/GT coordinates/features/masks/weights/credentials. Old paused queues, production, paper/Fig1 and every old frozen code/plan remain unchanged. Final remote publication verification is a separate completion gate.
