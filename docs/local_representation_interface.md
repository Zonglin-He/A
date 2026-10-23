# Native local representation and temporal evidence interfaces

The shared feedback algorithm accepts local feature values, their original-image
spatial supports and physical times. It has no required motion branch, textfusion,
class vocabulary, fixed feature dimension, or shared feature space across models.
Similarity is only within one frozen model and one video. Each adapter must provide
actual spatial correspondence, padding and temporal mapping; compressed query
vectors cannot silently be relabelled as spatial patch tokens.

`interfaces/local.py` aggregates trajectory-constrained native local representations.
Regular grids preserve the selected TA area/reduction arithmetic exactly; irregular
supports use explicit intersection/cell-area weights. `interfaces/prototype.py`
uses the same one-round centered cosine-kernel/common16 feedback with soft temporal
membership. No trained universal projection or additional encoder is added.

`interfaces/temporal.py` separates native temporal evidence from this feedback.
It exposes a joint-grid adapter and a candidate-list contract requiring real priors,
coverage and native readout. It does NOT manufacture MLLM probabilities/candidates.
A point-only timestamp distribution cannot gain mass on new intervals by a
multiplicative update; require an explicit separately validated candidate mechanism.
Two alternating fitting folds are not independent observations/validation.

`adapters/tastvg.py` is the only implemented native cache bridge. TA's selected
runtime reads the original postfusion motion-memory suffix. That is a frozen TA
implementation choice; other representations need their own measured comparison.
Current full runtime stays on its exact selected plan. Generic contracts are an
isolated tested library and do not alter bound inference modules during execution.

The [fixed TA local-visual comparison](../results/local_visual_v2/ANALYSIS.md)
actually tests a broader choice: original projected2Dappearance BEFORE textfusion,
with the same prior/trajectory/kernel/MBR/spatial branch. Controls record pre3D and
post2D from the same Source forward to distinguish representation differences.
All64outputs seal before officialCPUlabels. There are no CG/TubeDETR/MLLM results
and no claim of automatic portability/performance from an interface alone.

The follow-up [bounded κ study](../results/local_visual_tune_v2/ANALYSIS.md) tests only4/16/64 and a oncefrozen disjoint-parent confirmation. It retains16, does not demonstrate a better parameter or solve generalization, and does not promote Native2D to runtime.
